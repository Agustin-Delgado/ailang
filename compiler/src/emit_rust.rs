//! Backend: typed ail -> Rust source, compiled by rustc/LLVM.
//!
//! ail has value semantics with no references, so every value is owned by exactly one place.
//! The backend maps that onto Rust borrows: read-only params become `&T`, `mut` params `&mut T`
//! (which LLVM knows never alias), and copies are inserted only where a value is really duplicated.
//! A local whose final use is unambiguous is moved instead of cloned.

use crate::ast::*;
use crate::check::{BindId, BindKind, Checked, Res, pat_key};
use crate::diag::LineIndex;
use crate::types::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

pub struct Emitter<'a> {
    ck: &'a Checked,
    src: &'a str,
    moves: HashSet<u32>,
    tmp: usize,
    fn_name: String,
    /// Str locals emitted as borrowed views, and the bindings each one borrows from.
    views: HashMap<BindId, View>,
    view_roots: HashMap<BindId, Vec<BindId>>,
    /// Fns that return a view of one of their Str params: name -> (param index, kind).
    view_fns: HashMap<String, (usize, View)>,
    /// Set while emitting the body of a view fn: how its results are returned.
    cur_view: Option<View>,
    /// Expression ids whose code is fixed in advance (see `fused_update`).
    subst: HashMap<u32, String>,
    /// Pattern-key epoch of the fn being emitted (see `check::pat_key`).
    epoch: usize,
    /// `(at v i)` calls proven in range before the loop that contains them: id -> the element,
    /// read without a bounds check (see `for_unchecked`).
    unchecked: HashMap<u32, String>,
}

struct StructName {
    name: String,
}

/// How a borrowed string view is represented: `&str`, or a `Cow` when it is sometimes a copy
/// (`lower` of a string that has upper-case letters).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum View {
    Ref,
    Cow,
}

/// What the value of a block's last expression is turned into.
#[derive(Clone, Copy, PartialEq)]
enum Tail {
    Val,
    Unit,
    Ref,
    Cow,
}

/// The fns that are emitted: the non-generic ones and the concrete copies of generic ones.
fn concrete_fns<'p>(prog: &'p Program, ck: &'p Checked) -> Vec<&'p FnDef> {
    let mut v: Vec<&FnDef> = prog
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();
    v.extend(ck.instances.iter());
    v
}

pub fn mangle(n: &str) -> String {
    let mut o = String::new();
    for c in n.chars() {
        match c {
            '-' => o.push('_'),
            '?' => o.push_str("_q"),
            '.' => o.push_str("__"),
            // Instances of generic definitions: "(Pair I64 Str)", "largest<I64>".
            '(' | '<' => o.push_str("_L"),
            ')' | '>' => o.push_str("_R"),
            ' ' => o.push('_'),
            ',' => {}
            c => o.push(c),
        }
    }
    o
}

fn fn_name(n: &str) -> String {
    format!("f_{}", mangle(n))
}
fn ty_name(n: &str) -> String {
    format!("U_{}", mangle(n))
}

pub enum Mode {
    Run,
    Test,
    Web,
}

enum ModeKind {
    Native,
    Web,
}

/// `file_lines`: the line of `src` where each source file starts (the first is 1); `file` lists
/// the file names, one per line.
pub fn emit(prog: &Program, ck: &Checked, src: &str, file: &str, file_lines: &[usize], mode: Mode) -> String {
    let mode_kind = if matches!(mode, Mode::Web) {
        ModeKind::Web
    } else {
        ModeKind::Native
    };
    let mut em = Emitter {
        ck,
        src,
        moves: HashSet::new(),
        tmp: 0,
        fn_name: String::new(),
        views: HashMap::new(),
        view_roots: HashMap::new(),
        view_fns: HashMap::new(),
        cur_view: None,
        subst: HashMap::new(),
        epoch: 0,
        unchecked: HashMap::new(),
    };
    em.infer_view_fns(prog);
    let mut out = String::new();
    out.push_str("#![allow(warnings)]\n#![allow(unconditional_panic, arithmetic_overflow)]\n");
    for it in &prog.items {
        match it {
            Item::Struct(s) if s.type_params.is_empty() => em.emit_struct(&s.name, &mut out),
            Item::Enum(e) if e.type_params.is_empty() => em.emit_enum(&e.name, &mut out),
            Item::Const(c) => {
                let t = &ck.info.consts[&c.name];
                let v = em.val(&c.value);
                writeln!(out, "const C_{}: {} = {};", mangle(&c.name), rty(t), v).unwrap();
            }
            _ => {}
        }
    }
    let mut insts: Vec<&String> = ck.info.instances.keys().collect();
    insts.sort();
    for n in insts {
        if ck.info.structs.contains_key(n) {
            em.emit_struct(n, &mut out);
        } else {
            em.emit_enum(n, &mut out);
        }
    }
    for f in concrete_fns(prog, ck) {
        em.emit_fn(f, &mut out);
    }
    match mode {
        Mode::Run => {
            if let Some(main) = prog.items.iter().find_map(|i| match i {
                Item::Fn(f) if f.name == "main" => Some(f),
                _ => None,
            }) {
                let sig = &ck.info.fns["main"];
                let caps: Vec<String> = sig.params.iter().map(|(_, t, _)| rty(t)).collect();
                out.push_str("fn main() {\n    ail_rt::init(&AIL_LINES, AIL_FILE);\n");
                let call = format!("f_main({})", caps.join(", "));
                if sig.ret == Ty::Unit {
                    writeln!(out, "    {call};").unwrap();
                } else {
                    writeln!(
                        out,
                        "    if let Err(e) = {call} {{ ail_rt::main_error(&e); }}"
                    )
                    .unwrap();
                }
                out.push_str("    ail_rt::flush();\n}\n");
                let _ = main;
            } else {
                out.push_str("fn main() {}\n");
            }
        }
        Mode::Web => {
            let model = rty(&ck.info.fns["init"].ret);
            // Extra capability parameters of the app functions are supplied by the host.
            let caps = |f: &str, skip: usize| -> String {
                ck.info
                    .fns
                    .get(f)
                    .map(|s| {
                        s.params
                            .iter()
                            .skip(skip)
                            .map(|(_, t, _)| format!(", {}", rty(t)))
                            .collect::<String>()
                    })
                    .unwrap_or_default()
            };
            let mut ex = crate::runtime_web::exports(
                &model,
                ck.info.fns.contains_key("view"),
                ck.info.fns.contains_key("page"),
            );
            ex = ex.replace(
                "f_init()",
                &format!("f_init({})", caps("init", 0).trim_start_matches(", ")),
            );
            ex = ex.replace(
                "f_update(m, &ev)",
                &format!("f_update(m, &ev{})", caps("update", 2)),
            );
            ex = ex.replace(
                "ail_rt::Gfx);",
                &format!("ail_rt::Gfx{});", caps("view", 2)),
            );
            ex = ex.replace(
                "f_page(c.as_ref().expect(\"ail_init must run first\"))",
                &format!(
                    "f_page(c.as_ref().expect(\"ail_init must run first\"){})",
                    caps("page", 1)
                ),
            );
            if ck.info.is_copy(&ck.info.fns["init"].ret) {
                ex = ex
                    .replace("f_view(c.as_ref()", "f_view(*c.as_ref()")
                    .replace("f_page(c.as_ref()", "f_page(*c.as_ref()");
            }
            out.push_str(&ex);
        }
        Mode::Test => {
            let mut names = Vec::new();
            for (i, it) in prog.items.iter().enumerate() {
                if let Item::Test(t) = it {
                    em.moves = compute_moves(&t.body, ck);
                    em.epoch = 0;
                    em.views.clear();
                    em.view_roots.clear();
                    em.plan_views(&t.body);
                    em.fn_name = format!("test \"{}\"", t.name);
                    let body = em.block_as(&t.body, true);
                    writeln!(out, "fn t_{i}() {{\n    let _: () = {body};\n}}").unwrap();
                    names.push((i, t.name.clone()));
                }
            }
            out.push_str("fn main() {\n    ail_rt::init(&AIL_LINES, AIL_FILE);\n    let tests: &[(&str, fn())] = &[\n");
            for (i, n) in &names {
                writeln!(out, "        ({:?}, t_{i}),", n).unwrap();
            }
            out.push_str("    ];\n    ail_rt::run_tests(tests);\n}\n");
        }
    }
    let (body, table) = line_table(&out, src, file_lines);
    let mut fin = body;
    writeln!(fin, "static AIL_FILE: &str = {:?};", file).unwrap();
    writeln!(
        fin,
        "static AIL_LINES: [(u32, u32, u32); {}] = [{}];",
        table.len(),
        table
            .iter()
            .map(|(a, b, c)| format!("({a},{b},{c})"))
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    fin.push_str(crate::runtime::RUNTIME);
    fin.push_str(crate::runtime_net::RUNTIME_NET);
    if matches!(mode_kind, ModeKind::Web) {
        fin.push_str(crate::runtime_web::RUNTIME_WEB);
    } else {
        // Native builds of an app (e.g. `ail test` on a game): drawing does nothing.
        fin.push_str(crate::runtime_web::RUNTIME_WEB_STUB);
    }
    fin
}

/// Replace `/*@OFFSET*/` markers with a table (rust line -> ail line, col).
fn line_table(out: &str, src: &str, file_lines: &[usize]) -> (String, Vec<(u32, u32, u32)>) {
    let li = LineIndex::new(src);
    let mut table = Vec::new();
    let mut res = String::with_capacity(out.len());
    let mut cur: Option<usize> = None;
    for (n, line) in out.lines().enumerate() {
        let mut rest = line;
        let mut clean = String::new();
        let mut first_marker = None;
        while let Some(p) = rest.find("/*@") {
            clean.push_str(&rest[..p]);
            let end = rest[p..].find("*/").unwrap() + p;
            let off: usize = rest[p + 3..end].parse().unwrap();
            if first_marker.is_none() {
                first_marker = Some(off);
            }
            cur = Some(off);
            rest = &rest[end + 2..];
        }
        clean.push_str(rest);
        let mark = first_marker.or(cur);
        if let Some(off) = mark {
            let (l, c) = li.pos(off);
            let fi = file_lines.partition_point(|&s| s <= l).saturating_sub(1);
            let local = l + 1 - file_lines.get(fi).copied().unwrap_or(1);
            table.push(((n + 1) as u32, ((fi as u32) << 24) | local as u32, c as u32));
        }
        res.push_str(&clean);
        res.push('\n');
        if line.starts_with("fn ") || line.starts_with("}") {
            cur = None;
        }
    }
    (res, table)
}

pub fn rty(t: &Ty) -> String {
    match t {
        Ty::Int(k) => k.rust().to_string(),
        Ty::F32 => "f32".into(),
        Ty::F64 => "f64".into(),
        Ty::Bool => "bool".into(),
        Ty::Str => "String".into(),
        Ty::Unit => "()".into(),
        Ty::Never => "!".into(),
        Ty::Vec(t) => format!("Vec<{}>", rty(t)),
        Ty::Map(k, v) => format!("ail_rt::AilMap<{}, {}>", rty(k), rty(v)),
        Ty::Opt(t) => format!("Option<{}>", rty(t)),
        Ty::Res(a, b) => format!("Result<{}, {}>", rty(a), rty(b)),
        Ty::User(n) => ty_name(n),
        Ty::Cap(c) => format!("ail_rt::{}", c.name()),
        Ty::Chan(t, true) => format!("std::sync::mpsc::Sender<{}>", rty(t)),
        Ty::Chan(t, false) => format!("std::sync::mpsc::Receiver<{}>", rty(t)),
        Ty::Fn(..) | Ty::Param(_) | Ty::App(..) => "_".into(),
        Ty::Err => "_".into(),
    }
}

// ---------- move analysis ----------

struct UseInfo {
    id: u32,
    depth: u32,
    path: Vec<usize>,
}

/// Var uses that are the final, unambiguous use of an owned local: they move instead of cloning.
pub fn compute_moves(body: &[Expr], ck: &Checked) -> HashSet<u32> {
    let mut uses: HashMap<BindId, Vec<UseInfo>> = HashMap::new();
    let mut next_stmt = 0usize;
    let mut path = Vec::new();
    fn block(
        es: &[Expr],
        ck: &Checked,
        uses: &mut HashMap<BindId, Vec<UseInfo>>,
        next: &mut usize,
        path: &mut Vec<usize>,
        depth: u32,
    ) {
        for e in es {
            *next += 1;
            path.push(*next);
            expr(e, ck, uses, next, path, depth);
            path.pop();
        }
    }
    fn expr(
        e: &Expr,
        ck: &Checked,
        uses: &mut HashMap<BindId, Vec<UseInfo>>,
        next: &mut usize,
        path: &mut Vec<usize>,
        depth: u32,
    ) {
        match &e.kind {
            ExprKind::Var(_) => {
                if let Res::Local(b) = ck.res[e.id as usize] {
                    uses.entry(b).or_default().push(UseInfo {
                        id: e.id,
                        depth,
                        path: path.clone(),
                    });
                }
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Break
            | ExprKind::Continue => {}
            ExprKind::Call(_, _, args) => args
                .iter()
                .for_each(|a| expr(a.expr(), ck, uses, next, path, depth)),
            ExprKind::Let { value, .. } => expr(value, ck, uses, next, path, depth),
            ExprKind::Set(a, b) => {
                expr(b, ck, uses, next, path, depth);
                expr(a, ck, uses, next, path, depth);
            }
            ExprKind::If(c, t, f) => {
                expr(c, ck, uses, next, path, depth);
                block(std::slice::from_ref(t), ck, uses, next, path, depth);
                if let Some(f) = f {
                    block(std::slice::from_ref(f), ck, uses, next, path, depth);
                }
            }
            ExprKind::Do(es) => block(es, ck, uses, next, path, depth),
            ExprKind::While(c, b) => {
                expr(c, ck, uses, next, path, depth + 1);
                block(b, ck, uses, next, path, depth + 1);
            }
            ExprKind::For(_, _, it, b) => {
                match it {
                    ForIter::Range(a, c) => {
                        expr(a, ck, uses, next, path, depth);
                        expr(c, ck, uses, next, path, depth);
                    }
                    ForIter::Each(v) => expr(v, ck, uses, next, path, depth),
                }
                block(b, ck, uses, next, path, depth + 1);
            }
            ExprKind::Return(v) => {
                if let Some(v) = v {
                    expr(v, ck, uses, next, path, depth)
                }
            }
            ExprKind::Match(s, arms) => {
                expr(s, ck, uses, next, path, depth);
                for a in arms {
                    block(&a.body, ck, uses, next, path, depth);
                }
            }
            ExprKind::New(_, _, fs) => fs
                .iter()
                .for_each(|f| expr(&f.2, ck, uses, next, path, depth)),
            ExprKind::With(b, fs) => {
                expr(b, ck, uses, next, path, depth);
                fs.iter()
                    .for_each(|f| expr(&f.2, ck, uses, next, path, depth));
            }
            ExprKind::Field(b, _, _) => expr(b, ck, uses, next, path, depth),
            ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => {
                expr(x, ck, uses, next, path, depth)
            }
            ExprKind::Lambda(_, b) => block(b, ck, uses, next, path, depth + 1),
            ExprKind::Tasks(_, ts) => ts.iter().for_each(|t| block(&t.body, ck, uses, next, path, depth + 1)),
        }
    }
    block(body, ck, &mut uses, &mut next_stmt, &mut path, 0);
    let mut moves = HashSet::new();
    for (b, us) in uses {
        let bind = &ck.bindings[b];
        if !matches!(bind.kind, BindKind::Local { .. }) || ck.info.is_copy(&bind.ty) {
            continue;
        }
        let last = us.last().unwrap();
        if last.depth != bind.loop_depth {
            continue;
        }
        let p_last = *last.path.last().unwrap();
        let conflict = us[..us.len() - 1].iter().any(|w| {
            let q_last = *w.path.last().unwrap();
            w.path.contains(&p_last) || last.path.contains(&q_last)
        });
        if !conflict {
            moves.insert(last.id);
        }
    }
    moves
}

// ---------- emission ----------

impl<'a> Emitter<'a> {
    fn ty(&self, e: &Expr) -> &Ty {
        &self.ck.ty[e.id as usize]
    }
    fn is_copy(&self, t: &Ty) -> bool {
        self.ck.info.is_copy(t)
    }
    fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("__t{}", self.tmp)
    }
    fn mark(&self, e: &Expr) -> String {
        format!("/*@{}*/", e.span.start)
    }
    fn src_of(&self, e: &Expr) -> &str {
        &self.src[e.span.start..e.span.end]
    }

    fn emit_struct(&mut self, name: &str, out: &mut String) {
        let s = StructName { name: name.to_string() };
        let si = &self.ck.info.structs[&s.name];
        if crate::check::json_encodable(&self.ck.info, &Ty::User(s.name.clone()), &mut Vec::new()) {
            // (json-str value) writes the struct straight to text, fields in name order.
            let mut fs: Vec<&String> = si.fields.iter().map(|f| &f.0).collect();
            fs.sort();
            let mut body = String::from("o.push('{');");
            for (i, f) in fs.iter().enumerate() {
                let key = format!("{}\"{}\":", if i > 0 { "," } else { "" }, f);
                write!(body, " o.push_str({key:?}); ail_json::Enc::enc(&self.{}, o);", mangle(f)).unwrap();
            }
            body.push_str(" o.push('}');");
            writeln!(out, "impl ail_json::Enc for {} {{ fn enc(&self, o: &mut String) {{ {body} }} }}", ty_name(&s.name)).unwrap();
        }
        let copy = if self.ck.info.copy_types.contains(&s.name) {
            ", Copy"
        } else {
            ""
        };
        let copy = if self.ck.info.ord_types.contains(&s.name) {
            format!("{copy}, PartialOrd")
        } else {
            copy.to_string()
        };
        writeln!(
            out,
            "#[derive(Clone, Debug, PartialEq{copy})]\npub struct {} {{",
            ty_name(&s.name)
        )
        .unwrap();
        for (i, (f, t)) in si.fields.iter().enumerate() {
            let rt = if self.ck.info.is_boxed(&s.name, NO_VARIANT, i) {
                format!("Box<{}>", rty(t))
            } else {
                rty(t)
            };
            writeln!(out, "    pub {}: {},", mangle(f), rt).unwrap();
        }
        out.push_str("}\n");
    }

    fn emit_enum(&mut self, name: &str, out: &mut String) {
        let e = StructName { name: name.to_string() };
        let ei = &self.ck.info.enums[&e.name];
        let copy = if self.ck.info.copy_types.contains(&e.name) {
            ", Copy"
        } else {
            ""
        };
        let copy = if self.ck.info.ord_types.contains(&e.name) {
            format!("{copy}, PartialOrd")
        } else {
            copy.to_string()
        };
        writeln!(
            out,
            "#[derive(Clone, Debug, PartialEq{copy})]\npub enum {} {{",
            ty_name(&e.name)
        )
        .unwrap();
        for (vi, (v, ts)) in ei.variants.iter().enumerate() {
            if ts.is_empty() {
                writeln!(out, "    {},", mangle(v)).unwrap();
            } else {
                let fs: Vec<String> = ts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        if self.ck.info.is_boxed(&e.name, vi, i) {
                            format!("Box<{}>", rty(t))
                        } else {
                            rty(t)
                        }
                    })
                    .collect();
                writeln!(out, "    {}({}),", mangle(v), fs.join(", ")).unwrap();
            }
        }
        out.push_str("}\n");
    }

    fn param_rty(&self, t: &Ty, m: bool) -> String {
        if let Ty::Fn(ps, r) = t {
            let ps: Vec<String> = ps.iter().map(|p| self.conv_ty(p)).collect();
            format!("&impl Fn({}) -> {}", ps.join(", "), rty(r))
        } else if m {
            format!("&mut {}", rty(t))
        } else if *t == Ty::Str {
            "&str".into()
        } else if self.is_copy(t) {
            rty(t)
        } else {
            format!("&{}", rty(t))
        }
    }

    fn emit_fn(&mut self, f: &FnDef, out: &mut String) {
        let sig = &self.ck.info.fns[&f.name];
        self.epoch = self.ck.fn_epoch.get(&f.name).copied().unwrap_or(0);
        self.moves = compute_moves(&f.body, self.ck);
        self.views.clear();
        self.view_roots.clear();
        self.plan_views(&f.body);
        self.fn_name = f.name.clone();
        let vf = self.view_fns.get(&f.name).copied();
        self.cur_view = vf.map(|v| v.1);
        let params: Vec<String> = f
            .params
            .iter()
            .zip(sig.params.iter())
            .enumerate()
            .map(|(i, (p, (_, t, m)))| {
                let ty = if vf.is_some_and(|v| v.0 == i) {
                    "&'a str".to_string()
                } else {
                    self.param_rty(t, *m)
                };
                format!("{}: {}", self.local_name(&p.name), ty)
            })
            .collect();
        let (generics, ret) = match vf {
            Some((_, View::Ref)) => ("<'a>", "&'a str".to_string()),
            Some((_, View::Cow)) => ("<'a>", "std::borrow::Cow<'a, str>".to_string()),
            None => ("", rty(&sig.ret)),
        };
        writeln!(
            out,
            "fn {}{generics}({}) -> {} {{",
            fn_name(&f.name),
            params.join(", "),
            ret
        )
        .unwrap();
        for r in &f.requires {
            let c = self.val(r);
            writeln!(
                out,
                "    {}if !({c}) {{ ail_rt::contract(\"requires\", {:?}, {:?}); }}",
                self.mark(r),
                f.name,
                self.src_of(r)
            )
            .unwrap();
        }
        let tail = match vf {
            Some((_, View::Ref)) => Tail::Ref,
            Some((_, View::Cow)) => Tail::Cow,
            None if sig.ret == Ty::Unit => Tail::Unit,
            None => Tail::Val,
        };
        let body = self.block_tail(&f.body, tail);
        self.cur_view = None;
        if f.ensures.is_empty() {
            writeln!(out, "    let __r: {ret} = {body};\n    __r\n}}").unwrap();
        } else {
            writeln!(
                out,
                "    let __ret: {ret} = (|| -> {ret} {{ let __r: {ret} = {body}; __r }})();"
            )
            .unwrap();
            writeln!(out, "    {{ let v_result = &__ret;").unwrap();
            for en in &f.ensures {
                let c = self.val(en);
                writeln!(
                    out,
                    "    {}if !({c}) {{ ail_rt::contract(\"ensures\", {:?}, {:?}); }}",
                    self.mark(en),
                    f.name,
                    self.src_of(en)
                )
                .unwrap();
            }
            writeln!(out, "    }}\n    __ret\n}}").unwrap();
        }
    }

    fn local_name(&self, n: &str) -> String {
        format!("v_{}", mangle(n))
    }

    fn bind_place(&self, b: BindId) -> String {
        let bi = &self.ck.bindings[b];
        let n = self.local_name(&bi.name);
        match bi.kind {
            BindKind::Local { .. } => n,
            BindKind::Param { mutable: false } if self.is_copy(&bi.ty) => n,
            BindKind::Param { .. } | BindKind::Borrow => format!("(*{n})"),
            BindKind::BoxBorrow => format!("(**{n})"),
        }
    }

    /// A Rust lvalue for the expression, if it denotes a place.
    fn place(&mut self, e: &Expr) -> Option<String> {
        match &e.kind {
            ExprKind::Var(_) => match self.ck.res[e.id as usize] {
                Res::Local(b) if self.views.contains_key(&b) => None,
                Res::Local(b) => Some(self.bind_place(b)),
                _ => None,
            },
            ExprKind::Field(base, f, _) => {
                let p = self.place(base)?;
                Some(self.field_access(&p, base, f))
            }
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 => {
                if let Some(code) = self.unchecked.get(&e.id) {
                    return Some(code.clone());
                }
                let p = self.place(args[0].expr())?;
                let i = self.val(args[1].expr());
                Some(format!("{p}[ail_rt::ix({i})]"))
            }
            ExprKind::Copy(_) => None,
            _ => None,
        }
    }

    fn field_access(&self, base_code: &str, base: &Expr, f: &str) -> String {
        let Ty::User(sn) = self.ty(base) else {
            unreachable!()
        };
        let si = &self.ck.info.structs[sn];
        let idx = si.fields.iter().position(|x| x.0 == f).unwrap();
        if self.ck.info.is_boxed(sn, NO_VARIANT, idx) {
            format!("(*{base_code}.{})", mangle(f))
        } else {
            format!("{base_code}.{}", mangle(f))
        }
    }

    /// Owned value of the expression.
    pub fn val(&mut self, e: &Expr) -> String {
        if let (ExprKind::Var(_), Res::Local(b)) = (&e.kind, &self.ck.res[e.id as usize]) {
            if let Some(k) = self.views.get(b) {
                let n = self.local_name(&self.ck.bindings[*b].name);
                return if *k == View::Cow && self.moves.contains(&e.id) {
                    format!("{n}.into_owned()")
                } else {
                    format!("String::from(&{n}[..])")
                };
            }
        }
        if let Some(p) = self.place(e) {
            let t = self.ty(e).clone();
            if self.is_copy(&t) {
                return p;
            }
            if let ExprKind::Var(_) = e.kind {
                if self.moves.contains(&e.id) {
                    return p;
                }
            }
            if t == Ty::Str {
                // The place may be a borrowed `str` (a param), which has no `clone`.
                return format!("String::from(&{p}[..])");
            }
            return format!("{p}.clone()");
        }
        self.rvalue(e)
    }

    /// `&str` for a message argument: a literal stays static, so hot paths never allocate it.
    fn msg(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Str(s) => format!("{s:?}"),
            _ => self.rf(e),
        }
    }

    /// `&T` to the expression's value.
    fn rf(&mut self, e: &Expr) -> String {
        if self.ty(e) == &Ty::Str {
            return self.sv(e);
        }
        match self.place(e) {
            Some(p) => format!("&{p}"),
            None => format!("&({})", self.rvalue(e)),
        }
    }

    /// Something usable where Rust auto-refs (method receivers, format args): a place or an rvalue.
    fn pv(&mut self, e: &Expr) -> String {
        if self.ty(e) == &Ty::Str {
            return self.sv(e);
        }
        match self.place(e) {
            Some(p) => p,
            None => format!("({})", self.rvalue(e)),
        }
    }

    fn mut_place(&mut self, a: &Arg) -> String {
        self.place(a.expr())
            .expect("checker guarantees mutable places")
    }

    pub fn block(&mut self, es: &[Expr]) -> String {
        self.block_as(es, false)
    }

    /// A block; when `unit` is set, a non-Unit final value is discarded so the block is `()`.
    /// `(tasks ...)`: one scoped thread per task, std channels between them. A sender moves into
    /// its task, so the channel closes when that task ends and the receiver's loop finishes.
    fn emit_tasks(&mut self, id: u32, chans: &[ChanDecl], tasks: &[TaskDef]) -> String {
        let mut out = String::from("{ ail_rt::flush();
");
        for (i, c) in chans.iter().enumerate() {
            let ct = rty(&self.ck.chan_types[&id][i]);
            let m = mangle(&c.name);
            out.push_str(&format!("let (__tx_{m}, __rx_{m}) = std::sync::mpsc::channel::<{ct}>();
"));
        }
        out.push_str("std::thread::scope(|__s| {
");
        let saved = self.cur_view.take();
        for t in tasks {
            let mut ends = String::new();
            for (n, _) in &t.outs {
                ends.push_str(&format!("let {} = __tx_{}; ", self.local_name(n), mangle(n)));
            }
            for (n, _) in &t.ins {
                ends.push_str(&format!("let {} = __rx_{}; ", self.local_name(n), mangle(n)));
            }
            let body = self.block_as(&t.body, true);
            out.push_str(&format!("__s.spawn(|| {{ {ends}let _ = {body}; ail_rt::flush(); }});
"));
        }
        self.cur_view = saved;
        out.push_str("}); }");
        out
    }

    pub fn block_as(&mut self, es: &[Expr], unit: bool) -> String {
        self.block_tail(es, if unit { Tail::Unit } else { Tail::Val })
    }

    fn block_tail(&mut self, es: &[Expr], tail: Tail) -> String {
        let unit = tail == Tail::Unit;
        let mut o = String::from("{\n");
        for (i, e) in es.iter().enumerate() {
            let last = i + 1 == es.len();
            let m = self.mark(e);
            match &e.kind {
                ExprKind::Let { mutable, value, .. } => {
                    let b = self.ck.decl[&(e.id as usize)];
                    let bi = &self.ck.bindings[b];
                    let name = self.local_name(&bi.name);
                    let ann = if bi.ty.is_err() {
                        String::new()
                    } else {
                        format!(": {}", rty(&bi.ty))
                    };
                    let (ann, v) = match self.views.get(&b).copied() {
                        Some(View::Ref) => (": &str".to_string(), self.vref(value)),
                        Some(View::Cow) => (": std::borrow::Cow<str>".to_string(), self.vcow(value)),
                        None => (ann, self.val(value)),
                    };
                    let mu = if *mutable { "mut " } else { "" };
                    writeln!(o, "{m}let {mu}{name}{ann} = {v};").unwrap();
                    if last {
                        o.push_str("()\n");
                    }
                }
                _ => {
                    let code = match tail {
                        Tail::Ref if last => self.vref(e),
                        Tail::Cow if last => self.vcow(e),
                        _ => self.rvalue_or_place(e),
                    };
                    let discard = unit && !matches!(self.ty(e), Ty::Unit | Ty::Never);
                    if last && discard {
                        writeln!(o, "{m}let _ = {code};\n()").unwrap();
                    } else if last {
                        writeln!(o, "{m}{code}").unwrap();
                    } else {
                        writeln!(o, "{m}{code};").unwrap();
                    }
                }
            }
        }
        if es.is_empty() {
            o.push_str("()\n");
        }
        o.push('}');
        o
    }

    fn rvalue_or_place(&mut self, e: &Expr) -> String {
        self.val(e)
    }

    fn lit_int(&self, v: i128, t: &Ty) -> String {
        match t {
            Ty::F64 => format!("({v}f64)"),
            Ty::F32 => format!("({v}f32)"),
            Ty::Int(k) => format!("({v}{})", k.rust()),
            _ => format!("({v}i64)"),
        }
    }

    fn rvalue(&mut self, e: &Expr) -> String {
        let id = e.id as usize;
        if let Some(code) = self.subst.get(&e.id) {
            return code.clone();
        }
        let t = self.ty(e).clone();
        match &e.kind {
            ExprKind::Int(v) => self.lit_int(*v, &t),
            ExprKind::Float(v) => {
                let s = format!("{v:?}");
                format!("({}{})", s, if t == Ty::F32 { "f32" } else { "f64" })
            }
            ExprKind::Str(s) => format!("String::from({s:?})"),
            ExprKind::Var(_) => match &self.ck.res[id] {
                Res::Const(n) => format!("C_{}", mangle(n)),
                Res::Bool(b) => b.to_string(),
                Res::Unit => "()".into(),
                Res::None => "None".into(),
                Res::UnitVariant(en, vi) => {
                    let v = &self.ck.info.enums[en].variants[*vi].0;
                    format!("{}::{}", ty_name(en), mangle(v))
                }
                Res::Local(b) if self.views.contains_key(b) => self.val(e),
                Res::Local(b) => self.bind_place(*b),
                other => unreachable!("var resolved to {other:?}"),
            },
            ExprKind::Let { .. } => format!("{{ {} }}", self.block(std::slice::from_ref(e))),
            ExprKind::Set(place, value) => {
                if let Some(code) = self.append_in_place(place, value) {
                    return code;
                }
                let v = self.val(value);
                let p = self.place(place).expect("checked place");
                format!("{{ let __v = {v}; {p} = __v; }}")
            }
            ExprKind::If(c, th, el) => {
                let c = self.val(c);
                let unit = t == Ty::Unit;
                let tb = self.block_as(std::slice::from_ref(th), unit);
                match el {
                    Some(el) => {
                        let eb = self.block_as(std::slice::from_ref(el), unit);
                        format!("if {c} {tb} else {eb}")
                    }
                    None => format!("if {c} {{ let _ = {tb}; }}"),
                }
            }
            ExprKind::Do(es) => self.block_as(es, t == Ty::Unit),
            ExprKind::While(c, body) => {
                let c = self.val(c);
                let b = self.block(body);
                format!("while {c} {{ let _ = {b}; }}")
            }
            ExprKind::For(_, _, it, body) => {
                let b = self.ck.decl[&id];
                let var = self.local_name(&self.ck.bindings[b].name);
                if let ForIter::Range(a, z) = it {
                    // Bounds evaluated once, as Rust's a..z would; vectors the body pushes to on
                    // every iteration get their room up front.
                    let lo = self.val(a);
                    let hi = self.val(z);
                    let mut pre = format!("let __rlo: i64 = {lo}; let __rhi: i64 = {hi}; ");
                    for p in self.push_targets(body) {
                        pre.push_str(&format!("if __rhi > __rlo {{ let _ = {p}.try_reserve(((__rhi - __rlo) as u64).min(1 << 22) as usize); }} "));
                    }
                    let lp = match self.for_unchecked(b, &var, "__rlo", "__rhi", a, body) {
                        Some(code) => code,
                        None => {
                            let bd = self.block(body);
                            format!("for {var} in __rlo..__rhi {{ let _ = {bd}; }}")
                        }
                    };
                    return format!("{{ {pre}{lp} }}");
                }
                let head = match it {
                    ForIter::Range(a, z) => {
                        let a = self.val(a);
                        let z = self.val(z);
                        format!("{a}..{z}")
                    }
                    // (for w (split text sep) ...) streams the pieces instead of building a Vec.
                    ForIter::Each(v) => match &v.kind {
                        ExprKind::Call(n, _, args)
                            if n == "lines"
                                && args.len() == 1
                                && !matches!(self.ck.res.get(v.id as usize), Some(Res::UserFn(_)))
                                && !self.body_mutates(body, args[0].expr()) =>
                        {
                            let s = self.rf(args[0].expr());
                            let bd = self.block(body);
                            return format!("for __p in ail_rt::lines_iter({s}) {{ let {var}: &str = __p; let _ = {bd}; }}");
                        }
                        ExprKind::Call(n, _, args)
                            if n == "split"
                                && args.len() == 2
                                && !matches!(self.ck.res.get(v.id as usize), Some(Res::UserFn(_)))
                                && !self.body_mutates(body, args[0].expr()) =>
                        {
                            let s = self.rf(args[0].expr());
                            let sep = match &args[1].expr().kind {
                                ExprKind::Str(x) if x.len() == 1 => {
                                    let s = s.clone();
                                    let c = x.chars().next().unwrap();
                                    let bd = self.block(body);
                                    return format!(
                                        "for __p in {s}.split({c:?}) {{ let {var}: &str = __p; let _ = {bd}; }}"
                                    );
                                }
                                _ => self.rf(args[1].expr()),
                            };
                            let bd = self.block(body);
                            return format!(
                                "for __p in ail_rt::split_iter({s}, {sep}) {{ let {var}: &str = __p; let _ = {bd}; }}"
                            );
                        }
                        _ => format!("{}.iter()", self.pv(v)),
                    },
                };
                let bd = self.block(body);
                format!("for {var} in {head} {{ let _ = {bd}; }}")
            }
            ExprKind::Break => "break".into(),
            ExprKind::Continue => "continue".into(),
            ExprKind::Return(v) => match v {
                Some(v) if self.cur_view == Some(View::Ref) => format!("return {}", self.vref(v)),
                Some(v) if self.cur_view == Some(View::Cow) => format!("return {}", self.vcow(v)),
                Some(v) => format!("return {}", self.val(v)),
                None => "return ()".into(),
            },
            ExprKind::Match(scrut, arms) => {
                self.emit_match(scrut, arms, if t == Ty::Unit { Tail::Unit } else { Tail::Val })
            }
            ExprKind::New(_, _, fields) => {
                let Ty::User(name) = &t else { unreachable!("struct literal of type {t}") };
                let si = self.ck.info.structs[name].clone();
                let mut parts = Vec::new();
                for (f, _, v) in fields {
                    let idx = si.fields.iter().position(|x| &x.0 == f).unwrap();
                    let code = self.val(v);
                    let code = if self.ck.info.is_boxed(name, NO_VARIANT, idx) {
                        format!("Box::new({code})")
                    } else {
                        code
                    };
                    parts.push(format!("{}: {}", mangle(f), code));
                }
                format!("{} {{ {} }}", ty_name(name), parts.join(", "))
            }
            ExprKind::With(base, fields) => {
                let Ty::User(name) = self.ty(base).clone() else {
                    unreachable!()
                };
                let si = self.ck.info.structs[&name].clone();
                let mut parts = Vec::new();
                for (f, _, v) in fields {
                    let idx = si.fields.iter().position(|x| &x.0 == f).unwrap();
                    let code = self.val(v);
                    let code = if self.ck.info.is_boxed(&name, NO_VARIANT, idx) {
                        format!("Box::new({code})")
                    } else {
                        code
                    };
                    parts.push(format!("{}: {}", mangle(f), code));
                }
                let b = self.val(base);
                format!("{} {{ {}, ..{} }}", ty_name(&name), parts.join(", "), b)
            }
            ExprKind::Field(base, f, _) => {
                let b = self.rvalue(base);
                let code = self.field_access(&format!("({b})"), base, f);
                code
            }
            ExprKind::Try(inner) => format!("({})?", self.val(inner)),
            ExprKind::As(_, inner) => {
                let from = self.ty(inner).clone();
                let v = self.val(inner);
                cast(&v, &from, &t)
            }
            ExprKind::Copy(inner) => match self.place(inner) {
                Some(p) if t == Ty::Str => format!("String::from(&{p}[..])"),
                Some(p) => format!("{p}.clone()"),
                None => self.rvalue(inner),
            },
            ExprKind::Lambda(..) => unreachable!("lambdas are emitted where they are passed"),
            ExprKind::Tasks(chans, tasks) => self.emit_tasks(id as u32, chans, tasks),
            ExprKind::Call(name, _, args) => match self.ck.res[id].clone() {
                Res::LocalFn(b) => {
                    let f = self.local_name(&self.ck.bindings[b].name);
                    let parts: Vec<String> = args.iter().map(|a| self.conv_arg(a.expr())).collect();
                    format!("{f}({})", parts.join(", "))
                }
                Res::UserFn(n) => match self.view_fns.get(&n).copied() {
                    Some((_, View::Ref)) => format!("String::from({})", self.user_call(&n, args)),
                    Some((_, View::Cow)) => format!("({}).into_owned()", self.user_call(&n, args)),
                    None => self.user_call(&n, args),
                },
                Res::Variant(en, vi) => {
                    let v = self.ck.info.enums[&en].variants[vi].0.clone();
                    let mut parts = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        let code = self.val(a.expr());
                        parts.push(if self.ck.info.is_boxed(&en, vi, i) {
                            format!("Box::new({code})")
                        } else {
                            code
                        });
                    }
                    format!("{}::{}({})", ty_name(&en), mangle(&v), parts.join(", "))
                }
                Res::Builtin(_) => self.builtin(e, name, args),
                other => unreachable!("call resolved to {other:?}"),
            },
        }
    }

    fn user_call(&mut self, name: &str, args: &[Arg]) -> String {
        let sig = self.ck.info.fns[name].clone();
        let has_mut = sig.params.iter().any(|p| p.2);
        let mut pre = String::new();
        let mut parts = Vec::new();
        for (a, (_, pt, pm)) in args.iter().zip(sig.params.iter()) {
            let x = a.expr();
            if *pm {
                let p = self.place(x).expect("checked mutable place");
                parts.push(format!("&mut {p}"));
            } else if self.is_copy(pt) {
                let v = self.val(x);
                if has_mut {
                    let t = self.fresh();
                    write!(pre, "let {t} = {v}; ").unwrap();
                    parts.push(t);
                } else {
                    parts.push(v);
                }
            } else if let Ty::Fn(..) = pt {
                let cl = self.closure(x);
                parts.push(if matches!(x.kind, ExprKind::Var(_)) && matches!(self.ck.res[x.id as usize], Res::Local(_)) {
                    cl
                } else {
                    format!("&{cl}")
                });
            } else if *pt == Ty::Str && self.view_arg_ok(x, args, has_mut) {
                parts.push(self.sv(x));
            } else {
                match self.place(x) {
                    Some(p) => parts.push(format!("&{p}")),
                    None => {
                        let v = self.rvalue(x);
                        if has_mut {
                            let t = self.fresh();
                            write!(pre, "let {t} = {v}; ").unwrap();
                            parts.push(format!("&{t}"));
                        } else {
                            parts.push(format!("&({v})"));
                        }
                    }
                }
            }
        }
        let call = format!("{}({})", fn_name(name), parts.join(", "));
        if pre.is_empty() {
            call
        } else {
            format!("{{ {pre}{call} }}")
        }
    }

    /// The local a place-like expression reads from (`x`, `(. x f)`, `(at x i)`, `(get x k)`).
    fn root_of(&self, e: &Expr) -> Option<BindId> {
        match &e.kind {
            ExprKind::Var(_) => match self.ck.res[e.id as usize] {
                Res::Local(b) => Some(b),
                _ => None,
            },
            ExprKind::Field(b, _, _) => self.root_of(b),
            ExprKind::Call(n, _, args)
                if matches!(n.as_str(), "at" | "get" | "last") && !args.is_empty() =>
            {
                self.root_of(args[0].expr())
            }
            _ => None,
        }
    }

    /// True if any arm assigns to `b` or passes it as `(mut ...)`.
    fn arms_mutate(&self, arms: &[Arm], b: BindId) -> bool {
        let mut hit = false;
        for arm in arms {
            for x in &arm.body {
                crate::check::walk(x, &mut |e| match &e.kind {
                    ExprKind::Set(p, _) if self.root_of(p) == Some(b) => hit = true,
                    ExprKind::Call(_, _, args) => {
                        for a in args {
                            if let Arg::Mut(x, _) = a {
                                if self.root_of(x) == Some(b) {
                                    hit = true;
                                }
                            }
                        }
                    }
                    _ => {}
                });
            }
        }
        hit
    }

    /// True if `body` assigns to or passes as `(mut ...)` the variable `e` reads from.
    fn body_mutates(&self, body: &[Expr], e: &Expr) -> bool {
        // Any expression works as a source, as long as no variable it reads changes in the body
        // (the temporary it builds lives for the whole for statement).
        let mut roots = Vec::new();
        crate::check::walk(e, &mut |x| {
            if let (ExprKind::Var(_), Res::Local(b)) = (&x.kind, &self.ck.res[x.id as usize]) {
                roots.push(*b);
            }
        });
        let arm = Arm { pat: Pat::Wild(e.span), body: body.to_vec(), span: e.span };
        roots.iter().any(|b| self.arms_mutate(std::slice::from_ref(&arm), *b))
    }

    fn emit_match(&mut self, scrut: &Expr, arms: &[Arm], tail: Tail) -> String {
        // Pattern bindings are read-only views of the scrutinee. Borrow it unless an arm
        // mutates the variable it comes from; then the arms see a snapshot (value semantics).
        let mutated = self
            .root_of(scrut)
            .is_some_and(|b| self.arms_mutate(arms, b));
        let s = match &scrut.kind {
            ExprKind::Call(n, _, args)
                if !mutated
                    && (n == "get" || n == "last")
                    && self.place(args[0].expr()).is_some() =>
            {
                let base = self.pv(args[0].expr());
                if n == "get" {
                    format!("{base}.get({})", self.rf(args[1].expr()))
                } else {
                    format!("{base}.last()")
                }
            }
            _ if mutated => match self.place(scrut) {
                Some(p) => format!("&({p}.clone())"),
                None => self.rf(scrut),
            },
            _ => self.rf(scrut),
        };
        let st = self.ty(scrut).clone();
        let mut o = format!("match {s} {{\n");
        for arm in arms {
            let mut guards = Vec::new();
            let p = self.pat(&arm.pat, &st, &mut guards);
            let g = if guards.is_empty() {
                String::new()
            } else {
                format!(" if {}", guards.join(" && "))
            };
            let body = self.block_tail(&arm.body, tail);
            writeln!(o, "{p}{g} => {body},").unwrap();
        }
        if arms.is_empty() {
            o.push_str("_ => unreachable!(),\n");
        }
        o.push('}');
        o
    }

    fn pat(&mut self, p: &Pat, t: &Ty, guards: &mut Vec<String>) -> String {
        match p {
            Pat::Wild(_) => "_".into(),
            Pat::Bind(n, span) => {
                let b = self.ck.decl[&pat_key(*span, self.epoch)];
                let _ = n;
                self.local_name(&self.ck.bindings[b].name)
            }
            Pat::Int(v, _) => self
                .lit_int(*v, t)
                .trim_matches(|c| c == '(' || c == ')')
                .to_string(),
            Pat::Str(s, _) => {
                let v = self.fresh();
                guards.push(format!("&{v}[..] == {s:?}"));
                v
            }
            Pat::Ctor(name, subs, _) => match (name.as_str(), t) {
                ("true", _) => "true".into(),
                ("false", _) => "false".into(),
                ("none", _) => "None".into(),
                ("some", Ty::Opt(inner)) => format!("Some({})", self.pat(&subs[0], inner, guards)),
                ("ok", Ty::Res(a, _)) => format!("Ok({})", self.pat(&subs[0], a, guards)),
                ("err", Ty::Res(_, b)) => format!("Err({})", self.pat(&subs[0], b, guards)),
                (_, Ty::User(en)) => {
                    let (_, vn) = name.split_once('.').unwrap();
                    let ei = self.ck.info.enums[en].clone();
                    let vi = ei.variants.iter().position(|v| v.0 == vn).unwrap();
                    let path = format!("{}::{}", ty_name(en), mangle(vn));
                    if ei.variants[vi].1.is_empty() {
                        path
                    } else {
                        let ps: Vec<String> = subs
                            .iter()
                            .zip(ei.variants[vi].1.iter())
                            .map(|(sp, st)| self.pat(sp, st, guards))
                            .collect();
                        format!("{path}({})", ps.join(", "))
                    }
                }
                _ => unreachable!("pattern {name} on {t}"),
            },
        }
    }

    fn builtin(&mut self, e: &Expr, name: &str, args: &[Arg]) -> String {
        let t = self.ty(e).clone();
        let a = |i: usize| args[i].expr();
        let at = |s: &Self, i: usize| s.ty(args[i].expr()).clone();
        match name {
            "+" | "*" => {
                let parts: Vec<String> = args.iter().map(|x| self.val(x.expr())).collect();
                format!("({})", parts.join(&format!(" {name} ")))
            }
            "-" => {
                if args.len() == 1 {
                    let v = self.val(a(0));
                    match &t {
                        Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64) => {
                            format!("(0{} - {v})", k.rust())
                        }
                        _ => format!("(-{v})"),
                    }
                } else {
                    format!("({} - {})", self.val(a(0)), self.val(a(1)))
                }
            }
            "/" | "%" => format!("({} {name} {})", self.val(a(0)), self.val(a(1))),
            "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                let ot = at(self, 0);
                if self.is_copy(&ot) {
                    format!("({} {name} {})", self.val(a(0)), self.val(a(1)))
                } else {
                    format!("({} {name} {})", self.rf(a(0)), self.rf(a(1)))
                }
            }
            "and" | "or" => {
                let op = if name == "and" { " && " } else { " || " };
                let parts: Vec<String> = args.iter().map(|x| self.val(x.expr())).collect();
                format!("({})", parts.join(op))
            }
            "not" => format!("(!{})", self.val(a(0))),
            "wrapping-add" | "wrapping-sub" | "wrapping-mul" | "checked-add" | "checked-sub" | "checked-mul" => {
                let m = name.replace('-', "_");
                format!("({}).{m}({})", self.val(a(0)), self.val(a(1)))
            }
            "bit-and" => format!("({} & {})", self.val(a(0)), self.val(a(1))),
            "bit-or" => format!("({} | {})", self.val(a(0)), self.val(a(1))),
            "bit-xor" => format!("({} ^ {})", self.val(a(0)), self.val(a(1))),
            "shl" => format!(
                "ail_rt::ShiftAil::checked_shl_ail({}, ail_rt::icast::<_, u32>({}))",
                self.val(a(0)),
                self.val(a(1))
            ),
            "shr" => format!(
                "ail_rt::ShiftAil::checked_shr_ail({}, ail_rt::icast::<_, u32>({}))",
                self.val(a(0)),
                self.val(a(1))
            ),
            "abs" => match &t {
                Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64) => {
                    self.val(a(0))
                }
                _ => format!("({}).abs()", self.val(a(0))),
            },
            "min" | "max" => {
                if t.is_float() {
                    format!("({}).{name}({})", self.val(a(0)), self.val(a(1)))
                } else {
                    format!("std::cmp::{name}({}, {})", self.val(a(0)), self.val(a(1)))
                }
            }
            "pow" => {
                if t.is_int() {
                    format!("({}).pow({})", self.val(a(0)), self.val(a(1)))
                } else {
                    format!("({}).powf({})", self.val(a(0)), self.val(a(1)))
                }
            }
            "sqrt" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" | "exp" | "ln" => {
                format!("({}).{name}()", self.val(a(0)))
            }
            "atan2" => format!("({}).atan2({})", self.val(a(0)), self.val(a(1))),
            "str" => self.format(args),
            "print" | "println" | "eprintln" => {
                let f = self.format(&args[1..]);
                format!("ail_rt::{}(&{f})", name)
            }
            "len" => format!("({}.len() as i64)", self.pv(a(0))),
            "is-empty" => format!("{}.is_empty()", self.pv(a(0))),
            "at" if self.unchecked.contains_key(&e.id) => {
                let code = self.unchecked[&e.id].clone();
                if self.is_copy(&t) { code } else { format!("{code}.clone()") }
            }
            "at" => {
                let b = self.rvalue(a(0));
                let i = self.val(a(1));
                if self.is_copy(&t) {
                    format!("({b})[ail_rt::ix({i})]")
                } else {
                    format!("({b})[ail_rt::ix({i})].clone()")
                }
            }
            "vec" => {
                if args.is_empty() {
                    "Vec::new()".into()
                } else {
                    let parts: Vec<String> = args.iter().map(|x| self.val(x.expr())).collect();
                    format!("vec![{}]", parts.join(", "))
                }
            }
            "fill" => {
                let n = self.val(a(0));
                let x = self.val(a(1));
                format!("vec![{x}; ail_rt::count({n})]")
            }
            "push" => {
                let x = self.val(a(1));
                let p = self.mut_place(&args[0]);
                format!("{{ let __x = {x}; {p}.push(__x); }}")
            }
            "pop" => format!("{}.pop()", self.mut_place(&args[0])),
            "last" => format!("{}.last().cloned()", self.pv(a(0))),
            "insert-at" => {
                let i = self.val(a(1));
                let x = self.val(a(2));
                let p = self.mut_place(&args[0]);
                format!(
                    "{{ let __i = ail_rt::ix_incl({i}); let __x = {x}; {p}.insert(__i, __x); }}"
                )
            }
            "remove-at" => {
                let i = self.val(a(1));
                let p = self.mut_place(&args[0]);
                format!("{{ let __i = ail_rt::ix({i}); {p}.remove(__i) }}")
            }
            "swap" => {
                let i = self.val(a(1));
                let j = self.val(a(2));
                let p = self.mut_place(&args[0]);
                format!(
                    "{{ let __i = ail_rt::ix({i}); let __j = ail_rt::ix({j}); {p}.swap(__i, __j); }}"
                )
            }
            "clear" => format!("{}.clear()", self.mut_place(&args[0])),
            "extend" => {
                let x = self.val(a(1));
                let p = self.mut_place(&args[0]);
                format!("{{ let __x = {x}; {p}.extend(__x); }}")
            }
            "slice" => format!(
                "ail_rt::slice({}, {}, {})",
                self.rf(a(0)),
                self.val(a(1)),
                self.val(a(2))
            ),
            "concat" => format!("ail_rt::concat({}, {})", self.rf(a(0)), self.rf(a(1))),
            "contains" => {
                if at(self, 0) == Ty::Str {
                    format!("{}.contains({})", self.pv(a(0)), self.sv(a(1)))
                } else if at(self, 1) == Ty::Str {
                    format!("{}.iter().any(|__x| &__x[..] == {})", self.pv(a(0)), self.sv(a(1)))
                } else {
                    format!("{}.contains({})", self.pv(a(0)), self.rf(a(1)))
                }
            }
            "index-of" => {
                if at(self, 0) == Ty::Str {
                    format!(
                        "{}.find({}).map(|i| i as i64)",
                        self.pv(a(0)),
                        self.sv(a(1))
                    )
                } else if at(self, 1) == Ty::Str {
                    format!(
                        "{}.iter().position(|__x| &__x[..] == {}).map(|i| i as i64)",
                        self.pv(a(0)),
                        self.sv(a(1))
                    )
                } else {
                    format!("ail_rt::index_of({}, {})", self.rf(a(0)), self.rf(a(1)))
                }
            }
            "sort" => {
                let el = match at(self, 0) {
                    Ty::Vec(el) => *el,
                    _ => Ty::Err,
                };
                let p = self.mut_place(&args[0]);
                if el.is_float() {
                    format!("{p}.sort_unstable_by(|a, b| a.total_cmp(b))")
                } else if !matches!(el, Ty::Int(_) | Ty::Str | Ty::Bool) {
                    format!(
                        "{p}.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))"
                    )
                } else {
                    format!("{p}.sort_unstable()")
                }
            }
            "reverse" => format!("{}.reverse()", self.mut_place(&args[0])),
            "sort-desc" => {
                let el = match at(self, 0) {
                    Ty::Vec(el) => *el,
                    _ => Ty::Err,
                };
                let p = self.mut_place(&args[0]);
                if el.is_float() {
                    format!("{p}.sort_by(|a, b| b.total_cmp(a))")
                } else {
                    format!(
                        "{p}.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal))"
                    )
                }
            }
            "sum" => format!("{}.iter().sum::<{}>()", self.pv(a(0)), rty(&t)),
            "map" if args.len() == 2 => {
                let (f, xs) = (self.closure(a(1)), self.pv(a(0)));
                let c = self.conv_ref("__e", &self.elem_ty(a(0)));
                format!("{{ let __f = {f}; {xs}.iter().map(|__e| __f({c})).collect::<Vec<_>>() }}")
            }
            "send" => format!("{{ let _ = {}.send({}); }}", self.place(a(0)).expect("channel"), self.val(a(1))),
            "recv" => format!("{}.recv().ok()", self.place(a(0)).expect("channel")),
            "par-map" | "par-run" => {
                let (f, xs) = (self.closure(a(1)), self.pv(a(0)));
                let el = self.elem_ty(a(0));
                let c = self.conv_ref("__e", &el);
                let rt = if name == "par-map" { "par_map" } else { "par_run" };
                format!("{{ let __f = {f}; ail_rt::{rt}(&{xs}[..], &|__e: &{}| __f({c})) }}", rty(&el))
            }
            "filter" | "any" | "all" | "count" | "find" | "find-index" => {
                let (f, xs) = (self.closure(a(1)), self.pv(a(0)));
                let el = self.elem_ty(a(0));
                let c = self.conv_ref("__e", &el);
                let cc = self.conv_ref("(*__e)", &el);
                let body = match name {
                    "filter" => format!("{xs}.iter().filter(|__e| __f({cc})).cloned().collect::<Vec<_>>()"),
                    "any" => format!("{xs}.iter().any(|__e| __f({c}))"),
                    "all" => format!("{xs}.iter().all(|__e| __f({c}))"),
                    "count" => format!("({xs}.iter().filter(|__e| __f({cc})).count() as i64)"),
                    "find" => format!("{xs}.iter().find(|__e| __f({cc})).cloned()"),
                    _ => format!("{xs}.iter().position(|__e| __f({c})).map(|i| i as i64)"),
                };
                format!("{{ let __f = {f}; {body} }}")
            }
            "fold" => {
                let xs = self.pv(a(0));
                let init = self.val(a(1));
                let f = self.closure(a(2));
                let acc_t = self.ty(a(1)).clone();
                let el = self.elem_ty(a(0));
                let ca = self.conv_ref("(&__acc)", &acc_t);
                let ce = self.conv_ref("__e", &el);
                format!(
                    "{{ let __f = {f}; let mut __acc: {} = {init}; for __e in {xs}.iter() {{ __acc = __f({ca}, {ce}); }} __acc }}",
                    rty(&acc_t)
                )
            }
            "sort-by" => {
                let f = self.closure(a(1));
                let el = self.elem_ty(a(0));
                let k = match self.ty(a(1)) {
                    Ty::Fn(_, r) => (**r).clone(),
                    _ => Ty::Err,
                };
                let (ca, cb) = (self.conv_ref("__a", &el), self.conv_ref("__b", &el));
                let cmp = key_cmp(&k, "__ka", "__kb");
                let p = self.mut_place(&args[0]);
                if self.is_copy(&k) {
                    // Cheap keys are recomputed per comparison; others once per element.
                    format!("{{ let __f = {f}; {p}.sort_by(|__a, __b| {{ let __ka = __f({ca}); let __kb = __f({cb}); {cmp} }}); }}")
                } else {
                    let ce = self.conv_ref("__e", &el);
                    let cmp = key_cmp(&k, "(*__ka)", "(*__kb)");
                    format!("{{ let __f = {f}; let __keys: Vec<_> = {p}.iter().map(|__e| __f({ce})).collect(); ail_rt::sort_by_keys(&mut {p}, __keys, |__ka, __kb| {cmp}); }}")
                }
            }
            "min-by" | "max-by" => {
                let (f, xs) = (self.closure(a(1)), self.pv(a(0)));
                let el = self.elem_ty(a(0));
                let k = match self.ty(a(1)) {
                    Ty::Fn(_, r) => (**r).clone(),
                    _ => Ty::Err,
                };
                let c = self.conv_ref("__e", &el);
                // First element with the smallest (or largest) key.
                let cmp = if name == "min-by" { key_cmp(&k, "__x.0", "__y.0") } else { key_cmp(&k, "__y.0", "__x.0") };
                format!("{{ let __f = {f}; {xs}.iter().map(|__e| (__f({c}), __e)).min_by(|__x, __y| {cmp}).map(|(_, __e)| __e.clone()) }}")
            }
            "map" => "ail_rt::AilMap::default()".into(),
            "put" => {
                if let Some(code) = self.fused_update(args) {
                    return code;
                }
                if at(self, 1) == Ty::Str
                    && matches!(self.view_of(a(1)), Some((View::Ref, _)))
                {
                    // A view key is copied only when it is new to the map.
                    let v = self.val(a(2));
                    let k = self.vref(a(1));
                    let p = self.mut_place(&args[0]);
                    return format!(
                        "{{ let __v = {v}; let __k: &str = {k}; match {p}.get_mut(__k) {{ Some(__s) => *__s = __v, None => {{ {p}.insert(String::from(__k), __v); }} }} }}"
                    );
                }
                // Look up by reference first: an existing key is updated in place without
                // materializing (allocating) an owned copy of the key.
                let v = self.val(a(2));
                let kt = at(self, 1);
                if self.is_copy(&kt) || self.place(a(1)).is_none() {
                    let k = self.val(a(1));
                    let p = self.mut_place(&args[0]);
                    format!("{{ let __v = {v}; {p}.insert({k}, __v); }}")
                } else {
                    let kr = self.rf(a(1));
                    let k = self.val(a(1));
                    let p = self.mut_place(&args[0]);
                    format!(
                        "{{ let __v = {v}; match {p}.get_mut({kr}) {{ Some(__s) => *__s = __v, None => {{ {p}.insert({k}, __v); }} }} }}"
                    )
                }
            }
            "get" => format!("{}.get({}).cloned()", self.pv(a(0)), self.rf(a(1))),
            "has" => format!("{}.contains_key({})", self.pv(a(0)), self.rf(a(1))),
            "del" => {
                let k = self.val(a(1));
                let p = self.mut_place(&args[0]);
                format!("{{ let __k = {k}; {p}.remove(&__k) }}")
            }
            "keys" => format!("ail_rt::keys({})", self.rf(a(0))),
            "values" => format!("ail_rt::values({})", self.rf(a(0))),
            "some" => format!("Some({})", self.val(a(0))),
            "ok" => format!("Ok({})", self.val(a(0))),
            "err" => format!("Err({})", self.val(a(0))),
            "unwrap-or" => format!("({}).unwrap_or({})", self.val(a(0)), self.val(a(1))),
            "expect" => {
                let f = if matches!(at(self, 0), Ty::Opt(_)) {
                    "expect_opt"
                } else {
                    "expect_res"
                };
                format!("ail_rt::{f}({}, {})", self.val(a(0)), self.msg(a(1)))
            }
            "is-some" => format!("{}.is_some()", self.pv(a(0))),
            "is-none" => format!("{}.is_none()", self.pv(a(0))),
            "is-ok" => format!("{}.is_ok()", self.pv(a(0))),
            "is-err" => format!("{}.is_err()", self.pv(a(0))),
            "substr" => format!(
                "ail_rt::substr({}, {}, {})",
                self.rf(a(0)),
                self.val(a(1)),
                self.val(a(2))
            ),
            "split" => format!("ail_rt::split({}, {})", self.rf(a(0)), self.rf(a(1))),
            "trim" => format!("{}.trim().to_string()", self.pv(a(0))),
            // A temporary argument is converted in place instead of copied.
            "upper" | "lower" if self.place(a(0)).is_none() => format!("ail_rt::{name}_owned({})", self.val(a(0))),
            "upper" => format!("ail_rt::upper({})", self.rf(a(0))),
            "lower" => format!("ail_rt::lower({})", self.rf(a(0))),
            "starts-with" => format!("{}.starts_with({})", self.pv(a(0)), self.sv(a(1))),
            "ends-with" => format!("{}.ends_with({})", self.pv(a(0)), self.sv(a(1))),
            "replace" => format!(
                "{}.replace({}, {})",
                self.pv(a(0)),
                self.sv(a(1)),
                self.sv(a(2))
            ),
            "join" => format!("{}.join({})", self.pv(a(0)), self.sv(a(1))),
            "parse-i64" => format!("{}.trim().parse::<i64>().ok()", self.pv(a(0))),
            "parse-f64" => format!("{}.trim().parse::<f64>().ok()", self.pv(a(0))),
            "chars" => format!("ail_rt::chars({})", self.rf(a(0))),
            "byte-at" => format!("ail_rt::byte_at({}, {})", self.rf(a(0)), self.val(a(1))),
            "fmt-float" => format!(
                "format!(\"{{:.*}}\", ail_rt::count({}), {})",
                self.val(a(1)),
                self.val(a(0))
            ),
            "from-code" => format!("ail_rt::from_code({})", self.val(a(0))),
            "repeat" => format!(
                "{}.repeat(ail_rt::count({}))",
                self.pv(a(0)),
                self.val(a(1))
            ),
            "read-line" => "ail_rt::read_line()".into(),
            "read-file" => format!("ail_rt::read_file({}, {})", self.rf(a(1)), self.val(a(2))),
            "write-file" => format!("ail_rt::write_file({}, {})", self.rf(a(1)), self.rf(a(2))),
            "now-ms" => "ail_rt::now_ms()".into(),
            "now-ns" => "ail_rt::now_ns()".into(),
            "rand-int" => format!("ail_rt::rand_int({}, {})", self.val(a(1)), self.val(a(2))),
            "rand-f64" => "ail_rt::rand_f64()".into(),
            "args" => "ail_rt::args()".into(),
            "read-all" => "ail_rt::read_all()".into(),
            "draw-clear" => format!("ail_web::clear({})", self.msg(a(1))),
            "draw-rect" => format!(
                "ail_web::rect({}, {}, {}, {}, {})",
                self.val(a(1)),
                self.val(a(2)),
                self.val(a(3)),
                self.val(a(4)),
                self.msg(a(5))
            ),
            "draw-circle" => format!(
                "ail_web::circle({}, {}, {}, {})",
                self.val(a(1)),
                self.val(a(2)),
                self.val(a(3)),
                self.msg(a(4))
            ),
            "draw-line" => format!(
                "ail_web::line({}, {}, {}, {}, {}, {})",
                self.val(a(1)),
                self.val(a(2)),
                self.val(a(3)),
                self.val(a(4)),
                self.val(a(5)),
                self.msg(a(6))
            ),
            "draw-text" => format!(
                "ail_web::text({}, {}, {}, {}, {})",
                self.val(a(1)),
                self.val(a(2)),
                self.val(a(3)),
                self.msg(a(4)),
                self.msg(a(5))
            ),
            "lines" => format!("ail_rt::lines({})", self.rf(a(0))),
            "is-digit" => format!(
                "ail_rt::all_chars({}, |c| c.is_ascii_digit())",
                self.rf(a(0))
            ),
            "is-alpha" => format!(
                "ail_rt::all_chars({}, |c| c.is_alphabetic())",
                self.rf(a(0))
            ),
            "is-alnum" => format!(
                "ail_rt::all_chars({}, |c| c.is_alphanumeric())",
                self.rf(a(0))
            ),
            "is-space" => format!(
                "ail_rt::all_chars({}, |c| c.is_whitespace())",
                self.rf(a(0))
            ),
            "is-upper" => format!("ail_rt::all_chars({}, |c| c.is_uppercase())", self.rf(a(0))),
            "is-lower" => format!("ail_rt::all_chars({}, |c| c.is_lowercase())", self.rf(a(0))),
            "code-of" => format!("ail_rt::code_of({})", self.rf(a(0))),
            "json-parse" => format!("ail_json::parse({})", self.rf(a(0))),
            "json-str" if at(self, 0) == Ty::User("Json".into()) => format!("ail_json::to_string({})", self.rf(a(0))),
            // (json-str (slice v a b)) encodes the range in place, without copying it out.
            "json-str" if matches!(&a(0).kind, ExprKind::Call(n, _, xs) if n == "slice" && xs.len() == 3) => {
                let ExprKind::Call(_, _, xs) = &a(0).kind else { unreachable!() };
                format!("ail_json::encode(ail_rt::slice_ref({}, {}, {}))", self.rf(xs[0].expr()), self.val(xs[1].expr()), self.val(xs[2].expr()))
            }
            "json-str" => format!("ail_json::encode({})", self.rf(a(0))),
            "http-next" => format!("ail_http::next({}, {})", self.val(a(1)), self.val(a(2))),
            "http-respond" => format!(
                "ail_http::respond({}, {}, {}, {})",
                self.rf(a(1)),
                self.val(a(2)),
                self.rf(a(3)),
                self.rf(a(4))
            ),
            "http-request" => format!(
                "ail_http::request({}, {}, {}, {})",
                self.rf(a(1)),
                self.rf(a(2)),
                self.rf(a(3)),
                self.rf(a(4))
            ),
            "http-serve" => {
                let st = self.ty(a(3)).clone();
                let sty = rty(&st);
                let f = self.closure(a(4));
                let ExprKind::Var(w) = &a(5).kind else { unreachable!("write handler") };
                let c = self.conv_ref("__s", &st);
                format!(
                    "{{ let __f = {f}; ail_http::serve({}, {}, {}, &|__q: &U_Request, __s: &{sty}| __f(__q, {c}), &|__q: &U_Request, __s: &mut {sty}| {}(__q, __s)) }}",
                    self.val(a(1)),
                    self.val(a(2)),
                    self.val(a(3)),
                    fn_name(w)
                )
            }
            "fail" => format!("ail_rt::fail({})", self.msg(a(0))),
            "assert" => {
                let c = self.val(a(0));
                let msg = if args.len() == 2 {
                    self.msg(a(1))
                } else {
                    format!("{:?}", self.src_of(a(0)))
                };
                format!("if !{c} {{ ail_rt::assert_fail({msg}) }}")
            }
            other => unreachable!("builtin {other}"),
        }
    }

    /// How a value of type `t` is passed to a function: by value when it is cheap to copy,
    /// as `&str` for strings, by reference otherwise.
    fn conv_ty(&self, t: &Ty) -> String {
        if *t == Ty::Str {
            "&str".into()
        } else if self.is_copy(t) {
            rty(t)
        } else {
            format!("&{}", rty(t))
        }
    }

    /// Converts `r`, code of type `&T`, to the passing convention of `t`.
    fn conv_ref(&self, r: &str, t: &Ty) -> String {
        if *t == Ty::Str {
            format!("&{r}[..]")
        } else if self.is_copy(t) {
            format!("*{r}")
        } else {
            r.to_string()
        }
    }

    /// An argument passed by the convention of its type.
    fn conv_arg(&mut self, x: &Expr) -> String {
        let t = self.ty(x).clone();
        if t == Ty::Str {
            self.sv(x)
        } else if self.is_copy(&t) {
            self.val(x)
        } else {
            self.rf(x)
        }
    }

    fn elem_ty(&self, xs: &Expr) -> Ty {
        match self.ty(xs) {
            Ty::Vec(t) => (**t).clone(),
            _ => Ty::Err,
        }
    }

    /// A Rust callable for a function argument: a closure for a lambda or a named fn, or the
    /// Fn parameter itself (already a reference).
    fn closure(&mut self, f: &Expr) -> String {
        let Ty::Fn(ps, r) = self.ty(f).clone() else { unreachable!("closure of a non-function") };
        let ret = rty(&r);
        match &f.kind {
            ExprKind::Lambda(_, body) => {
                let ids = self.ck.lambda_params[&f.id].clone();
                let params: Vec<String> = ids
                    .iter()
                    .zip(ps.iter())
                    .map(|(b, t)| format!("{}: {}", self.local_name(&self.ck.bindings[*b].name), self.conv_ty(t)))
                    .collect();
                let saved = self.cur_view.take();
                let b = self.block_as(body, r.as_ref() == &Ty::Unit);
                self.cur_view = saved;
                format!("(|{}| -> {ret} {b})", params.join(", "))
            }
            ExprKind::Var(n) => match self.ck.res[f.id as usize].clone() {
                Res::Local(b) => self.local_name(&self.ck.bindings[b].name),
                Res::UserFn(_) => {
                    let names: Vec<String> = (0..ps.len()).map(|i| format!("__a{i}")).collect();
                    let params: Vec<String> = names.iter().zip(ps.iter()).map(|(n, t)| format!("{n}: {}", self.conv_ty(t))).collect();
                    let call = format!("{}({})", fn_name(n), names.join(", "));
                    let call = match self.view_fns.get(n).copied() {
                        Some((_, View::Ref)) => format!("String::from({call})"),
                        Some((_, View::Cow)) => format!("({call}).into_owned()"),
                        None => call,
                    };
                    format!("(|{}| -> {ret} {{ {call} }})", params.join(", "))
                }
                other => unreachable!("function value {other:?}"),
            },
            _ => unreachable!("function argument"),
        }
    }

    /// Vectors that a loop body pushes to unconditionally (a top-level `(push (mut v) x)`): one
    /// element per iteration at least, so reserving the trip count up front never wastes much.
    fn push_targets(&mut self, body: &[Expr]) -> Vec<String> {
        // Only a variable (or a field of one) that exists before the loop: anything indexed may
        // depend on the loop variable, and a variable declared in the body is not there yet.
        let mut inner = Vec::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let Some(b) = self.ck.decl.get(&(e.id as usize)) {
                    inner.push(*b);
                }
            });
        }
        fn simple(e: &Expr) -> bool {
            match &e.kind {
                ExprKind::Var(_) => true,
                ExprKind::Field(b, _, _) => simple(b),
                _ => false,
            }
        }
        let mut out = Vec::new();
        for x in body {
            if let ExprKind::Call(n, _, args) = &x.kind {
                if n == "push"
                    && args.len() == 2
                    && matches!(args[0], Arg::Mut(..))
                    && !matches!(self.ck.res.get(x.id as usize), Some(Res::UserFn(_)))
                    && matches!(self.ty(args[0].expr()), Ty::Vec(_))
                    && simple(args[0].expr())
                    && self.root_of(args[0].expr()).is_some_and(|b| !inner.contains(&b))
                {
                    let p = self.mut_place(&args[0]);
                    if !out.contains(&p) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// An innermost `(for j (range a b) ...)` whose `(at v i)` reads all have an index `j`,
    /// `j + c` or `j - c` (c fixed during the loop) on a Vec the loop does not change: one test
    /// before the loop shows every index in range, and then the loop runs without bounds checks
    /// (so LLVM can also vectorize it). If the test fails, the ordinary checked loop runs and
    /// traps exactly where it would have.
    fn for_unchecked(&mut self, lv: BindId, var: &str, lo: &str, hi: &str, a: &Expr, body: &[Expr]) -> Option<String> {
        let mut nested = false;
        let mut reads: Vec<(u32, &Expr, &Expr)> = Vec::new();
        for x in body {
            walk_ref(x, &mut |e| match &e.kind {
                ExprKind::For(..) | ExprKind::While(..) | ExprKind::Lambda(..) | ExprKind::Tasks(..) => nested = true,
                ExprKind::Call(n, _, args)
                    if n == "at"
                        && args.len() == 2
                        && !matches!(self.ck.res.get(e.id as usize), Some(Res::UserFn(_))) =>
                {
                    reads.push((e.id, args[0].expr(), args[1].expr()));
                }
                _ => {}
            });
        }
        if nested || reads.is_empty() {
            return None;
        }
        let arm = Arm { pat: Pat::Wild(a.span), body: body.to_vec(), span: a.span };
        let changes = |em: &Self, b: BindId| em.arms_mutate(std::slice::from_ref(&arm), b);
        // Vars read by an expression; None if it calls anything but arithmetic.
        fn vars_of(em: &Emitter, e: &Expr, out: &mut Vec<BindId>) -> bool {
            match &e.kind {
                ExprKind::Int(_) => true,
                ExprKind::Var(_) => match em.ck.res[e.id as usize] {
                    Res::Local(b) => {
                        out.push(b);
                        true
                    }
                    Res::Const(_) => true,
                    _ => false,
                },
                ExprKind::Call(n, _, xs)
                    if matches!(n.as_str(), "+" | "-" | "*" | "len")
                        && !matches!(em.ck.res.get(e.id as usize), Some(Res::UserFn(_))) =>
                {
                    xs.iter().all(|x| vars_of(em, x.expr(), out))
                }
                _ => false,
            }
        }
        let fixed = |em: &Self, e: &Expr| -> bool {
            let mut vs = Vec::new();
            vars_of(em, e, &mut vs) && vs.iter().all(|b| *b != lv && !changes(em, *b))
        };
        let is_lv = |em: &Self, e: &Expr| matches!((&e.kind, &em.ck.res[e.id as usize]), (ExprKind::Var(_), Res::Local(b)) if *b == lv);
        // Each read: (id, vector, offset sign and expression).
        let mut plan: Vec<(u32, &Expr, Option<(bool, &Expr)>, &Expr)> = Vec::new();
        for (rid, v, i) in reads {
            let root_ok = match &v.kind {
                ExprKind::Var(_) | ExprKind::Field(..) => self.root_of(v).is_some_and(|b| !changes(self, b) && b != lv),
                _ => false,
            };
            if !root_ok || !matches!(self.ty(v), Ty::Vec(_)) {
                return None;
            }
            let off = if is_lv(self, i) {
                None
            } else {
                match &i.kind {
                    ExprKind::Call(n, _, xs) if xs.len() == 2 && (n == "+" || n == "-") && !matches!(self.ck.res.get(i.id as usize), Some(Res::UserFn(_))) => {
                        let (x, y) = (xs[0].expr(), xs[1].expr());
                        if is_lv(self, x) && fixed(self, y) {
                            Some((n == "+", y))
                        } else if n == "+" && is_lv(self, y) && fixed(self, x) {
                            Some((true, x))
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                }
            };
            plan.push((rid, v, off, i));
        }
        // One slice per vector, taken once; a test per distinct (vector, offset).
        let mut slices: Vec<(String, String)> = Vec::new();
        let mut tests: Vec<String> = Vec::new();
        let mut codes: Vec<(u32, String)> = Vec::new();
        for (rid, v, off, i) in &plan {
            let place = self.pv(v);
            let elem = rty(&self.elem_ty(v));
            let sname = match slices.iter().position(|s| s.0 == place) {
                Some(k) => format!("__us{k}"),
                None => {
                    slices.push((place.clone(), elem));
                    format!("__us{}", slices.len() - 1)
                }
            };
            let o = match off {
                None => "0i128".to_string(),
                Some((plus, c)) => {
                    let c = self.val(c);
                    if *plus { format!("({c}) as i128") } else { format!("-(({c}) as i128)") }
                }
            };
            let t = format!("((__ulo as i128) + ({o}) >= 0 && (__uhi as i128) - 1 + ({o}) < ({sname}.len() as i128))");
            if !tests.contains(&t) {
                tests.push(t);
            }
            let idx = self.val(i);
            codes.push((*rid, format!("(*{sname}.get_unchecked(({idx}) as usize))")));
        }
        let mut out = format!("{{ let __ulo: i64 = {lo}; let __uhi: i64 = {hi}; ");
        for (k, (place, elem)) in slices.iter().enumerate() {
            out.push_str(&format!("let __us{k}: &[{elem}] = &{place}[..]; "));
        }
        let saved: Vec<(u32, Option<String>)> = codes.iter().map(|(id, _)| (*id, self.unchecked.get(id).cloned())).collect();
        for (id, c) in codes {
            self.unchecked.insert(id, c);
        }
        let fast = self.block(body);
        for (id, old) in saved {
            match old {
                Some(c) => self.unchecked.insert(id, c),
                None => self.unchecked.remove(&id),
            };
        }
        let slow = self.block(body);
        out.push_str(&format!(
            "if __ulo < __uhi && {} {{ for {var} in __ulo..__uhi {{ let _ = unsafe {{ {fast} }}; }} }} else {{ for {var} in __ulo..__uhi {{ let _ = {slow}; }} }} }}",
            tests.join(" && ")
        ));
        Some(out)
    }

    /// `(set s (str s a b ...))` appends to `s` in place instead of rebuilding it, so building
    /// a string in a loop is linear. Only when the appended parts don't read `s` itself.
    fn append_in_place(&mut self, place: &Expr, value: &Expr) -> Option<String> {
        let ExprKind::Call(n, _, args) = &value.kind else {
            return None;
        };
        if n != "str"
            || args.len() < 2
            || self.ty(place) != &Ty::Str
            || matches!(self.ck.res.get(value.id as usize), Some(Res::UserFn(_)))
        {
            return None;
        }
        let p = self.place(place)?;
        if self.place(args[0].expr()).as_deref() != Some(p.as_str()) {
            return None;
        }
        let root = self.root_of(place)?;
        let mut reads_root = false;
        for a in &args[1..] {
            crate::check::walk(a.expr(), &mut |e| {
                if matches!(e.kind, ExprKind::Var(_))
                    && matches!(self.ck.res[e.id as usize], Res::Local(b) if b == root)
                {
                    reads_root = true;
                }
            });
        }
        if reads_root {
            return None;
        }
        Some(format!("{{ {} }}", self.push_parts(&p, &args[1..])))
    }

    /// Statements appending each part to the String `target`: strings are copied in directly
    /// and integers written without going through the formatting machinery.
    fn push_parts(&mut self, target: &str, args: &[Arg]) -> String {
        let mut o = String::new();
        for x in args {
            let ex = x.expr();
            match (&ex.kind, self.ty(ex).clone()) {
                (ExprKind::Str(s), _) => write!(o, "{target}.push_str({s:?}); ").unwrap(),
                (_, Ty::Str) => {
                    let v = self.sv(ex);
                    write!(o, "{target}.push_str({v}); ").unwrap()
                }
                (_, Ty::Int(IntK::I64)) => {
                    let v = self.val(ex);
                    write!(o, "ail_rt::push_i64(&mut {target}, {v}); ").unwrap()
                }
                (_, t) => {
                    let v = self.pv(ex);
                    let f = if t.is_float() { "{:?}" } else { "{}" };
                    write!(o, "{{ use std::fmt::Write as _; let _ = write!({target}, {f:?}, {v}); }} ").unwrap()
                }
            }
        }
        o
    }

    /// True if every part of a `str` call is a string or an integer.
    fn simple_parts(&self, args: &[Arg]) -> bool {
        args.iter().all(|x| matches!(self.ty(x.expr()), Ty::Str | Ty::Int(IntK::I64)))
    }

    fn format(&mut self, args: &[Arg]) -> String {
        if args.is_empty() {
            return "String::new()".into();
        }
        let mut fmt = String::new();
        let mut vals = Vec::new();
        for x in args {
            let ex = x.expr();
            if let ExprKind::Str(s) = &ex.kind {
                fmt.push_str(&s.replace('{', "{{").replace('}', "}}"));
                continue;
            }
            let t = self.ty(ex).clone();
            fmt.push_str(if t.is_float() { "{:?}" } else { "{}" });
            vals.push(self.pv(ex));
        }
        if vals.is_empty() {
            format!("String::from({fmt:?})")
        } else if self.simple_parts(args) {
            format!("{{ let mut __s = String::new(); {}__s }}", self.push_parts("__s", args))
        } else {
            format!("format!({fmt:?}, {})", vals.join(", "))
        }
    }
}

/// An Ordering between two keys of type `k`, given as places of type K.
fn key_cmp(k: &Ty, a: &str, b: &str) -> String {
    match k {
        Ty::F32 | Ty::F64 => format!("({a}).total_cmp(&{b})"),
        Ty::Int(_) | Ty::Str | Ty::Bool => format!("({a}).cmp(&{b})"),
        _ => format!("({a}).partial_cmp(&{b}).unwrap_or(std::cmp::Ordering::Equal)"),
    }
}

fn cast(v: &str, from: &Ty, to: &Ty) -> String {
    if from == to {
        return v.to_string();
    }
    match (from, to) {
        (Ty::Int(_), Ty::Int(k)) => format!("ail_rt::icast::<_, {}>({v})", k.rust()),
        (Ty::Int(_), f) => format!("(({v}) as {})", rty(f)),
        (f, Ty::Int(k)) if f.is_float() => format!("ail_rt::f2i_{}(({v}) as f64)", k.rust()),
        (_, f) => format!("(({v}) as {})", rty(f)),
    }
}

// ---------- borrowed string views ----------
//
// A Str is a value, but reading part of one does not need a copy. An immutable local bound to a
// piece of another string (a substring, a literal, a trimmed or lower-cased string, a helper that
// returns part of its argument) becomes a borrowed `&str`, or a `Cow` when it is only sometimes a
// copy, as long as the strings it borrows from are neither changed nor moved while it lives.
// The language does not change: views are only how the program is emitted.

fn pat_binds(p: &Pat, ck: &Checked, epoch: usize, out: &mut HashSet<BindId>) {
    match p {
        Pat::Bind(_, span) => {
            if let Some(b) = ck.decl.get(&pat_key(*span, epoch)) {
                out.insert(*b);
            }
        }
        Pat::Ctor(_, subs, _) => subs.iter().for_each(|s| pat_binds(s, ck, epoch, out)),
        _ => {}
    }
}

impl<'a> Emitter<'a> {
    /// If `e` can be produced as a borrowed string with no temporaries, its kind and the
    /// bindings it borrows from.
    fn view_of(&self, e: &Expr) -> Option<(View, Vec<BindId>)> {
        if self.ty(e) == &Ty::Never {
            return Some((View::Ref, vec![]));
        }
        if self.ty(e) != &Ty::Str {
            return None;
        }
        match &e.kind {
            ExprKind::Str(_) => Some((View::Ref, vec![])),
            ExprKind::Var(_) => match self.ck.res[e.id as usize] {
                Res::Local(b) => {
                    let mut roots = vec![b];
                    if let Some(r) = self.view_roots.get(&b) {
                        roots.extend(r.iter().copied());
                    }
                    Some((View::Ref, roots))
                }
                _ => None,
            },
            ExprKind::Field(..) => {
                let mut cur = e;
                while let ExprKind::Field(b, _, _) = &cur.kind {
                    cur = b;
                }
                if !matches!(cur.kind, ExprKind::Var(_)) {
                    return None;
                }
                Some((View::Ref, vec![self.root_of(e)?]))
            }
            ExprKind::Do(es) if es.len() == 1 => self.view_of(&es[0]),
            ExprKind::If(_, t, Some(f)) => {
                let (k1, mut r1) = self.view_of(t)?;
                let (k2, r2) = self.view_of(f)?;
                r1.extend(r2);
                let k = if k1 == View::Ref && k2 == View::Ref { View::Ref } else { View::Cow };
                Some((k, r1))
            }
            ExprKind::Match(_, arms) if !arms.is_empty() => {
                let mut kind = View::Ref;
                let mut roots = vec![];
                let mut inner = HashSet::new();
                for a in arms {
                    pat_binds(&a.pat, self.ck, self.epoch, &mut inner);
                    if a.body.len() != 1 {
                        return None;
                    }
                    let (k, r) = self.view_of(&a.body[0])?;
                    if k == View::Cow {
                        kind = View::Cow;
                    }
                    roots.extend(r);
                }
                // A pattern binding lives only inside its arm.
                if roots.iter().any(|r| inner.contains(r)) {
                    return None;
                }
                Some((kind, roots))
            }
            ExprKind::Call(n, _, args) => match &self.ck.res[e.id as usize] {
                Res::Builtin(_) => match n.as_str() {
                    "substr" | "trim" => match self.view_of(args[0].expr())? {
                        (View::Ref, r) => Some((View::Ref, r)),
                        _ => None,
                    },
                    "lower" | "upper" => Some((View::Cow, self.view_of(args[0].expr())?.1)),
                    _ => None,
                },
                Res::UserFn(f) => {
                    let (k, kind) = *self.view_fns.get(f)?;
                    match self.view_of(args[k].expr())? {
                        (View::Ref, r) => Some((kind, r)),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// The place a view of a variable or field borrows from.
    fn view_place(&mut self, e: &Expr) -> String {
        if let (ExprKind::Var(_), Res::Local(b)) = (&e.kind, &self.ck.res[e.id as usize]) {
            if self.views.contains_key(b) {
                return self.local_name(&self.ck.bindings[*b].name);
            }
        }
        self.place(e).expect("view of a place")
    }

    /// `&str` code for an expression whose `view_of` kind is `Ref`.
    fn vref(&mut self, e: &Expr) -> String {
        if self.ty(e) == &Ty::Never {
            return self.rvalue(e);
        }
        match &e.kind {
            ExprKind::Str(s) => format!("{s:?}"),
            ExprKind::Var(_) => match self.ck.res[e.id as usize] {
                Res::Local(b) if self.views.get(&b) == Some(&View::Ref) => {
                    self.local_name(&self.ck.bindings[b].name)
                }
                _ => format!("(&{}[..])", self.view_place(e)),
            },
            ExprKind::Field(..) => format!("(&{}[..])", self.view_place(e)),
            ExprKind::Do(es) => self.vref(&es[0]),
            ExprKind::If(c, t, Some(f)) => {
                let c = self.val(c);
                let a = self.vref(t);
                let b = self.vref(f);
                format!("(if {c} {{ {a} }} else {{ {b} }})")
            }
            ExprKind::Match(s, arms) => self.emit_match(s, arms, Tail::Ref),
            ExprKind::Call(n, _, args) => match self.ck.res[e.id as usize].clone() {
                Res::Builtin(_) if n == "substr" => {
                    let s = self.vref(args[0].expr());
                    let i = self.val(args[1].expr());
                    let j = self.val(args[2].expr());
                    format!("ail_rt::substr_ref({s}, {i}, {j})")
                }
                Res::Builtin(_) if n == "trim" => format!("{}.trim()", self.vref(args[0].expr())),
                Res::UserFn(f) => self.user_call(&f, args),
                _ => unreachable!("vref of {n}"),
            },
            _ => unreachable!("vref"),
        }
    }

    /// `Cow<str>` code for an expression that has a view.
    fn vcow(&mut self, e: &Expr) -> String {
        let (k, _) = self.view_of(e).expect("vcow of a view");
        if k == View::Ref {
            return format!("std::borrow::Cow::Borrowed({})", self.vref(e));
        }
        match &e.kind {
            ExprKind::Do(es) => self.vcow(&es[0]),
            ExprKind::If(c, t, Some(f)) => {
                let c = self.val(c);
                let a = self.vcow(t);
                let b = self.vcow(f);
                format!("(if {c} {{ {a} }} else {{ {b} }})")
            }
            ExprKind::Match(s, arms) => self.emit_match(s, arms, Tail::Cow),
            ExprKind::Call(n, _, args) => match self.ck.res[e.id as usize].clone() {
                Res::Builtin(_) => format!("ail_rt::{n}_cow({})", self.vcow(args[0].expr())),
                Res::UserFn(f) => self.user_call(&f, args),
                _ => unreachable!("vcow of {n}"),
            },
            _ => unreachable!("vcow"),
        }
    }

    /// `&str` for any Str expression, inside a single expression (it may borrow a temporary).
    fn sv(&mut self, e: &Expr) -> String {
        match self.view_of(e) {
            Some((View::Ref, _)) => self.vref(e),
            Some((View::Cow, _)) => format!("(&{}[..])", self.vcow(e)),
            None => match self.place(e) {
                Some(p) => format!("(&{p}[..])"),
                None => format!("(&({})[..])", self.rvalue(e)),
            },
        }
    }

    /// A Str argument can be passed as a view unless it borrows from a `(mut ...)` argument of
    /// the same call.
    fn view_arg_ok(&self, x: &Expr, args: &[Arg], has_mut: bool) -> bool {
        let Some((_, roots)) = self.view_of(x) else { return false };
        !has_mut
            || !args.iter().any(|a| match a {
                Arg::Mut(m, _) => self.root_of(m).is_some_and(|r| roots.contains(&r)),
                _ => false,
            })
    }

    /// True if nothing in `value` or `rest` changes or moves any of `roots`.
    fn view_safe(&self, roots: &[BindId], value: &Expr, rest: &[Expr]) -> bool {
        let mut ok = true;
        let mut check = |e: &Expr| match &e.kind {
            ExprKind::Set(p, _) => {
                if self.root_of(p).is_some_and(|r| roots.contains(&r)) {
                    ok = false
                }
            }
            ExprKind::Call(_, _, args) => {
                for a in args {
                    if let Arg::Mut(x, _) = a {
                        if self.root_of(x).is_some_and(|r| roots.contains(&r)) {
                            ok = false
                        }
                    }
                }
            }
            ExprKind::Var(_) => {
                if let Res::Local(b) = self.ck.res[e.id as usize] {
                    // A `&str` view local is never moved: using it by value copies.
                    if roots.contains(&b)
                        && self.moves.contains(&e.id)
                        && self.views.get(&b) != Some(&View::Ref)
                    {
                        ok = false
                    }
                }
            }
            _ => {}
        };
        crate::check::walk(value, &mut check);
        for x in rest {
            crate::check::walk(x, &mut check);
        }
        ok
    }

    /// Decide which immutable Str locals of a block (and nested blocks) are views.
    fn plan_views(&mut self, es: &[Expr]) {
        for (i, e) in es.iter().enumerate() {
            self.plan_views_in(e);
            if let ExprKind::Let { mutable: false, value, .. } = &e.kind {
                let b = self.ck.decl[&(e.id as usize)];
                if self.ck.bindings[b].ty != Ty::Str || self.ty(value) != &Ty::Str {
                    continue;
                }
                if let Some((k, roots)) = self.view_of(value) {
                    // The view only has to stay valid up to the last statement that reads it.
                    let rest = &es[i + 1..];
                    let mut live = 0;
                    for (j, x) in rest.iter().enumerate() {
                        let mut uses = false;
                        crate::check::walk(x, &mut |e| {
                            if matches!(e.kind, ExprKind::Var(_))
                                && matches!(self.ck.res[e.id as usize], Res::Local(u) if u == b)
                            {
                                uses = true;
                            }
                        });
                        if uses {
                            live = j + 1;
                        }
                    }
                    if self.view_safe(&roots, value, &rest[..live]) {
                        self.views.insert(b, k);
                        self.view_roots.insert(b, roots);
                    }
                }
            }
        }
    }

    fn plan_views_in(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Let { value, .. } => self.plan_views_in(value),
            ExprKind::Set(a, b) => {
                self.plan_views_in(a);
                self.plan_views_in(b);
            }
            ExprKind::If(c, t, f) => {
                self.plan_views_in(c);
                self.plan_views(std::slice::from_ref(t));
                if let Some(f) = f {
                    self.plan_views(std::slice::from_ref(f));
                }
            }
            ExprKind::Do(es) => self.plan_views(es),
            ExprKind::While(c, b) => {
                self.plan_views_in(c);
                self.plan_views(b);
            }
            ExprKind::For(_, _, it, b) => {
                match it {
                    ForIter::Range(x, y) => {
                        self.plan_views_in(x);
                        self.plan_views_in(y);
                    }
                    ForIter::Each(v) => self.plan_views_in(v),
                }
                self.plan_views(b);
            }
            ExprKind::Match(s, arms) => {
                self.plan_views_in(s);
                for a in arms {
                    self.plan_views(&a.body);
                }
            }
            ExprKind::Call(_, _, args) => args.iter().for_each(|a| self.plan_views_in(a.expr())),
            ExprKind::New(_, _, fs) => fs.iter().for_each(|f| self.plan_views_in(&f.2)),
            ExprKind::With(b, fs) => {
                self.plan_views_in(b);
                fs.iter().for_each(|f| self.plan_views_in(&f.2));
            }
            ExprKind::Return(Some(x))
            | ExprKind::Field(x, _, _)
            | ExprKind::Try(x)
            | ExprKind::As(_, x)
            | ExprKind::Copy(x) => self.plan_views_in(x),
            ExprKind::Lambda(_, b) => self.plan_views(b),
            ExprKind::Tasks(_, ts) => ts.iter().for_each(|t| self.plan_views(&t.body)),
            _ => {}
        }
    }

    /// Find fns whose Str result is always a piece of one Str param (or a literal): they return
    /// a view of it instead of a copy.
    fn infer_view_fns(&mut self, prog: &Program) {
        let ck = self.ck;
        loop {
            let mut changed = false;
            for f in concrete_fns(prog, ck) {
                self.epoch = ck.fn_epoch.get(&f.name).copied().unwrap_or(0);
                if self.view_fns.contains_key(&f.name)
                    || !f.ensures.is_empty()
                    || f.body.is_empty()
                    || matches!(f.name.as_str(), "main" | "init" | "update" | "view" | "page")
                {
                    continue;
                }
                let sig = self.ck.info.fns[&f.name].clone();
                if sig.ret != Ty::Str {
                    continue;
                }
                self.moves = compute_moves(&f.body, self.ck);
                self.views.clear();
                self.view_roots.clear();
                self.plan_views(&f.body);
                let mut tails: Vec<&Expr> = vec![f.body.last().unwrap()];
                collect_returns(&f.body, &mut tails);
                let params = self.ck.fn_params[&f.name].clone();
                'p: for (k, (_, pt, pm)) in sig.params.iter().enumerate() {
                    if *pm || *pt != Ty::Str {
                        continue;
                    }
                    let pb = params[k];
                    let mut kind = View::Ref;
                    for t in &tails {
                        let Some((tk, roots)) = self.view_of(t) else { continue 'p };
                        if roots
                            .iter()
                            .any(|r| *r != pb && self.views.get(r) != Some(&View::Ref))
                        {
                            continue 'p;
                        }
                        if tk == View::Cow {
                            kind = View::Cow;
                        }
                    }
                    self.view_fns.insert(f.name.clone(), (k, kind));
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }
        self.views.clear();
        self.view_roots.clear();
        self.epoch = 0;
    }

    /// `(put (mut m) k (... (get m k) ...))` looks the key up once instead of twice, and copies
    /// a Str key only when it is new to the map.
    fn fused_update(&mut self, args: &[Arg]) -> Option<String> {
        let (m, k, v) = (args[0].expr(), args[1].expr(), args[2].expr());
        let local = |e: &Expr| match (&e.kind, &self.ck.res[e.id as usize]) {
            (ExprKind::Var(_), Res::Local(b)) => Some(*b),
            _ => None,
        };
        let mb = local(m)?;
        let kb = local(k)?;
        if mb == kb {
            return None;
        }
        let mut get = None;
        let mut map_uses = 0;
        let mut ok = true;
        crate::check::walk(v, &mut |e| match &e.kind {
            ExprKind::Call(n, _, xs)
                if n == "get"
                    && matches!(self.ck.res[e.id as usize], Res::Builtin(_))
                    && xs.len() == 2
                    && local(xs[0].expr()) == Some(mb)
                    && local(xs[1].expr()) == Some(kb) =>
            {
                if get.is_some() {
                    ok = false;
                }
                get = Some(e.id);
            }
            ExprKind::Var(_) if local(e) == Some(mb) => map_uses += 1,
            ExprKind::Match(s, _) => {
                if matches!(&s.kind, ExprKind::Call(n, _, _) if n == "get") {
                    ok = false;
                }
            }
            ExprKind::Set(p, _) if self.root_of(p) == Some(kb) => ok = false,
            ExprKind::Call(_, _, xs) => {
                if xs.iter().any(|a| matches!(a, Arg::Mut(x, _) if self.root_of(x) == Some(kb))) {
                    ok = false;
                }
            }
            _ => {}
        });
        let gid = get?;
        if !ok || map_uses != 1 {
            return None;
        }
        let vt = match self.ck.ty[gid as usize].clone() {
            Ty::Opt(t) => rty(&t),
            _ => return None,
        };
        let kr = if self.ty(k) == &Ty::Str {
            self.sv(k)
        } else {
            format!("&{}", self.place(k)?)
        };
        self.subst.insert(gid, format!("Some::<{vt}>(__s.clone())"));
        let vs = self.val(v);
        self.subst.insert(gid, format!("None::<{vt}>"));
        let vn = self.val(v);
        self.subst.remove(&gid);
        let kown = self.val(k);
        let p = self.mut_place(&args[0]);
        Some(format!(
            "{{ match {p}.get_mut({kr}) {{ Some(__s) => {{ let __v = {vs}; *__s = __v; }} None => {{ let __v = {vn}; {p}.insert({kown}, __v); }} }} }}"
        ))
    }
}

/// Every `(return x)` value in a fn body.
fn collect_returns<'e>(es: &'e [Expr], out: &mut Vec<&'e Expr>) {
    fn go<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
        match &e.kind {
            ExprKind::Return(Some(v)) => {
                out.push(v);
                go(v, out);
            }
            ExprKind::Let { value, .. } => go(value, out),
            ExprKind::Set(a, b) => {
                go(a, out);
                go(b, out);
            }
            ExprKind::If(c, t, f) => {
                go(c, out);
                go(t, out);
                if let Some(f) = f {
                    go(f, out);
                }
            }
            ExprKind::Do(es) => es.iter().for_each(|x| go(x, out)),
            ExprKind::While(c, b) => {
                go(c, out);
                b.iter().for_each(|x| go(x, out));
            }
            ExprKind::For(_, _, it, b) => {
                match it {
                    ForIter::Range(x, y) => {
                        go(x, out);
                        go(y, out);
                    }
                    ForIter::Each(v) => go(v, out),
                }
                b.iter().for_each(|x| go(x, out));
            }
            ExprKind::Match(s, arms) => {
                go(s, out);
                arms.iter().for_each(|a| a.body.iter().for_each(|x| go(x, out)));
            }
            ExprKind::Call(_, _, args) => args.iter().for_each(|a| go(a.expr(), out)),
            ExprKind::New(_, _, fs) => fs.iter().for_each(|f| go(&f.2, out)),
            ExprKind::With(b, fs) => {
                go(b, out);
                fs.iter().for_each(|f| go(&f.2, out));
            }
            ExprKind::Field(x, _, _) | ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => {
                go(x, out)
            }
            _ => {}
        }
    }
    es.iter().for_each(|e| go(e, out));
}

/// Pre-order walk like `check::walk`, but the visited expressions keep the tree's lifetime.
fn walk_ref<'e>(e: &'e Expr, f: &mut dyn FnMut(&'e Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Var(_) | ExprKind::Break | ExprKind::Continue => {}
        ExprKind::Call(_, _, args) => args.iter().for_each(|a| walk_ref(a.expr(), f)),
        ExprKind::Let { value, .. } => walk_ref(value, f),
        ExprKind::Set(a, b) => {
            walk_ref(a, f);
            walk_ref(b, f)
        }
        ExprKind::If(c, t, e2) => {
            walk_ref(c, f);
            walk_ref(t, f);
            if let Some(x) = e2 {
                walk_ref(x, f)
            }
        }
        ExprKind::Do(es) => es.iter().for_each(|x| walk_ref(x, f)),
        ExprKind::While(c, b) => {
            walk_ref(c, f);
            b.iter().for_each(|x| walk_ref(x, f))
        }
        ExprKind::For(_, _, it, b) => {
            match it {
                ForIter::Range(a, c) => {
                    walk_ref(a, f);
                    walk_ref(c, f)
                }
                ForIter::Each(v) => walk_ref(v, f),
            }
            b.iter().for_each(|x| walk_ref(x, f))
        }
        ExprKind::Return(v) => {
            if let Some(v) = v {
                walk_ref(v, f)
            }
        }
        ExprKind::Match(s, arms) => {
            walk_ref(s, f);
            for a in arms {
                a.body.iter().for_each(|x| walk_ref(x, f))
            }
        }
        ExprKind::New(_, _, fs) => fs.iter().for_each(|x| walk_ref(&x.2, f)),
        ExprKind::With(b, fs) => {
            walk_ref(b, f);
            fs.iter().for_each(|x| walk_ref(&x.2, f))
        }
        ExprKind::Field(b, _, _) => walk_ref(b, f),
        ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => walk_ref(x, f),
        ExprKind::Lambda(_, b) => b.iter().for_each(|x| walk_ref(x, f)),
        ExprKind::Tasks(_, ts) => ts.iter().for_each(|t| t.body.iter().for_each(|x| walk_ref(x, f))),
    }
}
