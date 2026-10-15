//! Backend: typed ail -> Rust source, compiled by rustc/LLVM.
//!
//! ail has value semantics with no references, so every value is owned by exactly one place.
//! The backend maps that onto Rust borrows: read-only params become `&T`, `mut` params `&mut T`
//! (which LLVM knows never alias), and copies are inserted only where a value is really duplicated.
//! A local whose final use is unambiguous is moved instead of cloned.

use crate::ast::*;
use crate::check::{pat_key, BindId, BindKind, Checked, Res};
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
}

pub fn mangle(n: &str) -> String {
    let mut o = String::new();
    for c in n.chars() {
        match c {
            '-' => o.push('_'),
            '?' => o.push_str("_q"),
            '.' => o.push_str("__"),
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
}

pub fn emit(prog: &Program, ck: &Checked, src: &str, file: &str, mode: Mode) -> String {
    let mut em = Emitter { ck, src, moves: HashSet::new(), tmp: 0, fn_name: String::new() };
    let mut out = String::new();
    out.push_str("#![allow(warnings)]\n#![allow(unconditional_panic, arithmetic_overflow)]\n");
    for it in &prog.items {
        match it {
            Item::Struct(s) => em.emit_struct(s, &mut out),
            Item::Enum(e) => em.emit_enum(e, &mut out),
            Item::Const(c) => {
                let t = &ck.info.consts[&c.name];
                let v = em.val(&c.value);
                writeln!(out, "const C_{}: {} = {};", mangle(&c.name), rty(t), v).unwrap();
            }
            _ => {}
        }
    }
    for it in &prog.items {
        if let Item::Fn(f) = it {
            em.emit_fn(f, &mut out);
        }
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
                    writeln!(out, "    if let Err(e) = {call} {{ ail_rt::main_error(&e); }}").unwrap();
                }
                out.push_str("    ail_rt::flush();\n}\n");
                let _ = main;
            } else {
                out.push_str("fn main() {}\n");
            }
        }
        Mode::Test => {
            let mut names = Vec::new();
            for (i, it) in prog.items.iter().enumerate() {
                if let Item::Test(t) = it {
                    em.moves = compute_moves(&t.body, ck);
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
    let (body, table) = line_table(&out, src);
    let mut fin = body;
    writeln!(fin, "static AIL_FILE: &str = {:?};", file).unwrap();
    writeln!(fin, "static AIL_LINES: [(u32, u32, u32); {}] = [{}];", table.len(), table.iter().map(|(a, b, c)| format!("({a},{b},{c})")).collect::<Vec<_>>().join(",")).unwrap();
    fin.push_str(crate::runtime::RUNTIME);
    fin.push_str(crate::runtime_net::RUNTIME_NET);
    fin
}

/// Replace `/*@OFFSET*/` markers with a table (rust line -> ail line, col).
fn line_table(out: &str, src: &str) -> (String, Vec<(u32, u32, u32)>) {
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
            table.push(((n + 1) as u32, l as u32, c as u32));
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
    fn block(es: &[Expr], ck: &Checked, uses: &mut HashMap<BindId, Vec<UseInfo>>, next: &mut usize, path: &mut Vec<usize>, depth: u32) {
        for e in es {
            *next += 1;
            path.push(*next);
            expr(e, ck, uses, next, path, depth);
            path.pop();
        }
    }
    fn expr(e: &Expr, ck: &Checked, uses: &mut HashMap<BindId, Vec<UseInfo>>, next: &mut usize, path: &mut Vec<usize>, depth: u32) {
        match &e.kind {
            ExprKind::Var(_) => {
                if let Res::Local(b) = ck.res[e.id as usize] {
                    uses.entry(b).or_default().push(UseInfo { id: e.id, depth, path: path.clone() });
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Break | ExprKind::Continue => {}
            ExprKind::Call(_, _, args) => args.iter().for_each(|a| expr(a.expr(), ck, uses, next, path, depth)),
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
            ExprKind::New(_, _, fs) => fs.iter().for_each(|f| expr(&f.2, ck, uses, next, path, depth)),
            ExprKind::With(b, fs) => {
                expr(b, ck, uses, next, path, depth);
                fs.iter().for_each(|f| expr(&f.2, ck, uses, next, path, depth));
            }
            ExprKind::Field(b, _, _) => expr(b, ck, uses, next, path, depth),
            ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => expr(x, ck, uses, next, path, depth),
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

    fn emit_struct(&mut self, s: &StructDef, out: &mut String) {
        let si = &self.ck.info.structs[&s.name];
        let copy = if self.ck.info.copy_types.contains(&s.name) { ", Copy" } else { "" };
        let copy = if self.ck.info.ord_types.contains(&s.name) { format!("{copy}, PartialOrd") } else { copy.to_string() };
        writeln!(out, "#[derive(Clone, Debug, PartialEq{copy})]\npub struct {} {{", ty_name(&s.name)).unwrap();
        for (i, (f, t)) in si.fields.iter().enumerate() {
            let rt = if self.ck.info.is_boxed(&s.name, NO_VARIANT, i) { format!("Box<{}>", rty(t)) } else { rty(t) };
            writeln!(out, "    pub {}: {},", mangle(f), rt).unwrap();
        }
        out.push_str("}\n");
    }

    fn emit_enum(&mut self, e: &EnumDef, out: &mut String) {
        let ei = &self.ck.info.enums[&e.name];
        let copy = if self.ck.info.copy_types.contains(&e.name) { ", Copy" } else { "" };
        let copy = if self.ck.info.ord_types.contains(&e.name) { format!("{copy}, PartialOrd") } else { copy.to_string() };
        writeln!(out, "#[derive(Clone, Debug, PartialEq{copy})]\npub enum {} {{", ty_name(&e.name)).unwrap();
        for (vi, (v, ts)) in ei.variants.iter().enumerate() {
            if ts.is_empty() {
                writeln!(out, "    {},", mangle(v)).unwrap();
            } else {
                let fs: Vec<String> = ts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| if self.ck.info.is_boxed(&e.name, vi, i) { format!("Box<{}>", rty(t)) } else { rty(t) })
                    .collect();
                writeln!(out, "    {}({}),", mangle(v), fs.join(", ")).unwrap();
            }
        }
        out.push_str("}\n");
    }

    fn param_rty(&self, t: &Ty, m: bool) -> String {
        if m {
            format!("&mut {}", rty(t))
        } else if self.is_copy(t) {
            rty(t)
        } else {
            format!("&{}", rty(t))
        }
    }

    fn emit_fn(&mut self, f: &FnDef, out: &mut String) {
        let sig = &self.ck.info.fns[&f.name];
        self.moves = compute_moves(&f.body, self.ck);
        self.fn_name = f.name.clone();
        let params: Vec<String> = f
            .params
            .iter()
            .zip(sig.params.iter())
            .map(|(p, (_, t, m))| format!("{}: {}", self.local_name(&p.name), self.param_rty(t, *m)))
            .collect();
        let ret = rty(&sig.ret);
        writeln!(out, "fn {}({}) -> {} {{", fn_name(&f.name), params.join(", "), ret).unwrap();
        for r in &f.requires {
            let c = self.val(r);
            writeln!(out, "    {}if !({c}) {{ ail_rt::contract(\"requires\", {:?}, {:?}); }}", self.mark(r), f.name, self.src_of(r)).unwrap();
        }
        let body = self.block_as(&f.body, sig.ret == Ty::Unit);
        if f.ensures.is_empty() {
            writeln!(out, "    let __r: {ret} = {body};\n    __r\n}}").unwrap();
        } else {
            writeln!(out, "    let __ret: {ret} = (|| -> {ret} {{ let __r: {ret} = {body}; __r }})();").unwrap();
            writeln!(out, "    {{ let v_result = &__ret;").unwrap();
            for en in &f.ensures {
                let c = self.val(en);
                writeln!(out, "    {}if !({c}) {{ ail_rt::contract(\"ensures\", {:?}, {:?}); }}", self.mark(en), f.name, self.src_of(en)).unwrap();
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
                Res::Local(b) => Some(self.bind_place(b)),
                _ => None,
            },
            ExprKind::Field(base, f, _) => {
                let p = self.place(base)?;
                Some(self.field_access(&p, base, f))
            }
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 => {
                let p = self.place(args[0].expr())?;
                let i = self.val(args[1].expr());
                Some(format!("{p}[ail_rt::ix({i})]"))
            }
            ExprKind::Copy(_) => None,
            _ => None,
        }
    }

    fn field_access(&self, base_code: &str, base: &Expr, f: &str) -> String {
        let Ty::User(sn) = self.ty(base) else { unreachable!() };
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
            return format!("{p}.clone()");
        }
        self.rvalue(e)
    }

    /// `&T` to the expression's value.
    fn rf(&mut self, e: &Expr) -> String {
        match self.place(e) {
            Some(p) => format!("&{p}"),
            None => format!("&({})", self.rvalue(e)),
        }
    }

    /// Something usable where Rust auto-refs (method receivers, format args): a place or an rvalue.
    fn pv(&mut self, e: &Expr) -> String {
        match self.place(e) {
            Some(p) => p,
            None => format!("({})", self.rvalue(e)),
        }
    }

    fn mut_place(&mut self, a: &Arg) -> String {
        self.place(a.expr()).expect("checker guarantees mutable places")
    }

    pub fn block(&mut self, es: &[Expr]) -> String {
        self.block_as(es, false)
    }

    /// A block; when `unit` is set, a non-Unit final value is discarded so the block is `()`.
    pub fn block_as(&mut self, es: &[Expr], unit: bool) -> String {
        let mut o = String::from("{\n");
        for (i, e) in es.iter().enumerate() {
            let last = i + 1 == es.len();
            let m = self.mark(e);
            match &e.kind {
                ExprKind::Let { mutable, value, .. } => {
                    let b = self.ck.decl[&(e.id as usize)];
                    let bi = &self.ck.bindings[b];
                    let name = self.local_name(&bi.name);
                    let ann = if bi.ty.is_err() { String::new() } else { format!(": {}", rty(&bi.ty)) };
                    let v = self.val(value);
                    let mu = if *mutable { "mut " } else { "" };
                    writeln!(o, "{m}let {mu}{name}{ann} = {v};").unwrap();
                    if last {
                        o.push_str("()\n");
                    }
                }
                _ => {
                    let code = self.rvalue_or_place(e);
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
                Res::Local(b) => self.bind_place(*b),
                other => unreachable!("var resolved to {other:?}"),
            },
            ExprKind::Let { .. } => format!("{{ {} }}", self.block(std::slice::from_ref(e))),
            ExprKind::Set(place, value) => {
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
                let head = match it {
                    ForIter::Range(a, z) => {
                        let a = self.val(a);
                        let z = self.val(z);
                        format!("{a}..{z}")
                    }
                    ForIter::Each(v) => format!("{}.iter()", self.pv(v)),
                };
                let bd = self.block(body);
                format!("for {var} in {head} {{ let _ = {bd}; }}")
            }
            ExprKind::Break => "break".into(),
            ExprKind::Continue => "continue".into(),
            ExprKind::Return(v) => match v {
                Some(v) => format!("return {}", self.val(v)),
                None => "return ()".into(),
            },
            ExprKind::Match(scrut, arms) => self.emit_match(scrut, arms, t == Ty::Unit),
            ExprKind::New(name, _, fields) => {
                let si = self.ck.info.structs[name].clone();
                let mut parts = Vec::new();
                for (f, _, v) in fields {
                    let idx = si.fields.iter().position(|x| &x.0 == f).unwrap();
                    let code = self.val(v);
                    let code = if self.ck.info.is_boxed(name, NO_VARIANT, idx) { format!("Box::new({code})") } else { code };
                    parts.push(format!("{}: {}", mangle(f), code));
                }
                format!("{} {{ {} }}", ty_name(name), parts.join(", "))
            }
            ExprKind::With(base, fields) => {
                let Ty::User(name) = self.ty(base).clone() else { unreachable!() };
                let si = self.ck.info.structs[&name].clone();
                let mut parts = Vec::new();
                for (f, _, v) in fields {
                    let idx = si.fields.iter().position(|x| &x.0 == f).unwrap();
                    let code = self.val(v);
                    let code = if self.ck.info.is_boxed(&name, NO_VARIANT, idx) { format!("Box::new({code})") } else { code };
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
                Some(p) => format!("{p}.clone()"),
                None => self.rvalue(inner),
            },
            ExprKind::Call(name, _, args) => match self.ck.res[id].clone() {
                Res::UserFn(n) => self.user_call(&n, args),
                Res::Variant(en, vi) => {
                    let v = self.ck.info.enums[&en].variants[vi].0.clone();
                    let mut parts = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        let code = self.val(a.expr());
                        parts.push(if self.ck.info.is_boxed(&en, vi, i) { format!("Box::new({code})") } else { code });
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

    fn emit_match(&mut self, scrut: &Expr, arms: &[Arm], unit: bool) -> String {
        let s = self.rf(scrut);
        let st = self.ty(scrut).clone();
        let mut o = format!("match {s} {{\n");
        for arm in arms {
            let mut guards = Vec::new();
            let p = self.pat(&arm.pat, &st, &mut guards);
            let g = if guards.is_empty() { String::new() } else { format!(" if {}", guards.join(" && ")) };
            let body = self.block_as(&arm.body, unit);
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
                let b = self.ck.decl[&pat_key(*span)];
                let _ = n;
                self.local_name(&self.ck.bindings[b].name)
            }
            Pat::Int(v, _) => self.lit_int(*v, t).trim_matches(|c| c == '(' || c == ')').to_string(),
            Pat::Str(s, _) => {
                let v = self.fresh();
                guards.push(format!("{v}.as_str() == {s:?}"));
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
                        let ps: Vec<String> = subs.iter().zip(ei.variants[vi].1.iter()).map(|(sp, st)| self.pat(sp, st, guards)).collect();
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
                        Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64) => format!("(0{} - {v})", k.rust()),
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
            "bit-and" => format!("({} & {})", self.val(a(0)), self.val(a(1))),
            "bit-or" => format!("({} | {})", self.val(a(0)), self.val(a(1))),
            "bit-xor" => format!("({} ^ {})", self.val(a(0)), self.val(a(1))),
            "shl" => format!("ail_rt::ShiftAil::checked_shl_ail({}, ail_rt::icast::<_, u32>({}))", self.val(a(0)), self.val(a(1))),
            "shr" => format!("ail_rt::ShiftAil::checked_shr_ail({}, ail_rt::icast::<_, u32>({}))", self.val(a(0)), self.val(a(1))),
            "abs" => match &t {
                Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64) => self.val(a(0)),
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
            "sqrt" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" | "exp" | "ln" => format!("({}).{name}()", self.val(a(0))),
            "atan2" => format!("({}).atan2({})", self.val(a(0)), self.val(a(1))),
            "str" => self.format(args),
            "print" | "println" | "eprintln" => {
                let f = self.format(&args[1..]);
                format!("ail_rt::{}(&{f})", name)
            }
            "len" => format!("({}.len() as i64)", self.pv(a(0))),
            "is-empty" => format!("{}.is_empty()", self.pv(a(0))),
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
                format!("{{ let __i = ail_rt::ix_incl({i}); let __x = {x}; {p}.insert(__i, __x); }}")
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
                format!("{{ let __i = ail_rt::ix({i}); let __j = ail_rt::ix({j}); {p}.swap(__i, __j); }}")
            }
            "clear" => format!("{}.clear()", self.mut_place(&args[0])),
            "extend" => {
                let x = self.val(a(1));
                let p = self.mut_place(&args[0]);
                format!("{{ let __x = {x}; {p}.extend(__x); }}")
            }
            "slice" => format!("ail_rt::slice({}, {}, {})", self.rf(a(0)), self.val(a(1)), self.val(a(2))),
            "concat" => format!("ail_rt::concat({}, {})", self.rf(a(0)), self.rf(a(1))),
            "contains" => {
                if at(self, 0) == Ty::Str {
                    format!("{}.contains({}.as_str())", self.pv(a(0)), self.pv(a(1)))
                } else {
                    format!("{}.contains({})", self.pv(a(0)), self.rf(a(1)))
                }
            }
            "index-of" => {
                if at(self, 0) == Ty::Str {
                    format!("{}.find({}.as_str()).map(|i| i as i64)", self.pv(a(0)), self.pv(a(1)))
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
                    format!("{p}.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))")
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
                    format!("{p}.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal))")
                }
            }
            "sum" => format!("{}.iter().sum::<{}>()", self.pv(a(0)), rty(&t)),
            "map" => "ail_rt::AilMap::default()".into(),
            "put" => {
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
                    format!("{{ let __v = {v}; match {p}.get_mut({kr}) {{ Some(__s) => *__s = __v, None => {{ {p}.insert({k}, __v); }} }} }}")
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
                let f = if matches!(at(self, 0), Ty::Opt(_)) { "expect_opt" } else { "expect_res" };
                format!("ail_rt::{f}({}, {})", self.val(a(0)), self.rf(a(1)))
            }
            "is-some" => format!("{}.is_some()", self.pv(a(0))),
            "is-none" => format!("{}.is_none()", self.pv(a(0))),
            "is-ok" => format!("{}.is_ok()", self.pv(a(0))),
            "is-err" => format!("{}.is_err()", self.pv(a(0))),
            "substr" => format!("ail_rt::substr({}, {}, {})", self.rf(a(0)), self.val(a(1)), self.val(a(2))),
            "split" => format!("ail_rt::split({}, {})", self.rf(a(0)), self.rf(a(1))),
            "trim" => format!("{}.trim().to_string()", self.pv(a(0))),
            "upper" => format!("{}.to_uppercase()", self.pv(a(0))),
            "lower" => format!("{}.to_lowercase()", self.pv(a(0))),
            "starts-with" => format!("{}.starts_with({}.as_str())", self.pv(a(0)), self.pv(a(1))),
            "ends-with" => format!("{}.ends_with({}.as_str())", self.pv(a(0)), self.pv(a(1))),
            "replace" => format!("{}.replace({}.as_str(), {}.as_str())", self.pv(a(0)), self.pv(a(1)), self.pv(a(2))),
            "join" => format!("{}.join({}.as_str())", self.pv(a(0)), self.pv(a(1))),
            "parse-i64" => format!("{}.trim().parse::<i64>().ok()", self.pv(a(0))),
            "parse-f64" => format!("{}.trim().parse::<f64>().ok()", self.pv(a(0))),
            "chars" => format!("ail_rt::chars({})", self.rf(a(0))),
            "byte-at" => format!("ail_rt::byte_at({}, {})", self.rf(a(0)), self.val(a(1))),
            "fmt-float" => format!("format!(\"{{:.*}}\", ail_rt::count({}), {})", self.val(a(1)), self.val(a(0))),
            "from-code" => format!("ail_rt::from_code({})", self.val(a(0))),
            "repeat" => format!("{}.repeat(ail_rt::count({}))", self.pv(a(0)), self.val(a(1))),
            "read-line" => "ail_rt::read_line()".into(),
            "read-file" => format!("ail_rt::read_file({}, {})", self.rf(a(1)), self.val(a(2))),
            "write-file" => format!("ail_rt::write_file({}, {})", self.rf(a(1)), self.rf(a(2))),
            "now-ms" => "ail_rt::now_ms()".into(),
            "now-ns" => "ail_rt::now_ns()".into(),
            "rand-int" => format!("ail_rt::rand_int({}, {})", self.val(a(1)), self.val(a(2))),
            "rand-f64" => "ail_rt::rand_f64()".into(),
            "args" => "ail_rt::args()".into(),
            "read-all" => "ail_rt::read_all()".into(),
            "lines" => format!("ail_rt::lines({})", self.rf(a(0))),
            "is-digit" => format!("ail_rt::all_chars({}, |c| c.is_ascii_digit())", self.rf(a(0))),
            "is-alpha" => format!("ail_rt::all_chars({}, |c| c.is_alphabetic())", self.rf(a(0))),
            "is-alnum" => format!("ail_rt::all_chars({}, |c| c.is_alphanumeric())", self.rf(a(0))),
            "is-space" => format!("ail_rt::all_chars({}, |c| c.is_whitespace())", self.rf(a(0))),
            "is-upper" => format!("ail_rt::all_chars({}, |c| c.is_uppercase())", self.rf(a(0))),
            "is-lower" => format!("ail_rt::all_chars({}, |c| c.is_lowercase())", self.rf(a(0))),
            "code-of" => format!("ail_rt::code_of({})", self.rf(a(0))),
            "json-parse" => format!("ail_json::parse({})", self.rf(a(0))),
            "json-str" => format!("ail_json::to_string({})", self.rf(a(0))),
            "http-next" => format!("ail_http::next({}, {})", self.val(a(1)), self.val(a(2))),
            "http-respond" => format!("ail_http::respond({}, {}, {}, {})", self.rf(a(1)), self.val(a(2)), self.rf(a(3)), self.rf(a(4))),
            "fail" => format!("ail_rt::fail({})", self.rf(a(0))),
            "assert" => {
                let c = self.val(a(0));
                let msg = if args.len() == 2 { self.rf(a(1)) } else { format!("&String::from({:?})", self.src_of(a(0))) };
                format!("if !{c} {{ ail_rt::assert_fail({msg}) }}")
            }
            other => unreachable!("builtin {other}"),
        }
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
        } else {
            format!("format!({fmt:?}, {})", vals.join(", "))
        }
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
