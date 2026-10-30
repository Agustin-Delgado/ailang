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
    /// Emitting the fast copy of a loop whose integer arithmetic records overflow in `__of`
    /// instead of trapping (see `for_unchecked`).
    defer: bool,
    /// Index arithmetic `j + c` in the fast copy of a loop, tested before the loop to stay in
    /// range of its type: emitted without a check, so LLVM sees plain affine addresses.
    affine: HashSet<u32>,
    /// Emitting the body of a loop that runs in parallel chunks (see `for_par`).
    in_par: bool,
    /// Statements of a parallel body emitted differently: update id -> what to emit.
    par_hooks: HashMap<u32, ParHook>,
    /// Vectors a parallel loop writes from several threads: root -> its `SharedSlice`.
    par_shared: HashMap<BindId, String>,
    /// Set while emitting the sequential copy of a loop that has a parallel one: loops inside
    /// it stay sequential (the outer loop already is the parallel one, and the code stays small).
    no_par: bool,
    /// Fns that loop, directly or through the fns they call, or call themselves: a loop body
    /// calling one is worth spreading even over few iterations.
    heavy_fns: HashSet<String>,
    /// Fns that use par-map, directly or through the fns they call.
    par_fns: HashSet<String>,
    /// Native builds may send loops to the GPU (see `gpu_attempt`).
    gpu_on: bool,
    /// Every fn definition, for kernels that call helpers.
    fn_defs: HashMap<String, FnDef>,
    /// Static descriptions of the kernels written so far.
    kernels: Vec<String>,
    /// Second emission pass: vectors in GPU-aware fns carry mirrors (see `gpu_sync`).
    pass2: bool,
    /// Found by the first pass: loops that became kernels, and the fns that hold them.
    kernel_loops: HashSet<u32>,
    kernel_fns: HashSet<String>,
    /// Fns with kernels and the fns that call them: their vectors keep device copies.
    aware: HashSet<String>,
    /// Mirror of each vector binding in the fn being emitted: (Rust name, kind).
    mirrors: HashMap<BindId, (String, MirrorKind)>,
    /// Above zero while emitting code that runs on the CPU as a whole (synced before).
    gpu_quiet: u32,
    /// Read-only parameters already shadowed by a CPU copy in the current block.
    viewed: HashSet<BindId>,
    /// The kernel loop being emitted as a statement of GPU-structured code: its launch uses
    /// the vectors' mirrors (elsewhere a launch gets its own and leaves the CPU copy current).
    kernel_stmt: Option<u32>,
    /// Emitting the fast copy of a loop that runs on 32-bit copies of its vectors (see
    /// `narrow_region`).
    nr: Option<NarrowCtx>,
    /// Set while emitting the ordinary copy of such a loop: no second region inside it.
    nr_off: bool,
    /// Above zero inside a lambda: a `break` out of the fast copy cannot be written there.
    nr_lam: u32,
    web: bool,
}

/// The fast copy of a loop that works on 32-bit copies of its I64 vectors.
struct NarrowCtx {
    /// The labeled block around the fast copy: `break` with false throws the run away.
    label: String,
    /// Vector binding -> its narrow copy and the range its elements must stay in.
    vecs: HashMap<BindId, NarrowVec>,
    /// `(at v i)` calls whose index is proven in range by those element ranges.
    unchecked: HashSet<u32>,
    /// Vectors declared inside the loop (`fill`): created narrow.
    inner: HashSet<BindId>,
    /// Stores whose value is known to be in range (see `NarrowPlan::copies`).
    nocheck: HashSet<u32>,
}

#[derive(Clone)]
struct NarrowVec {
    name: String,
    lo: String,
    hi: String,
}

/// What `narrow_plan` found a loop can run on narrow vectors with.
struct NarrowPlan {
    /// Vectors from before the loop, converted on entry; the written ones are copied back.
    outer: Vec<BindId>,
    written: Vec<BindId>,
    inner: Vec<BindId>,
    /// Plain variables from before the loop that it assigns: put back if the run is thrown away.
    scalars: Vec<BindId>,
    /// Bounds the elements of a vector must respect for the unchecked reads: lowest allowed
    /// value (a constant), and Rust expressions for the highest.
    lows: HashMap<BindId, i128>,
    highs: HashMap<BindId, Vec<String>>,
    unchecked: HashSet<u32>,
    /// `(set (at q i) y)` with y read from narrowed vector a: id -> a. The value is already in
    /// a's range, so the store needs no check when that range lies within q's.
    copies: HashMap<u32, BindId>,
}

/// How the fn being emitted holds a vector that has a mirror.
#[derive(Clone, Copy, PartialEq)]
enum MirrorKind {
    /// A local: the fn owns it and its mirror.
    Owned,
    /// A `mut` parameter: written in place, the caller's mirror comes along.
    MutParam,
    /// A read-only parameter: never written here; read on the CPU through a copy if needed.
    RefParam,
}

/// How a statement of a parallel loop body is emitted (see `for_par`).
#[derive(Clone)]
enum ParHook {
    /// `(set x (+ x e))`: add to the chunk's private sum and to its magnitude bound.
    Sum(String),
    /// `(set (at h k) (+ (at h k) e))`: the same on the chunk's private copy of `h`.
    Hist(String),
    /// The `if` of a running maximum or minimum: also note that it fired.
    Flag(String),
    /// `(set x (% (+ x e) m))`: a sum modulo `m` on the chunk's private copy; notes the
    /// largest term and any negative one (then the parallel result would differ).
    ModSum(String),
    /// A plain `(set x v)` of a variable the body never reads: note that it happened.
    Last(String),
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
        defer: false,
        affine: HashSet::new(),
        in_par: false,
        par_hooks: HashMap::new(),
        par_shared: HashMap::new(),
        no_par: false,
        heavy_fns: HashSet::new(),
        par_fns: HashSet::new(),
        gpu_on: !matches!(mode_kind, ModeKind::Web) && std::env::var("AIL_NO_GPU").is_err(),
        fn_defs: prog.items.iter().filter_map(|i| match i { Item::Fn(f) => Some((f.name.clone(), f.clone())), _ => None }).collect(),
        kernels: Vec::new(),
        pass2: false,
        kernel_loops: HashSet::new(),
        kernel_fns: HashSet::new(),
        aware: HashSet::new(),
        mirrors: HashMap::new(),
        gpu_quiet: 0,
        viewed: HashSet::new(),
        kernel_stmt: None,
        nr: None,
        nr_off: false,
        nr_lam: 0,
        web: matches!(mode_kind, ModeKind::Web),
    };
    em.infer_view_fns(prog);
    em.heavy_fns = fns_reaching(prog, |e| matches!(e.kind, ExprKind::For(..) | ExprKind::While(..)), true);
    em.par_fns = fns_reaching(prog, |e| matches!(&e.kind, ExprKind::Call(n, _, _) if n == "par-map" || n == "par-run"), false);
    let mut out = emit_items(&mut em, prog, ck, &mode);
    if !em.kernels.is_empty() {
        // Loops became kernels: emit again, now knowing which fns hold device copies.
        em.aware = gpu_aware(prog, ck, &em.kernel_fns);
        em.pass2 = true;
        em.kernels.clear();
        em.tmp = 0;
        out = emit_items(&mut em, prog, ck, &mode);
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
    if !em.kernels.is_empty() {
        for k in &em.kernels {
            fin.push_str(k);
            fin.push('\n');
        }
        fin.push_str(crate::runtime_gpu::RUNTIME_GPU);
    }
    if matches!(mode_kind, ModeKind::Web) {
        fin.push_str(crate::runtime_web::RUNTIME_WEB);
    } else {
        // Native builds of an app (e.g. `ail test` on a game): drawing does nothing.
        fin.push_str(crate::runtime_web::RUNTIME_WEB_STUB);
    }
    fin
}

/// Fns that hold kernels, and every fn that calls one of them.
fn gpu_aware(prog: &Program, ck: &Checked, kernel_fns: &HashSet<String>) -> HashSet<String> {
    let mut calls: HashMap<String, HashSet<String>> = HashMap::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            let mut cs = HashSet::new();
            for e in &f.body {
                walk_ref(e, &mut |x| {
                    if let (ExprKind::Call(..), Some(Res::UserFn(g))) = (&x.kind, ck.res.get(x.id as usize)) {
                        cs.insert(g.clone());
                    }
                });
            }
            calls.insert(f.name.clone(), cs);
        }
    }
    let mut aware = kernel_fns.clone();
    loop {
        let mut grew = false;
        for (f, cs) in &calls {
            if !aware.contains(f) && cs.iter().any(|g| aware.contains(g)) {
                aware.insert(f.clone());
                grew = true;
            }
        }
        if !grew {
            return aware;
        }
    }
}

fn emit_items(em: &mut Emitter, prog: &Program, ck: &Checked, mode: &Mode) -> String {
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
                if !em.kernels.is_empty() {
                    out.push_str("    ail_gpu::start();\n");
                }
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
                    em.mirrors.clear();
                    em.viewed.clear();
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
    out
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
        self.mirrors.clear();
        self.viewed.clear();
        let aware = self.pass2 && self.aware.contains(&f.name);
        let pbinds = self.ck.fn_params.get(&f.name).cloned().unwrap_or_default();
        let mut params: Vec<String> = Vec::new();
        for (i, (p, (_, t, m))) in f.params.iter().zip(sig.params.iter()).enumerate() {
            let ty = if vf.is_some_and(|v| v.0 == i) {
                "&'a str".to_string()
            } else {
                self.param_rty(t, *m)
            };
            params.push(format!("{}: {}", self.local_name(&p.name), ty));
            if aware && gpu_vec(t) {
                let name = format!("__mp{i}");
                params.push(format!("{name}: &ail_gpu::Mirror"));
                if let Some(b) = pbinds.get(i) {
                    self.mirrors.insert(*b, (name, if *m { MirrorKind::MutParam } else { MirrorKind::RefParam }));
                }
            }
        }
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
                Res::Local(b) if self.narrow_of(b).is_some() => Some(self.narrow_of(b).unwrap().name),
                Res::Local(b) => Some(self.bind_place(b)),
                _ => None,
            },
            ExprKind::Field(base, f, _) => {
                let p = self.place(base)?;
                Some(self.field_access(&p, base, f))
            }
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 && self.narrow_at(e).is_some() => {
                // A read: the narrow copy holds exactly the values the program stored.
                let nv = self.narrow_at(e).unwrap();
                let i = self.val(args[1].expr());
                if self.nr.as_ref().is_some_and(|c| c.unchecked.contains(&e.id)) {
                    Some(format!("((*unsafe {{ {}.get_unchecked(({i}) as usize) }}) as i64)", nv.name))
                } else {
                    Some(format!("({}[ail_rt::Ix({i})] as i64)", nv.name))
                }
            }
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 => {
                if let Some(code) = self.unchecked.get(&e.id) {
                    return Some(code.clone());
                }
                let p = self.place(args[0].expr())?;
                let i = self.val(args[1].expr());
                Some(format!("{p}[ail_rt::Ix({i})]"))
            }
            ExprKind::Copy(_) => None,
            _ => None,
        }
    }

    fn narrow_of(&self, b: BindId) -> Option<NarrowVec> {
        self.nr.as_ref().and_then(|c| c.vecs.get(&b).cloned())
    }

    /// The narrow copy behind `(at v i)`, if `v` has one.
    fn narrow_at(&self, e: &Expr) -> Option<NarrowVec> {
        let ExprKind::Call(n, _, args) = &e.kind else { return None };
        if n != "at" || args.len() != 2 || self.nr.is_none() {
            return None;
        }
        match (&args[0].expr().kind, self.ck.res.get(args[0].expr().id as usize)) {
            (ExprKind::Var(_), Some(Res::Local(b))) => self.narrow_of(*b),
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
        self.gpu_quiet += 1;
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
        self.gpu_quiet -= 1;
        out.push_str("}); }");
        out
    }

    pub fn block_as(&mut self, es: &[Expr], unit: bool) -> String {
        self.block_tail(es, if unit { Tail::Unit } else { Tail::Val })
    }

    fn block_tail(&mut self, es: &[Expr], tail: Tail) -> String {
        let unit = tail == Tail::Unit;
        let mut o = String::from("{\n");
        let hooks = self.gpu_hooks();
        let saved_viewed = if hooks { Some(self.viewed.clone()) } else { None };
        for (i, e) in es.iter().enumerate() {
            let last = i + 1 == es.len();
            let m = self.mark(e);
            // GPU-aware fns: make the CPU copies current before a statement that uses them.
            let quiet = if hooks {
                let (pre, quiet) = self.gpu_sync(e);
                o.push_str(&pre);
                quiet
            } else {
                false
            };
            if quiet {
                self.gpu_quiet += 1;
            }
            let saved_ks = self.kernel_stmt;
            if hooks && matches!(e.kind, ExprKind::For(..)) && self.kernel_loops.contains(&e.id) {
                self.kernel_stmt = Some(e.id);
            }
            match &e.kind {
                ExprKind::Let { value, .. }
                    if self.nr.as_ref().is_some_and(|c| c.inner.contains(&self.ck.decl[&(e.id as usize)])) =>
                {
                    // `(fill n x)` made narrow; x out of range stops the fast run.
                    let b = self.ck.decl[&(e.id as usize)];
                    let nv = self.narrow_of(b).unwrap();
                    let label = self.nr.as_ref().unwrap().label.clone();
                    let ExprKind::Call(_, _, fa) = &value.kind else { unreachable!() };
                    let x = self.val(fa[1].expr());
                    let n = self.val(fa[0].expr());
                    writeln!(
                        o,
                        "{m}let mut {}: Vec<i32> = {{ let __x: i64 = {x}; let __c = ail_rt::count({n}); match ail_rt::narrow_fill(__c, __x, {}, {}) {{ Some(__w) => __w, None => break {label} false }} }};",
                        nv.name, nv.lo, nv.hi
                    )
                    .unwrap();
                    if last {
                        o.push_str("()
");
                    }
                }
                ExprKind::Let { mutable, value, .. } => {
                    let b = self.ck.decl[&(e.id as usize)];
                    let bi = &self.ck.bindings[b];
                    let name = self.local_name(&bi.name);
                    let ann = if bi.ty.is_err() {
                        String::new()
                    } else {
                        format!(": {}", rty(&bi.ty))
                    };
                    let mirrored = hooks && gpu_vec(&bi.ty) && !self.views.contains_key(&b);
                    let dev = if mirrored { self.dev_copy(value, b) } else { None };
                    let (ann, v) = match (self.views.get(&b).copied(), dev) {
                        (_, Some((pre, code))) => {
                            o.push_str(&pre);
                            (ann, code)
                        }
                        (Some(View::Ref), _) => (": &str".to_string(), self.vref(value)),
                        (Some(View::Cow), _) => (": std::borrow::Cow<str>".to_string(), self.vcow(value)),
                        (None, _) => (ann, self.val(value)),
                    };
                    let mu = if *mutable || mirrored { "mut " } else { "" };
                    writeln!(o, "{m}let {mu}{name}{ann} = {v};").unwrap();
                    if mirrored && !self.mirrors.contains_key(&b) {
                        let mn = format!("__m{b}");
                        writeln!(o, "let {mn} = ail_gpu::Mirror::new();").unwrap();
                        self.mirrors.insert(b, (mn, MirrorKind::Owned));
                    }
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
            if quiet {
                self.gpu_quiet -= 1;
            }
            self.kernel_stmt = saved_ks;
        }
        if let Some(v) = saved_viewed {
            self.viewed = v;
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
            ExprKind::Set(place, value) if self.narrow_at(place).is_some() => {
                // Stored narrow only if it stays in range; otherwise the fast run stops here.
                let nv = self.narrow_at(place).unwrap();
                let ExprKind::Call(_, _, args) = &place.kind else { unreachable!() };
                let v = self.val(value);
                let i = self.val(args[1].expr());
                let label = self.nr.as_ref().unwrap().label.clone();
                let sure = self.nr.as_ref().unwrap().nocheck.contains(&e.id);
                let fast = self.nr.as_ref().unwrap().unchecked.contains(&place.id);
                if sure && fast {
                    format!("{{ let __v = {v}; let __i = ({i}) as usize; unsafe {{ *{}.get_unchecked_mut(__i) = __v as i32; }} }}", nv.name)
                } else if sure {
                    format!("{{ let __v = {v}; let __i = ail_rt::Ix({i}); {}[__i] = __v as i32; }}", nv.name)
                } else if fast {
                    format!(
                        "{{ let __v = {v}; let __i = ({i}) as usize; if !unsafe {{ ail_rt::nput_unchecked(&mut {}, __i, __v, {}, {}) }} {{ break {label} false; }} }}",
                        nv.name, nv.lo, nv.hi
                    )
                } else {
                    format!(
                        "{{ let __v = {v}; let __i = ail_rt::ix({i}); if !ail_rt::nput(&mut {}, __i, __v, {}, {}) {{ break {label} false; }} }}",
                        nv.name, nv.lo, nv.hi
                    )
                }
            }
            ExprKind::Set(place, value) => {
                if let Some(h) = self.par_hooks.get(&e.id).cloned() {
                    return self.par_update(h, place, value);
                }
                if let Some(code) = self.append_in_place(place, value) {
                    return code;
                }
                let v = self.val(value);
                let p = self.place(place).expect("checked place");
                format!("{{ let __v = {v}; {p} = __v; }}")
            }
            ExprKind::If(c, th, el) if matches!(self.par_hooks.get(&e.id), Some(ParHook::Flag(_))) => {
                let Some(ParHook::Flag(flag)) = self.par_hooks.get(&e.id).cloned() else { unreachable!() };
                let _ = el;
                let c = self.val(c);
                let tb = self.block_as(std::slice::from_ref(th), true);
                format!("if {c} {{ let _ = {tb}; {flag} = true; }}")
            }
            ExprKind::If(c, th, None) if self.select_store(c, th).is_some() => self.select_store(c, th).unwrap(),
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
                if self.narrow_ok() {
                    if let Some(code) = self.narrow_region(e) {
                        return code;
                    }
                }
                // Kernels in the body may change what the condition reads: sync it each time.
                let sync = if self.gpu_hooks() { self.gpu_sync_vars(c, false) } else { String::new() };
                let c = self.val(c);
                let b = self.block(body);
                if sync.is_empty() {
                    format!("while {c} {{ let _ = {b}; }}")
                } else {
                    format!("while {{ {sync}{c} }} {{ let _ = {b}; }}")
                }
            }
            ExprKind::For(_, _, it, body) => {
                let b = self.ck.decl[&id];
                let var = self.local_name(&self.ck.bindings[b].name);
                if !self.in_par && !self.defer && !self.no_par {
                    if let Some(code) = self.for_par(b, &var, e, it, body) {
                        return code;
                    }
                }
                if self.narrow_ok() {
                    if let Some(code) = self.narrow_region(e) {
                        return code;
                    }
                }
                if let ForIter::Range(a, z) = it {
                    let lo = self.val(a);
                    let hi = self.val(z);
                    if let Some(snap) = self.chunk_plan(b, body) {
                        return self.for_chunked(&var, &lo, &hi, body, &snap);
                    }
                    return self.for_range_seq(b, &var, &lo, &hi, a, body);
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
        let mut post = String::new();
        let mut parts = Vec::new();
        let aware = self.pass2 && self.aware.contains(name);
        for (a, (_, pt, pm)) in args.iter().zip(sig.params.iter()) {
            let x = a.expr();
            let n_before = parts.len();
            // After the vector itself, its mirror (the caller's, or one for this call).
            let mirror = if aware && gpu_vec(pt) {
                let own = match (&x.kind, self.ck.res.get(x.id as usize)) {
                    (ExprKind::Var(_), Some(Res::Local(b))) => self.mirrors.get(b).cloned(),
                    _ => None,
                };
                Some(match own {
                    Some((m, MirrorKind::Owned)) => format!("&{m}"),
                    Some((m, _)) => m,
                    None => {
                        let t = self.fresh();
                        write!(pre, "let {t} = ail_gpu::Mirror::new(); ").unwrap();
                        if *pm {
                            let p = self.place(x).expect("checked mutable place");
                            write!(post, "{t}.to_cpu(&mut {p}); ").unwrap();
                        }
                        format!("&{t}")
                    }
                })
            } else {
                None
            };
            let _ = n_before;
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
                        if has_mut || mirror.is_some() {
                            let t = self.fresh();
                            write!(pre, "let {t} = {v}; ").unwrap();
                            parts.push(format!("&{t}"));
                        } else {
                            parts.push(format!("&({v})"));
                        }
                    }
                }
            }
            if let Some(m) = mirror {
                parts.push(m);
            }
        }
        let call = format!("{}({})", fn_name(name), parts.join(", "));
        if pre.is_empty() && post.is_empty() {
            call
        } else if post.is_empty() {
            format!("{{ {pre}{call} }}")
        } else {
            format!("{{ {pre}let __cr = {call}; {post}__cr }}")
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
        if self.affine.contains(&e.id) && args.len() == 2 && (name == "+" || name == "-") {
            let op = if name == "+" { "add" } else { "sub" };
            return format!("{}.unchecked_{op}({})", self.val(a(0)), self.val(a(1)));
        }
        if self.defer && t.is_int() {
            if let Some(code) = self.deferred(name, &t, args) {
                return code;
            }
        }
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
                if t.is_int() && matches!(a(0).kind, ExprKind::Int(2)) {
                    // A shift while it fits: the general pow is a loop of checked multiplies.
                    format!("<{} as ail_rt::Pow2>::pow2_ail({})", rty(&t), self.val(a(1)))
                } else if t.is_int() {
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
            "print" | "println" if self.in_par => {
                let f = self.format(&args[1..]);
                let nl = if name == "println" { " __pout.push('\\n');" } else { "" };
                format!("{{ __pout.push_str(&{f});{nl} }}")
            }
            "print" | "println" if self.nr.is_some() => {
                // Held until the fast run is kept; a run that prints too much is not worth it.
                let f = self.format(&args[1..]);
                if self.nr_lam > 0 {
                    format!("ail_rt::hold_{name}(&{f})")
                } else {
                    let label = self.nr.as_ref().unwrap().label.clone();
                    format!("{{ ail_rt::hold_{name}(&{f}); if ail_rt::hold_len() > (1 << 26) {{ break {label} false; }} }}")
                }
            }
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
                    format!("({b})[ail_rt::Ix({i})]")
                } else {
                    format!("({b})[ail_rt::Ix({i})].clone()")
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
                // On all cores for long vectors; the same order a sequential sort gives.
                if el.is_float() {
                    format!("ail_rt::psort(&mut {p}, false, |a, b| a.total_cmp(b))")
                } else if !matches!(el, Ty::Int(_) | Ty::Str | Ty::Bool) {
                    format!("ail_rt::psort(&mut {p}, true, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))")
                } else {
                    format!("ail_rt::psort(&mut {p}, false, |a, b| a.cmp(b))")
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
                    format!("ail_rt::psort(&mut {p}, false, |a, b| b.total_cmp(a))")
                } else {
                    format!("ail_rt::psort(&mut {p}, true, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal))")
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
                self.gpu_quiet += 1;
                self.nr_lam += 1;
                let b = self.block_as(body, r.as_ref() == &Ty::Unit);
                self.nr_lam -= 1;
                self.gpu_quiet -= 1;
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

    /// `(for v (range lo hi) body)` run in order on this thread; `lo` and `hi` are code.
    fn for_range_seq(&mut self, b: BindId, var: &str, lo: &str, hi: &str, a: &Expr, body: &[Expr]) -> String {
        // Bounds evaluated once, as Rust's a..z would; vectors the body pushes to on every
        // iteration get their room up front.
        let mut pre = format!("let __rlo: i64 = {lo}; let __rhi: i64 = {hi}; ");
        for p in self.push_targets(body) {
            pre.push_str(&format!("if __rhi > __rlo {{ let _ = {p}.try_reserve(((__rhi - __rlo) as u64).min(1 << 22) as usize); }} "));
        }
        let lp = match self.for_unchecked(b, var, "__rlo", "__rhi", a, body) {
            Some(code) => code,
            None => {
                let bd = self.block(body);
                format!("for {var} in __rlo..__rhi {{ let _ = {bd}; }}")
            }
        };
        format!("{{ {pre}{lp} }}")
    }

    /// A statement of a parallel body that updates a private accumulator.
    fn par_update(&mut self, h: ParHook, place: &Expr, value: &Expr) -> String {
        if let ParHook::Last(flag) = &h {
            let v = self.val(value);
            let pl = self.place(place).unwrap();
            return format!("{{ let __v = {v}; {pl} = __v; {flag} = true; }}");
        }
        let ExprKind::Call(op, _, xs) = &value.kind else { unreachable!("parallel update") };
        match h {
            ParHook::Sum(abs) => {
                let x = self.root_of(place).unwrap();
                let pl = self.place(place).unwrap();
                // The terms other than the accumulator itself.
                let mut terms = Vec::new();
                let mut skipped = false;
                for a in xs {
                    let is_x = !skipped
                        && matches!((&a.expr().kind, &self.ck.res[a.expr().id as usize]), (ExprKind::Var(_), Res::Local(b)) if *b == x);
                    if is_x {
                        skipped = true;
                    } else {
                        terms.push(self.val(a.expr()));
                    }
                }
                let e = terms.join(" + ");
                let sign = if op == "-" { "-" } else { "+" };
                format!("{{ let __e = {e}; {abs} = {abs}.saturating_add(ail_rt::UAbs::uabs(__e)); {pl} = {pl} {sign} __e; }}")
            }
            ParHook::Hist(abs) => {
                let ExprKind::Call(_, _, pa) = &place.kind else { unreachable!() };
                let k = self.val(pa[1].expr());
                let base = self.place(pa[0].expr()).unwrap();
                let at_src = self.src_of(place).to_string();
                let mut terms = Vec::new();
                let mut skipped = false;
                for a in xs {
                    if !skipped && self.src_of(a.expr()) == at_src {
                        skipped = true;
                    } else {
                        terms.push(self.val(a.expr()));
                    }
                }
                let e = terms.join(" + ");
                format!(
                    "{{ let __k = ail_rt::Ix({k}); let __e = {e}; {abs} = {abs}.saturating_add(ail_rt::UAbs::uabs(__e)); {base}[__k] = {base}[__k] + __e; }}"
                )
            }
            ParHook::ModSum(k) => {
                let ExprKind::Call(_, _, ys) = &xs[0].expr().kind else { unreachable!() };
                let x = self.root_of(place).unwrap();
                let pl = self.place(place).unwrap();
                let m = self.val(xs[1].expr());
                let mut terms = Vec::new();
                let mut skipped = false;
                for a in ys {
                    let is_x = !skipped
                        && matches!((&a.expr().kind, &self.ck.res[a.expr().id as usize]), (ExprKind::Var(_), Res::Local(b)) if *b == x);
                    if is_x {
                        skipped = true;
                    } else {
                        terms.push(self.val(a.expr()));
                    }
                }
                let e = terms.join(" + ");
                format!(
                    "{{ let __e = {e}; if __e < 0 {{ __neg{k} = true; }} if __e > __mx{k} {{ __mx{k} = __e; }} {pl} = ({pl} + __e) % {m}; }}"
                )
            }
            ParHook::Last(flag) => {
                let v = self.val(value);
                let pl = self.place(place).unwrap();
                format!("{{ let __v = {v}; {pl} = __v; {flag} = true; }}")
            }
            ParHook::Flag(_) => unreachable!("flag hooks are on ifs"),
        }
    }

    /// A `(for v (range lo hi) ...)` whose iterations are independent, run in parallel chunks.
    /// The body may read anything; besides its own variables it may only
    /// - add to an integer variable: `(set x (+ x e))` (each chunk sums privately; the total is
    ///   used only if the sizes of the terms prove that no partial sum in program order could
    ///   have overflowed, else the loop runs sequentially),
    /// - add to elements of an integer vector the same way, a histogram,
    /// - keep a running maximum or minimum: `(if (> e m) (do (set m e) (set at i)))`,
    /// - write elements `(at v k)` where `k` is `i`, or `i * s + j` with `j` from an inner
    ///   `(range a b)` and `b - a <= s` (checked before the loop), and not read `v`,
    /// - push to a vector, and print to the console (both kept in iteration order).
    /// Chunks run on the thread pool and combine in order, so the result is the sequential one.
    /// A trap in any chunk discards the parallel run and the loop runs again sequentially,
    /// trapping where the program order says.
    fn for_par(&mut self, lv: BindId, var: &str, whole: &Expr, it: &ForIter, body: &[Expr]) -> Option<String> {
        // Over a range, the loop variable is an index; over the elements of a vector (or the
        // lines or pieces of a string, collected first), it is the element.
        let (range, each): (Option<(&Expr, &Expr)>, Option<&Expr>) = match it {
            ForIter::Range(a, z) => (Some((a, z)), None),
            ForIter::Each(v) => (None, Some(v)),
        };
        if let Some(v) = each {
            let ok = match &v.kind {
                ExprKind::Call(n, _, args) if (n == "lines" && args.len() == 1) || (n == "split" && args.len() == 2) => {
                    matches!(self.ck.res.get(v.id as usize), Some(Res::Builtin(_)))
                        && self.place(args[0].expr()).is_some()
                        && args.iter().skip(1).all(|x| matches!(x.expr().kind, ExprKind::Str(_)))
                        && !self.body_mutates(body, args[0].expr())
                }
                _ => matches!(self.ty(v), Ty::Vec(_)) && self.place(v).is_some() && !self.body_mutates(body, v),
            };
            if !ok {
                return None;
            }
        }
        // Everything declared inside the body, `let`s to look through, inner range loops.
        let mut inner: HashSet<BindId> = HashSet::new();
        let mut lets: HashMap<BindId, &Expr> = HashMap::new();
        let mut ranges: HashMap<BindId, (&Expr, &Expr)> = HashMap::new();
        let mut heavy = false;
        let mut bad = false;
        let mut prints = false;
        let mut cap_ok: HashSet<u32> = HashSet::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let Some(b) = self.ck.decl.get(&(e.id as usize)) {
                    inner.insert(*b);
                    if let ExprKind::Let { mutable: false, value, .. } = &e.kind {
                        lets.insert(*b, value);
                    }
                    if let ExprKind::For(_, _, ForIter::Range(lo, hi), _) = &e.kind {
                        ranges.insert(*b, (lo, hi));
                    }
                }
                match &e.kind {
                    ExprKind::For(..) | ExprKind::While(..) => heavy = true,
                    ExprKind::Return(_) | ExprKind::Try(_) | ExprKind::Tasks(..) => bad = true,
                    ExprKind::Match(_, arms) => {
                        for arm in arms {
                            pat_binds(&arm.pat, self.ck, self.epoch, &mut inner);
                        }
                    }
                    ExprKind::Call(n, _, args) => match self.ck.res.get(e.id as usize) {
                        Some(Res::UserFn(f)) => {
                            heavy |= self.heavy_fns.contains(f) || !self.ck.info.fns.contains_key(f);
                            // A fn that may use the GPU stays on the calling thread.
                            bad |= self.par_fns.contains(f) || (self.pass2 && self.aware.contains(f));
                        }
                        Some(Res::LocalFn(_)) => bad = true,
                        Some(Res::Builtin(_)) => match n.as_str() {
                            "print" | "println" => {
                                prints = true;
                                if matches!(self.ty(args[0].expr()), Ty::Cap(crate::types::Cap::Console)) {
                                    cap_ok.insert(args[0].expr().id);
                                }
                            }
                            // The body already spreads its work: keep that parallelism.
                            "par-map" => bad = true,
                            "sort-by" | "map" | "filter" => heavy = true,
                            "par-run" | "http-serve" | "http-request" | "send" | "recv" | "eprintln" => bad = true,
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            });
        }
        if bad || breaks_out(body) {
            return None;
        }
        // Scratch variables: from outside the body, but every iteration overwrites them before
        // reading them, with `(set v ...)` or `(for j (range 0 n) (set (at v j) x))` where n is
        // the length of v (checked before the loop). Each chunk then works on its own copy.
        let mut scratch: Vec<(BindId, Option<(&Expr, &Expr)>)> = Vec::new();
        {
            let locals_in = |e: &Expr| {
                let mut out = HashSet::new();
                walk_ref(e, &mut |x| {
                    if let (ExprKind::Var(_), Some(Res::Local(b))) = (&x.kind, self.ck.res.get(x.id as usize)) {
                        out.insert(*b);
                    }
                });
                out
            };
            let is_len_of = |e: &Expr, v: BindId| {
                matches!(&e.kind, ExprKind::Call(n, _, xs) if n == "len" && xs.len() == 1
                    && matches!((&xs[0].expr().kind, self.ck.res.get(xs[0].expr().id as usize)), (ExprKind::Var(_), Some(Res::Local(b))) if *b == v))
            };
            let mut before: HashSet<BindId> = HashSet::new();
            for st in body {
                let refs = locals_in(st);
                match &st.kind {
                    ExprKind::Set(pl, v) => {
                        if let (ExprKind::Var(_), Some(Res::Local(b))) = (&pl.kind, self.ck.res.get(pl.id as usize)) {
                            if !inner.contains(b) && *b != lv && !before.contains(b) && !locals_in(v).contains(b) {
                                scratch.push((*b, None));
                            }
                        }
                    }
                    ExprKind::For(_, _, ForIter::Range(flo, fhi), fb) if fb.len() == 1 => {
                        let j = self.ck.decl.get(&(st.id as usize)).copied();
                        if let ExprKind::Set(pl, x) = &fb[0].kind {
                            if let ExprKind::Call(n, _, pa) = &pl.kind {
                                let v = match (&pa[0].expr().kind, self.ck.res.get(pa[0].expr().id as usize)) {
                                    (ExprKind::Var(_), Some(Res::Local(b))) => Some(*b),
                                    _ => None,
                                };
                                let jx = match (&pa[1].expr().kind, self.ck.res.get(pa[1].expr().id as usize)) {
                                    (ExprKind::Var(_), Some(Res::Local(b))) => Some(*b),
                                    _ => None,
                                };
                                if let (Some(v), Some(_)) = (v, j) {
                                    if n == "at"
                                        && jx == j
                                        && !inner.contains(&v)
                                        && v != lv
                                        && !before.contains(&v)
                                        && !locals_in(x).contains(&v)
                                        && !locals_in(flo).contains(&v)
                                        && (!locals_in(fhi).contains(&v) || is_len_of(fhi, v))
                                    {
                                        scratch.push((v, Some((flo, fhi))));
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                before.extend(refs);
            }
            // A vector killed element by element must keep its length in the body: it is only
            // read and written element by element.
            scratch.retain(|(v, kill)| {
                if kill.is_none() {
                    return true;
                }
                let mut ok = true;
                for x in body {
                    walk_ref(x, &mut |e| match &e.kind {
                        ExprKind::Set(pl, _) => {
                            if matches!((&pl.kind, self.ck.res.get(pl.id as usize)), (ExprKind::Var(_), Some(Res::Local(b))) if b == v) {
                                ok = false;
                            }
                        }
                        ExprKind::Call(_, _, args) => {
                            for a in args {
                                if let Arg::Mut(m, _) = a {
                                    if self.root_of(m) == Some(*v) {
                                        ok = false;
                                    }
                                }
                            }
                        }
                        _ => {}
                    });
                }
                ok
            });
            for (v, _) in &scratch {
                inner.insert(*v);
            }
        }
        // Capabilities: only the console, only to print, and not inside a lambda.
        let mut caps_bad = false;
        for x in body {
            walk_ref(x, &mut |e| {
                if matches!(e.kind, ExprKind::Var(_)) && matches!(self.ty(e), Ty::Cap(_)) && !cap_ok.contains(&e.id) {
                    caps_bad = true;
                }
                if let ExprKind::Lambda(_, lb) = &e.kind {
                    for y in lb {
                        walk_ref(y, &mut |f| {
                            if let ExprKind::Call(n, _, _) = &f.kind {
                                if n == "print" || n == "println" {
                                    caps_bad = true;
                                }
                            }
                        });
                    }
                }
            });
        }
        if caps_bad {
            return None;
        }
        let is_local = |e: &Expr, b: BindId| matches!((&e.kind, &self.ck.res[e.id as usize]), (ExprKind::Var(_), Res::Local(x)) if *x == b);
        let local_of = |e: &Expr| match (&e.kind, &self.ck.res[e.id as usize]) {
            (ExprKind::Var(_), Res::Local(x)) => Some(*x),
            _ => None,
        };
        let builtin = |e: &Expr| matches!(self.ck.res.get(e.id as usize), Some(Res::Builtin(_)));
        // Classify every change to a variable from outside the body.
        let mut sums: Vec<(BindId, Vec<u32>)> = Vec::new();
        let mut hists: Vec<(BindId, Vec<u32>)> = Vec::new();
        let mut writes: Vec<(BindId, Vec<(u32, &Expr)>)> = Vec::new();
        let mut pushes: Vec<BindId> = Vec::new();
        let mut plain_sets: Vec<(BindId, u32)> = Vec::new();
        let mut modsums: Vec<(BindId, Vec<u32>, &Expr)> = Vec::new();
        let mut ifs: Vec<&Expr> = Vec::new();
        let mut fail = false;
        let add = |v: &mut Vec<(BindId, Vec<u32>)>, b: BindId, id: u32| match v.iter_mut().find(|x| x.0 == b) {
            Some(x) => x.1.push(id),
            None => v.push((b, vec![id])),
        };
        for x in body {
            walk_ref(x, &mut |e| match &e.kind {
                ExprKind::If(_, _, None) => ifs.push(e),
                ExprKind::Call(n, _, args) => {
                    for (i, arg) in args.iter().enumerate() {
                        let Arg::Mut(m, _) = arg else { continue };
                        let Some(r) = self.root_of(m) else {
                            fail = true;
                            continue;
                        };
                        if inner.contains(&r) {
                            continue;
                        }
                        if n == "push" && i == 0 && builtin(e) && local_of(m) == Some(r) {
                            if !pushes.contains(&r) {
                                pushes.push(r);
                            }
                        } else {
                            fail = true;
                        }
                    }
                }
                ExprKind::Set(pl, v) => {
                    let Some(r) = self.root_of(pl) else {
                        fail = true;
                        return;
                    };
                    if inner.contains(&r) {
                        return;
                    }
                    if r == lv {
                        fail = true;
                        return;
                    }
                    match &pl.kind {
                        ExprKind::Var(_) => {
                            let sum = match &v.kind {
                                ExprKind::Call(op, _, xs) if builtin(v) && self.ty(pl).is_int() => {
                                    (op == "+" && xs.iter().filter(|a| is_local(a.expr(), r)).count() == 1)
                                        || (op == "-" && xs.len() == 2 && is_local(xs[0].expr(), r) && !self.unsigned(self.ty(pl)))
                                }
                                _ => false,
                            };
                            let modsum = match &v.kind {
                                ExprKind::Call(op, _, xs) if op == "%" && xs.len() == 2 && builtin(v) && self.ty(pl).is_int() && !self.unsigned(self.ty(pl)) => {
                                    matches!(&xs[0].expr().kind, ExprKind::Call(op2, _, ys) if op2 == "+" && builtin(xs[0].expr())
                                        && ys.iter().filter(|a| is_local(a.expr(), r)).count() == 1)
                                }
                                _ => false,
                            };
                            if sum {
                                add(&mut sums, r, e.id);
                            } else if modsum {
                                let ExprKind::Call(_, _, xs) = &v.kind else { unreachable!() };
                                match modsums.iter_mut().find(|x| x.0 == r) {
                                    Some(x) => {
                                        if self.src_of(x.2) != self.src_of(xs[1].expr()) {
                                            fail = true;
                                        }
                                        x.1.push(e.id);
                                    }
                                    None => modsums.push((r, vec![e.id], xs[1].expr())),
                                }
                            } else {
                                plain_sets.push((r, e.id));
                            }
                        }
                        ExprKind::Call(n, _, pa) if n == "at" && pa.len() == 2 => {
                            let at_src = self.src_of(pl);
                            // `(set (at v i) (+ (at v i) e))` at the loop index is a write to the
                            // iteration's own element, not a histogram.
                            let hist = local_of(pa[0].expr()) == Some(r)
                                && local_of(pa[1].expr()) != Some(lv)
                                && self.ty(pl).is_int()
                                && !self.unsigned(self.ty(pl))
                                && matches!(&v.kind, ExprKind::Call(op, _, xs) if op == "+" && builtin(v)
                                    && xs.iter().filter(|a| self.src_of(a.expr()) == at_src).count() == 1);
                            if hist {
                                add(&mut hists, r, e.id);
                                return;
                            }
                            // A write: the at directly on the variable decides which element.
                            let mut first = &**pl;
                            loop {
                                let ExprKind::Call(n2, _, p2) = &first.kind else {
                                    fail = true;
                                    return;
                                };
                                if n2 != "at" || p2.len() != 2 {
                                    fail = true;
                                    return;
                                }
                                if local_of(p2[0].expr()) == Some(r) {
                                    break;
                                }
                                first = p2[0].expr();
                            }
                            let ExprKind::Call(_, _, p2) = &first.kind else { unreachable!() };
                            match writes.iter_mut().find(|x| x.0 == r) {
                                Some(x) => x.1.push((first.id, p2[1].expr())),
                                None => writes.push((r, vec![(first.id, p2[1].expr())])),
                            }
                        }
                        _ => fail = true,
                    }
                }
                _ => {}
            });
        }
        if fail {
            return None;
        }
        // Running maxima and minima: `(if (op e m) (do (set m e) (set w x)...))`, also with a
        // condition that does not read m: `(if (and g (op e m)) ...)`. A chunk starts from m as
        // it was before the loop, so it evaluates `g` wherever the sequential loop would (and
        // maybe more often): a trap there is still found, by the sequential run.
        let mut maxes: Vec<(u32, Vec<BindId>, bool, u32)> = Vec::new(); // if id, [m, w...], m on the left, comparison id
        let mut grouped: HashSet<u32> = HashSet::new();
        for &f in &ifs {
            let ExprKind::If(c0, th, None) = &f.kind else { continue };
            let cands: Vec<&Expr> = match &c0.kind {
                ExprKind::Call(n, _, xs) if n == "and" && xs.len() == 2 && builtin(c0) => vec![xs[1].expr(), xs[0].expr()],
                _ => vec![&**c0],
            };
            let sets: Vec<&Expr> = match &th.kind {
                ExprKind::Set(..) => vec![&**th],
                ExprKind::Do(es) => es.iter().collect(),
                _ => continue,
            };
            if !sets.iter().all(|x| matches!(&x.kind, ExprKind::Set(p, _) if local_of(p).is_some_and(|b| !inner.contains(&b)))) {
                continue;
            }
            'cand: for c in cands {
                let ExprKind::Call(op, _, cx) = &c.kind else { continue };
                if !builtin(c) || !matches!(op.as_str(), "<" | ">" | "<=" | ">=") || cx.len() != 2 {
                    continue;
                }
                for (left, (mx, ev)) in [(true, (cx[0].expr(), cx[1].expr())), (false, (cx[1].expr(), cx[0].expr()))] {
                    let Some(m) = local_of(mx) else { continue };
                    if inner.contains(&m) || m == lv {
                        continue;
                    }
                    let ev_src = self.src_of(ev);
                    let mut group = vec![m];
                    let mut ok = false;
                    for x in &sets {
                        let ExprKind::Set(p, v) = &x.kind else { unreachable!() };
                        let b = local_of(p).unwrap();
                        if b == m {
                            ok |= self.src_of(v) == ev_src;
                        } else if !group.contains(&b) {
                            group.push(b);
                        }
                    }
                    if ok && group.iter().all(|b| self.is_copy(&self.ck.bindings[*b].ty)) {
                        maxes.push((f.id, group, left, c.id));
                        for x in &sets {
                            grouped.insert(x.id);
                        }
                        break 'cand;
                    }
                }
            }
        }
        let mut lasts: Vec<(BindId, Vec<u32>)> = Vec::new();
        for (b, id) in &plain_sets {
            if grouped.contains(id) {
                continue;
            }
            if !self.is_copy(&self.ck.bindings[*b].ty) {
                return None;
            }
            add(&mut lasts, *b, *id);
        }
        // Each changed variable in one role only, and used nowhere else in the body.
        let mut uses_allowed: HashMap<BindId, usize> = HashMap::new();
        let mut claim = |b: BindId, uses: usize| {
            *uses_allowed.entry(b).or_insert(0) += uses;
        };
        let mut roles: Vec<BindId> = Vec::new();
        for (b, ids) in &sums {
            roles.push(*b);
            claim(*b, 2 * ids.len());
        }
        for (b, ids) in &hists {
            roles.push(*b);
            claim(*b, 2 * ids.len());
        }
        for (b, ws) in &writes {
            roles.push(*b);
            claim(*b, ws.len());
        }
        for b in &pushes {
            roles.push(*b);
            claim(*b, 0);
        }
        for (b, ids, _) in &modsums {
            roles.push(*b);
            claim(*b, 2 * ids.len());
        }
        for (b, ids) in &lasts {
            roles.push(*b);
            claim(*b, ids.len());
        }
        let mut push_uses: HashMap<BindId, usize> = HashMap::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let ExprKind::Call(n, _, args) = &e.kind {
                    if n == "push" {
                        if let Some(Arg::Mut(m, _)) = args.first() {
                            if let Some(r) = local_of(m) {
                                *push_uses.entry(r).or_insert(0) += 1;
                            }
                        }
                    }
                }
            });
        }
        for b in &pushes {
            claim(*b, push_uses.get(b).copied().unwrap_or(0));
        }
        // Reads of a vector the loop writes: allowed (when its elements are plain values) if
        // each read falls in the window its own iteration writes, tested before the loop.
        let write_ids: HashSet<u32> = writes.iter().flat_map(|(_, ws)| ws.iter().map(|w| w.0)).collect();
        let mut own_reads: Vec<(BindId, u32, &Expr)> = Vec::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let ExprKind::Call(n, _, args) = &e.kind {
                    if n == "at" && args.len() == 2 && builtin(e) && !write_ids.contains(&e.id) {
                        if let Some(r) = local_of(args[0].expr()) {
                            if writes.iter().any(|w| w.0 == r) {
                                own_reads.push((r, e.id, args[1].expr()));
                            }
                        }
                    }
                }
            });
        }
        for (r, _, _) in &own_reads {
            let Ty::Vec(et) = &self.ck.bindings[*r].ty else { return None };
            if !self.is_copy(et) || matches!(**et, Ty::Vec(_)) {
                return None;
            }
            claim(*r, 1);
        }
        for (fid, group, _, _) in &maxes {
            let ExprKind::If(_, th, _) = &find_expr(body, *fid)?.kind else { unreachable!() };
            let n_sets = match &th.kind {
                ExprKind::Do(es) => es.len(),
                _ => 1,
            };
            for (k, b) in group.iter().enumerate() {
                roles.push(*b);
                let sets_of_b = match &th.kind {
                    ExprKind::Do(es) => es.iter().filter(|x| matches!(&x.kind, ExprKind::Set(p, _) if local_of(p) == Some(*b))).count(),
                    _ => 1,
                };
                let _ = n_sets;
                claim(*b, sets_of_b + if k == 0 { 1 } else { 0 });
            }
        }
        let mut seen = HashSet::new();
        if !roles.iter().all(|b| seen.insert(*b)) {
            return None;
        }
        let mut counts: HashMap<BindId, usize> = HashMap::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let Some(b) = local_of(e) {
                    if roles.contains(&b) {
                        *counts.entry(b).or_insert(0) += 1;
                    }
                }
            });
        }
        if roles.iter().any(|b| counts.get(b).copied().unwrap_or(0) != uses_allowed[b]) {
            return None;
        }
        // Nothing worth parallelizing.
        if sums.is_empty() && hists.is_empty() && writes.is_empty() && pushes.is_empty() && maxes.is_empty() && modsums.is_empty() && !prints && !heavy {
            return None;
        }
        // Elements, not indices: no index-based writes.
        if each.is_some() && !writes.is_empty() {
            return None;
        }
        let changed: HashSet<BindId> = roles.iter().copied().collect();
        let fixed = |e: &Expr| -> bool {
            let mut ok = true;
            walk_ref(e, &mut |x| match (&x.kind, self.ck.res.get(x.id as usize)) {
                (ExprKind::Var(_), Some(Res::Local(b))) => {
                    if *b == lv || inner.contains(b) || changed.contains(b) {
                        ok = false;
                    }
                }
                (ExprKind::Var(_), _) | (ExprKind::Int(_), _) => {}
                (ExprKind::Call(n, _, _), Some(Res::Builtin(_))) if matches!(n.as_str(), "+" | "-" | "*" | "len") => {}
                _ => ok = false,
            });
            ok
        };
        if !modsums.iter().all(|x| fixed(x.2) || matches!(x.2.kind, ExprKind::Var(_) if matches!(self.ck.res[x.2.id as usize], Res::Const(_)))) {
            return None;
        }
        // Disjoint writes: per vector, the windows each iteration writes.
        let mut win_tests: Vec<String> = Vec::new();
        // Reads that see the vector as it was before the loop (from a copy): a vector written
        // once per iteration, by the last statement, at `c ± i`, whose other reads come before
        // that write and hit `c ± i + d` with d on the side of later iterations (so no earlier
        // iteration has written there). That is the in-place dynamic programming over a row
        // walked backwards, as in a 0/1 knapsack.
        let mut stale: Vec<(BindId, u32, &Expr)> = Vec::new();
        let mut stale_tests: Vec<String> = Vec::new();
        for (wb, ws) in &writes {
            let reads: Vec<(BindId, u32, &Expr)> = own_reads.iter().filter(|x| x.0 == *wb).copied().collect();
            if ws.len() != 1 || reads.is_empty() {
                continue;
            }
            if !matches!(body.last().map(|x| &x.kind), Some(ExprKind::Set(p, _)) if p.id == ws[0].0) {
                continue;
            }
            // Each index as (decreasing?, its fixed terms as source text, the same terms wide).
            let mut forms: Vec<Option<(bool, String, String)>> = Vec::new();
            for idx in std::iter::once(ws[0].1).chain(reads.iter().map(|r| r.2)) {
                let mut af = Aff::default();
                if !aff(self, idx, lv, &lets, &ranges, &fixed, true, &mut af, 0) || !af.lv || af.j.is_some() || af.stride.is_some() {
                    forms.push(None);
                    continue;
                }
                let (mut src, mut wide) = (String::from("0"), String::from("0i128"));
                for (plus, t) in &af.consts {
                    let sg = if *plus { " + " } else { " - " };
                    src.push_str(&format!("{sg}({})", self.src_of(t)));
                    wide.push_str(&format!("{sg}{}", wide_of(self, t)));
                }
                forms.push(Some((af.neg, src, wide)));
            }
            let Some(Some((wneg, wsrc, wwide))) = forms.first().cloned() else { continue };
            let rs: Vec<_> = forms[1..].iter().cloned().collect();
            if rs.iter().any(|f| !matches!(f, Some((n, _, _)) if *n == wneg)) {
                continue;
            }
            // Only worth a copy when some read is not the iteration's own element.
            if rs.iter().all(|f| f.as_ref().is_some_and(|f| f.1 == wsrc)) {
                continue;
            }
            for (r, f) in reads.iter().zip(rs.iter()) {
                stale.push(*r);
                let rw = &f.as_ref().unwrap().2;
                // read - write must be <= 0 for a decreasing index, >= 0 for an increasing one.
                stale_tests.push(if wneg { format!("(({rw}) - ({wwide}) <= 0)") } else { format!("(({rw}) - ({wwide}) >= 0)") });
            }
        }
        let stale_vecs: HashSet<BindId> = stale.iter().map(|x| x.0).collect();
        own_reads.retain(|x| !stale_vecs.contains(&x.0));
        win_tests.extend(stale_tests);
        for (wb, ws) in &writes {
            if stale_vecs.contains(wb) {
                continue;
            }
            let mut stride: Option<Option<&Expr>> = None;
            let mut wins: Vec<String> = Vec::new();
            let mut bounds: Vec<(String, String)> = Vec::new();
            let reads: Vec<&Expr> = own_reads.iter().filter(|x| x.0 == *wb).map(|x| x.2).collect();
            let n_writes = ws.len();
            let mut dir: Option<bool> = None;
            for (wi, idx) in ws.iter().map(|w| w.1).chain(reads.iter().copied()).enumerate() {
                let mut af = Aff::default();
                if !aff(self, idx, lv, &lets, &ranges, &fixed, true, &mut af, 0) || !af.lv {
                    return None;
                }
                // All of a vector's indices move the same way with the loop.
                if *dir.get_or_insert(af.neg) != af.neg {
                    return None;
                }
                let s_src = af.stride.map(|e| self.src_of(e).to_string());
                match &stride {
                    None => stride = Some(af.stride),
                    Some(prev) => {
                        if prev.map(|e| self.src_of(e).to_string()) != s_src {
                            return None;
                        }
                    }
                }
                let mut c = String::from("0i128");
                for (plus, t) in &af.consts {
                    c.push_str(if *plus { " + " } else { " - " });
                    c.push_str(&wide_of(self, t));
                }
                let (lo, hi) = match af.j {
                    Some(j) => {
                        let (jl, jh) = ranges[&j];
                        if !fixed(jl) || !fixed(jh) {
                            return None;
                        }
                        let (jl, jh) = (wide_of(self, jl), wide_of(self, jh));
                        (format!("{c} + {jl}"), format!("{c} + {jh} - 1"))
                    }
                    None => (c.clone(), c.clone()),
                };
                if wi < n_writes {
                    wins.push(format!("({lo}, {hi})"));
                    bounds.push((lo, hi));
                } else {
                    // An empty read window needs nothing; otherwise one write window holds it.
                    let inside: Vec<String> = bounds.iter().map(|(wl, wh)| format!("(({lo}) >= ({wl}) && ({hi}) <= ({wh}))")).collect();
                    win_tests.push(format!("(({lo}) > ({hi}) || {})", inside.join(" || ")));
                }
            }
            let s = match stride.flatten() {
                Some(e) => wide_of(self, e),
                None => "1i128".to_string(),
            };
            win_tests.push(format!("ail_rt::disjoint({s}, &[{}])", wins.join(", ")));
        }
        // Emit.
        let (mut pre, head) = match (range, each) {
            (Some((a, z)), _) => {
                let lo = self.val(a);
                let hi = self.val(z);
                (format!("let __plo: i64 = {lo}; let __phi: i64 = {hi}; let mut __pdone = false; "), format!("for {var} in __a..__b {{"))
            }
            (_, Some(v)) => match &v.kind {
                ExprKind::Call(n, _, args) if n == "lines" || n == "split" => {
                    let sv = self.rf(args[0].expr());
                    let it = if n == "lines" {
                        format!("ail_rt::lines_iter({sv})")
                    } else {
                        let sep = self.rf(args[1].expr());
                        format!("ail_rt::split_iter({sv}, {sep})")
                    };
                    (
                        format!("let __pxs: Vec<&str> = if {sv}.len() >= 4096 {{ {it}.collect() }} else {{ Vec::new() }}; let __plo: i64 = 0; let __phi: i64 = __pxs.len() as i64; let mut __pdone = false; "),
                        format!("for __pi in __a..__b {{ let {var}: &str = __pxs[__pi as usize];"),
                    )
                }
                _ => {
                    let xs = self.pv(v);
                    (
                        format!("let __plo: i64 = 0; let __phi: i64 = {xs}.len() as i64; let mut __pdone = false; "),
                        format!("for __pi in __a..__b {{ let {var} = &{xs}[__pi as usize];"),
                    )
                }
            },
            _ => unreachable!(),
        };
        let mut setup = String::new();
        let mut privs = String::new();
        let mut ret: Vec<String> = Vec::new();
        let mut combine_check = String::new();
        let mut combine_ref = String::new();
        let mut combine_own = String::new();
        let mut hooks: Vec<(u32, ParHook)> = Vec::new();
        let mut shared: Vec<(BindId, String)> = Vec::new();
        let mut unchecked: Vec<(u32, String)> = Vec::new();
        let mut tests: Vec<String> = win_tests;
        let mut chunks = "0";
        // A private variable shadowing `b` inside the chunk, initialized with `init`; returns
        // the owned name to hand back.
        let shadow = |em: &Self, b: BindId, ty: &str, init: &str, k: &str, out: &mut String| -> String {
            let pl = em.bind_place(b);
            let name = em.local_name(&em.ck.bindings[b].name);
            if pl.starts_with("(*") {
                out.push_str(&format!("let mut __pv{k}: {ty} = {init}; let {name} = &mut __pv{k}; "));
                format!("__pv{k}")
            } else {
                out.push_str(&format!("let mut {name}: {ty} = {init}; "));
                name
            }
        };
        for (i, (b, ids)) in sums.iter().enumerate() {
            let ty = rty(&self.ck.bindings[*b].ty);
            let k = format!("s{i}");
            let own = shadow(self, *b, &ty, &format!("0 as {ty}"), &k, &mut privs);
            privs.push_str(&format!("let mut __abs{k}: u64 = 0; "));
            let (pi, ai) = (ret.len(), ret.len() + 1);
            ret.push(own);
            ret.push(format!("__abs{k}"));
            for id in ids {
                hooks.push((*id, ParHook::Sum(format!("__abs{k}"))));
            }
            let pl = self.bind_place(*b);
            combine_check.push_str(&format!(
                "if ail_rt::UAbs::uabs({pl}) as i128 + __parts.iter().map(|p| p.{ai} as i128).sum::<i128>() > {ty}::MAX as i128 {{ __ok = false; }} "
            ));
            combine_ref.push_str(&format!(
                "{pl} = (ail_rt::UAbs::wide({pl}) + __parts.iter().map(|p| ail_rt::UAbs::wide(p.{pi})).sum::<i128>()) as {ty}; "
            ));
        }
        for (i, (b, ids)) in hists.iter().enumerate() {
            let vty = rty(&self.ck.bindings[*b].ty);
            let Ty::Vec(et) = &self.ck.bindings[*b].ty else { return None };
            let ety = rty(et);
            let k = format!("h{i}");
            let pl = self.bind_place(*b);
            pre.push_str(&format!("let __hlen{k} = {pl}.len(); "));
            tests.push(format!("__hlen{k} <= (1 << 18)"));
            let own = shadow(self, *b, &vty, &format!("vec![0 as {ety}; __hlen{k}]"), &k, &mut privs);
            privs.push_str(&format!("let mut __abs{k}: u64 = 0; "));
            let (pi, ai) = (ret.len(), ret.len() + 1);
            ret.push(own);
            ret.push(format!("__abs{k}"));
            for id in ids {
                hooks.push((*id, ParHook::Hist(format!("__abs{k}"))));
            }
            combine_check.push_str(&format!(
                "if {pl}.iter().map(|x| ail_rt::UAbs::uabs(*x) as i128).max().unwrap_or(0) + __parts.iter().map(|p| p.{ai} as i128).sum::<i128>() > {ety}::MAX as i128 {{ __ok = false; }} "
            ));
            combine_ref.push_str(&format!(
                "for __p in __parts.iter() {{ for (__d, __s) in {pl}.iter_mut().zip(__p.{pi}.iter()) {{ *__d += *__s; }} }} "
            ));
            chunks = "1";
        }
        for (i, (fid, group, left, cid)) in maxes.iter().enumerate() {
            let k = format!("m{i}");
            let mut fields = Vec::new();
            for (gi, b) in group.iter().enumerate() {
                let ty = rty(&self.ck.bindings[*b].ty);
                let pl = self.bind_place(*b);
                setup.push_str(&format!("let __sn{k}_{gi}: {ty} = {pl}; "));
                let own = shadow(self, *b, &ty, &format!("__sn{k}_{gi}"), &format!("{k}_{gi}"), &mut privs);
                fields.push(ret.len());
                ret.push(own);
            }
            privs.push_str(&format!("let mut __upd{k} = false; "));
            let ui = ret.len();
            ret.push(format!("__upd{k}"));
            hooks.push((*fid, ParHook::Flag(format!("__upd{k}"))));
            let ExprKind::Call(op, _, _) = &find_expr(body, *cid)?.kind else { unreachable!() };
            let mpl = self.bind_place(group[0]);
            let cond = if *left { format!("{mpl} {op} __p.{}", fields[0]) } else { format!("__p.{} {op} {mpl}", fields[0]) };
            let mut assign = String::new();
            for (gi, b) in group.iter().enumerate() {
                assign.push_str(&format!("{} = __p.{}; ", self.bind_place(*b), fields[gi]));
            }
            combine_ref.push_str(&format!("for __p in __parts.iter() {{ if __p.{ui} && {cond} {{ {assign}}} }} "));
        }
        if !scratch.is_empty() && chunks == "0" {
            // Each running chunk holds its own copies: no more chunks at once than cores.
            chunks = "2";
        }
        for (i, (b, kill)) in scratch.iter().enumerate() {
            let ty = rty(&self.ck.bindings[*b].ty);
            let k = format!("w{i}");
            let pl = self.bind_place(*b);
            if let Some((flo, fhi)) = kill {
                let whole = matches!(&fhi.kind, ExprKind::Call(n, _, xs) if n == "len" && xs.len() == 1 && self.root_of(xs[0].expr()) == Some(*b));
                if !fixed(flo) || (!whole && !fixed(fhi)) {
                    return None;
                }
                let wl = wide_of(self, flo);
                if whole {
                    tests.push(format!("{wl} == 0"));
                } else {
                    let wh = wide_of(self, fhi);
                    tests.push(format!("{wl} == 0 && {wh} == ({pl}.len() as i128)"));
                }
            }
            let own = shadow(self, *b, &ty, &format!("{pl}.clone()"), &k, &mut privs);
            let pi = ret.len();
            ret.push(own);
            combine_ref.push_str(&format!("if let Some(__l) = __parts.last() {{ {pl} = __l.{pi}.clone(); }} "));
        }
        for (i, (b, ids, m)) in modsums.iter().enumerate() {
            let ty = rty(&self.ck.bindings[*b].ty);
            let k = format!("r{i}");
            let own = shadow(self, *b, &ty, &format!("0 as {ty}"), &k, &mut privs);
            privs.push_str(&format!("let mut __neg{k} = false; let mut __mx{k}: {ty} = 0; "));
            let (pi, ni, mi) = (ret.len(), ret.len() + 1, ret.len() + 2);
            ret.push(own);
            ret.push(format!("__neg{k}"));
            ret.push(format!("__mx{k}"));
            for id in ids {
                hooks.push((*id, ParHook::ModSum(k.clone())));
            }
            let pl = self.bind_place(*b);
            let mc = self.val(m);
            // In order, x becomes (x + e) % m at every step; with nothing negative that is
            // (x0 + all the terms) mod m, and no step overflows if no term is near the top.
            combine_check.push_str(&format!(
                "{{ let __m: {ty} = {mc}; if {pl} < 0 || __m <= 0 || __parts.iter().any(|p| p.{ni} || p.{mi} as i128 + __m as i128 > {ty}::MAX as i128) {{ __ok = false; }} }} "
            ));
            combine_ref.push_str(&format!(
                "{{ let __m = ({mc}) as i128; {pl} = ((({pl} as i128) + __parts.iter().map(|p| p.{pi} as i128).sum::<i128>()) % __m) as {ty}; }} "
            ));
        }
        for (i, (b, ids)) in lasts.iter().enumerate() {
            let ty = rty(&self.ck.bindings[*b].ty);
            let k = format!("l{i}");
            let pl = self.bind_place(*b);
            setup.push_str(&format!("let __snl{k}: {ty} = {pl}; "));
            let own = shadow(self, *b, &ty, &format!("__snl{k}"), &k, &mut privs);
            privs.push_str(&format!("let mut __set{k} = false; "));
            let (pi, fi) = (ret.len(), ret.len() + 1);
            ret.push(own);
            ret.push(format!("__set{k}"));
            for id in ids {
                hooks.push((*id, ParHook::Last(format!("__set{k}"))));
            }
            combine_ref.push_str(&format!("for __p in __parts.iter() {{ if __p.{fi} {{ {pl} = __p.{pi}; }} }} "));
        }
        for (i, b) in pushes.iter().enumerate() {
            let ty = rty(&self.ck.bindings[*b].ty);
            let k = format!("p{i}");
            let own = shadow(self, *b, &ty, "Vec::new()", &k, &mut privs);
            let pi = ret.len();
            ret.push(own);
            combine_own.push_str(&format!("{}.append(&mut __p.{pi}); ", self.bind_place(*b)));
        }
        if prints {
            privs.push_str("let mut __pout = String::new(); ");
            let pi = ret.len();
            ret.push("__pout".into());
            combine_own.push_str(&format!("ail_rt::print(&__p.{pi}); "));
        }
        for (i, (b, ws)) in writes.iter().enumerate() {
            let pl = self.bind_place(*b);
            let name = format!("__pw{i}");
            setup.push_str(&format!("let {name} = ail_rt::SharedSlice::new(&mut {pl}[..]); "));
            shared.push((*b, name.clone()));
            for (fid, idx) in ws {
                let ic = self.val(idx);
                unchecked.push((*fid, format!("(*{name}.at({ic}))")));
            }
            for (_, rid, idx) in own_reads.iter().filter(|x| x.0 == *b) {
                let ic = self.val(idx);
                unchecked.push((*rid, format!("(*{name}.at({ic}))")));
            }
            if stale.iter().any(|x| x.0 == *b) {
                // The copy is taken before the shared view of the vector is made.
                let sn = format!("__pst{i}");
                setup.insert_str(0, &format!("let {sn} = {pl}.clone(); "));
                for (_, rid, idx) in stale.iter().filter(|x| x.0 == *b) {
                    let ic = self.val(idx);
                    unchecked.push((*rid, format!("{sn}[ail_rt::Ix({ic})]")));
                }
            }
        }
        // The chunk body, with the hooks and shared vectors in place.
        let saved_hooks = std::mem::take(&mut self.par_hooks);
        let saved_shared = std::mem::take(&mut self.par_shared);
        self.par_hooks = hooks.into_iter().collect();
        self.par_shared = shared.into_iter().collect();
        let mut saved_unchecked = Vec::new();
        for (id, c) in &unchecked {
            saved_unchecked.push((*id, self.unchecked.insert(*id, c.clone())));
        }
        self.in_par = true;
        let pbody = self.block(body);
        self.in_par = false;
        self.par_hooks = saved_hooks;
        self.par_shared = saved_shared;
        for (id, old) in saved_unchecked {
            match old {
                Some(c) => self.unchecked.insert(id, c),
                None => self.unchecked.remove(&id),
            };
        }
        let was = self.no_par;
        self.no_par = true;
        let seq = match range {
            Some((a, _)) => self.for_range_seq(lv, var, "__plo", "__phi", a, body),
            None => format!("{{ {} }}", self.rvalue(whole)),
        };
        self.no_par = was;
        let tuple = if ret.is_empty() { "()".to_string() } else { format!("({},)", ret.join(", ")) };
        let gpu = if range.is_some() && self.gpu_on && hists.is_empty() && pushes.is_empty() && maxes.is_empty() && modsums.is_empty() && lasts.is_empty() && scratch.is_empty() && !prints {
            self.gpu_attempt(whole.id, lv, body, &sums, &writes, &own_reads, &stale, &tests)
        } else {
            None
        };
        if let Some(g) = &gpu {
            pre.push_str(g);
        }
        let worth = format!("!__pdone && ail_rt::par_worth(__plo, __phi, {heavy})");
        let cond = std::iter::once(worth).chain(tests).collect::<Vec<_>>().join(" && ");
        pre.push_str(&format!(
            "if {cond} {{ {setup}let __pr = ail_rt::par_for(__plo, __phi, {chunks}, &|__a: i64, __b: i64| {{ {privs}{head} let _ = {pbody}; }} {tuple} }}); \
             if let Some(mut __parts) = __pr {{ let mut __ok = true; {combine_check}if __ok {{ {combine_ref}for __p in __parts.iter_mut() {{ let _ = &__p; {combine_own}}} __pdone = true; }} }} }} "
        ));
        if gpu.is_some() {
            // The CPU run after a kernel trap without backups must trap too.
            return Some(format!("{{ let mut __gtrap = false; {pre}if !__pdone {seq} if __gtrap {{ ail_gpu::fatal_mismatch(); }} }}"));
        }
        Some(format!("{{ {pre}if !__pdone {seq} }}"))
    }

    /// The loop as a GPU kernel, tried before the CPU: code that sets `__pdone` when the kernel
    /// ran and its results are in place. None if the body is outside what kernels can do.
    #[allow(clippy::too_many_arguments)]
    fn gpu_attempt(
        &mut self,
        loop_id: u32,
        lv: BindId,
        body: &[Expr],
        sums: &[(BindId, Vec<u32>)],
        writes: &[(BindId, Vec<(u32, &Expr)>)],
        own_reads: &[(BindId, u32, &Expr)],
        stale: &[(BindId, u32, &Expr)],
        tests: &[String],
    ) -> Option<String> {
        let mut sum_map = HashMap::new();
        let mut sum_vars = Vec::new();
        for (r, (b, ids)) in sums.iter().enumerate() {
            sum_vars.push(*b);
            for id in ids {
                sum_map.insert(*id, r);
            }
        }
        let collapse = self.collapse_plan(lv, body, &sum_vars);
        let collapsed: Option<(&Expr, &Expr)> = collapse.as_ref().map(|c| (c.jlo, c.jhi));
        let inp = crate::gpu::Input {
            ck: self.ck,
            fns: &self.fn_defs,
            lv,
            body,
            sums: sum_map,
            sum_vars: sum_vars.clone(),
            writes: writes.iter().flat_map(|(_, ws)| ws.iter().map(|w| w.0)).collect(),
            own_reads: own_reads.iter().map(|x| x.1).collect(),
            stale: stale.iter().map(|x| x.1).collect(),
            collapse,
        };
        let k = crate::gpu::build(&inp);
        drop(inp);
        let k = k?;
        self.kernel_loops.insert(loop_id);
        self.kernel_fns.insert(self.fn_name.clone());
        let id = self.kernels.len();
        let words: Vec<String> = k.words.iter().map(|w| format!("{w:#x}")).collect();
        self.kernels.push(format!(
            "static AIL_K{id}: ail_gpu::KernelDesc = ail_gpu::KernelDesc {{ id: {id}, words: &[{}], nbufs: {}, nargs: {}, nred: {}, f64: {}, exact: {}, narrow: {} }};",
            words.join(","),
            k.bufs.len(),
            k.args.len(),
            k.nred,
            k.f64,
            k.exact,
            k.bufs.iter().enumerate().filter(|(_, b)| b.narrow).map(|(i, _)| 1u32 << i).sum::<u32>()
        ));
        let mut args = Vec::new();
        for (a, sc) in &k.args {
            let v = match a {
                crate::gpu::KArg::Local(b) => self.bind_place(*b),
                crate::gpu::KArg::Const(n) => format!("C_{}", mangle(n)),
                crate::gpu::KArg::Len(bi) => format!("{}.len()", self.bind_place(k.bufs[*bi].bind)),
                crate::gpu::KArg::Raw(r) => r.clone(),
            };
            args.push(match sc {
                crate::spirv::Scalar::Float(_) => format!("({v}).to_bits()"),
                crate::spirv::Scalar::Bool => format!("(({v}) as u64)"),
                crate::spirv::Scalar::Int(32, true) => format!("((({v}) as i32 as u32) as u64)"),
                _ => format!("(({v}) as u64)"),
            });
        }
        let mut mirrors = String::new();
        let mut bufs = Vec::new();
        let mut after = String::new();
        let mut fallback = String::new();
        let mut shadows = String::new();
        let mut mstr: Vec<String> = Vec::new();
        for (i, b) in k.bufs.iter().enumerate() {
            let pl = self.bind_place(b.bind);
            if b.snap {
                // Copied on the device from the written vector's buffer at launch (filled in
                // below, once that buffer's mirror is known).
                let w = k.bufs.iter().position(|x| x.bind == b.bind && !x.snap)?;
                bufs.push(format!("\u{0}SNAP{w}"));
                mstr.push(String::new());
                continue;
            }
            // A vector with a mirror keeps its device copy; any other gets one for the launch.
            let resident = self.kernel_stmt == Some(loop_id);
            let m = match self.mirrors.get(&b.bind).cloned().filter(|_| resident) {
                Some((name, kind)) => {
                    match kind {
                        MirrorKind::RefParam => {
                            // Read on the CPU (if the GPU path is not taken) through a copy.
                            let n = self.local_name(&self.ck.bindings[b.bind].name);
                            let et = rty(&self.ck.bindings[b.bind].ty);
                            shadows.push_str(&format!(
                                "let __gv{i} = if __pdone {{ None }} else {{ Some({name}.view(&{pl})) }}; let {n}: &{et} = match &__gv{i} {{ Some(c) => &**c, None => {n} }}; "
                            ));
                        }
                        _ => {
                            fallback.push_str(&format!("{name}.to_cpu(&mut {pl}); "));
                            if b.write {
                                fallback.push_str(&format!("{name}.cpu_changed(); "));
                            }
                        }
                    }
                    if kind == MirrorKind::Owned { format!("&{name}") } else { name }
                }
                None => {
                    mirrors.push_str(&format!("let __gm{i} = ail_gpu::Mirror::new(); "));
                    if b.write {
                        after.push_str(&format!("__gm{i}.to_cpu(&mut {pl}); "));
                    }
                    format!("&__gm{i}")
                }
            };
            mstr.push(m.clone());
            if b.write {
                bufs.push(format!("ail_gpu::BufArg::write(&mut {pl}, {m}, {})", b.read));
            } else {
                bufs.push(format!("ail_gpu::BufArg::read(&{pl}, {m})"));
            }
        }
        for x in bufs.iter_mut() {
            if let Some(w) = x.strip_prefix("\u{0}SNAP") {
                let w: usize = w.parse().unwrap();
                *x = format!("ail_gpu::BufArg::snap_of({w}, {})", mstr[w]);
            }
        }
        let starts: Vec<String> = sum_vars.iter().map(|b| format!("({}) as i64", self.bind_place(*b))).collect();
        for (r, b) in sum_vars.iter().enumerate() {
            after.push_str(&format!("{} = __gs[{r}]; ", self.bind_place(*b)));
        }
        // The work items: the iterations, or (i, j) pairs when the inner loop is folded in. The
        // inner range is computed wide; if it does not fit, or either range is empty, the CPU
        // runs the loop (and traps or does nothing exactly as the program would).
        let (count, setup) = match collapsed {
            Some((jlo, jhi)) => {
                let (wl, wh) = (wide_of(self, jlo), wide_of(self, jhi));
                (
                    "__gn".to_string(),
                    format!(
                        "let __gn: i64 = {{ let __ni = (__phi as i128) - (__plo as i128); if __ni <= 0 {{ 0 }} else {{ let (__jl, __jh) = ({wl}, {wh}); let __nj = __jh - __jl; if __jl < i64::MIN as i128 || __jh > i64::MAX as i128 || __nj <= 0 || __ni * __nj > (1i128 << 40) {{ 0 }} else {{ (__ni * __nj) as i64 }} }} }}; "
                    ),
                )
            }
            None => ("__phi.wrapping_sub(__plo)".to_string(), String::new()),
        };
        let (glo, ghi) = if collapsed.is_some() { ("0".to_string(), "__gn".to_string()) } else { ("__plo".to_string(), "__phi".to_string()) };
        // Where the data already is on the device, the GPU is the cheaper place to run.
        let resident: Vec<String> = k
            .bufs
            .iter()
            .filter_map(|b| self.mirrors.get(&b.bind).filter(|_| self.kernel_stmt == Some(loop_id)).cloned())
            .map(|(m, kind)| if kind == MirrorKind::Owned { format!("&{m}") } else { m })
            .collect();
        let mut cond = vec![if resident.is_empty() {
            format!("ail_gpu::worth({id}, {count}, {})", k.cost)
        } else {
            format!("ail_gpu::worth_at({id}, {count}, {}, &[{}])", k.cost, resident.join(", "))
        }];
        if collapsed.is_some() {
            cond.insert(0, "__gn > 0".into());
        }
        cond.extend(tests.iter().cloned());
        cond.push("ail_gpu::ready()".into());
        Some(format!(
            "{{ {setup}if {} {{ {mirrors}match ail_gpu::launch(&AIL_K{id}, {glo}, {ghi}, &[{}], &mut [{}], &[{}]) {{ Ok(__gs) => {{ {after}__pdone = true; }} Err(__t) => __gtrap = __t }} }} }} if !__pdone {{ {fallback}}} {shadows}",
            cond.join(" && "),
            args.join(", "),
            bufs.join(", "),
            starts.join(", ")
        ))
    }

    /// True where statements get syncs: a GPU-aware fn, outside code that runs on the CPU as a
    /// whole (a CPU statement, a lambda, a task, a chunk of a parallel loop).
    fn gpu_hooks(&self) -> bool {
        self.pass2 && self.gpu_quiet == 0 && !self.in_par && self.aware.contains(&self.fn_name)
    }

    /// `(slice v a b)` or `(copy v)` of a vector with a mirror, bound to new vector `b`: copied on
    /// the device when that is where v is current. Returns the declaration of b's mirror (to go
    /// before the `let`) and the value.
    fn dev_copy(&mut self, value: &Expr, b: BindId) -> Option<(String, String)> {
        let (v, range) = match &value.kind {
            ExprKind::Call(n, _, xs) if n == "slice" && xs.len() == 3 && matches!(self.ck.res.get(value.id as usize), Some(Res::Builtin(_))) => {
                (xs[0].expr(), Some((xs[1].expr(), xs[2].expr())))
            }
            ExprKind::Copy(x) => (&**x, None),
            _ => return None,
        };
        let Some(Res::Local(src)) = self.ck.res.get(v.id as usize) else { return None };
        if !matches!(v.kind, ExprKind::Var(_)) {
            return None;
        }
        let (m, kind) = self.mirrors.get(src)?.clone();
        let mref = if kind == MirrorKind::Owned { format!("&{m}") } else { m.clone() };
        let pl = self.bind_place(*src);
        let mn = format!("__m{b}");
        let (a, z) = match range {
            Some((a, z)) => (self.val(a), self.val(z)),
            None => ("0".to_string(), format!("({pl}.len() as i64)")),
        };
        let cpu = self.val(value);
        self.mirrors.insert(b, (mn.clone(), MirrorKind::Owned));
        Some((
            format!("let {mn} = ail_gpu::Mirror::new();\n"),
            format!("if {m}.on_cpu() {{ {cpu} }} else {{ ail_gpu::slice_dev(&{pl}, {mref}, {a}, {z}, &{mn}) }}"),
        ))
    }

    /// A loop that may run in chunks (see `ail_gpu::chunk_begin`): it launches kernels, has no
    /// effects but computing (no capability is used, nothing leaves the loop early), and changes
    /// outside it only variables that can be saved by copying (returned here) and vectors with
    /// mirrors (which the runtime saves), the latter only element by element or through a call.
    fn chunk_plan(&self, lv: BindId, body: &[Expr]) -> Option<Vec<BindId>> {
        if !self.gpu_hooks() || !body.iter().any(|e| self.has_gpu(e)) {
            return None;
        }
        let mut inner: HashSet<BindId> = HashSet::from([lv]);
        let mut nested: HashSet<u32> = HashSet::new();
        for e in body {
            walk_ref(e, &mut |x| match &x.kind {
                ExprKind::Let { .. } | ExprKind::For(..) => {
                    if let Some(b) = self.ck.decl.get(&(x.id as usize)) {
                        inner.insert(*b);
                    }
                    if let ExprKind::For(_, _, _, fb) = &x.kind {
                        fb.iter().for_each(|y| walk_ref(y, &mut |z| { nested.insert(z.id); }));
                    }
                }
                ExprKind::While(_, wb) => wb.iter().for_each(|y| walk_ref(y, &mut |z| { nested.insert(z.id); })),
                _ => {}
            });
        }
        let mut snap: Vec<BindId> = Vec::new();
        let mut ok = true;
        let outer = |p: &Expr| self.root_of(p).filter(|r| !inner.contains(r));
        for e in body {
            walk_ref(e, &mut |x| {
                if !ok {
                    return;
                }
                match &x.kind {
                    ExprKind::Return(_) | ExprKind::Try(_) | ExprKind::Tasks(..) => ok = false,
                    ExprKind::Break | ExprKind::Continue if !nested.contains(&x.id) => ok = false,
                    ExprKind::Var(_) if matches!(self.ty(x), Ty::Cap(_)) => ok = false,
                    ExprKind::Set(pl, _) => {
                        if let Some(r) = outer(pl) {
                            let ty = &self.ck.bindings[r].ty;
                            let elem = matches!(&pl.kind, ExprKind::Call(n, _, a) if n == "at" && matches!(a[0].expr().kind, ExprKind::Var(_)));
                            if self.is_copy(ty) {
                                if !snap.contains(&r) {
                                    snap.push(r);
                                }
                            } else if !(elem && gpu_vec(ty) && self.mirrors.contains_key(&r)) {
                                ok = false;
                            }
                        } else if self.root_of(pl).is_none() {
                            ok = false;
                        }
                    }
                    ExprKind::Call(_, _, args) => {
                        let user = matches!(self.ck.res.get(x.id as usize), Some(Res::UserFn(_)));
                        for a in args {
                            let Arg::Mut(m, _) = a else { continue };
                            let Some(r) = outer(m) else {
                                if self.root_of(m).is_none() {
                                    ok = false;
                                }
                                continue;
                            };
                            let ty = &self.ck.bindings[r].ty;
                            let var = matches!(m.kind, ExprKind::Var(_));
                            if var && self.is_copy(ty) {
                                if !snap.contains(&r) {
                                    snap.push(r);
                                }
                            } else if !(var && user && gpu_vec(ty) && self.mirrors.contains_key(&r)) {
                                ok = false;
                            }
                        }
                    }
                    _ => {}
                }
            });
        }
        ok.then_some(snap)
    }

    /// `(for var (range lo hi) body)` in chunks: each chunk runs with its GPU work queued and is
    /// checked at its end; one that traps is undone (the saved variables here, the mirrors in
    /// the runtime) and runs again the ordinary way. Chunks grow from one iteration.
    fn for_chunked(&mut self, var: &str, lo: &str, hi: &str, body: &[Expr], snap: &[BindId]) -> String {
        let bd = self.block(body);
        let places: Vec<String> = snap.iter().map(|b| self.bind_place(*b)).collect();
        let save = places.iter().map(|p| format!("{p}, ")).collect::<String>();
        let restore = places.iter().enumerate().map(|(i, p)| format!("{p} = __snap.{i}; ")).collect::<String>();
        format!(
            "{{ let __chi: i64 = {hi}; let mut __ct: i64 = {lo}; let mut __cd: i64 = 0; \
             while __ct < __chi {{ let __cn = ail_gpu::chunk_len(__cd, __chi - __ct); \
             if __cn > 1 && ail_gpu::chunk_begin() {{ let __snap = ({save}); \
             let __r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {{ for {var} in __ct..__ct + __cn {{ let _ = {bd}; }} }})); \
             if ail_gpu::chunk_end(__r) {{ __ct += __cn; __cd += __cn; continue; }} \
             {restore}__cd = ail_gpu::chunk_retry(__cd); for {var} in __ct..__ct + __cn {{ let _ = {bd}; }} __ct += __cn; \
             }} else {{ let {var}: i64 = __ct; let _ = {bd}; __ct += 1; if __cd >= 0 {{ __cd += 1; }} }} }} }}"
        )
    }

    /// True if running `e` may launch a kernel: it holds a kernel loop or calls a GPU-aware fn.
    fn has_gpu(&self, e: &Expr) -> bool {
        let mut found = false;
        walk_ref(e, &mut |x| match &x.kind {
            ExprKind::For(..) if self.kernel_loops.contains(&x.id) => found = true,
            ExprKind::Call(..) => {
                if let Some(Res::UserFn(g)) = self.ck.res.get(x.id as usize) {
                    if self.aware.contains(g) {
                        found = true;
                    }
                }
            }
            _ => {}
        });
        found
    }

    /// The syncs to emit before statement `e`, and whether `e` then runs as plain CPU code.
    /// A statement that may launch kernels is not synced as a whole: only the parts it runs
    /// before its body (an `if` condition, a range), and its body's statements get their own.
    fn gpu_sync(&mut self, e: &Expr) -> (String, bool) {
        if self.mirrors.is_empty() {
            return (String::new(), false);
        }
        if let ExprKind::For(..) = &e.kind {
            if self.kernel_loops.contains(&e.id) {
                // The launch and its CPU fallback do their own syncing.
                return (String::new(), true);
            }
        }
        if let ExprKind::Let { value, .. } = &e.kind {
            let b = self.ck.decl.get(&(e.id as usize)).copied();
            let src_mirrored = |x: &Expr| matches!((&x.kind, self.ck.res.get(x.id as usize)), (ExprKind::Var(_), Some(Res::Local(s))) if self.mirrors.contains_key(s));
            if b.is_some_and(|b| gpu_vec(&self.ck.bindings[b].ty)) {
                match &value.kind {
                    ExprKind::Call(n, _, xs) if n == "slice" && xs.len() == 3 && src_mirrored(xs[0].expr()) => {
                        let mut s = self.gpu_sync_vars(xs[1].expr(), false);
                        s.push_str(&self.gpu_sync_vars(xs[2].expr(), false));
                        return (s, true);
                    }
                    ExprKind::Copy(x) if src_mirrored(x) => return (String::new(), true),
                    _ => {}
                }
            }
        }
        if !self.has_gpu(e) {
            return (self.gpu_sync_vars(e, false), true);
        }
        let pre = match &e.kind {
            ExprKind::If(c, _, _) => self.gpu_sync_vars(c, false),
            ExprKind::Match(sc, _) => self.gpu_sync_vars(sc, false),
            ExprKind::For(_, _, ForIter::Range(a, b), _) => {
                let mut s = self.gpu_sync_vars(a, false);
                s.push_str(&self.gpu_sync_vars(b, false));
                s
            }
            ExprKind::For(_, _, ForIter::Each(v), _) => self.gpu_sync_vars(v, false),
            ExprKind::Do(_) | ExprKind::While(..) => String::new(),
            // A statement with a call of a GPU-aware fn: its vector arguments go with their
            // mirrors; everything else it uses is synced.
            _ => self.gpu_sync_vars(e, true),
        };
        (pre, false)
    }

    /// Syncs for every vector with a mirror that `e` uses (skipping the vector arguments of
    /// calls to GPU-aware fns when `skip_aware_args`): its own memory made current, and its
    /// device copy marked stale if `e` changes it. Read-only parameters are shadowed by a CPU
    /// copy instead (they cannot be written to).
    fn gpu_sync_vars(&mut self, e: &Expr, skip_aware_args: bool) -> String {
        let mut used: Vec<BindId> = Vec::new();
        let mut skip: HashSet<u32> = HashSet::new();
        if skip_aware_args {
            walk_ref(e, &mut |x| {
                if let (ExprKind::Call(_, _, args), Some(Res::UserFn(g))) = (&x.kind, self.ck.res.get(x.id as usize)) {
                    if self.aware.contains(g) {
                        for a in args {
                            if matches!(a.expr().kind, ExprKind::Var(_)) {
                                skip.insert(a.expr().id);
                            }
                        }
                    }
                }
            });
        }
        let mut changed: HashSet<BindId> = HashSet::new();
        walk_ref(e, &mut |x| {
            if let (ExprKind::Var(_), Some(Res::Local(b))) = (&x.kind, self.ck.res.get(x.id as usize)) {
                if self.mirrors.contains_key(b) && !skip.contains(&x.id) && !used.contains(b) {
                    used.push(*b);
                }
            }
            match &x.kind {
                ExprKind::Set(pl, _) => {
                    if let Some(r) = self.root_of(pl) {
                        changed.insert(r);
                    }
                }
                ExprKind::Call(_, _, args) => {
                    for a in args {
                        if let Arg::Mut(m, _) = a {
                            if !skip.contains(&m.id) {
                                if let Some(r) = self.root_of(m) {
                                    changed.insert(r);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        });
        let mut out = String::new();
        for b in used {
            let (name, kind) = self.mirrors[&b].clone();
            let pl = self.bind_place(b);
            match kind {
                MirrorKind::RefParam => {
                    if self.viewed.insert(b) {
                        let n = self.local_name(&self.ck.bindings[b].name);
                        let et = rty(&self.ck.bindings[b].ty);
                        let t = self.fresh();
                        out.push_str(&format!("let {t} = {name}.view(&{pl}); let {n}: &{et} = &*{t}; "));
                    }
                }
                _ => {
                    out.push_str(&format!("{name}.to_cpu(&mut {pl}); "));
                    if changed.contains(&b) {
                        out.push_str(&format!("{name}.cpu_changed(); "));
                    }
                }
            }
        }
        out
    }

    /// The body as lets, one inner range loop, and sums of the inner loop folded into outer ones
    /// (see `gpu::Collapse`); None if it has another shape.
    fn collapse_plan<'b>(&self, lv: BindId, body: &'b [Expr], sum_vars: &[BindId]) -> Option<crate::gpu::Collapse<'b>> {
        let local = |e: &Expr| match (&e.kind, self.ck.res.get(e.id as usize)) {
            (ExprKind::Var(_), Some(Res::Local(b))) => Some(*b),
            _ => None,
        };
        // The inner loop, alone or as the only thing under an `(if c ...)` without else.
        let inner_of = |e: &'b Expr| -> Option<(&'b Expr, Option<&'b Expr>)> {
            match &e.kind {
                ExprKind::For(_, _, ForIter::Range(..), _) => Some((e, None)),
                ExprKind::If(c, t, None) => match &t.kind {
                    ExprKind::For(_, _, ForIter::Range(..), _) => Some((&**t, Some(&**c))),
                    ExprKind::Do(es) if es.len() == 1 && matches!(es[0].kind, ExprKind::For(_, _, ForIter::Range(..), _)) => Some((&es[0], Some(&**c))),
                    _ => None,
                },
                _ => None,
            }
        };
        let at = body.iter().position(|e| inner_of(e).is_some())?;
        let (fe, guard) = inner_of(&body[at])?;
        let ExprKind::For(_, _, ForIter::Range(jlo, jhi), inner) = &fe.kind else { return None };
        let j = *self.ck.decl.get(&(fe.id as usize))?;
        if breaks_out(inner) {
            return None;
        }
        let mut prefix = Vec::new();
        let mut cands: Vec<BindId> = Vec::new();
        let mut declared: HashSet<BindId> = HashSet::new();
        let mut muts: Vec<BindId> = Vec::new();
        for e in &body[..at] {
            let ExprKind::Let { mutable: true, .. } = &e.kind else { continue };
            muts.push(*self.ck.decl.get(&(e.id as usize))?);
        }
        // A sliding window: `(set x1 x2) ... (set xk e)` ending the inner body (see gpu::Rot).
        let mut tail = inner.len();
        while tail > 0 && matches!(&inner[tail - 1].kind, ExprKind::Set(pl, v)
            if local(pl).is_some_and(|b| muts.contains(&b)) && local(v).is_some())
        {
            tail -= 1;
        }
        let mut rot: Option<crate::gpu::Rot<'b>> = None;
        if tail < inner.len() {
            let sets: Vec<(BindId, BindId)> = inner[tail..]
                .iter()
                .map(|e| match &e.kind {
                    ExprKind::Set(pl, v) => (local(pl).unwrap(), local(v).unwrap()),
                    _ => unreachable!(),
                })
                .collect();
            let vars: Vec<BindId> = sets.iter().map(|x| x.0).collect();
            let e = sets.last()?.1;
            let chained = sets.windows(2).all(|w| w[0].1 == w[1].0);
            let distinct = vars.iter().collect::<HashSet<_>>().len() == vars.len();
            // The lets that compute e: the start of the inner body, up to e's own.
            let p = inner.iter().position(|x| matches!(x.kind, ExprKind::Let { mutable: false, .. })
                && self.ck.decl.get(&(x.id as usize)) == Some(&e))?;
            let lets = &inner[..=p];
            if !chained || !distinct || muts.contains(&e) || !lets.iter().all(|x| matches!(x.kind, ExprKind::Let { mutable: false, .. })) {
                return None;
            }
            // e depends only on j and values fixed for the row: no mutable variable, and no
            // vector the loop writes.
            let mut written: HashSet<BindId> = HashSet::new();
            for x in body.iter() {
                walk_ref(x, &mut |y| {
                    if let ExprKind::Set(pl, _) = &y.kind {
                        if let ExprKind::Call(n, _, xs) = &pl.kind {
                            if n == "at" {
                                if let Some(v) = local(xs[0].expr()) {
                                    written.insert(v);
                                }
                            }
                        }
                    }
                });
            }
            let mut pure = true;
            for x in lets {
                walk_ref(x, &mut |y| {
                    if let Some(b) = local(y) {
                        if muts.contains(&b) || written.contains(&b) {
                            pure = false;
                        }
                    }
                });
            }
            // The window variables are set only by the chain.
            for x in &inner[..tail] {
                walk_ref(x, &mut |y| {
                    if let ExprKind::Set(pl, _) = &y.kind {
                        if local(pl).is_some_and(|b| vars.contains(&b)) {
                            pure = false;
                        }
                    }
                });
            }
            if !pure {
                return None;
            }
            rot = Some(crate::gpu::Rot { vars, lets, e });
        }
        let rot_vars: Vec<BindId> = rot.as_ref().map(|r| r.vars.clone()).unwrap_or_default();
        for e in &body[..at] {
            let ExprKind::Let { mutable, value, .. } = &e.kind else { return None };
            let b = *self.ck.decl.get(&(e.id as usize))?;
            declared.insert(b);
            if rot_vars.contains(&b) {
                prefix.push(e);
            } else if *mutable {
                if !matches!(value.kind, ExprKind::Int(0)) || self.ck.bindings[b].ty != Ty::Int(IntK::I64) {
                    return None;
                }
                cands.push(b);
            } else {
                prefix.push(e);
            }
        }
        let mut inner_sums: HashMap<BindId, usize> = HashMap::new();
        for e in &body[at + 1..] {
            let ExprKind::Set(pl, v) = &e.kind else { return None };
            let s = local(pl)?;
            let si = sum_vars.iter().position(|x| *x == s)?;
            let ExprKind::Call(op, _, xs) = &v.kind else { return None };
            if op != "+" || xs.len() != 2 {
                return None;
            }
            let (a, b) = (local(xs[0].expr()), local(xs[1].expr()));
            let r = if a == Some(s) { b? } else if b == Some(s) { a? } else { return None };
            if !cands.contains(&r) || inner_sums.insert(r, si).is_some() {
                return None;
            }
        }
        if cands.len() != inner_sums.len() {
            return None;
        }
        // Inside the inner loop an inner sum is only added to.
        let mut uses: HashMap<BindId, usize> = HashMap::new();
        let mut sets: HashMap<BindId, usize> = HashMap::new();
        let mut bad = false;
        for x in inner[..tail].iter() {
            walk_ref(x, &mut |e| {
                if let Some(b) = local(e) {
                    if inner_sums.contains_key(&b) {
                        *uses.entry(b).or_insert(0) += 1;
                    }
                    if b == lv || declared.contains(&b) && !inner_sums.contains_key(&b) {
                        // reading the outer index or a prefix value is fine
                    }
                }
                if let ExprKind::Set(pl, v) = &e.kind {
                    if let Some(b) = local(pl) {
                        if inner_sums.contains_key(&b) {
                            let ok = matches!(&v.kind, ExprKind::Call(op, _, xs) if op == "+"
                                && xs.iter().filter(|a| local(a.expr()) == Some(b)).count() == 1);
                            if !ok {
                                bad = true;
                            }
                            *sets.entry(b).or_insert(0) += 1;
                        } else if b == lv || declared.contains(&b) {
                            bad = true;
                        }
                    }
                }
            });
        }
        if bad || inner_sums.keys().any(|r| uses.get(r).copied().unwrap_or(0) != 2 * sets.get(r).copied().unwrap_or(0)) {
            return None;
        }
        // The inner range does not depend on the outer index or on the body.
        let mut fixed = true;
        for x in [&**jlo, &**jhi] {
            walk_ref(x, &mut |e| {
                if let Some(b) = local(e) {
                    if b == lv || declared.contains(&b) || sum_vars.contains(&b) {
                        fixed = false;
                    }
                }
                if let ExprKind::Call(n, _, _) = &e.kind {
                    if !matches!(n.as_str(), "+" | "-" | "*" | "len") {
                        fixed = false;
                    }
                }
            });
        }
        if !fixed {
            return None;
        }
        if guard.is_some() && !inner_sums.is_empty() {
            return None;
        }
        Some(crate::gpu::Collapse { j, jlo, jhi, prefix, guard, inner, inner_sums, rot })
    }

    /// `(if c (set (at v i) x))` on an element proven in range, where neither c nor x can trap:
    /// the element is written either way (with its own value when c is false), so the loop has
    /// no branch around a store and LLVM can vectorize it.
    fn select_store(&mut self, c: &Expr, th: &Expr) -> Option<String> {
        let ExprKind::Set(place, value) = &th.kind else { return None };
        // Only where the loop vectorizes (its arithmetic defers overflow): elsewhere the extra
        // stores just cost.
        if !self.defer {
            return None;
        }
        if !matches!(&place.kind, ExprKind::Call(n, _, xs) if n == "at" && xs.len() == 2) || self.par_hooks.contains_key(&th.id) {
            return None;
        }
        let slot = self.unchecked.get(&place.id)?.clone();
        if !self.is_copy(self.ty(place)) || !self.no_trap(c) || !self.no_trap(value) {
            return None;
        }
        let (c, v) = (self.val(c), self.val(value));
        Some(format!("{{ let __c = {c}; let __n = {v}; let __s = &mut {slot}; *__s = if __c {{ __n }} else {{ *__s }}; }}"))
    }

    /// True if evaluating `e` cannot trap: literals, variables, element reads proven in range,
    /// comparisons and logic on those, and arithmetic where it records overflow instead.
    fn no_trap(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Var(_) => true,
            ExprKind::Call(n, _, xs) if matches!(self.ck.res.get(e.id as usize), Some(Res::Builtin(_))) => {
                let args_ok = xs.iter().all(|x| matches!(x, Arg::Val(_)) && self.no_trap(x.expr()));
                match n.as_str() {
                    "at" => self.unchecked.contains_key(&e.id),
                    "<" | ">" | "<=" | ">=" | "==" | "!=" | "and" | "or" | "not" | "min" | "max" => args_ok,
                    "+" | "-" | "*" => args_ok && (self.ty(e).is_float() || (self.defer && self.ty(e).is_int())),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn unsigned(&self, t: &Ty) -> bool {
        matches!(t, Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64))
    }

    /// Integer arithmetic in the fast copy of a loop: wraps and records overflow in `__of`.
    fn deferred(&mut self, name: &str, t: &Ty, args: &[Arg]) -> Option<String> {
        let unsigned = matches!(t, Ty::Int(k) if matches!(k, IntK::U8 | IntK::U16 | IntK::U32 | IntK::U64));
        let op = match (name, args.len()) {
            ("+", _) => "dadd",
            ("*", _) => "dmul",
            ("-", 1) => return Some(format!("ail_rt::Dov::dneg({}, &mut __of)", self.val(args[0].expr()))),
            ("-", 2) => "dsub",
            ("/", 2) => "ddiv",
            ("%", 2) => "drem",
            ("abs", 1) if !unsigned => return Some(format!("ail_rt::Dov::dabs({}, &mut __of)", self.val(args[0].expr()))),
            _ => return None,
        };
        let lit = |e: &Expr| matches!(e.kind, ExprKind::Int(_));
        if op == "dmul" && args.len() == 2 && (lit(args[0].expr()) || lit(args[1].expr())) {
            let (x, c) = if lit(args[1].expr()) { (args[0].expr(), args[1].expr()) } else { (args[1].expr(), args[0].expr()) };
            let (x, c) = (self.val(x), self.val(c));
            return Some(format!("ail_rt::Dov::dmulc({x}, {c}, &mut __of)"));
        }
        let mut acc = self.val(args[0].expr());
        for x in &args[1..] {
            let v = self.val(x.expr());
            acc = format!("ail_rt::Dov::{op}({acc}, {v}, &mut __of)");
        }
        Some(acc)
    }

    /// An innermost `(for j (range a b) ...)` whose vector accesses `(at v i)` (reads, and
    /// writes `(set (at v i) x)`) all have an index `j + c`, with `c` fixed during the loop
    /// (seen through the body's `let`s, so `(let k (+ row j))` then `(at v (- k n))` counts).
    /// One test before the loop shows every index in range, and then the loop runs without
    /// bounds checks. If the body is plain arithmetic and the vectors it writes are not read,
    /// the integer operations also defer their overflow check to the end of the loop, so LLVM
    /// can vectorize it; on overflow the loop runs again with every check and traps where it
    /// would have. If the test fails, the ordinary checked loop runs.
    fn for_unchecked(&mut self, lv: BindId, var: &str, lo: &str, hi: &str, a: &Expr, body: &[Expr]) -> Option<String> {
        if self.nr.is_some() {
            return None;
        }
        let mut nested = false;
        let mut accs: Vec<(u32, &Expr, &Expr)> = Vec::new();
        let mut writes: HashSet<u32> = HashSet::new();
        let mut inner: HashSet<BindId> = HashSet::new();
        let mut lets: HashMap<BindId, &Expr> = HashMap::new();
        for x in body {
            walk_ref(x, &mut |e| {
                if let Some(b) = self.ck.decl.get(&(e.id as usize)) {
                    inner.insert(*b);
                    if let ExprKind::Let { mutable: false, value, .. } = &e.kind {
                        lets.insert(*b, value);
                    }
                }
                match &e.kind {
                    ExprKind::For(..) | ExprKind::While(..) | ExprKind::Lambda(..) | ExprKind::Tasks(..) => nested = true,
                    ExprKind::Match(_, arms) => {
                        for arm in arms {
                            pat_binds(&arm.pat, self.ck, self.epoch, &mut inner);
                        }
                    }
                    ExprKind::Set(p, _) => {
                        if let ExprKind::Call(n, _, xs) = &p.kind {
                            if n == "at" && xs.len() == 2 {
                                writes.insert(p.id);
                            }
                        }
                    }
                    ExprKind::Call(n, _, args)
                        if n == "at" && args.len() == 2 && !matches!(self.ck.res.get(e.id as usize), Some(Res::UserFn(_))) =>
                    {
                        accs.push((e.id, args[0].expr(), args[1].expr()));
                    }
                    _ => {}
                }
            });
        }
        if nested || accs.is_empty() {
            return None;
        }
        let arm = Arm { pat: Pat::Wild(a.span), body: body.to_vec(), span: a.span };
        let changes = |em: &Self, b: BindId| em.arms_mutate(std::slice::from_ref(&arm), b);
        // Vars read by an expression; false if it calls anything but arithmetic.
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
        // Fixed during the loop: reads only variables that exist before it and do not change.
        let fixed = |e: &Expr| -> bool {
            let mut vs = Vec::new();
            vars_of(self, e, &mut vs) && vs.iter().all(|b| *b != lv && !inner.contains(b) && !changes(self, *b))
        };
        // `j + c1 - c2 ...` as its terms (sign, fixed expression), looking through `let`s.
        fn offset<'e>(
            em: &Emitter,
            e: &'e Expr,
            lv: BindId,
            lets: &HashMap<BindId, &'e Expr>,
            fixed: &dyn Fn(&Expr) -> bool,
            depth: usize,
            rec: &mut Vec<(u32, Vec<(bool, &'e Expr)>)>,
        ) -> Option<Vec<(bool, &'e Expr)>> {
            if depth > 8 {
                return None;
            }
            match &e.kind {
                ExprKind::Var(_) => match em.ck.res[e.id as usize] {
                    Res::Local(b) if b == lv => Some(Vec::new()),
                    Res::Local(b) => offset(em, lets.get(&b)?, lv, lets, fixed, depth + 1, rec),
                    _ => None,
                },
                ExprKind::Call(n, _, xs)
                    if xs.len() == 2 && (n == "+" || n == "-") && !matches!(em.ck.res.get(e.id as usize), Some(Res::UserFn(_))) =>
                {
                    let (x, y) = (xs[0].expr(), xs[1].expr());
                    if fixed(y) {
                        let mut o = offset(em, x, lv, lets, fixed, depth + 1, rec)?;
                        o.push((n == "+", y));
                        rec.push((e.id, o.clone()));
                        Some(o)
                    } else if n == "+" && fixed(x) {
                        let mut o = offset(em, y, lv, lets, fixed, depth + 1, rec)?;
                        o.push((true, x));
                        rec.push((e.id, o.clone()));
                        Some(o)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        // The vector an access reads: a variable or field, or a row `(at m r)` of one with `r`
        // fixed (its own index is tested before the loop too). Returns the root.
        fn path_ok<'e>(em: &Emitter, v: &'e Expr, fixed: &dyn Fn(&Expr) -> bool, rows: &mut Vec<(&'e Expr, &'e Expr)>) -> Option<BindId> {
            match &v.kind {
                ExprKind::Var(_) => match em.ck.res[v.id as usize] {
                    Res::Local(b) => Some(b),
                    _ => None,
                },
                ExprKind::Field(base, _, _) => path_ok(em, base, fixed, rows),
                ExprKind::Call(n, _, xs)
                    if n == "at" && xs.len() == 2 && !matches!(em.ck.res.get(v.id as usize), Some(Res::UserFn(_))) && fixed(xs[1].expr()) =>
                {
                    // The prefix is tested first, then this row index.
                    let root = path_ok(em, xs[0].expr(), fixed, rows)?;
                    rows.push((xs[0].expr(), xs[1].expr()));
                    Some(root)
                }
                _ => None,
            }
        }
        let fixed_ref: &dyn Fn(&Expr) -> bool = &fixed;
        // Each access: (id, vector, offset terms, index, write?, root).
        let mut plan: Vec<(u32, &Expr, Vec<(bool, &Expr)>, &Expr, bool, BindId)> = Vec::new();
        let mut row_tests: Vec<(&Expr, &Expr)> = Vec::new();
        let mut aff_nodes: Vec<(u32, Vec<(bool, &Expr)>)> = Vec::new();
        for (rid, v, i) in accs {
            let root = path_ok(self, v, fixed_ref, &mut row_tests)?;
            if root == lv || inner.contains(&root) || !matches!(self.ty(v), Ty::Vec(_)) {
                return None;
            }
            let off = offset(self, i, lv, &lets, fixed_ref, 0, &mut aff_nodes)?;
            plan.push((rid, v, off, i, writes.contains(&rid), root));
        }
        // A written vector changes only through the planned element writes, and is reached
        // through one path only (it is borrowed mutably for the whole loop).
        let written: HashSet<BindId> = plan.iter().filter(|x| x.4).map(|x| x.5).collect();
        for x in &plan {
            if !written.contains(&x.5) && changes(self, x.5) {
                return None;
            }
        }
        if !written.is_empty() {
            let mut paths: HashMap<BindId, String> = HashMap::new();
            for x in &plan {
                if written.contains(&x.5) {
                    let pl = self.pv(x.1);
                    if paths.insert(x.5, pl.clone()).is_some_and(|old| old != pl) {
                        return None;
                    }
                }
            }
            // Root occurrences allowed: the ones inside planned accesses' vector paths.
            let mut allowed: HashSet<u32> = HashSet::new();
            for x in &plan {
                walk_ref(x.1, &mut |e| {
                    allowed.insert(e.id);
                });
            }
            let planned_writes: HashSet<u32> = plan.iter().filter(|x| x.4).map(|x| x.0).collect();
            let mut bad = false;
            for x in body {
                walk_ref(x, &mut |e| match &e.kind {
                    ExprKind::Var(_) => {
                        if let Res::Local(b) = self.ck.res[e.id as usize] {
                            if written.contains(&b) && !allowed.contains(&e.id) {
                                bad = true;
                            }
                        }
                    }
                    ExprKind::Set(p, _) => {
                        if self.root_of(p).is_some_and(|b| written.contains(&b)) && !planned_writes.contains(&p.id) {
                            bad = true;
                        }
                    }
                    _ => {}
                });
            }
            if bad {
                return None;
            }
        }
        // Deferred overflow: the body only computes, the vectors it writes are not read, and
        // outer variables it sets are plain values restored before the checked run.
        let reads_written = plan.iter().any(|x| !x.4 && written.contains(&x.5));
        let mut outer_sets: Vec<BindId> = Vec::new();
        let defer = !reads_written && body.iter().all(|e| self.defer_ok(e, &inner, &mut outer_sets));
        let defer = defer && outer_sets.iter().all(|b| self.is_copy(&self.ck.bindings[*b].ty));
        // i128 form of a fixed expression, for tests that cannot overflow themselves.
        fn wide(em: &mut Emitter, e: &Expr) -> String {
            match &e.kind {
                ExprKind::Call(n, _, xs) if matches!(n.as_str(), "+" | "-" | "*") && !xs.is_empty() => {
                    if n == "-" && xs.len() == 1 {
                        return format!("(-{})", wide(em, xs[0].expr()));
                    }
                    let parts: Vec<String> = xs.iter().map(|x| wide(em, x.expr())).collect();
                    format!("({})", parts.join(&format!(" {n} ")))
                }
                ExprKind::Call(n, _, xs) if n == "len" => format!("({}.len() as i128)", em.pv(xs[0].expr())),
                _ => format!("(({}) as i128)", em.val(e)),
            }
        }
        // One slice per vector, taken once; a test per distinct (vector, offset).
        // (place, element type, written, shared by a parallel loop: a raw pointer)
        let mut slices: Vec<(String, String, bool, bool)> = Vec::new();
        let mut tests: Vec<String> = Vec::new();
        for (m, r) in &row_tests {
            let t = format!("({} >= 0 && {} < ({}.len() as i128))", wide(self, r), wide(self, r), self.pv(m));
            if !tests.contains(&t) {
                tests.push(t);
            }
        }
        let row_t: Vec<String> = tests.clone();
        let mut codes: Vec<(u32, String)> = Vec::new();
        let mut idxs: Vec<&Expr> = Vec::new();
        // The iterations whose accesses are all in range: j >= -o and j < len - o for each.
        let mut lows: Vec<String> = Vec::new();
        let mut highs: Vec<String> = Vec::new();
        for (rid, v, off, i, _, root) in &plan {
            let shared = match &v.kind {
                ExprKind::Var(_) => self.par_shared.get(root).cloned(),
                _ => None,
            };
            let place = match &shared {
                Some(n) => n.clone(),
                None => self.pv(v),
            };
            let elem = rty(&self.elem_ty(v));
            let w = written.contains(root);
            let k = match slices.iter().position(|s| s.0 == place) {
                Some(k) => k,
                None => {
                    slices.push((place.clone(), elem, w, shared.is_some()));
                    slices.len() - 1
                }
            };
            let mut o = String::from("0i128");
            for (plus, c) in off {
                let c = wide(self, c);
                o.push_str(if *plus { " + " } else { " - " });
                o.push_str(&c);
            }
            let t = format!("((__ulo as i128) + ({o}) >= 0 && (__uhi as i128) - 1 + ({o}) < ({place}.len() as i128))");
            if !tests.contains(&t) {
                tests.push(t);
            }
            let l = format!("-({o})");
            if !lows.contains(&l) {
                lows.push(l);
            }
            let h = format!("({place}.len() as i128) - ({o})");
            if !highs.contains(&h) {
                highs.push(h);
            }
            let get = if slices[k].3 { "add" } else if w { "get_unchecked_mut" } else { "get_unchecked" };
            codes.push((*rid, format!("__us{k}.{get}")));
            idxs.push(*i);
        }
        // Every `j + c` node on the way to an index stays within its type for the iterations
        // that run fast, so the fast copy computes it without a check.
        let mut aff_ids: HashSet<u32> = HashSet::new();
        let mut aff_t: Vec<String> = Vec::new();
        for (id, terms) in &aff_nodes {
            if !aff_ids.insert(*id) {
                continue;
            }
            let Ty::Int(k) = self.ck.ty[*id as usize].clone() else { continue };
            let mut o = String::from("0i128");
            for (plus, c) in terms {
                o.push_str(if *plus { " + " } else { " - " });
                o.push_str(&wide(self, c));
            }
            let r = k.rust();
            aff_t.push(format!("((__flo as i128) + ({o}) >= ({r}::MIN as i128) && (__fhi as i128) - 1 + ({o}) <= ({r}::MAX as i128))"));
        }
        let aff_ok = if aff_t.is_empty() { String::from("true") } else { aff_t.join(" && ") };
        let was_aff = std::mem::replace(&mut self.affine, aff_ids);
        let was = self.defer;
        self.defer = defer;
        for ((_, c), i) in codes.iter_mut().zip(idxs) {
            let idx = self.val(i);
            *c = format!("(*{c}(({idx}) as usize))");
        }
        let mut take = String::new();
        for (k, (place, elem, w, sh)) in slices.iter().enumerate() {
            if *sh {
                take.push_str(&format!("let __us{k}: *mut {elem} = {place}.ptr(); "));
            } else if *w {
                take.push_str(&format!("let __us{k}: &mut [{elem}] = &mut {place}[..]; "));
            } else {
                take.push_str(&format!("let __us{k}: &[{elem}] = &{place}[..]; "));
            }
        }
        let saved: Vec<(u32, Option<String>)> = codes.iter().map(|(id, _)| (*id, self.unchecked.get(id).cloned())).collect();
        for (id, c) in codes {
            self.unchecked.insert(id, c);
        }
        let fast = self.block(body);
        self.affine = was_aff;
        self.defer = was;
        for (id, old) in saved {
            match old {
                Some(c) => self.unchecked.insert(id, c),
                None => self.unchecked.remove(&id),
            };
        }
        let slow = self.block(body);
        if !breaks_out(body) {
            // Split the range: checked iterations at the edges, where some access would be out
            // of range (an `(at v (- j 1))` guarded by `(> j 0)`, say), unchecked in between.
            let slow_in = |a: &str, b: &str| format!("for {var} in {a}..{b} {{ let _ = {slow}; }}");
            let fast_loop = format!("for {var} in __flo..__fhi {{ let _ = unsafe {{ {fast} }}; }}");
            let fast_run = if defer {
                let mut snap = String::new();
                let mut restore = String::new();
                for (k, b) in outer_sets.iter().enumerate() {
                    let pl = self.bind_place(*b);
                    snap.push_str(&format!("let __sn{k} = {pl}; "));
                    restore.push_str(&format!("{pl} = __sn{k}; "));
                }
                format!(
                    "{snap}let __of = {{ {take}let mut __of = false; {fast_loop} __of }}; if __of {{ {restore}{} }}",
                    slow_in("__flo", "__fhi")
                )
            } else {
                format!("{{ {take}{fast_loop} }}")
            };
            let row_ok = if row_t.is_empty() { String::from("true") } else { row_t.join(" && ") };
            let lo_b: String = lows.iter().map(|l| format!(".max({l})")).collect();
            let hi_b: String = highs.iter().map(|h| format!(".min({h})")).collect();
            return Some(format!(
                "{{ let __ulo: i64 = {lo}; let __uhi: i64 = {hi};                  let (__flo, __fhi): (i64, i64) = if __ulo < __uhi && {row_ok} {{ let __a = (__ulo as i128){lo_b}; let __b = (__uhi as i128){hi_b}; if __a < __b {{ (__a as i64, __b as i64) }} else {{ (__uhi, __uhi) }} }} else {{ (__uhi, __uhi) }};                  {} if __flo < __fhi {{ if {aff_ok} {{ {fast_run} }} else {{ {} }} }} {} }}",
                slow_in("__ulo", "__flo"),
                slow_in("__flo", "__fhi"),
                slow_in("__fhi", "__uhi")
            ));
        }
        let slow_loop = format!("for {var} in __ulo..__uhi {{ let _ = {slow}; }}");
        let fast_run = if defer {
            let mut snap = String::new();
            let mut restore = String::new();
            for (k, b) in outer_sets.iter().enumerate() {
                let pl = self.bind_place(*b);
                snap.push_str(&format!("let __sn{k} = {pl}; "));
                restore.push_str(&format!("{pl} = __sn{k}; "));
            }
            format!(
                "{snap}let __of = {{ {take}let mut __of = false; for {var} in __ulo..__uhi {{ let _ = unsafe {{ {fast} }}; }} __of }}; if __of {{ {restore}{slow_loop} }}"
            )
        } else {
            format!("{take}for {var} in __ulo..__uhi {{ let _ = unsafe {{ {fast} }}; }}")
        };
        Some(format!(
            "{{ let __ulo: i64 = {lo}; let __uhi: i64 = {hi}; let (__flo, __fhi) = (__ulo, __uhi); if __ulo < __uhi && {} && {aff_ok} {{ {fast_run} }} else {{ {slow_loop} }} }}",
            tests.join(" && ")
        ))
    }

    fn narrow_ok(&self) -> bool {
        self.nr.is_none()
            && !self.nr_off
            && !self.in_par
            && !self.defer
            && !self.no_par
            && !self.web
            && !(self.pass2 && self.aware.contains(&self.fn_name))
    }

    /// A loop whose inner loops read vectors at indices loaded from other vectors (a graph walk,
    /// say) is limited by memory, and both the 64-bit elements and the bounds checks on those
    /// loaded indices cost there. Such a loop runs on 32-bit copies of its I64 vectors, and each
    /// vector whose elements serve as indices of another keeps them within that one's length,
    /// so those reads need no check. Every store checks that its value stays in range; one that
    /// does not throws the fast run away (its output is held until the end, the variables it
    /// changed are put back) and the loop runs again as written, which gives the same result.
    /// A trap inside the fast run is the program's own: the run computes exactly what the loop
    /// computes, in the same order, as long as every value fits.
    fn narrow_plan(&self, whole: &Expr) -> Option<NarrowPlan> {
        let i64v = |t: &Ty| matches!(t, Ty::Vec(et) if **et == Ty::Int(IntK::I64));
        let var_of = |e: &Expr| -> Option<BindId> {
            match (&e.kind, self.ck.res.get(e.id as usize)) {
                (ExprKind::Var(_), Some(Res::Local(b))) => Some(*b),
                _ => None,
            }
        };
        let builtin = |e: &Expr| matches!(self.ck.res.get(e.id as usize), Some(Res::Builtin(_)));
        let mut decl: HashSet<BindId> = HashSet::new();
        let mut lam_decl: HashSet<BindId> = HashSet::new();
        let mut lets: HashMap<BindId, &Expr> = HashMap::new();
        let mut fills: HashMap<BindId, &Expr> = HashMap::new();
        let mut ranges: HashMap<BindId, (&Expr, &Expr)> = HashMap::new();
        let mut set_vars: HashSet<BindId> = HashSet::new();
        let mut set_roots: HashSet<BindId> = HashSet::new();
        let mut mut_roots: HashSet<BindId> = HashSet::new();
        let mut ok_occ: HashSet<u32> = HashSet::new();
        let mut occ: Vec<(u32, BindId)> = Vec::new();
        let mut uses: Vec<(u32, BindId, &Expr, u32)> = Vec::new();
        let mut vars: HashMap<BindId, &Expr> = HashMap::new();
        let mut set_count: HashMap<BindId, u32> = HashMap::new();
        let mut stores: Vec<(u32, BindId, &Expr)> = Vec::new();
        let mut counted: Vec<(BindId, &Expr, &[Expr])> = Vec::new();
        let mut nested = false;
        let mut bad = false;
        walk_depth(whole, 0, 0, &mut |e, d, lam| {
            if let Some(&b) = self.ck.decl.get(&(e.id as usize)) {
                decl.insert(b);
                if lam > 0 {
                    lam_decl.insert(b);
                }
                match &e.kind {
                    ExprKind::Let { mutable, value, .. } => {
                        if !mutable {
                            lets.insert(b, value);
                        } else {
                            vars.insert(b, value);
                        }
                        if let ExprKind::Call(n, _, fa) = &value.kind {
                            if n == "fill" && fa.len() == 2 && builtin(value) && i64v(&self.ck.bindings[b].ty) {
                                fills.insert(b, fa[0].expr());
                            }
                        }
                    }
                    ExprKind::For(_, _, ForIter::Range(lo, hi), _) => {
                        ranges.insert(b, (lo, hi));
                    }
                    _ => {}
                }
            }
            // `(while (< e end) ... (set e (+ e 1)))`: inside the body, e is below end.
            if let ExprKind::While(c, wb) = &e.kind {
                if let (ExprKind::Call(n, _, cs), Some(ExprKind::Set(sp, sv))) = (&c.kind, wb.last().map(|x| &x.kind)) {
                    if n == "<" && cs.len() == 2 && builtin(c) {
                        if let (Some(ev), Some(sb)) = (var_of(cs[0].expr()), var_of(sp)) {
                            let inc = matches!(&sv.kind, ExprKind::Call(p, _, ps) if p == "+" && ps.len() == 2 && builtin(sv)
                                && var_of(ps[0].expr()) == Some(ev) && matches!(ps[1].expr().kind, ExprKind::Int(1)));
                            if ev == sb && inc {
                                counted.push((ev, cs[1].expr(), &wb[..wb.len() - 1]));
                            }
                        }
                    }
                }
            }
            match &e.kind {
                ExprKind::While(..) | ExprKind::For(..) if e.id != whole.id => nested = true,
                ExprKind::Return(_) | ExprKind::Try(_) | ExprKind::Tasks(..) => bad = true,
                ExprKind::Match(_, arms) => {
                    for arm in arms {
                        pat_binds(&arm.pat, self.ck, self.epoch, &mut decl);
                    }
                }
                ExprKind::Lambda(..) => {
                    if let Some(ps) = self.ck.lambda_params.get(&e.id) {
                        decl.extend(ps.iter().copied());
                    }
                }
                ExprKind::Var(_) => {
                    if let Some(b) = var_of(e) {
                        if i64v(&self.ck.bindings[b].ty) {
                            occ.push((e.id, b));
                        }
                    }
                }
                ExprKind::Set(pl, value) => match &pl.kind {
                    ExprKind::Var(_) => match var_of(pl) {
                        Some(b) => {
                            set_vars.insert(b);
                            *set_count.entry(b).or_insert(0) += 1;
                        }
                        None => bad = true,
                    },
                    ExprKind::Call(n, _, xs) if n == "at" && xs.len() == 2 && var_of(xs[0].expr()).is_some() => {
                        let q = var_of(xs[0].expr()).unwrap();
                        set_roots.insert(q);
                        stores.push((e.id, q, value));
                    }
                    _ => match self.root_of(pl) {
                        Some(r) => {
                            set_roots.insert(r);
                        }
                        None => bad = true,
                    },
                },
                ExprKind::Call(n, _, args) => {
                    let caps = args.iter().any(|a| matches!(self.ty(a.expr()), Ty::Cap(_)));
                    match self.ck.res.get(e.id as usize) {
                        Some(Res::Builtin(_)) => match n.as_str() {
                            "at" if args.len() == 2 => {
                                if let Some(b) = var_of(args[0].expr()) {
                                    ok_occ.insert(args[0].expr().id);
                                    uses.push((e.id, b, args[1].expr(), d));
                                }
                            }
                            "len" if args.len() == 1 => {
                                if var_of(args[0].expr()).is_some() {
                                    ok_occ.insert(args[0].expr().id);
                                }
                            }
                            "print" | "println" => {
                                bad |= !matches!(self.ty(args[0].expr()), Ty::Cap(crate::types::Cap::Console))
                                    || args[1..].iter().any(|a| matches!(self.ty(a.expr()), Ty::Cap(_)));
                            }
                            "eprintln" | "par-map" | "par-run" | "http-serve" | "http-request" | "send" | "recv" => bad = true,
                            _ => bad |= caps,
                        },
                        Some(Res::UserFn(_)) => bad |= caps,
                        Some(Res::Variant(..)) => {}
                        _ => bad = true,
                    }
                    for a in args {
                        if let Arg::Mut(m, _) = a {
                            match self.root_of(m) {
                                Some(r) => {
                                    mut_roots.insert(r);
                                }
                                None => bad = true,
                            }
                        }
                    }
                }
                _ => {}
            }
        });
        if bad || !nested {
            return None;
        }
        let outer = |b: &BindId| !decl.contains(b);
        // Plain variables from before the loop may change: they are put back on a bail.
        let mut scalars: Vec<BindId> = set_vars.iter().filter(|b| outer(b)).copied().collect();
        scalars.sort_unstable();
        for b in &scalars {
            if !matches!(self.ck.bindings[*b].ty, Ty::Int(_) | Ty::F64 | Ty::F32 | Ty::Bool) {
                return None;
            }
        }
        if mut_roots.iter().any(outer) {
            return None;
        }
        // Candidates: I64 vectors the loop touches only through `at`, `set at` and `len`.
        let mut by_bind: HashMap<BindId, bool> = HashMap::new();
        for (id, b) in &occ {
            let ok = by_bind.entry(*b).or_insert(true);
            *ok &= ok_occ.contains(id);
        }
        let cand = |b: &BindId| -> bool {
            by_bind.get(b).copied().unwrap_or(false)
                && !mut_roots.contains(b)
                && !lam_decl.contains(b)
                && if decl.contains(b) {
                    fills.contains_key(b)
                } else {
                    matches!(self.ck.bindings[*b].kind, BindKind::Local { .. } | BindKind::Param { .. })
                        && !self.views.contains_key(b)
                        && !self.mirrors.contains_key(b)
                }
        };
        // Vectors from before the loop that it writes must have a narrow copy (that is what
        // makes the fast run disposable); the ones it only reads, if read in an inner loop.
        if set_roots.iter().any(|r| outer(r) && !cand(r)) {
            return None;
        }
        let hot: HashSet<BindId> = uses.iter().filter(|u| u.3 >= 2).map(|u| u.1).collect();
        let mut outer_n: Vec<BindId> =
            by_bind.keys().filter(|b| outer(b) && cand(b) && (set_roots.contains(b) || hot.contains(b))).copied().collect();
        outer_n.sort_unstable();
        let mut inner_n: Vec<BindId> = by_bind.keys().filter(|b| !outer(b) && cand(b)).copied().collect();
        inner_n.sort_unstable();
        let narrowed: HashSet<BindId> = outer_n.iter().chain(inner_n.iter()).copied().collect();
        // The length of a narrowed vector, known before the loop starts.
        let len_of = |b: BindId| -> Option<String> {
            if outer(&b) {
                return Some(format!("({}.len() as i128)", self.bind_place(b)));
            }
            let n = fills.get(&b)?;
            match &n.kind {
                ExprKind::Int(v) => Some(format!("({v}i128)")),
                ExprKind::Var(_) => {
                    let x = var_of(n)?;
                    (outer(&x) && !set_vars.contains(&x) && self.ck.bindings[x].ty.is_int())
                        .then(|| format!("({} as i128)", self.bind_place(x)))
                }
                ExprKind::Call(l, _, xs) if l == "len" && xs.len() == 1 && builtin(n) => {
                    let v = var_of(xs[0].expr())?;
                    (outer(&v) && !set_vars.contains(&v) && !mut_roots.contains(&v))
                        .then(|| format!("({}.len() as i128)", self.bind_place(v)))
                }
                _ => None,
            }
        };
        // An index read from a narrowed vector, plus a constant: (vector, constant).
        let base = |e: &Expr| -> Option<BindId> {
            let e = match var_of(e) {
                Some(y) => *lets.get(&y)?,
                None => e,
            };
            match &e.kind {
                ExprKind::Call(n, _, xs) if n == "at" && xs.len() == 2 && builtin(e) => {
                    var_of(xs[0].expr()).filter(|a| narrowed.contains(a))
                }
                _ => None,
            }
        };
        let load_of = |e: &Expr| -> Option<(BindId, i128)> {
            if let Some(a) = base(e) {
                return Some((a, 0));
            }
            let ExprKind::Call(n, _, xs) = &e.kind else { return None };
            if xs.len() != 2 || !builtin(e) {
                return None;
            }
            match (n.as_str(), &xs[0].expr().kind, &xs[1].expr().kind) {
                ("+", _, ExprKind::Int(c)) => base(xs[0].expr()).map(|a| (a, *c)),
                ("+", ExprKind::Int(c), _) => base(xs[1].expr()).map(|a| (a, *c)),
                ("-", _, ExprKind::Int(c)) => base(xs[0].expr()).map(|a| (a, -*c)),
                _ => None,
            }
        };
        let mut lows: HashMap<BindId, i128> = HashMap::new();
        let mut highs: HashMap<BindId, Vec<String>> = HashMap::new();
        let mut unchecked: HashSet<u32> = HashSet::new();
        for (aid, bv, idx, d) in &uses {
            if *d < 2 || !narrowed.contains(bv) {
                continue;
            }
            let Some(len) = len_of(*bv) else { continue };
            if let Some((a, c)) = load_of(idx) {
                let lo = lows.entry(a).or_insert(i128::MIN);
                *lo = (*lo).max(-c);
                highs.entry(a).or_default().push(format!("({len} - 1 - ({c}))"));
                unchecked.insert(*aid);
                continue;
            }
            // A range loop variable between two narrowed reads (lo <= k < hi), or a variable
            // that starts at a narrowed read, only grows by one at the end of a while loop
            // whose condition keeps it below another one, read inside that loop.
            let Some(k) = var_of(idx) else { continue };
            if self.ck.bindings[k].ty != Ty::Int(IntK::I64) {
                continue;
            }
            let (lo, hi) = if let Some((lo, hi)) = ranges.get(&k) {
                if set_vars.contains(&k) {
                    continue;
                }
                (*lo, *hi)
            } else {
                let Some((_, end, _)) = counted.iter().find(|(v, _, wb)| {
                    *v == k && wb.iter().any(|x| {
                        let mut hit = false;
                        walk_ref(x, &mut |y| hit |= y.id == *aid);
                        hit
                    })
                }) else {
                    continue;
                };
                let (Some(init), Some(1)) = (vars.get(&k), set_count.get(&k)) else { continue };
                (*init, *end)
            };
            let lo_c = match &lo.kind {
                ExprKind::Int(v) if *v >= 0 => Some(None),
                _ => load_of(lo).map(|(a, c)| Some((a, -c))),
            };
            let (Some(lo_c), Some((ha, hc))) = (lo_c, load_of(hi)) else { continue };
            if let Some((a, m)) = lo_c {
                let l = lows.entry(a).or_insert(i128::MIN);
                *l = (*l).max(m);
            }
            highs.entry(ha).or_default().push(format!("({len} - ({hc}))"));
            unchecked.insert(*aid);
        }
        if unchecked.is_empty() {
            return None;
        }
        let mut copies: HashMap<u32, BindId> = HashMap::new();
        for (sid, q, v) in &stores {
            if let (true, Some(a)) = (narrowed.contains(q), base(v)) {
                copies.insert(*sid, a);
            }
        }
        let mut written: Vec<BindId> = outer_n.iter().filter(|b| set_roots.contains(b)).copied().collect();
        written.sort_unstable();
        Some(NarrowPlan { outer: outer_n, written, inner: inner_n, scalars, lows, highs, unchecked, copies })
    }

    fn narrow_region(&mut self, e: &Expr) -> Option<String> {
        let plan = self.narrow_plan(e)?;
        let k = e.id;
        let label = format!("'__nr{k}");
        let mut vecs: HashMap<BindId, NarrowVec> = HashMap::new();
        let mut bounds = String::new();
        for &b in plan.outer.iter().chain(plan.inner.iter()) {
            let lo = plan.lows.get(&b).copied().unwrap_or(i128::MIN).max(i32::MIN as i128);
            let hi = match plan.highs.get(&b) {
                Some(hs) if !hs.is_empty() => {
                    let h: String = hs.iter().map(|h| format!(".min({h})")).collect();
                    bounds.push_str(&format!("let __nhi{b}: i64 = (i32::MAX as i128){h}.max(i32::MIN as i128 - 1) as i64; "));
                    format!("__nhi{b}")
                }
                _ => "(i32::MAX as i64)".to_string(),
            };
            vecs.insert(b, NarrowVec { name: format!("__nv{b}"), lo: format!("({lo}i64)"), hi });
        }
        // A copy between narrowed vectors needs no check if the source's range lies within the
        // target's (both known now); otherwise the loop runs as written.
        let mut subs: Vec<String> = Vec::new();
        let mut nocheck: HashSet<u32> = HashSet::new();
        let mut pairs: Vec<(u32, BindId)> = plan.copies.iter().map(|(a, b)| (*a, *b)).collect();
        pairs.sort_unstable();
        for (sid, a) in pairs {
            let Some(q) = find_expr(std::slice::from_ref(e), sid).and_then(|st| match &st.kind {
                ExprKind::Set(pl, _) => match &pl.kind {
                    ExprKind::Call(_, _, xs) => match self.ck.res.get(xs[0].expr().id as usize) {
                        Some(Res::Local(q)) => Some(*q),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }) else {
                continue;
            };
            let (va, vq) = (&vecs[&a], &vecs[&q]);
            if vq.hi != "(i32::MAX as i64)" || vq.lo != format!("({}i64)", i32::MIN) {
                subs.push(format!("{} >= {} && {} <= {}", va.lo, vq.lo, va.hi, vq.hi));
            }
            nocheck.insert(sid);
        }
        let sub = if subs.is_empty() { "true".to_string() } else { subs.join(" && ") };
        let size = if plan.outer.is_empty() {
            "true".to_string()
        } else {
            let lens: Vec<String> = plan.outer.iter().map(|b| format!("{}.len()", self.bind_place(*b))).collect();
            format!("({}) >= (1 << 15)", lens.join(" + "))
        };
        let tys: String = plan.outer.iter().map(|_| "Vec<i32>, ").collect();
        let convs: String = plan
            .outer
            .iter()
            .map(|b| format!("ail_rt::narrow(&{}, {}, {})?, ", self.bind_place(*b), vecs[b].lo, vecs[b].hi))
            .collect();
        let pat: String = plan.outer.iter().map(|b| format!("mut {}, ", vecs[b].name)).collect();
        let mut snap = String::new();
        let mut restore = String::new();
        for (i, b) in plan.scalars.iter().enumerate() {
            let pl = self.bind_place(*b);
            snap.push_str(&format!("let __ns{k}_{i} = {pl}; "));
            restore.push_str(&format!("{pl} = __ns{k}_{i}; "));
        }
        let back: String = plan
            .written
            .iter()
            .map(|b| format!("ail_rt::widen_into(&mut {}, &{}); ", self.bind_place(*b), vecs[b].name))
            .collect();
        self.nr = Some(NarrowCtx { label: label.clone(), vecs, unchecked: plan.unchecked, inner: plan.inner.iter().copied().collect(), nocheck });
        let was = self.no_par;
        self.no_par = true;
        let fast = self.rvalue(e);
        self.no_par = was;
        self.nr = None;
        let was_off = self.nr_off;
        self.nr_off = true;
        let slow = self.rvalue(e);
        self.nr_off = was_off;
        Some(format!(
            "{{ let mut __nrd{k} = false; if {size} {{ {bounds}let __nrc{k} = if {sub} {{ (|| -> Option<({tys})> {{ Some(({convs})) }})() }} else {{ None }}; \
             if let Some(({pat})) = __nrc{k} {{ {snap}let __nh{k} = ail_rt::hold_begin(); \
             let __nok{k}: bool = {label}: {{ let _ = {fast}; true }}; \
             if __nok{k} {{ ail_rt::hold_commit(); {back}__nrd{k} = true; }} else {{ ail_rt::hold_discard(__nh{k}); {restore}}} }} }} \
             if !__nrd{k} {{ let _ = {slow}; }} }}"
        ))
    }

    /// True if `e` only computes (arithmetic, comparisons, planned vector accesses, `let`,
    /// `set` of locals and of planned elements, `if`, `do`), so running it twice changes
    /// nothing but the variables `outer` collects (set here, declared before the loop).
    fn defer_ok(&self, e: &Expr, inner: &HashSet<BindId>, outer: &mut Vec<BindId>) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) => true,
            ExprKind::Var(_) => true,
            ExprKind::Let { value, .. } => self.defer_ok(value, inner, outer),
            ExprKind::Set(p, v) => {
                let p_ok = match &p.kind {
                    ExprKind::Var(_) => match self.ck.res[p.id as usize] {
                        Res::Local(b) if inner.contains(&b) => true,
                        Res::Local(b) => {
                            // Outside variables only as sums (a reduction LLVM vectorizes); a
                            // running minimum or the like cannot vectorize, and there the
                            // deferred checks only cost.
                            let sum = matches!(&v.kind, ExprKind::Call(op, _, xs) if (op == "+" || op == "-")
                                && matches!((&xs[0].expr().kind, &self.ck.res[xs[0].expr().id as usize]), (ExprKind::Var(_), Res::Local(x)) if *x == b));
                            if sum && !outer.contains(&b) {
                                outer.push(b);
                            }
                            sum
                        }
                        _ => false,
                    },
                    ExprKind::Call(n, _, _) => n == "at",
                    _ => false,
                };
                p_ok && self.defer_ok(v, inner, outer)
            }
            ExprKind::If(c, t, f) => {
                self.defer_ok(c, inner, outer)
                    && self.defer_ok(t, inner, outer)
                    && f.as_ref().map_or(true, |f| self.defer_ok(f, inner, outer))
            }
            ExprKind::Do(es) => es.iter().all(|x| self.defer_ok(x, inner, outer)),
            ExprKind::Field(b, _, _) => self.defer_ok(b, inner, outer),
            ExprKind::As(_, x) => {
                let from = self.ty(x);
                let to = self.ty(e);
                let plain = !(matches!(to, Ty::Int(_)) && (from.is_int() || from.is_float()) && from != to);
                plain && self.defer_ok(x, inner, outer)
            }
            ExprKind::Call(n, _, args) => {
                if !matches!(self.ck.res.get(e.id as usize), Some(Res::Builtin(_))) {
                    return false;
                }
                let ok_name = matches!(
                    n.as_str(),
                    "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "and" | "or" | "not" | "min" | "max" | "abs" | "bit-and" | "bit-or" | "bit-xor"
                ) || n == "at";
                ok_name && args.iter().all(|a| matches!(a, Arg::Val(_)) && self.defer_ok(a.expr(), inner, outer))
            }
            _ => false,
        }
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
/// A vector whose elements a kernel can use in place: it gets a mirror in GPU-aware fns.
fn gpu_vec(t: &Ty) -> bool {
    matches!(t, Ty::Vec(el) if crate::gpu::elem_ok(el).is_some())
}

pub fn walk_ref_pub<'e>(e: &'e Expr, f: &mut dyn FnMut(&'e Expr)) {
    walk_ref(e, f)
}

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

/// `walk_ref` that also passes how many loops enclose each expression (a loop's condition and
/// body count it, its range does not) and how many lambdas.
fn walk_depth<'e>(e: &'e Expr, d: u32, lam: u32, f: &mut dyn FnMut(&'e Expr, u32, u32)) {
    f(e, d, lam);
    match &e.kind {
        ExprKind::While(c, b) => {
            walk_depth(c, d + 1, lam, f);
            b.iter().for_each(|x| walk_depth(x, d + 1, lam, f))
        }
        ExprKind::For(_, _, it, b) => {
            match it {
                ForIter::Range(a, z) => {
                    walk_depth(a, d, lam, f);
                    walk_depth(z, d, lam, f)
                }
                ForIter::Each(v) => walk_depth(v, d, lam, f),
            }
            b.iter().for_each(|x| walk_depth(x, d + 1, lam, f))
        }
        ExprKind::Lambda(_, b) => b.iter().for_each(|x| walk_depth(x, d, lam + 1, f)),
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Var(_) | ExprKind::Break | ExprKind::Continue => {}
        ExprKind::Call(_, _, args) => args.iter().for_each(|a| walk_depth(a.expr(), d, lam, f)),
        ExprKind::Let { value, .. } => walk_depth(value, d, lam, f),
        ExprKind::Set(a, b) => {
            walk_depth(a, d, lam, f);
            walk_depth(b, d, lam, f)
        }
        ExprKind::If(c, t, e2) => {
            walk_depth(c, d, lam, f);
            walk_depth(t, d, lam, f);
            if let Some(x) = e2 {
                walk_depth(x, d, lam, f)
            }
        }
        ExprKind::Do(es) => es.iter().for_each(|x| walk_depth(x, d, lam, f)),
        ExprKind::Return(v) => {
            if let Some(v) = v {
                walk_depth(v, d, lam, f)
            }
        }
        ExprKind::Match(s, arms) => {
            walk_depth(s, d, lam, f);
            for a in arms {
                a.body.iter().for_each(|x| walk_depth(x, d, lam, f))
            }
        }
        ExprKind::New(_, _, fs) => fs.iter().for_each(|x| walk_depth(&x.2, d, lam, f)),
        ExprKind::With(b, fs) => {
            walk_depth(b, d, lam, f);
            fs.iter().for_each(|x| walk_depth(&x.2, d, lam, f))
        }
        ExprKind::Field(b, _, _) => walk_depth(b, d, lam, f),
        ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => walk_depth(x, d, lam, f),
        ExprKind::Tasks(_, ts) => ts.iter().for_each(|t| t.body.iter().for_each(|x| walk_depth(x, d, lam, f))),
    }
}

/// True if a `break` in `es` would leave the loop whose body `es` is (not an inner loop).
fn breaks_out(es: &[Expr]) -> bool {
    fn go(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Break => true,
            ExprKind::For(..) | ExprKind::While(..) | ExprKind::Lambda(..) => false,
            ExprKind::Let { value, .. } => go(value),
            ExprKind::Set(a, b) => go(a) || go(b),
            ExprKind::If(c, t, f) => go(c) || go(t) || f.as_ref().is_some_and(|f| go(f)),
            ExprKind::Do(es) => es.iter().any(go),
            ExprKind::Match(s, arms) => go(s) || arms.iter().any(|a| a.body.iter().any(go)),
            ExprKind::Call(_, _, args) => args.iter().any(|a| go(a.expr())),
            _ => false,
        }
    }
    es.iter().any(go)
}

/// The expression with this id inside `es`.
fn find_expr(es: &[Expr], id: u32) -> Option<&Expr> {
    let mut found = None;
    for x in es {
        walk_ref(x, &mut |e| {
            if e.id == id {
                found = Some(e);
            }
        });
    }
    found
}

/// An index as `i * stride + j + consts` (see `aff`).
#[derive(Default)]
struct Aff<'e> {
    lv: bool,
    /// The loop variable is subtracted: the index goes down as the loop goes up.
    neg: bool,
    stride: Option<&'e Expr>,
    j: Option<BindId>,
    consts: Vec<(bool, &'e Expr)>,
}

/// Splits `e` (seen through the body's `let`s) into the loop variable `lv` times a fixed
/// stride, at most one inner range variable, and fixed terms. False if it has another form.
#[allow(clippy::too_many_arguments)]
fn aff<'e>(
    em: &Emitter,
    e: &'e Expr,
    lv: BindId,
    lets: &HashMap<BindId, &'e Expr>,
    ranges: &HashMap<BindId, (&'e Expr, &'e Expr)>,
    fixed: &dyn Fn(&Expr) -> bool,
    plus: bool,
    out: &mut Aff<'e>,
    depth: usize,
) -> bool {
    if depth > 12 {
        return false;
    }
    let builtin = |x: &Expr| matches!(em.ck.res.get(x.id as usize), Some(Res::Builtin(_)));
    let local = |x: &Expr| match (&x.kind, &em.ck.res[x.id as usize]) {
        (ExprKind::Var(_), Res::Local(b)) => Some(*b),
        _ => None,
    };
    if fixed(e) {
        out.consts.push((plus, e));
        return true;
    }
    if let Some(b) = local(e) {
        if b == lv {
            if out.lv {
                return false;
            }
            out.lv = true;
            out.neg = !plus;
            return true;
        }
        if ranges.contains_key(&b) {
            if out.j.is_some() || !plus {
                return false;
            }
            out.j = Some(b);
            return true;
        }
        return match lets.get(&b) {
            Some(v) => aff(em, v, lv, lets, ranges, fixed, plus, out, depth + 1),
            None => false,
        };
    }
    let ExprKind::Call(n, _, xs) = &e.kind else { return false };
    if !builtin(e) {
        return false;
    }
    match (n.as_str(), xs.len()) {
        ("+", _) => xs.iter().all(|x| aff(em, x.expr(), lv, lets, ranges, fixed, plus, out, depth + 1)),
        ("-", 2) => {
            aff(em, xs[0].expr(), lv, lets, ranges, fixed, plus, out, depth + 1)
                && aff(em, xs[1].expr(), lv, lets, ranges, fixed, !plus, out, depth + 1)
        }
        ("*", 2) => {
            let (x, y) = (xs[0].expr(), xs[1].expr());
            let (v, s) = if local(x) == Some(lv) && fixed(y) {
                (x, y)
            } else if local(y) == Some(lv) && fixed(x) {
                (y, x)
            } else {
                return false;
            };
            let _ = v;
            if out.lv {
                return false;
            }
            out.lv = true;
            out.neg = !plus;
            out.stride = Some(s);
            true
        }
        _ => false,
    }
}

/// i128 form of a fixed expression, for tests that cannot overflow themselves.
fn wide_of(em: &mut Emitter, e: &Expr) -> String {
    match &e.kind {
        ExprKind::Call(n, _, xs) if matches!(n.as_str(), "+" | "-" | "*") && !xs.is_empty() => {
            if n == "-" && xs.len() == 1 {
                return format!("(-{})", wide_of(em, xs[0].expr()));
            }
            let parts: Vec<String> = xs.iter().map(|x| wide_of(em, x.expr())).collect();
            format!("({})", parts.join(&format!(" {n} ")))
        }
        ExprKind::Call(n, _, xs) if n == "len" => format!("({}.len() as i128)", em.pv(xs[0].expr())),
        _ => format!("(({}) as i128)", em.val(e)),
    }
}

/// Names of the fns whose body has an expression `hit` matches, directly or through the fns
/// they call; `recursive` also counts fns that call themselves.
fn fns_reaching(prog: &Program, hit: impl Fn(&Expr) -> bool, recursive: bool) -> HashSet<String> {
    let mut calls: HashMap<String, HashSet<String>> = HashMap::new();
    let mut heavy: HashSet<String> = HashSet::new();
    for it in &prog.items {
        let Item::Fn(f) = it else { continue };
        let mut cs = HashSet::new();
        let mut loops = false;
        for x in &f.body {
            walk_ref(x, &mut |e| {
                if hit(e) {
                    loops = true;
                }
                if let ExprKind::Call(n, _, _) = &e.kind {
                    cs.insert(n.clone());
                }
            });
        }
        if loops || (recursive && cs.contains(&f.name)) {
            heavy.insert(f.name.clone());
        }
        calls.insert(f.name.clone(), cs);
    }
    loop {
        let before = heavy.len();
        for (f, cs) in &calls {
            if !heavy.contains(f) && cs.iter().any(|c| heavy.contains(c)) {
                heavy.insert(f.clone());
            }
        }
        if heavy.len() == before {
            return heavy;
        }
    }
}
