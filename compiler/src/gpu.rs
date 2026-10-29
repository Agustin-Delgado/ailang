//! Kernels: the body of a parallel loop written as a SPIR-V compute shader, with the same
//! meaning as the loop on the CPU. Every operation that traps on the CPU (overflow, an index out
//! of range, division by zero, a value that does not fit) marks the invocation as trapped
//! instead; the runtime then throws the whole launch away and runs the loop on the CPU, which
//! traps where the program would have. Loops inside the body stop as soon as an invocation has
//! trapped or has run too many iterations (the GPU must not hang on garbage values), and that too
//! sends the loop to the CPU.

use std::collections::{HashMap, HashSet};

use crate::ast::{Arg, Expr, ExprKind, FnDef, ForIter};
use crate::check::{BindId, Checked, Res};
use crate::spirv::{op, FnBuilder, Id, Module, Scalar};
use crate::types::{IntK, Ty};

/// One vector a kernel reads or writes: binding `index + 1`.
pub struct KBuf {
    pub bind: BindId,
    /// A copy of the vector taken before the loop, for reads of elements that only later
    /// iterations write (they must see the old values).
    pub snap: bool,
    pub elem: Scalar,
    pub write: bool,
    pub read: bool,
    /// Stored narrow: the device copy holds each element in 1, 2, 4 or 8 bytes (whatever the
    /// values needed when it was uploaded), packed in 32-bit words; the width comes in the
    /// argument buffer. A store that does not fit marks the launch as needing a wider copy.
    pub narrow: bool,
}

/// Buffers that can be narrow: the first 16 (their widths share one 64-bit word).
pub const MAX_NARROW: usize = 16;

/// Slot of the argument buffer holding the narrow buffers' widths (log2 of the bytes per
/// element, 4 bits per buffer); the kernel's own arguments follow it.
pub const WIDTH_SLOT: u32 = 4;
pub const ARGS_SLOT: u32 = 5;

/// A value the kernel gets from the program when it starts (slot `4 + index` of the argument
/// buffer, as 64 bits).
#[derive(Clone, PartialEq)]
pub enum KArg {
    Local(BindId),
    Const(String),
    Len(usize),
    /// A value the generated Rust already holds (the loop's first index).
    Raw(String),
}

/// A loop whose body is `lets...`, an inner `(for j (range a b) ...)` and nothing else but
/// folding the inner loop's sums into the outer ones: one work item per (i, j), so the GPU gets
/// enough of them (and neighbouring items touch neighbouring elements).
pub struct Collapse<'a> {
    pub j: BindId,
    pub jlo: &'a Expr,
    pub jhi: &'a Expr,
    pub prefix: Vec<&'a Expr>,
    /// `(if c (for j ...))`: the condition, checked by every item of row i.
    pub guard: Option<&'a Expr>,
    pub inner: &'a [Expr],
    /// Inner sums: `(var r 0)` before the inner loop, `(set r (+ r e))` inside it, and
    /// `(set S (+ S r))` after it: r -> the reduction index of S.
    pub inner_sums: HashMap<BindId, usize>,
    /// A sliding window carried across the inner loop (see `Rot`).
    pub rot: Option<Rot<'a>>,
}

/// `(var x1 a1) ... (var xk ak)` before the inner loop, whose body starts with lets `lets` that
/// end in `(let e ...)`, depend only on j and on values fixed for the row, and ends with
/// `(set x1 x2) ... (set xk e)`. At item j (s = j - jlo), x_m holds e as computed at j - (k-m+1)
/// when s reaches that far, and a_{m+s} before: each item computes it again from the lets, so the
/// items of a row do not depend on each other.
pub struct Rot<'a> {
    pub vars: Vec<BindId>,
    pub lets: &'a [Expr],
    pub e: BindId,
}

pub struct Kernel {
    pub words: Vec<u32>,
    pub bufs: Vec<KBuf>,
    pub args: Vec<(KArg, Scalar)>,
    pub nred: usize,
    pub f64: bool,
    /// Rough operations per iteration, for deciding whether the GPU is worth it.
    pub cost: u64,
    /// No loops inside (in helpers neither) and no sums: traps exactly when the CPU loop would.
    pub exact: bool,
}

/// What the parallel-loop analysis found, handed to the kernel writer.
pub struct Input<'a> {
    pub ck: &'a Checked,
    pub fns: &'a HashMap<String, FnDef>,
    pub lv: BindId,
    pub body: &'a [Expr],
    /// `(set s (+ s e))` statements: id -> reduction index.
    pub sums: HashMap<u32, usize>,
    pub sum_vars: Vec<BindId>,
    /// `(at v k)` ids that are targets of the loop's disjoint writes.
    pub writes: HashSet<u32>,
    /// Vectors whose reads fall in the iteration's own write window (already tested).
    pub own_reads: HashSet<u32>,
    /// Reads of a written vector that see its contents from before the loop (already tested
    /// to hit only elements no earlier iteration writes): they read the `snap` buffer.
    pub stale: HashSet<u32>,
    pub collapse: Option<Collapse<'a>>,
}

/// Iterations of the loops inside one invocation before it gives up (and the loop goes back to
/// the CPU): keeps every launch far from the driver's watchdog.
const BUDGET: i64 = 1 << 22;

/// Lanes per workgroup (matches the runtime's `WG`).
const WG: u32 = 256;

fn short(e: &Expr) -> String {
    let s = format!("{:?}", e.kind);
    s.chars().take(160).collect()
}

pub fn scalar_of(t: &Ty) -> Option<Scalar> {
    match t {
        Ty::Int(k) => Some(match k {
            IntK::I64 => Scalar::Int(64, true),
            IntK::U64 => Scalar::Int(64, false),
            IntK::I32 => Scalar::Int(32, true),
            IntK::U32 => Scalar::Int(32, false),
            _ => return None,
        }),
        Ty::F64 => Some(Scalar::Float(64)),
        Ty::Bool => Some(Scalar::Bool),
        _ => None,
    }
}

/// Element types a kernel can read and write in place.
pub fn elem_ok(t: &Ty) -> Option<Scalar> {
    match scalar_of(t)? {
        Scalar::Bool => None,
        s => Some(s),
    }
}

struct Loop {
    merge: Id,
    cont: Id,
}

struct HelperFn {
    id: Id,
    ty: Id,
    ret: Option<Scalar>,
    params: Vec<Scalar>,
}

struct T<'a> {
    inp: &'a Input<'a>,
    m: Module,
    f: FnBuilder,
    /// Code that runs first in the kernel (loading arguments): goes before `f.code`.
    prologue: Vec<u32>,
    locals: HashMap<BindId, (Id, Scalar)>,
    bufs: Vec<KBuf>,
    buf_vars: Vec<Id>,
    args: Vec<(KArg, Scalar)>,
    arg_vars: Vec<Id>,
    io_var: Id,
    loops: Vec<Loop>,
    trapped: Id,
    /// Set by a store that did not fit its narrow buffer (with `trapped`, to stop the lane).
    widen: Id,
    widths: HashMap<usize, Id>,
    budget: Id,
    accs: Vec<(Id, Id)>,
    helpers: HashMap<String, HelperFn>,
    helper_stack: Vec<String>,
    in_helper: bool,
    f64: bool,
    cost: u64,
    weight: u64,
    has_loops: bool,
    decorated: HashSet<Id>,
}

type V = (Id, Scalar);

pub fn build(inp: &Input) -> Option<Kernel> {
    let mut m = Module::new();
    m.capability(1); // Shader
    m.capability(11); // Int64
    let u64t = m.t_int(64, false);
    let io_var = buffer_var(&mut m, u64t, 8, 0, &mut HashSet::new());
    let mut t = T {
        inp,
        m,
        f: FnBuilder::new(),
        prologue: Vec::new(),
        locals: HashMap::new(),
        bufs: Vec::new(),
        buf_vars: Vec::new(),
        args: Vec::new(),
        arg_vars: Vec::new(),
        io_var,
        loops: Vec::new(),
        trapped: 0,
        widen: 0,
        widths: HashMap::new(),
        budget: 0,
        accs: Vec::new(),
        helpers: HashMap::new(),
        helper_stack: Vec::new(),
        in_helper: false,
        f64: false,
        cost: 0,
        weight: 1,
        has_loops: false,
        decorated: HashSet::new(),
    };
    t.kernel()?;
    let nred = inp.sum_vars.len();
    let exact = !t.has_loops && nred == 0;
    Some(Kernel { words: t.m.finish(), bufs: t.bufs, args: t.args, nred, f64: t.f64, cost: t.cost.max(1), exact })
}

/// A storage buffer `{ T data[]; }` at set 0, binding `binding`.
fn buffer_var(m: &mut Module, elem: Id, stride: u32, binding: u32, decorated: &mut HashSet<Id>) -> Id {
    let rt = m.unique_ty(op::TYPE_RUNTIME_ARRAY, &[elem]);
    m.decorate(rt, 6, &[stride]); // ArrayStride
    let st = m.unique_ty(op::TYPE_STRUCT, &[rt]);
    m.decorate(st, 2, &[]); // Block
    m.member_decorate(st, 0, 35, &[0]); // Offset 0
    decorated.insert(st);
    let pt = m.t_ptr(12, st);
    let v = m.global_var(pt, 12);
    m.decorate(v, 34, &[0]); // DescriptorSet
    m.decorate(v, 33, &[binding]); // Binding
    v
}

impl<'a> T<'a> {
    fn ty(&mut self, s: Scalar) -> Id {
        self.m.t_scalar(s)
    }
    fn bool_t(&mut self) -> Id {
        self.m.t_bool()
    }
    fn op(&mut self, opcode: u16, s: Scalar, operands: &[u32]) -> Id {
        let t = self.ty(s);
        self.cost += self.weight;
        self.f.op(&mut self.m, opcode, t, operands)
    }
    fn bop(&mut self, opcode: u16, operands: &[u32]) -> Id {
        let t = self.bool_t();
        self.cost += self.weight;
        self.f.op(&mut self.m, opcode, t, operands)
    }
    fn fop(&mut self, opcode: u16, s: Scalar, operands: &[u32]) -> Id {
        let r = self.op(opcode, s, operands);
        self.m.decorate(r, 42, &[]); // NoContraction: no fused multiply-add
        r
    }
    fn var(&mut self, s: Scalar) -> Id {
        let t = self.ty(s);
        let p = self.m.t_ptr(7, t);
        self.f.var(&mut self.m, p)
    }
    fn load(&mut self, ptr: Id, s: Scalar) -> Id {
        let t = self.ty(s);
        self.f.load(&mut self.m, t, ptr)
    }
    fn label(&mut self) -> Id {
        self.m.id()
    }
    /// Sets the window variables of `r` for item j (see `Rot`); `jv` holds j.
    fn rotate(&mut self, r: &Rot, jv: Id, j: Id, jl: Id) -> Option<()> {
        let s64 = Scalar::Int(64, true);
        let k = r.vars.len();
        let mut inits = Vec::new();
        for b in &r.vars {
            let (p, sc) = *self.locals.get(b)?;
            inits.push((self.load(p, sc), sc));
        }
        let s = self.op(op::I_SUB, s64, &[j, jl]);
        // t[d]: e at j - d, for the items that have it.
        let mut t = vec![0; k + 1];
        for d in 1..=k {
            let sc = inits[k - d].1;
            let tv = self.var(sc);
            let dc = self.m.c_i64(d as i64);
            let has = self.bop(op::S_GREATER_THAN_EQUAL, &[s, dc]);
            let (yes, done) = (self.label(), self.label());
            self.f.emit(op::SELECTION_MERGE, &[done, 0]);
            self.f.branch_cond(has, yes, done);
            self.f.label(yes);
            let jd = self.op(op::I_SUB, s64, &[j, dc]);
            self.f.store(jv, jd);
            for st in r.lets {
                self.stmt(st)?;
            }
            let (ep, es) = *self.locals.get(&r.e)?;
            if es != sc {
                return None;
            }
            let ev = self.load(ep, es);
            self.f.store(tv, ev);
            self.f.branch(done);
            self.f.label(done);
            t[d] = self.load(tv, sc);
        }
        self.f.store(jv, j);
        for m in 0..k {
            let d = k - m;
            let (_, sc) = inits[m];
            let mut v = t[d];
            for tt in (0..d).rev() {
                let tc = self.m.c_i64(tt as i64);
                let at = self.bop(op::I_EQUAL, &[s, tc]);
                v = self.op(op::SELECT, sc, &[at, inits[m + tt].0, v]);
            }
            let (p, _) = *self.locals.get(&r.vars[m])?;
            self.f.store(p, v);
        }
        Some(())
    }

    /// Marks the invocation trapped where `cond` holds.
    fn trap_if(&mut self, cond: Id) {
        let b = Scalar::Bool;
        let cur = self.load(self.trapped, b);
        let nv = self.bop(op::LOGICAL_OR, &[cur, cond]);
        self.f.store(self.trapped, nv);
    }
    fn ty_of(&self, e: &Expr) -> &Ty {
        &self.inp.ck.ty[e.id as usize]
    }
    fn scalar(&self, e: &Expr) -> Option<Scalar> {
        scalar_of(self.ty_of(e))
    }
    fn res(&self, e: &Expr) -> Option<&Res> {
        self.inp.ck.res.get(e.id as usize)
    }

    // ---- the kernel's entry point ----

    fn kernel(&mut self) -> Option<()> {
        let void = self.m.t_void();
        let fnty = self.m.t_fn(void, &[]);
        let main = self.m.id();
        let entry = self.m.id();
        let u32t = self.m.t_int(32, false);
        let v3 = self.m.t_vector(u32t, 3);
        let pin = self.m.t_ptr(1, v3);
        let wg_id = self.m.global_var(pin, 1);
        self.m.decorate(wg_id, 11, &[26]); // WorkgroupId
        let nwg = self.m.global_var(pin, 1);
        self.m.decorate(nwg, 11, &[24]); // NumWorkgroups
        let lid = self.m.global_var(pin, 1);
        self.m.decorate(lid, 11, &[27]); // LocalInvocationId
        // Per-invocation globals, so helper functions share them.
        let bt = self.m.t_bool();
        let pb = self.m.t_ptr(6, bt);
        self.trapped = self.m.global_var(pb, 6);
        self.widen = self.m.global_var(pb, 6);
        let i64t0 = self.m.t_int(64, true);
        let pi = self.m.t_ptr(6, i64t0);
        self.budget = self.m.global_var(pi, 6);
        let fls = self.m.c_bool(false);
        let zero = self.m.c_i64(0);
        crate::spirv::inst(&mut self.prologue, op::STORE, &[self.trapped, fls]);
        crate::spirv::inst(&mut self.prologue, op::STORE, &[self.widen, fls]);
        crate::spirv::inst(&mut self.prologue, op::STORE, &[self.budget, zero]);
        for _ in &self.inp.sum_vars {
            let a = self.var(Scalar::Int(64, true));
            let b = self.var(Scalar::Int(64, false));
            let z64 = self.m.c_u64(0);
            crate::spirv::inst(&mut self.prologue, op::STORE, &[a, zero]);
            crate::spirv::inst(&mut self.prologue, op::STORE, &[b, z64]);
            self.accs.push((a, b));
        }
        // idx = (wg.y * nwg.x + wg.x) * WG + lid.x
        let w = self.f.load(&mut self.m, v3, wg_id);
        let nw = self.f.load(&mut self.m, v3, nwg);
        let l = self.f.load(&mut self.m, v3, lid);
        let wx = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, u32t, &[w, 0]);
        let wy = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, u32t, &[w, 1]);
        let nx = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, u32t, &[nw, 0]);
        let lx = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, u32t, &[l, 0]);
        let row = self.f.op(&mut self.m, op::I_MUL, u32t, &[wy, nx]);
        let lin = self.f.op(&mut self.m, op::I_ADD, u32t, &[row, wx]);
        let wgc = self.m.c_u32(WG);
        let base = self.f.op(&mut self.m, op::I_MUL, u32t, &[lin, wgc]);
        let idx32 = self.f.op(&mut self.m, op::I_ADD, u32t, &[base, lx]);
        let u64t = self.m.t_int(64, false);
        let i64t = self.m.t_int(64, true);
        let idxu = self.f.op(&mut self.m, op::U_CONVERT, u64t, &[idx32]);
        let idx = self.f.op(&mut self.m, op::BITCAST, i64t, &[idxu]);
        let lo = self.io_load(2, Scalar::Int(64, true));
        let n = self.io_load(3, Scalar::Int(64, true));
        let inr = self.bop(op::S_LESS_THAN, &[idx, n]);
        let body_l = self.label();
        let after = self.label();
        self.f.emit(op::SELECTION_MERGE, &[after, 0]);
        self.f.branch_cond(inr, body_l, after);
        self.f.label(body_l);
        let s64 = Scalar::Int(64, true);
        let lvv = self.var(s64);
        self.locals.insert(self.inp.lv, (lvv, s64));
        if let Some(c) = &self.inp.collapse {
            // Item idx is (i, j) = (ilo + idx / nj, jlo + idx % nj); lo is 0 here and the
            // outer loop's first index comes as an argument.
            let (ilo, _) = self.arg(KArg::Raw("__plo".into()), s64);
            let (jl, jls) = self.expr(c.jlo)?;
            let (jh, jhs) = self.expr(c.jhi)?;
            if jls != s64 || jhs != s64 {
                return None;
            }
            let nj = self.f.op(&mut self.m, op::I_SUB, i64t, &[jh, jl]);
            // In 32 bits (64-bit division is long on a GPU): idx fits in 32 bits, and when nj
            // does not, idx < nj.
            let bt = self.m.t_bool();
            let lim = self.m.c_i64(u32::MAX as i64);
            let small = self.f.op(&mut self.m, op::S_LESS_THAN_EQUAL, bt, &[nj, lim]);
            let one64 = self.m.c_i64(1);
            let njs = self.f.op(&mut self.m, op::SELECT, i64t, &[small, nj, one64]);
            let nj32 = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[njs]);
            let q32 = self.f.op(&mut self.m, op::U_DIV, u32t, &[idx32, nj32]);
            let r32 = self.f.op(&mut self.m, op::U_MOD, u32t, &[idx32, nj32]);
            let q64 = self.f.op(&mut self.m, op::U_CONVERT, u64t, &[q32]);
            let r64 = self.f.op(&mut self.m, op::U_CONVERT, u64t, &[r32]);
            let qs = self.f.op(&mut self.m, op::BITCAST, i64t, &[q64]);
            let rs = self.f.op(&mut self.m, op::BITCAST, i64t, &[r64]);
            let zero64 = self.m.c_i64(0);
            let q = self.f.op(&mut self.m, op::SELECT, i64t, &[small, qs, zero64]);
            let r = self.f.op(&mut self.m, op::SELECT, i64t, &[small, rs, idx]);
            let i = self.f.op(&mut self.m, op::I_ADD, i64t, &[ilo, q]);
            self.f.store(lvv, i);
            let j = self.f.op(&mut self.m, op::I_ADD, i64t, &[jl, r]);
            let jv = self.var(s64);
            self.f.store(jv, j);
            self.locals.insert(c.j, (jv, s64));
            let _ = lo;
            for s in c.prefix.clone() {
                self.stmt(s)?;
            }
            let guard = match c.guard {
                Some(g) => {
                    let (gv, _) = self.cond(g)?;
                    let (yes, done) = (self.label(), self.label());
                    self.f.emit(op::SELECTION_MERGE, &[done, 0]);
                    self.f.branch_cond(gv, yes, done);
                    self.f.label(yes);
                    Some(done)
                }
                None => None,
            };
            if let Some(r) = &c.rot {
                self.rotate(r, jv, j, jl)?;
            }
            for s in c.inner {
                self.stmt(s)?;
            }
            if let Some(done) = guard {
                self.f.branch(done);
                self.f.label(done);
            }
        } else {
            let i = self.f.op(&mut self.m, op::I_ADD, i64t, &[lo, idx]);
            self.f.store(lvv, i);
            for s in self.inp.body {
                self.stmt(s)?;
            }
        }
        self.f.branch(after);
        self.f.label(after);
        if !self.accs.is_empty() {
            self.reduce(lin, lx)?;
        }
        // The trap flag: any trapped lane writes 1, or 2 if it stopped at a store too wide for
        // its narrow buffer (then the runtime widens the copy and runs the launch again).
        let tv = self.load(self.trapped, Scalar::Bool);
        let st = self.label();
        let end = self.label();
        self.f.emit(op::SELECTION_MERGE, &[end, 0]);
        self.f.branch_cond(tv, st, end);
        self.f.label(st);
        let wv = self.load(self.widen, Scalar::Bool);
        let one = self.m.c_u64(1);
        let two = self.m.c_u64(2);
        let u64t_f = self.m.t_int(64, false);
        let fv = self.f.op(&mut self.m, op::SELECT, u64t_f, &[wv, two, one]);
        let p0 = self.io_ptr_const(0);
        self.f.store(p0, fv);
        self.f.branch(end);
        self.f.label(end);
        self.f.ret();
        // Assemble: arguments are loaded at the top of the entry block.
        let mut f = std::mem::replace(&mut self.f, FnBuilder::new());
        let mut code = std::mem::take(&mut self.prologue);
        code.extend(f.code.drain(..));
        f.code = code;
        f.finish(&mut self.m, main, void, fnty, &[], entry);
        self.m.entry_point(main, "main", &[wg_id, nwg, lid]);
        self.m.execution_mode(main, 17, &[WG, 1, 1]); // LocalSize
        if self.f64 {
            self.m.capability(10); // Float64
            self.m.extension("SPV_KHR_float_controls");
            self.m.capability(4464); // DenormPreserve
            self.m.capability(4466); // SignedZeroInfNanPreserve
            self.m.capability(4467); // RoundingModeRTE
            self.m.execution_mode(main, 4459, &[64]);
            self.m.execution_mode(main, 4461, &[64]);
            self.m.execution_mode(main, 4462, &[64]);
        }
        Some(())
    }

    fn io_ptr(&mut self, index: Id) -> Id {
        let u64t = self.m.t_int(64, false);
        let p = self.m.t_ptr(12, u64t);
        let z = self.m.c_u32(0);
        let io = self.io_var;
        self.f.op(&mut self.m, op::ACCESS_CHAIN, p, &[io, z, index])
    }
    fn io_ptr_const(&mut self, slot: u32) -> Id {
        let c = self.m.c_u32(slot);
        self.io_ptr(c)
    }
    fn io_load(&mut self, slot: u32, s: Scalar) -> Id {
        let p = self.io_ptr_const(slot);
        let u64t = self.m.t_int(64, false);
        let raw = self.f.load(&mut self.m, u64t, p);
        let t = self.ty(s);
        self.f.op(&mut self.m, op::BITCAST, t, &[raw])
    }

    /// Each lane hands its sums to lane 0 of its workgroup through shared memory; lane 0 adds
    /// them (checked) and writes the workgroup's partial sums.
    fn reduce(&mut self, lin: Id, lx: Id) -> Option<()> {
        let nred = self.accs.len() as u32;
        let i64t = self.m.t_int(64, true);
        let u64t = self.m.t_int(64, false);
        let u32t = self.m.t_int(32, false);
        let len = self.m.c_u32(WG * nred);
        let arr_s = self.m.t_array(i64t, len);
        let arr_a = self.m.t_array(u64t, len);
        let ps = self.m.t_ptr(4, arr_s);
        let pa = self.m.t_ptr(4, arr_a);
        let sh_s = self.m.global_var(ps, 4);
        let sh_a = self.m.global_var(pa, 4);
        let pse = self.m.t_ptr(4, i64t);
        let pae = self.m.t_ptr(4, u64t);
        let wgc = self.m.c_u32(WG);
        for r in 0..nred {
            let (acc, abs) = self.accs[r as usize];
            let s = self.load(acc, Scalar::Int(64, true));
            let a = self.load(abs, Scalar::Int(64, false));
            let off = self.m.c_u32(r * WG);
            let k = self.f.op(&mut self.m, op::I_ADD, u32t, &[off, lx]);
            let p1 = self.f.op(&mut self.m, op::ACCESS_CHAIN, pse, &[sh_s, k]);
            self.f.store(p1, s);
            let p2 = self.f.op(&mut self.m, op::ACCESS_CHAIN, pae, &[sh_a, k]);
            self.f.store(p2, a);
        }
        let wscope = self.m.c_u32(2);
        let sem = self.m.c_u32(0x8 | 0x100);
        self.f.emit(op::CONTROL_BARRIER, &[wscope, wscope, sem]);
        let zero32 = self.m.c_u32(0);
        let is0 = self.bop(op::I_EQUAL, &[lx, zero32]);
        let do_l = self.label();
        let end = self.label();
        self.f.emit(op::SELECTION_MERGE, &[end, 0]);
        self.f.branch_cond(is0, do_l, end);
        self.f.label(do_l);
        for r in 0..nred {
            let ts = self.var(Scalar::Int(64, true));
            let ta = self.var(Scalar::Int(64, false));
            let z = self.m.c_i64(0);
            let zu = self.m.c_u64(0);
            self.f.store(ts, z);
            self.f.store(ta, zu);
            let k = self.var(Scalar::Int(32, false));
            self.f.store(k, zero32);
            let (head, bodyl, cont, merge, cond_l) = (self.label(), self.label(), self.label(), self.label(), self.label());
            self.f.branch(head);
            self.f.label(head);
            self.f.emit(op::LOOP_MERGE, &[merge, cont, 0]);
            self.f.branch(cond_l);
            self.f.label(cond_l);
            let kv = self.load(k, Scalar::Int(32, false));
            let c = self.bop(op::U_LESS_THAN, &[kv, wgc]);
            self.f.branch_cond(c, bodyl, merge);
            self.f.label(bodyl);
            let off = self.m.c_u32(r * WG);
            let kk = self.f.op(&mut self.m, op::I_ADD, u32t, &[off, kv]);
            let p1 = self.f.op(&mut self.m, op::ACCESS_CHAIN, pse, &[sh_s, kk]);
            let s = self.f.load(&mut self.m, i64t, p1);
            let p2 = self.f.op(&mut self.m, op::ACCESS_CHAIN, pae, &[sh_a, kk]);
            let a = self.f.load(&mut self.m, u64t, p2);
            let cs = self.load(ts, Scalar::Int(64, true));
            let ns = self.add(cs, s, Scalar::Int(64, true));
            self.f.store(ts, ns);
            let ca = self.load(ta, Scalar::Int(64, false));
            let na = self.add(ca, a, Scalar::Int(64, false));
            self.f.store(ta, na);
            self.f.branch(cont);
            self.f.label(cont);
            let one = self.m.c_u32(1);
            let k2 = self.f.op(&mut self.m, op::I_ADD, u32t, &[kv, one]);
            self.f.store(k, k2);
            self.f.branch(head);
            self.f.label(merge);
            // io[pbase + (lin * nred + r) * 2 + {0, 1}], pbase = 4 + nargs (patched below).
            let nr = self.m.c_u32(nred);
            let g = self.f.op(&mut self.m, op::I_MUL, u32t, &[lin, nr]);
            let rc = self.m.c_u32(r);
            let gr = self.f.op(&mut self.m, op::I_ADD, u32t, &[g, rc]);
            let two = self.m.c_u32(2);
            let slot = self.f.op(&mut self.m, op::I_MUL, u32t, &[gr, two]);
            let pb = self.pbase_marker();
            let s0 = self.f.op(&mut self.m, op::I_ADD, u32t, &[slot, pb]);
            let one = self.m.c_u32(1);
            let s1 = self.f.op(&mut self.m, op::I_ADD, u32t, &[s0, one]);
            let fs = self.load(ts, Scalar::Int(64, true));
            let fsu = self.f.op(&mut self.m, op::BITCAST, u64t, &[fs]);
            let p = self.io_ptr(s0);
            self.f.store(p, fsu);
            let fa = self.load(ta, Scalar::Int(64, false));
            let p = self.io_ptr(s1);
            self.f.store(p, fa);
        }
        self.f.branch(end);
        self.f.label(end);
        Some(())
    }
    /// The first slot of the partial sums depends on how many arguments the kernel ends up
    /// with, known only at the end: a 32-bit argument holds it (the runtime writes pbase).
    fn pbase_marker(&mut self) -> Id {
        // Slot 1 of the argument buffer holds pbase (as u64); load its low 32 bits.
        let raw = self.io_load(1, Scalar::Int(64, false));
        let u32t = self.m.t_int(32, false);
        self.f.op(&mut self.m, op::U_CONVERT, u32t, &[raw])
    }

    // ---- arguments and buffers ----

    fn arg(&mut self, a: KArg, s: Scalar) -> V {
        let i = match self.args.iter().position(|x| x.0 == a) {
            Some(i) => i,
            None => {
                let v = self.var(s);
                // Loaded at the start of the kernel.
                let u64t = self.m.t_int(64, false);
                let pt = self.m.t_ptr(12, u64t);
                let z = self.m.c_u32(0);
                let slot = self.m.c_u32(ARGS_SLOT + self.args.len() as u32);
                let p = self.m.id();
                crate::spirv::inst(&mut self.prologue, op::ACCESS_CHAIN, &[pt, p, self.io_var, z, slot]);
                let raw = self.m.id();
                crate::spirv::inst(&mut self.prologue, op::LOAD, &[u64t, raw, p]);
                let val = match s {
                    Scalar::Bool => {
                        let bt = self.m.t_bool();
                        let zero = self.m.c_u64(0);
                        let b = self.m.id();
                        crate::spirv::inst(&mut self.prologue, op::I_NOT_EQUAL, &[bt, b, raw, zero]);
                        b
                    }
                    Scalar::Int(64, false) => raw,
                    Scalar::Int(32, sg) => {
                        let t = self.m.t_int(32, sg);
                        let u32t = self.m.t_int(32, false);
                        let lo = self.m.id();
                        crate::spirv::inst(&mut self.prologue, op::U_CONVERT, &[u32t, lo, raw]);
                        if sg {
                            let b = self.m.id();
                            crate::spirv::inst(&mut self.prologue, op::BITCAST, &[t, b, lo]);
                            b
                        } else {
                            lo
                        }
                    }
                    _ => {
                        let t = self.ty(s);
                        let b = self.m.id();
                        crate::spirv::inst(&mut self.prologue, op::BITCAST, &[t, b, raw]);
                        b
                    }
                };
                crate::spirv::inst(&mut self.prologue, op::STORE, &[v, val]);
                self.args.push((a, s));
                self.arg_vars.push(v);
                self.args.len() - 1
            }
        };
        let v = self.arg_vars[i];
        (self.load(v, s), s)
    }

    /// The buffer of vector `b`, declared on first use; with its length argument.
    fn buffer(&mut self, b: BindId, write: bool) -> Option<usize> {
        self.buffer_of(b, write, false)
    }

    fn buffer_of(&mut self, b: BindId, write: bool, snap: bool) -> Option<usize> {
        if let Some(i) = self.bufs.iter().position(|x| x.bind == b && x.snap == snap) {
            if write {
                self.bufs[i].write = true;
            } else {
                self.bufs[i].read = true;
            }
            return Some(i);
        }
        let Ty::Vec(el) = &self.inp.ck.bindings[b].ty else { return None };
        let s = elem_ok(el)?;
        if s == Scalar::Float(64) {
            self.f64 = true;
        }
        let narrow = matches!(s, Scalar::Int(64, true) | Scalar::Int(32, true)) && self.bufs.len() < MAX_NARROW;
        let (t, stride) = if narrow {
            (self.m.t_int(32, false), 4)
        } else {
            let stride = match s {
                Scalar::Int(w, _) | Scalar::Float(w) => w / 8,
                Scalar::Bool => return None,
            };
            (self.ty(s), stride)
        };
        let binding = self.bufs.len() as u32 + 1;
        let v = buffer_var(&mut self.m, t, stride, binding, &mut self.decorated);
        self.bufs.push(KBuf { bind: b, snap, elem: s, write, read: !write, narrow });
        self.buf_vars.push(v);
        Some(self.bufs.len() - 1)
    }

    /// `idx` (i64) checked against the length of buffer `bi`: trapping (and giving element 0)
    /// when out of range.
    fn checked_index(&mut self, bi: usize, idx: Id) -> Id {
        let (len, _) = self.arg(KArg::Len(bi), Scalar::Int(64, true));
        let zero = self.m.c_i64(0);
        let neg = self.bop(op::S_LESS_THAN, &[idx, zero]);
        let big = self.bop(op::S_GREATER_THAN_EQUAL, &[idx, len]);
        let bad = self.bop(op::LOGICAL_OR, &[neg, big]);
        self.trap_if(bad);
        let i64t = self.m.t_int(64, true);
        self.f.op(&mut self.m, op::SELECT, i64t, &[bad, zero, idx])
    }

    /// log2 of the bytes per element of narrow buffer `bi` in this launch (u32).
    fn width_log(&mut self, bi: usize) -> Id {
        if let Some(v) = self.widths.get(&bi) {
            let v = *v;
            let u32t = self.m.t_int(32, false);
            return self.f.load(&mut self.m, u32t, v);
        }
        let u32t = self.m.t_int(32, false);
        let u64t = self.m.t_int(64, false);
        let pv = self.m.t_ptr(7, u32t);
        let var = self.f.var(&mut self.m, pv);
        let pt = self.m.t_ptr(12, u64t);
        let z = self.m.c_u32(0);
        let slot = self.m.c_u32(WIDTH_SLOT);
        let p = self.m.id();
        crate::spirv::inst(&mut self.prologue, op::ACCESS_CHAIN, &[pt, p, self.io_var, z, slot]);
        let raw = self.m.id();
        crate::spirv::inst(&mut self.prologue, op::LOAD, &[u64t, raw, p]);
        let sh = self.m.c_u64(4 * bi as u64);
        let shifted = self.m.id();
        crate::spirv::inst(&mut self.prologue, op::SHIFT_RIGHT_LOGICAL, &[u64t, shifted, raw, sh]);
        let fifteen = self.m.c_u64(15);
        let masked = self.m.id();
        crate::spirv::inst(&mut self.prologue, op::BITWISE_AND, &[u64t, masked, shifted, fifteen]);
        let w32 = self.m.id();
        crate::spirv::inst(&mut self.prologue, op::U_CONVERT, &[u32t, w32, masked]);
        crate::spirv::inst(&mut self.prologue, op::STORE, &[var, w32]);
        self.widths.insert(bi, var);
        self.f.load(&mut self.m, u32t, var)
    }

    /// Where element `safe` of narrow buffer `bi` lives: (word index, bit offset in it, width
    /// log2), all u32.
    fn narrow_at(&mut self, bi: usize, safe: Id) -> (Id, Id, Id) {
        let l = self.width_log(bi);
        let i64t = self.m.t_int(64, true);
        let u32t = self.m.t_int(32, false);
        let boff = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, i64t, &[safe, l]);
        let two = self.m.c_u32(2);
        let wi64 = self.f.op(&mut self.m, op::SHIFT_RIGHT_LOGICAL, i64t, &[boff, two]);
        let wi = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[wi64]);
        let b32 = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[boff]);
        let three = self.m.c_u32(3);
        let low = self.f.op(&mut self.m, op::BITWISE_AND, u32t, &[b32, three]);
        let sh = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u32t, &[low, three]);
        self.cost += 4 * self.weight;
        (wi, sh, l)
    }

    fn word_ptr(&mut self, bi: usize, wi: Id) -> Id {
        let u32t = self.m.t_int(32, false);
        let pt = self.m.t_ptr(12, u32t);
        let z = self.m.c_u32(0);
        let bv = self.buf_vars[bi];
        self.f.op(&mut self.m, op::ACCESS_CHAIN, pt, &[bv, z, wi])
    }

    /// Bits per element (8, 16 or 32; 32 also for 64-bit elements), u32.
    fn field_bits(&mut self, l: Id) -> Id {
        let u32t = self.m.t_int(32, false);
        let eight = self.m.c_u32(8);
        let bits = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u32t, &[eight, l]);
        let c32 = self.m.c_u32(32);
        let bt = self.m.t_bool();
        let big = self.f.op(&mut self.m, op::U_GREATER_THAN, bt, &[bits, c32]);
        self.f.op(&mut self.m, op::SELECT, u32t, &[big, c32, bits])
    }

    /// Element `idx` of buffer `bi`, as its element type.
    fn load_elem(&mut self, bi: usize, idx: Id) -> Id {
        let s = self.bufs[bi].elem;
        if !self.bufs[bi].narrow {
            let p = self.elem_ptr(bi, idx);
            return self.load(p, s);
        }
        let safe = self.checked_index(bi, idx);
        let (wi, sh, l) = self.narrow_at(bi, safe);
        let u32t = self.m.t_int(32, false);
        let i32t = self.m.t_int(32, true);
        let bt = self.m.t_bool();
        let p0 = self.word_ptr(bi, wi);
        let w0 = self.f.load(&mut self.m, u32t, p0);
        let bits = self.field_bits(l);
        let w0s = self.f.op(&mut self.m, op::BITCAST, i32t, &[w0]);
        let nx = self.f.op(&mut self.m, op::BITFIELD_S_EXTRACT, i32t, &[w0s, sh, bits]);
        if s == Scalar::Int(32, true) {
            return nx;
        }
        // 64-bit elements: the next word too (clamped to the buffer, for narrow copies).
        let i64t = self.m.t_int(64, true);
        let u64t = self.m.t_int(64, false);
        let nw = self.f.op(&mut self.m, op::ARRAY_LENGTH, u32t, &[self.buf_vars[bi], 0]);
        let one = self.m.c_u32(1);
        let wi1 = self.f.op(&mut self.m, op::I_ADD, u32t, &[wi, one]);
        let inr = self.f.op(&mut self.m, op::U_LESS_THAN, bt, &[wi1, nw]);
        let wj = self.f.op(&mut self.m, op::SELECT, u32t, &[inr, wi1, wi]);
        let p1 = self.word_ptr(bi, wj);
        let w1 = self.f.load(&mut self.m, u32t, p1);
        let lo = self.f.op(&mut self.m, op::U_CONVERT, u64t, &[w0]);
        let hi = self.f.op(&mut self.m, op::U_CONVERT, u64t, &[w1]);
        let c32 = self.m.c_u32(32);
        let hs = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u64t, &[hi, c32]);
        let wide_u = self.f.op(&mut self.m, op::BITWISE_OR, u64t, &[hs, lo]);
        let wide = self.f.op(&mut self.m, op::BITCAST, i64t, &[wide_u]);
        let narrow = self.f.op(&mut self.m, op::S_CONVERT, i64t, &[nx]);
        let c3 = self.m.c_u32(3);
        let is8 = self.f.op(&mut self.m, op::I_EQUAL, bt, &[l, c3]);
        self.cost += 6 * self.weight;
        self.f.op(&mut self.m, op::SELECT, i64t, &[is8, wide, narrow])
    }

    /// Stores `v` (of the element type) at element `idx` of buffer `bi`.
    fn store_elem(&mut self, bi: usize, idx: Id, v: Id) {
        if !self.bufs[bi].narrow {
            let p = self.elem_ptr(bi, idx);
            self.f.store(p, v);
            return;
        }
        let s = self.bufs[bi].elem;
        let safe = self.checked_index(bi, idx);
        let (wi, sh, l) = self.narrow_at(bi, safe);
        let u32t = self.m.t_int(32, false);
        let i32t = self.m.t_int(32, true);
        let u64t = self.m.t_int(64, false);
        let i64t = self.m.t_int(64, true);
        let bt = self.m.t_bool();
        let p0 = self.word_ptr(bi, wi);
        let c3 = self.m.c_u32(3);
        let is8 = self.f.op(&mut self.m, op::I_EQUAL, bt, &[l, c3]);
        let (wide_l, narrow_l, done) = (self.label(), self.label(), self.label());
        self.f.emit(op::SELECTION_MERGE, &[done, 0]);
        self.f.branch_cond(is8, wide_l, narrow_l);
        // 8 bytes: two words.
        self.f.label(wide_l);
        if s == Scalar::Int(64, true) {
            let vu = self.f.op(&mut self.m, op::BITCAST, u64t, &[v]);
            let lo = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[vu]);
            let c32 = self.m.c_u32(32);
            let hs = self.f.op(&mut self.m, op::SHIFT_RIGHT_LOGICAL, u64t, &[vu, c32]);
            let hi = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[hs]);
            self.f.store(p0, lo);
            let one = self.m.c_u32(1);
            let wi1 = self.f.op(&mut self.m, op::I_ADD, u32t, &[wi, one]);
            let p1 = self.word_ptr(bi, wi1);
            self.f.store(p1, hi);
        }
        self.f.branch(done);
        // 1, 2 or 4 bytes: if the value fits, into its bits of the word (atomically, since
        // neighbouring elements may belong to other lanes).
        self.f.label(narrow_l);
        let bits = self.field_bits(l);
        let (t32, fits) = if s == Scalar::Int(64, true) {
            let vu = self.f.op(&mut self.m, op::BITCAST, u64t, &[v]);
            let t = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[vu]);
            let ts = self.f.op(&mut self.m, op::BITCAST, i32t, &[t]);
            let z = self.m.c_u32(0);
            let sx = self.f.op(&mut self.m, op::BITFIELD_S_EXTRACT, i32t, &[ts, z, bits]);
            let back = self.f.op(&mut self.m, op::S_CONVERT, i64t, &[sx]);
            let fits = self.f.op(&mut self.m, op::I_EQUAL, bt, &[back, v]);
            (t, fits)
        } else {
            let t = self.f.op(&mut self.m, op::BITCAST, u32t, &[v]);
            let z = self.m.c_u32(0);
            let sx = self.f.op(&mut self.m, op::BITFIELD_S_EXTRACT, i32t, &[v, z, bits]);
            let fits = self.f.op(&mut self.m, op::I_EQUAL, bt, &[sx, v]);
            (t, fits)
        };
        let (put, wide, after) = (self.label(), self.label(), self.label());
        self.f.emit(op::SELECTION_MERGE, &[after, 0]);
        self.f.branch_cond(fits, put, wide);
        self.f.label(put);
        let c32 = self.m.c_u32(32);
        let full = self.f.op(&mut self.m, op::I_EQUAL, bt, &[bits, c32]);
        let one = self.m.c_u32(1);
        let lowbits = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u32t, &[one, bits]);
        let lowmask = self.f.op(&mut self.m, op::I_SUB, u32t, &[lowbits, one]);
        let all = self.m.c_u32(u32::MAX);
        let maskbits = self.f.op(&mut self.m, op::SELECT, u32t, &[full, all, lowmask]);
        let mask = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u32t, &[maskbits, sh]);
        let tv = self.f.op(&mut self.m, op::BITWISE_AND, u32t, &[t32, maskbits]);
        let val = self.f.op(&mut self.m, op::SHIFT_LEFT_LOGICAL, u32t, &[tv, sh]);
        let (plain, atom, stored) = (self.label(), self.label(), self.label());
        self.f.emit(op::SELECTION_MERGE, &[stored, 0]);
        self.f.branch_cond(full, plain, atom);
        self.f.label(plain);
        self.f.store(p0, val);
        self.f.branch(stored);
        self.f.label(atom);
        let nmask = self.f.op(&mut self.m, op::NOT, u32t, &[mask]);
        let scope = self.m.c_u32(1); // Device
        let sem = self.m.c_u32(0); // Relaxed
        self.f.op(&mut self.m, op::ATOMIC_AND, u32t, &[p0, scope, sem, nmask]);
        self.f.op(&mut self.m, op::ATOMIC_OR, u32t, &[p0, scope, sem, val]);
        self.f.branch(stored);
        self.f.label(stored);
        self.f.branch(after);
        self.f.label(wide);
        let tru = self.m.c_bool(true);
        self.f.store(self.widen, tru);
        self.f.store(self.trapped, tru);
        self.f.branch(after);
        self.f.label(after);
        self.f.branch(done);
        self.f.label(done);
        self.cost += 6 * self.weight;
    }

    /// Pointer to element `idx` (i64) of a buffer stored at full width, trapping (and using
    /// element 0) when the index is out of range.
    fn elem_ptr(&mut self, bi: usize, idx: Id) -> Id {
        let safe = self.checked_index(bi, idx);
        let u32t = self.m.t_int(32, false);
        let s32 = self.f.op(&mut self.m, op::U_CONVERT, u32t, &[safe]);
        let s = self.bufs[bi].elem;
        let et = self.ty(s);
        let pt = self.m.t_ptr(12, et);
        let z = self.m.c_u32(0);
        let bv = self.buf_vars[bi];
        self.f.op(&mut self.m, op::ACCESS_CHAIN, pt, &[bv, z, s32])
    }

    // ---- checked arithmetic ----

    fn int_min(&mut self, s: Scalar) -> Id {
        match s {
            Scalar::Int(64, true) => self.m.c_i64(i64::MIN),
            Scalar::Int(32, true) => self.m.c_i32(i32::MIN),
            _ => self.m.c_int(s, 0),
        }
    }
    fn add(&mut self, a: Id, b: Id, s: Scalar) -> Id {
        let r = self.op(op::I_ADD, s, &[a, b]);
        let ov = match s {
            Scalar::Int(_, true) => {
                let x1 = self.op(op::BITWISE_XOR, s, &[a, r]);
                let x2 = self.op(op::BITWISE_XOR, s, &[b, r]);
                let x = self.op(op::BITWISE_AND, s, &[x1, x2]);
                let z = self.m.c_int(s, 0);
                self.bop(op::S_LESS_THAN, &[x, z])
            }
            _ => self.bop(op::U_LESS_THAN, &[r, a]),
        };
        self.trap_if(ov);
        r
    }
    fn sub(&mut self, a: Id, b: Id, s: Scalar) -> Id {
        let r = self.op(op::I_SUB, s, &[a, b]);
        let ov = match s {
            Scalar::Int(_, true) => {
                let x1 = self.op(op::BITWISE_XOR, s, &[a, b]);
                let x2 = self.op(op::BITWISE_XOR, s, &[a, r]);
                let x = self.op(op::BITWISE_AND, s, &[x1, x2]);
                let z = self.m.c_int(s, 0);
                self.bop(op::S_LESS_THAN, &[x, z])
            }
            _ => self.bop(op::U_GREATER_THAN, &[b, a]),
        };
        self.trap_if(ov);
        r
    }
    fn mul(&mut self, a: Id, b: Id, s: Scalar) -> Id {
        if s == Scalar::Int(64, true) {
            // Both factors in 32 bits (the usual case) cannot overflow: a plain multiply, and
            // the full 128-bit product only otherwise (it is long on a GPU).
            let i32t = self.m.t_int(32, true);
            let t = self.ty(s);
            let bt = self.m.t_bool();
            let a32 = self.f.op(&mut self.m, op::S_CONVERT, i32t, &[a]);
            let a64 = self.f.op(&mut self.m, op::S_CONVERT, t, &[a32]);
            let b32 = self.f.op(&mut self.m, op::S_CONVERT, i32t, &[b]);
            let b64 = self.f.op(&mut self.m, op::S_CONVERT, t, &[b32]);
            let fa = self.f.op(&mut self.m, op::I_EQUAL, bt, &[a, a64]);
            let fb = self.f.op(&mut self.m, op::I_EQUAL, bt, &[b, b64]);
            let small = self.f.op(&mut self.m, op::LOGICAL_AND, bt, &[fa, fb]);
            let out = self.var(s);
            let (fast, slow, done) = (self.label(), self.label(), self.label());
            self.f.emit(op::SELECTION_MERGE, &[done, 0]);
            self.f.branch_cond(small, fast, slow);
            self.f.label(fast);
            let p = self.f.op(&mut self.m, op::I_MUL, t, &[a, b]);
            self.f.store(out, p);
            self.f.branch(done);
            self.f.label(slow);
            let st = self.m.t_struct(&[t, t]);
            let ext = self.f.op(&mut self.m, op::S_MUL_EXTENDED, st, &[a, b]);
            let lo = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, t, &[ext, 0]);
            let hi = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, t, &[ext, 1]);
            let sh = self.m.c_u32(63);
            let sign = self.op(op::SHIFT_RIGHT_ARITHMETIC, s, &[lo, sh]);
            let ov = self.bop(op::I_NOT_EQUAL, &[hi, sign]);
            self.trap_if(ov);
            self.f.store(out, lo);
            self.f.branch(done);
            self.f.label(done);
            self.cost += 6 * self.weight;
            return self.load(out, s);
        }
        let Scalar::Int(w, sg) = s else { unreachable!() };
        let t = self.ty(s);
        let st = self.m.t_struct(&[t, t]);
        let ext = self.f.op(&mut self.m, if sg { op::S_MUL_EXTENDED } else { op::U_MUL_EXTENDED }, st, &[a, b]);
        let lo = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, t, &[ext, 0]);
        let hi = self.f.op(&mut self.m, op::COMPOSITE_EXTRACT, t, &[ext, 1]);
        self.cost += 4 * self.weight;
        let ov = if sg {
            let sh = self.m.c_u32(w - 1);
            let sign = self.op(op::SHIFT_RIGHT_ARITHMETIC, s, &[lo, sh]);
            self.bop(op::I_NOT_EQUAL, &[hi, sign])
        } else {
            let z = self.m.c_int(s, 0);
            self.bop(op::I_NOT_EQUAL, &[hi, z])
        };
        self.trap_if(ov);
        lo
    }
    fn divrem(&mut self, a: Id, b: Id, s: Scalar, rem: bool) -> Id {
        let z = self.m.c_int(s, 0);
        let one = self.m.c_int(s, 1);
        let bz = self.bop(op::I_EQUAL, &[b, z]);
        let bad = match s {
            Scalar::Int(_, true) => {
                let mn = self.int_min(s);
                let m1 = self.m.c_int(s, -1);
                let am = self.bop(op::I_EQUAL, &[a, mn]);
                let bm = self.bop(op::I_EQUAL, &[b, m1]);
                let both = self.bop(op::LOGICAL_AND, &[am, bm]);
                self.bop(op::LOGICAL_OR, &[bz, both])
            }
            _ => bz,
        };
        self.trap_if(bad);
        let t = self.ty(s);
        let sb = self.f.op(&mut self.m, op::SELECT, t, &[bad, one, b]);
        let signed = matches!(s, Scalar::Int(_, true));
        let opc = match (rem, signed) {
            (false, true) => op::S_DIV,
            (false, false) => op::U_DIV,
            (true, true) => op::S_REM,
            (true, false) => op::U_MOD,
        };
        self.cost += 8 * self.weight;
        self.op(opc, s, &[a, sb])
    }

    // ---- statements ----

    fn stmt(&mut self, e: &Expr) -> Option<()> {
        let r = self.stmt_inner(e);
        if r.is_none() && std::env::var("AIL_GPU_DEBUG").is_ok() {
            eprintln!("gpu: statement not supported: {:?}", short(e));
        }
        r
    }

    fn stmt_inner(&mut self, e: &Expr) -> Option<()> {
        if self.f.terminated {
            return Some(());
        }
        match &e.kind {
            ExprKind::Let { value, ty: _, .. } => {
                let b = *self.inp.ck.decl.get(&(e.id as usize))?;
                let s = scalar_of(&self.inp.ck.bindings[b].ty)?;
                let (v, vs) = self.expr(value)?;
                if vs != s {
                    return None;
                }
                let p = self.var(s);
                self.f.store(p, v);
                self.locals.insert(b, (p, s));
                Some(())
            }
            ExprKind::Set(place, value) => self.set(e, place, value),
            ExprKind::If(c, t, f) => {
                let (cv, _) = self.cond(c)?;
                let tl = self.label();
                let merge = self.label();
                let fl = if f.is_some() { self.label() } else { merge };
                self.f.emit(op::SELECTION_MERGE, &[merge, 0]);
                self.f.branch_cond(cv, tl, fl);
                self.f.label(tl);
                self.stmt(t)?;
                self.f.branch(merge);
                if let Some(f) = f {
                    self.f.label(fl);
                    self.stmt(f)?;
                    self.f.branch(merge);
                }
                self.f.label(merge);
                Some(())
            }
            ExprKind::Do(es) => {
                for x in es {
                    self.stmt(x)?;
                }
                Some(())
            }
            ExprKind::While(c, body) => self.lp(None, Some(c), body),
            ExprKind::For(_, _, ForIter::Range(lo, hi), body) => {
                let b = *self.inp.ck.decl.get(&(e.id as usize))?;
                if scalar_of(&self.inp.ck.bindings[b].ty)? != Scalar::Int(64, true) {
                    return None;
                }
                self.lp(Some((b, lo, hi)), None, body)
            }
            ExprKind::Break => {
                let l = self.loops.last()?.merge;
                self.f.branch(l);
                Some(())
            }
            ExprKind::Continue => {
                let l = self.loops.last()?.cont;
                self.f.branch(l);
                Some(())
            }
            ExprKind::Return(v) if self.in_helper => {
                match v {
                    Some(v) => {
                        let (x, _) = self.expr(v)?;
                        self.f.ret_value(x);
                    }
                    None => self.f.ret(),
                }
                Some(())
            }
            // A call kept for its effect (its value, if any, is dropped).
            ExprKind::Call(..) => {
                if matches!(self.res(e), Some(Res::UserFn(_))) {
                    self.call(e).map(|_| ())
                } else if matches!(self.ty_of(e), Ty::Unit) {
                    None
                } else {
                    self.expr(e).map(|_| ())
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Var(_) => Some(()),
            _ => None,
        }
    }

    fn set(&mut self, e: &Expr, place: &Expr, value: &Expr) -> Option<()> {
        let inner = match (&place.kind, self.res(place), &self.inp.collapse) {
            (ExprKind::Var(_), Some(Res::Local(b)), Some(c)) => c.inner_sums.get(b).map(|r| (*r, *b)),
            _ => None,
        };
        if let Some((r, rb)) = inner {
            // (set r (+ r x)) of an inner sum: x goes to the outer sum.
            let ExprKind::Call(opn, _, xs) = &value.kind else { return None };
            if opn != "+" {
                return None;
            }
            let s = Scalar::Int(64, true);
            let (acc, abs) = self.accs[r];
            let is_r = |x: &Expr| matches!((&x.kind, self.inp.ck.res.get(x.id as usize)), (ExprKind::Var(_), Some(Res::Local(b))) if *b == rb);
            let terms: Vec<&Expr> = xs.iter().map(|a| a.expr()).filter(|x| !is_r(x)).collect();
            for x in terms {
                let (v, vs) = self.expr(x)?;
                if vs != s {
                    return None;
                }
                self.accumulate(acc, abs, v, false);
            }
            return Some(());
        }
        if let Some(&r) = self.inp.sums.get(&e.id) {
            // (set s (+ s x)) or (set s (- s x)): add x (or -x) to this invocation's sum.
            let ExprKind::Call(opn, _, xs) = &value.kind else { return None };
            let sv = self.inp.sum_vars[r];
            let is_s = |x: &Expr| matches!((&x.kind, self.inp.ck.res.get(x.id as usize)), (ExprKind::Var(_), Some(Res::Local(b))) if *b == sv);
            let s = Scalar::Int(64, true);
            if scalar_of(&self.inp.ck.bindings[sv].ty)? != s {
                return None;
            }
            let (acc, abs) = self.accs[r];
            let terms: Vec<&Expr> = xs.iter().map(|a| a.expr()).filter(|x| !is_s(x)).collect();
            let minus = opn == "-";
            for x in terms {
                let (v, vs) = self.expr(x)?;
                if vs != s {
                    return None;
                }
                self.accumulate(acc, abs, v, minus);
            }
            return Some(());
        }
        match &place.kind {
            ExprKind::Var(_) => {
                let Some(Res::Local(b)) = self.res(place) else { return None };
                let (p, s) = *self.locals.get(b)?;
                if *b == self.inp.lv {
                    return None;
                }
                let (v, vs) = self.expr(value)?;
                if vs != s {
                    return None;
                }
                self.f.store(p, v);
                Some(())
            }
            ExprKind::Call(n, _, xs) if n == "at" && xs.len() == 2 && self.inp.writes.contains(&place.id) => {
                let Some(Res::Local(b)) = self.res(xs[0].expr()) else { return None };
                if !matches!(xs[0].expr().kind, ExprKind::Var(_)) {
                    return None;
                }
                let bi = self.buffer(*b, true)?;
                let (iv, is) = self.expr(xs[1].expr())?;
                if is != Scalar::Int(64, true) {
                    return None;
                }
                let (v, vs) = self.expr(value)?;
                if vs != self.bufs[bi].elem {
                    return None;
                }
                self.store_elem(bi, iv, v);
                self.cost += 2 * self.weight;
                Some(())
            }
            _ => None,
        }
    }

    /// Adds x (or -x) to an invocation's sum and |x| to its magnitude bound, both checked.
    fn accumulate(&mut self, acc: Id, abs: Id, v: Id, minus: bool) {
        let s = Scalar::Int(64, true);
        let cur = self.load(acc, s);
        let nv = if minus { self.sub(cur, v, s) } else { self.add(cur, v, s) };
        self.f.store(acc, nv);
        // |x| as unsigned (exact for i64::MIN too).
        let z = self.m.c_i64(0);
        let neg = self.bop(op::S_LESS_THAN, &[v, z]);
        let nvv = self.op(op::S_NEGATE, s, &[v]);
        let av = self.op(op::SELECT, s, &[neg, nvv, v]);
        let u = Scalar::Int(64, false);
        let ut = self.ty(u);
        let au = self.f.op(&mut self.m, op::BITCAST, ut, &[av]);
        let ca = self.load(abs, u);
        let na = self.add(ca, au, u);
        self.f.store(abs, na);
    }

    /// A loop: `(for j (range lo hi) body)` when `range` is given, else `(while c body)`. It
    /// also stops when the invocation has trapped or used up its budget of iterations.
    fn lp(&mut self, range: Option<(BindId, &Expr, &Expr)>, cond: Option<&Expr>, body: &[Expr]) -> Option<()> {
        self.has_loops = true;
        let s64 = Scalar::Int(64, true);
        let mut jv = 0;
        let mut hiv = 0;
        if let Some((b, lo, hi)) = range {
            let (l, ls) = self.expr(lo)?;
            let (h, hs) = self.expr(hi)?;
            if ls != s64 || hs != s64 {
                return None;
            }
            jv = self.var(s64);
            self.f.store(jv, l);
            hiv = self.var(s64);
            self.f.store(hiv, h);
            self.locals.insert(b, (jv, s64));
        }
        let (head, cond_l, body_l, cont, merge) = (self.label(), self.label(), self.label(), self.label(), self.label());
        self.f.branch(head);
        self.f.label(head);
        self.f.emit(op::LOOP_MERGE, &[merge, cont, 0]);
        self.f.branch(cond_l);
        self.f.label(cond_l);
        let old_w = self.weight;
        self.weight = (self.weight * 16).min(1 << 20);
        let mut c = match (range, cond) {
            (Some(_), _) => {
                let j = self.load(jv, s64);
                let h = self.load(hiv, s64);
                self.bop(op::S_LESS_THAN, &[j, h])
            }
            (None, Some(c)) => self.cond(c)?.0,
            _ => unreachable!(),
        };
        // ... and not trapped, and within budget.
        let tv = self.load(self.trapped, Scalar::Bool);
        let nt = self.bop(op::LOGICAL_NOT, &[tv]);
        c = self.bop(op::LOGICAL_AND, &[c, nt]);
        let bv = self.load(self.budget, s64);
        let one = self.m.c_i64(1);
        let b2 = self.op(op::I_ADD, s64, &[bv, one]);
        self.f.store(self.budget, b2);
        let lim = self.m.c_i64(BUDGET);
        let over = self.bop(op::S_GREATER_THAN, &[b2, lim]);
        self.trap_if(over);
        let ok = self.bop(op::LOGICAL_NOT, &[over]);
        c = self.bop(op::LOGICAL_AND, &[c, ok]);
        self.f.branch_cond(c, body_l, merge);
        self.f.label(body_l);
        self.loops.push(Loop { merge, cont });
        for s in body {
            self.stmt(s)?;
        }
        self.loops.pop();
        self.f.branch(cont);
        self.f.label(cont);
        if range.is_some() {
            let j = self.load(jv, s64);
            let j2 = self.op(op::I_ADD, s64, &[j, one]);
            self.f.store(jv, j2);
        }
        self.f.branch(head);
        self.f.label(merge);
        self.weight = old_w;
        Some(())
    }

    // ---- expressions ----

    /// A Bool condition.
    fn cond(&mut self, e: &Expr) -> Option<V> {
        let v = self.expr(e)?;
        if v.1 != Scalar::Bool {
            return None;
        }
        Some(v)
    }

    fn expr(&mut self, e: &Expr) -> Option<V> {
        let r = self.expr_inner(e);
        if r.is_none() && std::env::var("AIL_GPU_DEBUG").is_ok() {
            eprintln!("gpu: expression not supported: {:?} : {:?}", short(e), self.ty_of(e));
        }
        r
    }

    fn expr_inner(&mut self, e: &Expr) -> Option<V> {
        let s = self.scalar(e);
        match &e.kind {
            ExprKind::Int(v) => {
                let s = s?;
                if let Scalar::Float(_) = s {
                    return Some((self.m.c_f64(*v as f64), s));
                }
                Some((self.m.c_int(s, *v), s))
            }
            ExprKind::Float(v) => {
                if s? != Scalar::Float(64) {
                    return None;
                }
                self.f64 = true;
                Some((self.m.c_f64(*v), Scalar::Float(64)))
            }
            ExprKind::Var(_) => match self.res(e)?.clone() {
                Res::Bool(b) => Some((self.m.c_bool(b), Scalar::Bool)),
                Res::Const(n) => {
                    let s = s?;
                    if s == Scalar::Float(64) {
                        self.f64 = true;
                    }
                    Some(self.arg(KArg::Const(n), s))
                }
                Res::Local(b) => {
                    if let Some(&(p, s)) = self.locals.get(&b) {
                        return Some((self.load(p, s), s));
                    }
                    if self.in_helper || self.inp.sum_vars.contains(&b) {
                        return None;
                    }
                    let s = scalar_of(&self.inp.ck.bindings[b].ty)?;
                    if s == Scalar::Float(64) {
                        self.f64 = true;
                    }
                    Some(self.arg(KArg::Local(b), s))
                }
                _ => None,
            },
            ExprKind::If(c, t, Some(f)) => {
                let s = s?;
                let r = self.var(s);
                let (cv, _) = self.cond(c)?;
                let (tl, fl, merge) = (self.label(), self.label(), self.label());
                self.f.emit(op::SELECTION_MERGE, &[merge, 0]);
                self.f.branch_cond(cv, tl, fl);
                self.f.label(tl);
                let (tv, ts) = self.value_of(t)?;
                if ts != s {
                    return None;
                }
                self.f.store(r, tv);
                self.f.branch(merge);
                self.f.label(fl);
                let (fv, fs) = self.value_of(f)?;
                if fs != s {
                    return None;
                }
                self.f.store(r, fv);
                self.f.branch(merge);
                self.f.label(merge);
                Some((self.load(r, s), s))
            }
            ExprKind::Do(es) => {
                let (last, rest) = es.split_last()?;
                for x in rest {
                    self.stmt(x)?;
                }
                self.expr(last)
            }
            ExprKind::As(_, x) => self.cast(x, s?),
            ExprKind::Call(..) if matches!(self.res(e), Some(Res::UserFn(_))) => self.call(e)?,
            ExprKind::Call(n, _, args) if matches!(self.res(e), Some(Res::Builtin(_))) => self.builtin(e, n, args),
            _ => None,
        }
    }

    /// The value of a branch of an `if`: an expression, or a block whose last item is one.
    fn value_of(&mut self, e: &Expr) -> Option<V> {
        self.expr(e)
    }

    fn cast(&mut self, x: &Expr, to: Scalar) -> Option<V> {
        let (v, from) = self.expr(x)?;
        if from == to {
            return Some((v, to));
        }
        let t = self.ty(to);
        match (from, to) {
            (Scalar::Int(fw, fs), Scalar::Int(tw, ts)) => {
                // Checked: the value must fit the target type.
                let (tlo, thi): (i128, i128) = match (tw, ts) {
                    (64, true) => (i64::MIN as i128, i64::MAX as i128),
                    (64, false) => (0, u64::MAX as i128),
                    (32, true) => (i32::MIN as i128, i32::MAX as i128),
                    (32, false) => (0, u32::MAX as i128),
                    _ => return None,
                };
                let (flo, fhi): (i128, i128) = match (fw, fs) {
                    (64, true) => (i64::MIN as i128, i64::MAX as i128),
                    (64, false) => (0, u64::MAX as i128),
                    (32, true) => (i32::MIN as i128, i32::MAX as i128),
                    (32, false) => (0, u32::MAX as i128),
                    _ => return None,
                };
                if flo < tlo {
                    let c = self.m.c_int(from, tlo.max(flo));
                    let bad = self.bop(if fs { op::S_LESS_THAN } else { op::U_LESS_THAN }, &[v, c]);
                    self.trap_if(bad);
                }
                if fhi > thi {
                    let c = self.m.c_int(from, thi);
                    let bad = self.bop(if fs { op::S_GREATER_THAN } else { op::U_GREATER_THAN }, &[v, c]);
                    self.trap_if(bad);
                }
                let r = if fw == tw {
                    self.f.op(&mut self.m, op::BITCAST, t, &[v])
                } else if fs {
                    let st = self.m.t_int(tw, true);
                    let c = self.f.op(&mut self.m, op::S_CONVERT, st, &[v]);
                    if ts { c } else { self.f.op(&mut self.m, op::BITCAST, t, &[c]) }
                } else {
                    let ut = self.m.t_int(tw, false);
                    let c = self.f.op(&mut self.m, op::U_CONVERT, ut, &[v]);
                    if !ts { c } else { self.f.op(&mut self.m, op::BITCAST, t, &[c]) }
                };
                Some((r, to))
            }
            (Scalar::Int(_, sg), Scalar::Float(64)) => {
                self.f64 = true;
                let r = self.f.op(&mut self.m, if sg { op::CONVERT_S_TO_F } else { op::CONVERT_U_TO_F }, t, &[v]);
                Some((r, to))
            }
            (Scalar::Float(64), Scalar::Int(w, sg)) => {
                // In range (and not NaN), else trap; then truncate toward zero.
                let (lo, hi): (f64, f64) = match (w, sg) {
                    (64, true) => (i64::MIN as f64, i64::MAX as f64 + 1.0),
                    (64, false) => (0.0, u64::MAX as f64 + 1.0),
                    (32, true) => (i32::MIN as f64, i32::MAX as f64 + 1.0),
                    (32, false) => (0.0, u32::MAX as f64 + 1.0),
                    _ => return None,
                };
                let cl = self.m.c_f64(lo);
                let ch = self.m.c_f64(hi);
                let a = self.bop(op::F_ORD_GREATER_THAN_EQUAL, &[v, cl]);
                let b = self.bop(op::F_ORD_LESS_THAN, &[v, ch]);
                let ok = self.bop(op::LOGICAL_AND, &[a, b]);
                let bad = self.bop(op::LOGICAL_NOT, &[ok]);
                self.trap_if(bad);
                let zf = self.m.c_f64(0.0);
                let ft = self.ty(Scalar::Float(64));
                let safe = self.f.op(&mut self.m, op::SELECT, ft, &[bad, zf, v]);
                let r = self.f.op(&mut self.m, if sg { op::CONVERT_F_TO_S } else { op::CONVERT_F_TO_U }, t, &[safe]);
                Some((r, to))
            }
            _ => None,
        }
    }

    fn builtin(&mut self, e: &Expr, n: &str, args: &[Arg]) -> Option<V> {
        if args.iter().any(|a| matches!(a, Arg::Mut(..))) {
            return None;
        }
        let a = |i: usize| args[i].expr();
        let rs = self.scalar(e);
        match n {
            "+" | "*" | "-" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                if n == "-" && args.len() == 1 {
                    let (v, _) = self.expr(a(0))?;
                    return Some(match s {
                        Scalar::Int(_, true) => {
                            let mn = self.int_min(s);
                            let bad = self.bop(op::I_EQUAL, &[v, mn]);
                            self.trap_if(bad);
                            (self.op(op::S_NEGATE, s, &[v]), s)
                        }
                        _ => {
                            let z = self.m.c_int(s, 0);
                            (self.sub(z, v, s), s)
                        }
                    });
                }
                let (mut acc, _) = self.expr(a(0))?;
                for x in &args[1..] {
                    let (v, vs) = self.expr(x.expr())?;
                    if vs != s {
                        return None;
                    }
                    acc = match n {
                        "+" => self.add(acc, v, s),
                        "-" => self.sub(acc, v, s),
                        _ => self.mul(acc, v, s),
                    };
                }
                Some((acc, s))
            }
            "+" | "*" | "-" if rs == Some(Scalar::Float(64)) => {
                let s = Scalar::Float(64);
                self.f64 = true;
                if n == "-" && args.len() == 1 {
                    let (v, _) = self.expr(a(0))?;
                    return Some((self.fop(op::F_NEGATE, s, &[v]), s));
                }
                let (mut acc, _) = self.expr(a(0))?;
                for x in &args[1..] {
                    let (v, vs) = self.expr(x.expr())?;
                    if vs != s {
                        return None;
                    }
                    let opc = match n {
                        "+" => op::F_ADD,
                        "-" => op::F_SUB,
                        _ => op::F_MUL,
                    };
                    acc = self.fop(opc, s, &[acc, v]);
                }
                Some((acc, s))
            }
            "/" | "%" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) && args.len() == 2 => {
                let s = rs?;
                let (x, _) = self.expr(a(0))?;
                let (y, _) = self.expr(a(1))?;
                Some((self.divrem(x, y, s, n == "%"), s))
            }
            "wrapping-add" | "wrapping-sub" | "wrapping-mul" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                let (x, _) = self.expr(a(0))?;
                let (y, _) = self.expr(a(1))?;
                let opc = match n {
                    "wrapping-add" => op::I_ADD,
                    "wrapping-sub" => op::I_SUB,
                    _ => op::I_MUL,
                };
                Some((self.op(opc, s, &[x, y]), s))
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                let (x, xs) = self.expr(a(0))?;
                let (y, ys) = self.expr(a(1))?;
                if xs != ys {
                    return None;
                }
                let opc = match (xs, n) {
                    (Scalar::Bool, "==") => op::LOGICAL_EQUAL,
                    (Scalar::Bool, "!=") => op::LOGICAL_NOT_EQUAL,
                    (Scalar::Bool, _) => return None,
                    (Scalar::Int(..), "==") => op::I_EQUAL,
                    (Scalar::Int(..), "!=") => op::I_NOT_EQUAL,
                    (Scalar::Int(_, true), "<") => op::S_LESS_THAN,
                    (Scalar::Int(_, true), "<=") => op::S_LESS_THAN_EQUAL,
                    (Scalar::Int(_, true), ">") => op::S_GREATER_THAN,
                    (Scalar::Int(_, true), ">=") => op::S_GREATER_THAN_EQUAL,
                    (Scalar::Int(_, false), "<") => op::U_LESS_THAN,
                    (Scalar::Int(_, false), "<=") => op::U_LESS_THAN_EQUAL,
                    (Scalar::Int(_, false), ">") => op::U_GREATER_THAN,
                    (Scalar::Int(_, false), ">=") => op::U_GREATER_THAN_EQUAL,
                    (Scalar::Float(_), "==") => op::F_ORD_EQUAL,
                    (Scalar::Float(_), "!=") => op::F_UNORD_NOT_EQUAL,
                    (Scalar::Float(_), "<") => op::F_ORD_LESS_THAN,
                    (Scalar::Float(_), "<=") => op::F_ORD_LESS_THAN_EQUAL,
                    (Scalar::Float(_), ">") => op::F_ORD_GREATER_THAN,
                    (Scalar::Float(_), ">=") => op::F_ORD_GREATER_THAN_EQUAL,
                    _ => return None,
                };
                Some((self.bop(opc, &[x, y]), Scalar::Bool))
            }
            "and" | "or" => {
                // Short-circuit, as on the CPU: later operands may trap.
                let r = self.var(Scalar::Bool);
                let (first, _) = self.cond(a(0))?;
                self.f.store(r, first);
                let mut merges = Vec::new();
                for x in &args[1..] {
                    let cur = self.load(r, Scalar::Bool);
                    let (more, merge) = (self.label(), self.label());
                    self.f.emit(op::SELECTION_MERGE, &[merge, 0]);
                    if n == "and" {
                        self.f.branch_cond(cur, more, merge);
                    } else {
                        self.f.branch_cond(cur, merge, more);
                    }
                    self.f.label(more);
                    let (v, _) = self.cond(x.expr())?;
                    self.f.store(r, v);
                    self.f.branch(merge);
                    self.f.label(merge);
                    merges.push(merge);
                }
                Some((self.load(r, Scalar::Bool), Scalar::Bool))
            }
            "not" => {
                let (v, _) = self.cond(a(0))?;
                Some((self.bop(op::LOGICAL_NOT, &[v]), Scalar::Bool))
            }
            "min" | "max" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                let (mut acc, _) = self.expr(a(0))?;
                for x in &args[1..] {
                    let (v, _) = self.expr(x.expr())?;
                    let lt = self.bop(if matches!(s, Scalar::Int(_, true)) { op::S_LESS_THAN } else { op::U_LESS_THAN }, &[v, acc]);
                    acc = if n == "min" { self.op(op::SELECT, s, &[lt, v, acc]) } else {
                        let gt = self.bop(if matches!(s, Scalar::Int(_, true)) { op::S_GREATER_THAN } else { op::U_GREATER_THAN }, &[v, acc]);
                        let _ = lt;
                        self.op(op::SELECT, s, &[gt, v, acc])
                    };
                }
                Some((acc, s))
            }
            "abs" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                let (v, _) = self.expr(a(0))?;
                if let Scalar::Int(_, false) = s {
                    return Some((v, s));
                }
                let mn = self.int_min(s);
                let bad = self.bop(op::I_EQUAL, &[v, mn]);
                self.trap_if(bad);
                let z = self.m.c_int(s, 0);
                let neg = self.bop(op::S_LESS_THAN, &[v, z]);
                let nv = self.op(op::S_NEGATE, s, &[v]);
                Some((self.op(op::SELECT, s, &[neg, nv, v]), s))
            }
            "bit-and" | "bit-or" | "bit-xor" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                let (x, _) = self.expr(a(0))?;
                let (y, _) = self.expr(a(1))?;
                let opc = match n {
                    "bit-and" => op::BITWISE_AND,
                    "bit-or" => op::BITWISE_OR,
                    _ => op::BITWISE_XOR,
                };
                Some((self.op(opc, s, &[x, y]), s))
            }
            "shl" | "shr" if rs.is_some_and(|s| matches!(s, Scalar::Int(..))) => {
                let s = rs?;
                let Scalar::Int(w, sg) = s else { return None };
                let (x, _) = self.expr(a(0))?;
                let (k, ks) = self.expr(a(1))?;
                let Scalar::Int(_, ksg) = ks else { return None };
                // The amount must be in 0..width (it is converted to u32 first).
                let z = self.m.c_int(ks, 0);
                let wc = self.m.c_int(ks, w as i128);
                let neg = if ksg { self.bop(op::S_LESS_THAN, &[k, z]) } else { self.m.c_bool(false) };
                let big = self.bop(if ksg { op::S_GREATER_THAN_EQUAL } else { op::U_GREATER_THAN_EQUAL }, &[k, wc]);
                let bad = self.bop(op::LOGICAL_OR, &[neg, big]);
                self.trap_if(bad);
                let kt = self.ty(ks);
                let safe = self.f.op(&mut self.m, op::SELECT, kt, &[bad, z, k]);
                let opc = if n == "shl" { op::SHIFT_LEFT_LOGICAL } else if sg { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL };
                Some((self.op(opc, s, &[x, safe]), s))
            }
            "at" if args.len() == 2 => {
                let v = a(0);
                let Some(Res::Local(b)) = self.res(v) else { return None };
                if !matches!(v.kind, ExprKind::Var(_)) || self.locals.contains_key(b) || self.in_helper {
                    return None;
                }
                // A vector the loop writes is read only at its own elements.
                let b = *b;
                let written = self.bufs.iter().any(|x| x.bind == b && x.write) || self.writes_vec(b);
                let stale = self.inp.stale.contains(&e.id);
                if written && !self.inp.own_reads.contains(&e.id) && !stale {
                    return None;
                }
                let bi = self.buffer_of(b, false, stale)?;
                let (iv, is) = self.expr(a(1))?;
                if is != Scalar::Int(64, true) {
                    return None;
                }
                let s = self.bufs[bi].elem;
                self.cost += 2 * self.weight;
                Some((self.load_elem(bi, iv), s))
            }
            "len" if args.len() == 1 => {
                let v = a(0);
                let Some(Res::Local(b)) = self.res(v) else { return None };
                if !matches!(v.kind, ExprKind::Var(_)) || self.in_helper {
                    return None;
                }
                let bi = self.buffer(*b, false)?;
                Some(self.arg(KArg::Len(bi), Scalar::Int(64, true)))
            }
            _ => None,
        }
    }

    /// True if the loop writes vector `b` somewhere (even where not yet translated).
    fn writes_vec(&self, b: BindId) -> bool {
        let mut found = false;
        for x in self.inp.body {
            crate::emit_rust::walk_ref_pub(x, &mut |e| {
                if let ExprKind::Call(n, _, xs) = &e.kind {
                    if n == "at" && self.inp.writes.contains(&e.id) {
                        if let Some(Res::Local(r)) = self.inp.ck.res.get(xs[0].expr().id as usize) {
                            if *r == b {
                                found = true;
                            }
                        }
                    }
                }
            });
        }
        found
    }

    /// A call of a user fn whose parameters and result are scalars: a SPIR-V function.
    fn call(&mut self, e: &Expr) -> Option<Option<V>> {
        let ExprKind::Call(_, _, args) = &e.kind else { return None };
        let Some(Res::UserFn(name)) = self.res(e).cloned() else { return None };
        if self.helper_stack.contains(&name) {
            return None;
        }
        let mut vals = Vec::new();
        for a in args {
            if matches!(a, Arg::Mut(..)) {
                return None;
            }
            vals.push(self.expr(a.expr())?);
        }
        if !self.helpers.contains_key(&name) {
            self.helper(&name)?;
        }
        let h = &self.helpers[&name];
        if h.params.len() != vals.len() || h.params.iter().zip(&vals).any(|(p, v)| *p != v.1) {
            return None;
        }
        let (fid, ret) = (h.id, h.ret);
        let rt = match ret {
            Some(s) => self.ty(s),
            None => self.m.t_void(),
        };
        let mut ops = vec![fid];
        ops.extend(vals.iter().map(|v| v.0));
        self.cost += 4 * self.weight;
        let r = self.f.op(&mut self.m, op::FUNCTION_CALL, rt, &ops);
        Some(ret.map(|s| (r, s)))
    }

    fn helper(&mut self, name: &str) -> Option<()> {
        let def = self.inp.fns.get(name)?;
        if !def.type_params.is_empty() || !def.requires.is_empty() || !def.ensures.is_empty() {
            return None;
        }
        let sig = self.inp.ck.info.fns.get(name)?;
        let mut ps = Vec::new();
        for (_, t, m) in &sig.params {
            if *m {
                return None;
            }
            ps.push(scalar_of(t)?);
        }
        let ret = match &sig.ret {
            Ty::Unit => None,
            t => Some(scalar_of(t)?),
        };
        let param_binds = self.inp.ck.fn_params.get(name)?.clone();
        let rt = match ret {
            Some(s) => self.ty(s),
            None => self.m.t_void(),
        };
        let pts: Vec<Id> = ps.iter().map(|s| self.m.t_scalar(*s)).collect();
        let fty = self.m.t_fn(rt, &pts);
        let fid = self.m.id();
        self.helpers.insert(name.to_string(), HelperFn { id: fid, ty: fty, ret, params: ps.clone() });
        // Translate the body into a fresh function.
        let saved_f = std::mem::replace(&mut self.f, FnBuilder::new());
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_loops = std::mem::take(&mut self.loops);
        let saved_in = self.in_helper;
        self.in_helper = true;
        self.helper_stack.push(name.to_string());
        let entry = self.m.id();
        let mut params = Vec::new();
        for (i, b) in param_binds.iter().enumerate() {
            let pid = self.m.id();
            params.push((pts[i], pid));
            let v = self.var(ps[i]);
            self.f.store(v, pid);
            self.locals.insert(*b, (v, ps[i]));
        }
        let ok = (|| -> Option<()> {
            let (last, rest) = def.body.split_last()?;
            for s in rest {
                self.stmt(s)?;
            }
            match ret {
                Some(s) => {
                    if self.f.terminated {
                        return Some(());
                    }
                    let (v, vs) = self.expr(last)?;
                    if vs != s {
                        return None;
                    }
                    self.f.ret_value(v);
                }
                None => {
                    self.stmt(last)?;
                    self.f.ret();
                }
            }
            Some(())
        })();
        self.helper_stack.pop();
        self.in_helper = saved_in;
        let fb = std::mem::replace(&mut self.f, saved_f);
        self.locals = saved_locals;
        self.loops = saved_loops;
        ok?;
        fb.finish(&mut self.m, fid, rt, fty, &params, entry);
        Some(())
    }

}
