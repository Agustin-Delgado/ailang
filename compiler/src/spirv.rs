//! A small SPIR-V builder: enough to write compute kernels (types and constants without
//! repeats, global variables, functions made of blocks, and the binary layout Vulkan expects).

use std::collections::HashMap;

pub type Id = u32;

/// Opcodes used by the kernel generator.
#[allow(dead_code)]
pub mod op {
    pub const EXTENSION: u16 = 10;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const F_CONVERT: u16 = 115;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const S_REM: u16 = 138;
    pub const U_MUL_EXTENDED: u16 = 151;
    pub const S_MUL_EXTENDED: u16 = 152;
    pub const IS_NAN: u16 = 156;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const RETURN: u16 = 253;
    pub const RETURN_VALUE: u16 = 254;
}

/// Scalar types a kernel computes with.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scalar {
    Bool,
    Int(u32, bool), // width, signed
    Float(u32),
}

#[derive(Default)]
pub struct Module {
    bound: Id,
    caps: Vec<u32>,
    exts: Vec<u32>,
    entry: Vec<u32>,
    modes: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>,
    pub functions: Vec<u32>,
    types: HashMap<Vec<u32>, Id>,
    consts: HashMap<(Id, Vec<u32>), Id>,
    cap_set: Vec<u32>,
}

pub fn inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | opcode as u32);
    out.extend_from_slice(operands);
}

/// A literal string operand: UTF-8, nul-terminated, padded to whole words.
pub fn string_words(s: &str) -> Vec<u32> {
    let mut b = s.as_bytes().to_vec();
    b.push(0);
    while b.len() % 4 != 0 {
        b.push(0);
    }
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

impl Module {
    pub fn new() -> Self {
        Module { bound: 1, ..Default::default() }
    }
    pub fn id(&mut self) -> Id {
        let i = self.bound;
        self.bound += 1;
        i
    }
    pub fn capability(&mut self, c: u32) {
        if !self.cap_set.contains(&c) {
            self.cap_set.push(c);
            inst(&mut self.caps, op::CAPABILITY, &[c]);
        }
    }
    pub fn extension(&mut self, name: &str) {
        let w = string_words(name);
        inst(&mut self.exts, op::EXTENSION, &w);
    }
    pub fn entry_point(&mut self, f: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![5, f];
        ops.extend(string_words(name));
        ops.extend_from_slice(interface);
        inst(&mut self.entry, op::ENTRY_POINT, &ops);
    }
    pub fn execution_mode(&mut self, f: Id, mode: u32, args: &[u32]) {
        let mut ops = vec![f, mode];
        ops.extend_from_slice(args);
        inst(&mut self.modes, op::EXECUTION_MODE, &ops);
    }
    pub fn decorate(&mut self, target: Id, dec: u32, args: &[u32]) {
        let mut ops = vec![target, dec];
        ops.extend_from_slice(args);
        inst(&mut self.annotations, op::DECORATE, &ops);
    }
    pub fn member_decorate(&mut self, ty: Id, member: u32, dec: u32, args: &[u32]) {
        let mut ops = vec![ty, member, dec];
        ops.extend_from_slice(args);
        inst(&mut self.annotations, op::MEMBER_DECORATE, &ops);
    }
    fn ty(&mut self, opcode: u16, operands: &[u32]) -> Id {
        let mut key = vec![opcode as u32];
        key.extend_from_slice(operands);
        if let Some(&id) = self.types.get(&key) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(operands);
        inst(&mut self.globals, opcode, &ops);
        self.types.insert(key, id);
        id
    }
    /// A type that must not be shared with another of the same shape (a decorated struct).
    pub fn unique_ty(&mut self, opcode: u16, operands: &[u32]) -> Id {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(operands);
        inst(&mut self.globals, opcode, &ops);
        id
    }
    pub fn t_void(&mut self) -> Id {
        self.ty(op::TYPE_VOID, &[])
    }
    pub fn t_bool(&mut self) -> Id {
        self.ty(op::TYPE_BOOL, &[])
    }
    pub fn t_int(&mut self, width: u32, signed: bool) -> Id {
        self.ty(op::TYPE_INT, &[width, signed as u32])
    }
    pub fn t_float(&mut self, width: u32) -> Id {
        self.ty(op::TYPE_FLOAT, &[width])
    }
    pub fn t_scalar(&mut self, s: Scalar) -> Id {
        match s {
            Scalar::Bool => self.t_bool(),
            Scalar::Int(w, sg) => self.t_int(w, sg),
            Scalar::Float(w) => self.t_float(w),
        }
    }
    pub fn t_struct(&mut self, members: &[Id]) -> Id {
        self.ty(op::TYPE_STRUCT, members)
    }
    pub fn t_vector(&mut self, el: Id, n: u32) -> Id {
        self.ty(op::TYPE_VECTOR, &[el, n])
    }
    pub fn t_array(&mut self, el: Id, len: Id) -> Id {
        self.ty(op::TYPE_ARRAY, &[el, len])
    }
    pub fn t_ptr(&mut self, class: u32, ty: Id) -> Id {
        self.ty(op::TYPE_POINTER, &[class, ty])
    }
    pub fn t_fn(&mut self, ret: Id, params: &[Id]) -> Id {
        let mut ops = vec![ret];
        ops.extend_from_slice(params);
        self.ty(op::TYPE_FUNCTION, &ops)
    }
    pub fn constant(&mut self, ty: Id, words: &[u32]) -> Id {
        let key = (ty, words.to_vec());
        if let Some(&id) = self.consts.get(&key) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![ty, id];
        ops.extend_from_slice(words);
        inst(&mut self.globals, op::CONSTANT, &ops);
        self.consts.insert(key, id);
        id
    }
    pub fn c_bool(&mut self, v: bool) -> Id {
        let t = self.t_bool();
        let key = (t, vec![v as u32, 0xffff_ffff]);
        if let Some(&id) = self.consts.get(&key) {
            return id;
        }
        let id = self.id();
        inst(&mut self.globals, if v { op::CONSTANT_TRUE } else { op::CONSTANT_FALSE }, &[t, id]);
        self.consts.insert(key, id);
        id
    }
    pub fn c_i64(&mut self, v: i64) -> Id {
        let t = self.t_int(64, true);
        self.constant(t, &[v as u64 as u32, ((v as u64) >> 32) as u32])
    }
    pub fn c_u64(&mut self, v: u64) -> Id {
        let t = self.t_int(64, false);
        self.constant(t, &[v as u32, (v >> 32) as u32])
    }
    pub fn c_u32(&mut self, v: u32) -> Id {
        let t = self.t_int(32, false);
        self.constant(t, &[v])
    }
    pub fn c_i32(&mut self, v: i32) -> Id {
        let t = self.t_int(32, true);
        self.constant(t, &[v as u32])
    }
    pub fn c_f64(&mut self, v: f64) -> Id {
        let t = self.t_float(64);
        let b = v.to_bits();
        self.constant(t, &[b as u32, (b >> 32) as u32])
    }
    /// A constant of scalar type `s` from an integer value (truncated to its width).
    pub fn c_int(&mut self, s: Scalar, v: i128) -> Id {
        match s {
            Scalar::Int(64, true) => self.c_i64(v as i64),
            Scalar::Int(64, false) => self.c_u64(v as u64),
            Scalar::Int(32, true) => self.c_i32(v as i32),
            Scalar::Int(32, false) => self.c_u32(v as u32),
            Scalar::Int(w, sg) => {
                let t = self.t_int(w, sg);
                self.constant(t, &[(v as u64 as u32) & ((1u64 << w) - 1) as u32])
            }
            Scalar::Float(64) => self.c_f64(v as f64),
            Scalar::Float(w) => {
                let t = self.t_float(w);
                self.constant(t, &[(v as f32).to_bits()])
            }
            Scalar::Bool => self.c_bool(v != 0),
        }
    }
    /// A global variable (in the types section, where Vulkan wants it).
    pub fn global_var(&mut self, ptr_ty: Id, class: u32) -> Id {
        let id = self.id();
        inst(&mut self.globals, op::VARIABLE, &[ptr_ty, id, class]);
        id
    }
    pub fn finish(self) -> Vec<u32> {
        let mut out = vec![0x0723_0203, 0x0001_0300, 0, self.bound, 0];
        out.extend(self.caps);
        out.extend(self.exts);
        inst(&mut out, op::MEMORY_MODEL, &[0, 1]);
        out.extend(self.entry);
        out.extend(self.modes);
        out.extend(self.annotations);
        out.extend(self.globals);
        out.extend(self.functions);
        out
    }
}

/// The body of one function being written: its `Function` variables (which SPIR-V wants at
/// the top of the first block) and its instructions.
pub struct FnBuilder {
    pub vars: Vec<u32>,
    pub code: Vec<u32>,
    /// The current block already ended with a branch or return: nothing more goes in it.
    pub terminated: bool,
}

impl FnBuilder {
    pub fn new() -> Self {
        FnBuilder { vars: Vec::new(), code: Vec::new(), terminated: false }
    }
    pub fn var(&mut self, m: &mut Module, ptr_ty: Id) -> Id {
        let id = m.id();
        inst(&mut self.vars, op::VARIABLE, &[ptr_ty, id, 7]);
        id
    }
    pub fn emit(&mut self, opcode: u16, operands: &[u32]) {
        if !self.terminated {
            inst(&mut self.code, opcode, operands);
        }
    }
    /// An instruction with a result: `opcode result_type result operands...`.
    pub fn op(&mut self, m: &mut Module, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let id = m.id();
        let mut ops = vec![ty, id];
        ops.extend_from_slice(operands);
        self.emit(opcode, &ops);
        id
    }
    pub fn label(&mut self, l: Id) {
        inst(&mut self.code, op::LABEL, &[l]);
        self.terminated = false;
    }
    pub fn branch(&mut self, to: Id) {
        self.emit(op::BRANCH, &[to]);
        self.terminated = true;
    }
    pub fn branch_cond(&mut self, c: Id, t: Id, f: Id) {
        self.emit(op::BRANCH_CONDITIONAL, &[c, t, f]);
        self.terminated = true;
    }
    pub fn ret(&mut self) {
        self.emit(op::RETURN, &[]);
        self.terminated = true;
    }
    pub fn ret_value(&mut self, v: Id) {
        self.emit(op::RETURN_VALUE, &[v]);
        self.terminated = true;
    }
    pub fn load(&mut self, m: &mut Module, ty: Id, ptr: Id) -> Id {
        self.op(m, op::LOAD, ty, &[ptr])
    }
    pub fn store(&mut self, ptr: Id, v: Id) {
        self.emit(op::STORE, &[ptr, v]);
    }
    /// Appends the finished function to the module.
    pub fn finish(self, m: &mut Module, f: Id, ret_ty: Id, fn_ty: Id, params: &[(Id, Id)], entry: Id) {
        inst(&mut m.functions, op::FUNCTION, &[ret_ty, f, 0, fn_ty]);
        for (ty, p) in params {
            inst(&mut m.functions, op::FUNCTION_PARAMETER, &[*ty, *p]);
        }
        inst(&mut m.functions, op::LABEL, &[entry]);
        m.functions.extend(self.vars);
        m.functions.extend(self.code);
        inst(&mut m.functions, op::FUNCTION_END, &[]);
    }
}
