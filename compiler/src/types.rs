//! Semantic types and program-wide type information.

use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntK {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntK {
    pub fn range(self) -> (i128, i128) {
        match self {
            IntK::I8 => (i8::MIN as i128, i8::MAX as i128),
            IntK::I16 => (i16::MIN as i128, i16::MAX as i128),
            IntK::I32 => (i32::MIN as i128, i32::MAX as i128),
            IntK::I64 => (i64::MIN as i128, i64::MAX as i128),
            IntK::U8 => (0, u8::MAX as i128),
            IntK::U16 => (0, u16::MAX as i128),
            IntK::U32 => (0, u32::MAX as i128),
            IntK::U64 => (0, u64::MAX as i128),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            IntK::I8 => "I8",
            IntK::I16 => "I16",
            IntK::I32 => "I32",
            IntK::I64 => "I64",
            IntK::U8 => "U8",
            IntK::U16 => "U16",
            IntK::U32 => "U32",
            IntK::U64 => "U64",
        }
    }
    pub fn rust(self) -> &'static str {
        match self {
            IntK::I8 => "i8",
            IntK::I16 => "i16",
            IntK::I32 => "i32",
            IntK::I64 => "i64",
            IntK::U8 => "u8",
            IntK::U16 => "u16",
            IntK::U32 => "u32",
            IntK::U64 => "u64",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cap {
    Console,
    Fs,
    Clock,
    Rand,
    Env,
    Net,
    Gfx,
}

impl Cap {
    pub fn name(self) -> &'static str {
        match self {
            Cap::Console => "Console",
            Cap::Fs => "Fs",
            Cap::Clock => "Clock",
            Cap::Rand => "Rand",
            Cap::Env => "Env",
            Cap::Net => "Net",
            Cap::Gfx => "Gfx",
        }
    }
    pub fn default_param(self) -> &'static str {
        match self {
            Cap::Console => "con",
            Cap::Fs => "fs",
            Cap::Clock => "clock",
            Cap::Rand => "rng",
            Cap::Env => "env",
            Cap::Net => "net",
            Cap::Gfx => "g",
        }
    }
    pub fn all() -> [Cap; 7] {
        [
            Cap::Console,
            Cap::Fs,
            Cap::Clock,
            Cap::Rand,
            Cap::Env,
            Cap::Net,
            Cap::Gfx,
        ]
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Int(IntK),
    F32,
    F64,
    Bool,
    Str,
    Unit,
    Never,
    Vec(Box<Ty>),
    Map(Box<Ty>, Box<Ty>),
    Opt(Box<Ty>),
    Res(Box<Ty>, Box<Ty>),
    User(String),
    Cap(Cap),
    /// A function passed as a parameter: argument types and result.
    Fn(Vec<Ty>, Box<Ty>),
    /// Poison type after an error; compatible with everything to avoid cascades.
    Err,
}

impl Ty {
    pub fn is_int(&self) -> bool {
        matches!(self, Ty::Int(_))
    }
    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }
    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }
    pub fn is_err(&self) -> bool {
        matches!(self, Ty::Err)
    }
    pub fn is_displayable(&self) -> bool {
        matches!(
            self,
            Ty::Int(_) | Ty::F32 | Ty::F64 | Ty::Bool | Ty::Str | Ty::Err | Ty::Never
        )
    }
    pub fn is_ordered(&self) -> bool {
        self.is_numeric() || matches!(self, Ty::Str | Ty::Bool | Ty::Err)
    }
    pub fn is_hashable(&self) -> bool {
        matches!(self, Ty::Int(_) | Ty::Str | Ty::Bool | Ty::Err)
    }
    pub fn contains_fn(&self) -> bool {
        match self {
            Ty::Fn(..) => true,
            Ty::Vec(a) | Ty::Opt(a) => a.contains_fn(),
            Ty::Map(a, b) | Ty::Res(a, b) => a.contains_fn() || b.contains_fn(),
            _ => false,
        }
    }
    pub fn contains_cap(&self) -> bool {
        match self {
            Ty::Cap(_) => true,
            Ty::Vec(a) | Ty::Opt(a) => a.contains_cap(),
            Ty::Map(a, b) | Ty::Res(a, b) => a.contains_cap() || b.contains_cap(),
            _ => false,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Int(k) => write!(f, "{}", k.name()),
            Ty::F32 => write!(f, "F32"),
            Ty::F64 => write!(f, "F64"),
            Ty::Bool => write!(f, "Bool"),
            Ty::Str => write!(f, "Str"),
            Ty::Unit => write!(f, "Unit"),
            Ty::Never => write!(f, "Never"),
            Ty::Vec(t) => write!(f, "(Vec {t})"),
            Ty::Map(k, v) => write!(f, "(Map {k} {v})"),
            Ty::Opt(t) => write!(f, "(Option {t})"),
            Ty::Res(t, e) => write!(f, "(Result {t} {e})"),
            Ty::User(n) => write!(f, "{n}"),
            Ty::Cap(c) => write!(f, "{}", c.name()),
            Ty::Fn(ps, r) => {
                write!(f, "(Fn")?;
                for p in ps {
                    write!(f, " {p}")?;
                }
                write!(f, " {r})")
            }
            Ty::Err => write!(f, "?"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructInfo {
    pub fields: Vec<(String, Ty)>,
}

#[derive(Clone, Debug)]
pub struct EnumInfo {
    pub variants: Vec<(String, Vec<Ty>)>,
}

#[derive(Clone, Debug)]
pub struct FnSig {
    pub params: Vec<(String, Ty, bool)>,
    pub ret: Ty,
}

pub const NO_VARIANT: usize = usize::MAX;

#[derive(Default, Debug)]
pub struct TypeInfo {
    pub structs: HashMap<String, StructInfo>,
    pub enums: HashMap<String, EnumInfo>,
    pub fns: HashMap<String, FnSig>,
    pub consts: HashMap<String, Ty>,
    /// (type name, variant index or NO_VARIANT for structs, field index) stored behind a Box
    /// because the type is recursive.
    pub boxed: HashSet<(String, usize, usize)>,
    pub copy_types: HashSet<String>,
    /// User types with a total field-by-field ordering (no maps inside).
    pub ord_types: HashSet<String>,
}

impl TypeInfo {
    pub fn is_copy(&self, t: &Ty) -> bool {
        match t {
            Ty::Int(_)
            | Ty::F32
            | Ty::F64
            | Ty::Bool
            | Ty::Unit
            | Ty::Never
            | Ty::Cap(_)
            | Ty::Err => true,
            Ty::Opt(a) => self.is_copy(a),
            Ty::Res(a, b) => self.is_copy(a) && self.is_copy(b),
            Ty::User(n) => self.copy_types.contains(n),
            _ => false,
        }
    }

    /// Values that can be compared with < and sorted.
    pub fn orderable(&self, t: &Ty) -> bool {
        match t {
            Ty::Int(_) | Ty::F32 | Ty::F64 | Ty::Str | Ty::Bool | Ty::Unit | Ty::Err => true,
            Ty::Vec(a) | Ty::Opt(a) => self.orderable(a),
            Ty::Res(a, b) => self.orderable(a) && self.orderable(b),
            Ty::User(n) => self.ord_types.contains(n),
            _ => false,
        }
    }

    pub fn is_boxed(&self, ty: &str, variant: usize, field: usize) -> bool {
        self.boxed.contains(&(ty.to_string(), variant, field))
    }

    pub fn field_types(&self, name: &str) -> Vec<Ty> {
        if let Some(s) = self.structs.get(name) {
            s.fields.iter().map(|f| f.1.clone()).collect()
        } else if let Some(e) = self.enums.get(name) {
            e.variants
                .iter()
                .flat_map(|v| v.1.iter().cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Compute which fields need boxing (recursive types) and which user types are Copy.
    pub fn finalize(&mut self) {
        fn direct_users(t: &Ty, out: &mut Vec<String>) {
            match t {
                Ty::User(n) => out.push(n.clone()),
                Ty::Opt(a) => direct_users(a, out),
                Ty::Res(a, b) => {
                    direct_users(a, out);
                    direct_users(b, out);
                }
                _ => {}
            }
        }
        let names: Vec<String> = self
            .structs
            .keys()
            .chain(self.enums.keys())
            .cloned()
            .collect();
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        for n in &names {
            let mut out = Vec::new();
            for t in self.field_types(n) {
                direct_users(&t, &mut out);
            }
            edges.insert(n.clone(), out);
        }
        let reaches = |from: &str, to: &str| -> bool {
            let mut seen = HashSet::new();
            let mut stack = vec![from.to_string()];
            while let Some(x) = stack.pop() {
                if x == to {
                    return true;
                }
                if seen.insert(x.clone()) {
                    if let Some(es) = edges.get(&x) {
                        stack.extend(es.iter().cloned());
                    }
                }
            }
            false
        };
        let mut slots: Vec<(String, usize, usize, Ty)> = Vec::new();
        for (n, s) in &self.structs {
            for (i, (_, t)) in s.fields.iter().enumerate() {
                slots.push((n.clone(), NO_VARIANT, i, t.clone()));
            }
        }
        for (n, e) in &self.enums {
            for (vi, (_, ts)) in e.variants.iter().enumerate() {
                for (i, t) in ts.iter().enumerate() {
                    slots.push((n.clone(), vi, i, t.clone()));
                }
            }
        }
        let mut boxed = HashSet::new();
        for (n, vi, i, t) in slots {
            let mut out = Vec::new();
            direct_users(&t, &mut out);
            if out.iter().any(|u| reaches(u, &n)) {
                boxed.insert((n, vi, i));
            }
        }
        self.boxed = boxed;
        // Copy: fixpoint, assuming all user types copy then removing violators.
        let mut copy: HashSet<String> = names.iter().cloned().collect();
        for (n, _, _) in &self.boxed {
            copy.remove(n);
        }
        loop {
            self.copy_types = copy.clone();
            let mut changed = false;
            for n in &names {
                if copy.contains(n) && !self.field_types(n).iter().all(|t| self.is_copy(t)) {
                    copy.remove(n);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        self.copy_types = copy;
        let mut ord: HashSet<String> = names.iter().cloned().collect();
        loop {
            self.ord_types = ord.clone();
            let mut changed = false;
            for n in &names {
                if ord.contains(n) && !self.field_types(n).iter().all(|t| self.orderable(t)) {
                    ord.remove(n);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        self.ord_types = ord;
    }
}
