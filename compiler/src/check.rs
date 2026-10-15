//! Type, capability (effect), mutability and aliasing checker.
//! Produces side tables keyed by expression id that the backends consume.

use crate::ast::*;
use crate::diag::{Diag, Edit, Span};
use crate::types::*;
use std::collections::HashMap;

pub type BindId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindKind {
    /// `let` (immutable) or `var` (mutable) local, owned.
    Local { mutable: bool },
    /// Function parameter. Non-copy params are borrowed; mutable params are exclusive borrows.
    Param { mutable: bool },
    /// Element binding of a for-each loop or a pattern binding: a read-only borrow.
    Borrow,
    /// Pattern binding of a boxed (recursive) field: a read-only borrow through a Box.
    BoxBorrow,
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub name: String,
    pub ty: Ty,
    pub kind: BindKind,
    pub loop_depth: u32,
    pub decl_kw: Option<Span>,
    pub param_ty_span: Option<Span>,
}

#[derive(Clone, Debug)]
pub enum Res {
    Local(BindId),
    Const(String),
    Bool(bool),
    Unit,
    None,
    UnitVariant(String, usize),
    UserFn(String),
    Variant(String, usize),
    Builtin(String),
    Nothing,
}

pub struct Checked {
    pub info: TypeInfo,
    pub ty: Vec<Ty>,
    pub res: Vec<Res>,
    pub bindings: Vec<Binding>,
    /// Let/For expr id or pattern span start -> binding.
    pub decl: HashMap<usize, BindId>,
    /// Params of each fn, in order.
    pub fn_params: HashMap<String, Vec<BindId>>,
    /// Binding of `result` in ensures clauses, per fn.
    pub result_binding: HashMap<String, BindId>,
}

pub fn pat_key(span: Span) -> usize {
    // Pattern spans and expr ids live in the same map; offset pattern keys to avoid collisions.
    (1usize << 40) + span.start
}

pub fn check(prog: &Program) -> (Checked, Vec<Diag>) {
    let n = prog.expr_count as usize;
    let mut c = Checker {
        info: TypeInfo::default(),
        ty: vec![Ty::Err; n],
        res: vec![Res::Nothing; n],
        bindings: Vec::new(),
        decl: HashMap::new(),
        fn_params: HashMap::new(),
        result_binding: HashMap::new(),
        diags: Vec::new(),
        scopes: Vec::new(),
        loop_depth: 0,
        ret: Ty::Unit,
        cur_fn: None,
        in_contract: false,
        in_test: false,
        iter_roots: Vec::new(),
        clash_uses: HashMap::new(),
        stmt_ids: std::collections::HashSet::new(),
    };
    c.collect(prog);
    for it in &prog.items {
        match it {
            Item::Fn(f) => c.check_fn(f),
            Item::Test(t) => c.check_test(t),
            Item::Const(k) => {
                let t = c.info.consts.get(&k.name).cloned().unwrap_or(Ty::Err);
                c.scopes = vec![HashMap::new()];
                c.check_expect(&k.value, &t);
                if !(t.is_numeric() || t == Ty::Bool || t.is_err()) {
                    c.diags.push(Diag::error(
                        "E0110",
                        k.ty.span,
                        "const types must be numeric or Bool; use a fn for other values",
                    ));
                }
            }
            _ => {}
        }
    }
    let mut clashes: Vec<(String, Vec<Span>)> = c.clash_uses.drain().collect();
    clashes.sort_by_key(|x| x.1[0].start);
    for (name, spans) in clashes {
        let new = format!("my-{name}");
        let edits = spans
            .iter()
            .map(|s| Edit {
                span: *s,
                text: new.clone(),
            })
            .collect();
        c.diags.push(
            Diag::warning(
                "W0003",
                spans[0],
                &format!(
                    "'{name}' shadows the builtin of the same name; calls use your definition"
                ),
            )
            .with_fix(&format!("rename it to '{new}' everywhere"), edits),
        );
    }
    let checked = Checked {
        info: c.info,
        ty: c.ty,
        res: c.res,
        bindings: c.bindings,
        decl: c.decl,
        fn_params: c.fn_params,
        result_binding: c.result_binding,
    };
    (checked, c.diags)
}

struct FnCtx {
    name: String,
    name_span: Span,
    param_insert_at: usize,
    has_params_clause: bool,
}

struct Checker {
    info: TypeInfo,
    ty: Vec<Ty>,
    res: Vec<Res>,
    bindings: Vec<Binding>,
    decl: HashMap<usize, BindId>,
    fn_params: HashMap<String, Vec<BindId>>,
    result_binding: HashMap<String, BindId>,
    diags: Vec<Diag>,
    scopes: Vec<HashMap<String, BindId>>,
    loop_depth: u32,
    ret: Ty,
    cur_fn: Option<FnCtx>,
    in_contract: bool,
    in_test: bool,
    /// Roots currently being iterated by for-each loops (cannot be mutated inside).
    iter_roots: Vec<(BindId, Vec<String>)>,
    /// User fns whose name clashes with a builtin: definition span and call-site spans.
    clash_uses: HashMap<String, Vec<Span>>,
    /// Expressions whose value is discarded (statement position): their branches may differ in type.
    stmt_ids: std::collections::HashSet<u32>,
}

/// Names that cannot be variables or functions. Clause names (params, body...) are not here:
/// they only mean something in their fixed position inside (fn ...).
const KEYWORDS: &[&str] = &[
    "let", "var", "set", "if", "do", "while", "for", "break", "continue", "return", "match",
    "case", "new", "with", "try", "as", "copy", "mut", "range", "true", "false", "unit", "none",
];

pub const BUILTINS: &[&str] = &[
    "+",
    "-",
    "*",
    "/",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
    "not",
    "bit-and",
    "bit-or",
    "bit-xor",
    "shl",
    "shr",
    "abs",
    "min",
    "max",
    "pow",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "sin",
    "cos",
    "tan",
    "atan2",
    "exp",
    "ln",
    "str",
    "len",
    "is-empty",
    "at",
    "vec",
    "fill",
    "push",
    "pop",
    "last",
    "insert-at",
    "remove-at",
    "swap",
    "clear",
    "extend",
    "slice",
    "concat",
    "contains",
    "index-of",
    "sort",
    "reverse",
    "sum",
    "map",
    "put",
    "get",
    "has",
    "del",
    "keys",
    "values",
    "some",
    "ok",
    "err",
    "unwrap-or",
    "expect",
    "is-some",
    "is-none",
    "is-ok",
    "is-err",
    "substr",
    "split",
    "trim",
    "starts-with",
    "ends-with",
    "replace",
    "upper",
    "lower",
    "join",
    "parse-i64",
    "parse-f64",
    "chars",
    "byte-at",
    "repeat",
    "fmt-float",
    "from-code",
    "print",
    "println",
    "eprintln",
    "read-line",
    "read-file",
    "write-file",
    "now-ms",
    "now-ns",
    "rand-int",
    "rand-f64",
    "args",
    "fail",
    "json-parse",
    "json-str",
    "http-next",
    "http-respond",
    "read-all",
    "lines",
    "is-digit",
    "is-alpha",
    "is-alnum",
    "is-space",
    "is-upper",
    "is-lower",
    "code-of",
    "sort-desc",
    "draw-clear",
    "draw-rect",
    "draw-circle",
    "draw-line",
    "draw-text",
    "assert",
];

fn lev(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1].eq_ignore_ascii_case(&b[j - 1]) {
                0
            } else {
                1
            };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn closest<'a>(name: &str, cands: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in cands {
        let d = lev(name, c);
        let limit = (name.len() / 3).max(1) + if name.eq_ignore_ascii_case(c) { 1 } else { 0 };
        if d <= limit && best.map_or(true, |(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    best.map(|(_, c)| c.to_string())
}

/// Mistakes models make when importing habits from other languages.
fn known_alias(name: &str) -> Option<&'static str> {
    Some(match name {
        "i64" | "int" | "Int" | "isize" | "long" => "I64",
        "i32" | "Int32" => "I32",
        "u8" | "byte" => "U8",
        "u32" => "U32",
        "u64" | "usize" => "U64",
        "f64" | "float" | "Float" | "double" => "F64",
        "f32" => "F32",
        "bool" | "boolean" | "Boolean" => "Bool",
        "String" | "string" | "str" | "&str" => "Str",
        "void" | "()" | "Void" => "Unit",
        "List" | "Array" | "Vector" => "Vec",
        "HashMap" | "Dict" | "Hash" | "BTreeMap" => "Map",
        "Maybe" | "Optional" => "Option",
        "Either" => "Result",
        "null" | "nil" | "None" | "Nothing" => "none",
        "True" => "true",
        "False" => "false",
        "Some" | "Just" => "some",
        "Ok" => "ok",
        "Err" | "Error" => "err",
        "println!" | "puts" | "console.log" | "printf" => "println",
        "length" | "size" | "count" => "len",
        "append" | "push-back" | "add" => "push",
        "panic" | "error" | "throw" | "raise" | "abort" => "fail",
        "concat-str" | "format" | "string-append" | "to-string" | "to-str" => "str",
        "get-at" | "nth" | "index" | "aref" => "at",
        "set!" | "assign" => "set",
        "&&" => "and",
        "||" => "or",
        "!" => "not",
        "=" => "==",
        "<>" => "!=",
        "else" | "elif" | "unless" => "if",
        "loop" => "while",
        "begin" | "progn" | "block" => "do",
        "lambda" | "defun" | "define" | "def" | "func" => "fn",
        "insert" | "assoc" => "put",
        "remove" | "delete" | "dissoc" => "del",
        "contains-key" | "has-key" => "has",
        "unwrap" => "expect",
        _ => return None,
    })
}

impl Checker {
    fn err(&mut self, d: Diag) {
        self.diags.push(d);
    }

    // ---------- declarations ----------

    fn collect(&mut self, prog: &Program) {
        let mut seen: HashMap<String, Span> = HashMap::new();
        for it in &prog.items {
            let (name, span) = match it {
                Item::Fn(f) => (&f.name, f.name_span),
                Item::Struct(s) => (&s.name, s.span),
                Item::Enum(e) => (&e.name, e.span),
                Item::Const(c) => (&c.name, c.span),
                Item::Test(_) => continue,
            };
            if let Some(_prev) = seen.get(name) {
                self.err(Diag::error(
                    "E0104",
                    span,
                    &format!("'{name}' is defined more than once"),
                ));
            }
            seen.insert(name.clone(), span);
            if KEYWORDS.contains(&name.as_str()) {
                self.err(Diag::error(
                    "E0105",
                    span,
                    &format!("'{name}' is a keyword"),
                ));
            } else if BUILTINS.contains(&name.as_str()) {
                self.clash_uses.entry(name.clone()).or_default().push(span);
            }
            if name.contains('.') {
                self.err(Diag::error("E0105", span, "names cannot contain '.'"));
            }
        }
        // First register type names so field types can refer to each other.
        for it in &prog.items {
            match it {
                Item::Struct(s) => {
                    if !s.name.starts_with(|c: char| c.is_ascii_uppercase()) {
                        self.err(Diag::error(
                            "E0106",
                            s.span,
                            "type names start with an uppercase letter",
                        ));
                    }
                    self.info
                        .structs
                        .insert(s.name.clone(), StructInfo { fields: Vec::new() });
                }
                Item::Enum(e) => {
                    if !e.name.starts_with(|c: char| c.is_ascii_uppercase()) {
                        self.err(Diag::error(
                            "E0106",
                            e.span,
                            "type names start with an uppercase letter",
                        ));
                    }
                    self.info.enums.insert(
                        e.name.clone(),
                        EnumInfo {
                            variants: Vec::new(),
                        },
                    );
                }
                _ => {}
            }
        }
        for it in &prog.items {
            match it {
                Item::Struct(s) => {
                    let mut fields = Vec::new();
                    for (fname, te) in &s.fields {
                        if fields.iter().any(|(n, _): &(String, Ty)| n == fname) {
                            self.err(Diag::error(
                                "E0104",
                                te.span,
                                &format!("duplicate field '{fname}'"),
                            ));
                        }
                        let t = self.resolve_type(te);
                        if t.contains_cap() {
                            self.err(Diag::error(
                                "E0304",
                                te.span,
                                "capabilities cannot be stored in data; pass them as parameters",
                            ));
                        }
                        fields.push((fname.clone(), t));
                    }
                    self.info
                        .structs
                        .insert(s.name.clone(), StructInfo { fields });
                }
                Item::Enum(e) => {
                    let mut variants = Vec::new();
                    for (vname, tys, vspan) in &e.variants {
                        if !vname.starts_with(|c: char| c.is_ascii_uppercase()) {
                            self.err(Diag::error(
                                "E0106",
                                *vspan,
                                "variant names start with an uppercase letter",
                            ));
                        }
                        let ts: Vec<Ty> = tys.iter().map(|t| self.resolve_type(t)).collect();
                        if ts.iter().any(|t| t.contains_cap()) {
                            self.err(Diag::error(
                                "E0304",
                                *vspan,
                                "capabilities cannot be stored in data; pass them as parameters",
                            ));
                        }
                        variants.push((vname.clone(), ts));
                    }
                    self.info
                        .enums
                        .insert(e.name.clone(), EnumInfo { variants });
                }
                _ => {}
            }
        }
        for it in &prog.items {
            match it {
                Item::Fn(f) => {
                    let params = f
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), self.resolve_type(&p.ty), p.mutable))
                        .collect::<Vec<_>>();
                    let ret = f
                        .ret
                        .as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(Ty::Unit);
                    if ret.contains_cap() {
                        self.err(Diag::error(
                            "E0304",
                            f.ret.as_ref().unwrap().span,
                            "functions cannot return capabilities",
                        ));
                    }
                    self.info.fns.insert(f.name.clone(), FnSig { params, ret });
                }
                Item::Const(k) => {
                    let t = self.resolve_type(&k.ty);
                    self.info.consts.insert(k.name.clone(), t);
                }
                _ => {}
            }
        }
        self.info.finalize();
        let is_app = self.info.fns.contains_key("init") && self.info.fns.contains_key("update");
        if is_app {
            self.check_app_signatures(prog);
        }
        if !is_app
            && !self.info.fns.contains_key("main")
            && !prog.items.iter().any(|i| matches!(i, Item::Test(_)))
        {
            self.err(
                Diag::error(
                    "E0107",
                    Span::new(0, 0),
                    "program has no (fn main ...) and no tests",
                )
                .note(
                    "entry point: (fn main (params (con Console)) (body (println con \"hello\")))",
                ),
            );
        }
    }

    /// An app (web or game) is init/update plus view (canvas) or page (HTML).
    fn check_app_signatures(&mut self, prog: &Program) {
        let span_of = |n: &str| {
            prog.items.iter().find_map(|i| match i {
                Item::Fn(f) if f.name == n => Some(f.name_span),
                _ => None,
            })
        };
        let model = self.info.fns["init"].ret.clone();
        let only_caps =
            |ps: &[(String, Ty, bool)]| ps.iter().all(|p| matches!(p.1, Ty::Cap(_)) && !p.2);
        let bad = |c: &mut Checker, f: &str, want: String| {
            let sp = span_of(f).unwrap_or_default();
            c.err(Diag::error("E0308", sp, &format!("app function '{f}' must have the signature {want}")).note("an app is (fn init (returns Model)), (fn update (params (m (mut Model)) (e Event))) and (fn view (params (m Model) (g Gfx))) for a canvas or (fn page (params (m Model)) (returns Str)) for HTML; extra capability parameters may follow"));
        };
        if model == Ty::Unit || !only_caps(&self.info.fns["init"].params) {
            bad(self, "init", "(fn init (returns Model) ...)".into());
        }
        let u = self.info.fns["update"].clone();
        let ok_u = u.params.len() >= 2
            && u.params[0].1 == model
            && u.params[0].2
            && u.params[1].1 == Ty::User("Event".into())
            && !u.params[1].2
            && only_caps(&u.params[2..])
            && u.ret == Ty::Unit;
        if !ok_u {
            bad(
                self,
                "update",
                format!("(fn update (params (m (mut {model})) (e Event)) ...)"),
            );
        }
        match (
            self.info.fns.get("view").cloned(),
            self.info.fns.get("page").cloned(),
        ) {
            (None, None) => {
                let sp = span_of("init").unwrap_or_default();
                self.err(Diag::error("E0308", sp, "an app needs (fn view (params (m Model) (g Gfx)) ...) to draw on a canvas or (fn page (params (m Model)) (returns Str) ...) for an HTML page"));
            }
            (Some(v), _) => {
                let ok = v.params.len() >= 2
                    && v.params[0].1 == model
                    && !v.params[0].2
                    && v.params[1].1 == Ty::Cap(Cap::Gfx)
                    && only_caps(&v.params[2..]);
                if !ok {
                    bad(
                        self,
                        "view",
                        format!("(fn view (params (m {model}) (g Gfx)) ...)"),
                    );
                }
            }
            (None, Some(p)) => {
                let ok = !p.params.is_empty()
                    && p.params[0].1 == model
                    && !p.params[0].2
                    && only_caps(&p.params[1..])
                    && p.ret == Ty::Str;
                if !ok {
                    bad(
                        self,
                        "page",
                        format!("(fn page (params (m {model})) (returns Str) ...)"),
                    );
                }
            }
        }
    }

    fn resolve_type(&mut self, te: &TypeExpr) -> Ty {
        let arity = |c: &mut Checker, n: usize, shape: &str| -> bool {
            if te.args.len() != n {
                c.err(Diag::error(
                    "E0108",
                    te.span,
                    &format!("type {} takes {n} argument(s): {shape}", te.name),
                ));
                false
            } else {
                true
            }
        };
        let simple = te.args.is_empty();
        let t = match te.name.as_str() {
            "I8" => Ty::Int(IntK::I8),
            "I16" => Ty::Int(IntK::I16),
            "I32" => Ty::Int(IntK::I32),
            "I64" => Ty::Int(IntK::I64),
            "U8" => Ty::Int(IntK::U8),
            "U16" => Ty::Int(IntK::U16),
            "U32" => Ty::Int(IntK::U32),
            "U64" => Ty::Int(IntK::U64),
            "F32" => Ty::F32,
            "F64" => Ty::F64,
            "Bool" => Ty::Bool,
            "Str" => Ty::Str,
            "Unit" => Ty::Unit,
            "Console" => Ty::Cap(Cap::Console),
            "Fs" => Ty::Cap(Cap::Fs),
            "Clock" => Ty::Cap(Cap::Clock),
            "Rand" => Ty::Cap(Cap::Rand),
            "Env" => Ty::Cap(Cap::Env),
            "Net" => Ty::Cap(Cap::Net),
            "Gfx" => Ty::Cap(Cap::Gfx),
            "Vec" => {
                if !arity(self, 1, "(Vec T)") {
                    return Ty::Err;
                }
                return Ty::Vec(Box::new(self.resolve_type(&te.args[0])));
            }
            "Option" => {
                if !arity(self, 1, "(Option T)") {
                    return Ty::Err;
                }
                return Ty::Opt(Box::new(self.resolve_type(&te.args[0])));
            }
            "Map" => {
                if !arity(self, 2, "(Map K V)") {
                    return Ty::Err;
                }
                let k = self.resolve_type(&te.args[0]);
                let v = self.resolve_type(&te.args[1]);
                if !k.is_hashable() {
                    self.err(Diag::error(
                        "E0109",
                        te.args[0].span,
                        "map keys must be an integer type, Str or Bool",
                    ));
                }
                return Ty::Map(Box::new(k), Box::new(v));
            }
            "Result" => {
                if !arity(self, 2, "(Result T E)") {
                    return Ty::Err;
                }
                let a = self.resolve_type(&te.args[0]);
                let b = self.resolve_type(&te.args[1]);
                return Ty::Res(Box::new(a), Box::new(b));
            }
            n if self.info.structs.contains_key(n) || self.info.enums.contains_key(n) => {
                Ty::User(n.to_string())
            }
            n => {
                let mut d = Diag::error("E0103", te.span, &format!("unknown type '{n}'"));
                let fix = known_alias(n).map(|s| s.to_string()).or_else(|| {
                    let mut names: Vec<&str> = vec![
                        "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "F32", "F64", "Bool",
                        "Str", "Unit", "Vec", "Map", "Option", "Result", "Console", "Fs", "Clock",
                        "Rand", "Env", "Net", "Gfx",
                    ];
                    names.extend(self.info.structs.keys().map(|s| s.as_str()));
                    names.extend(self.info.enums.keys().map(|s| s.as_str()));
                    closest(n, names.into_iter())
                });
                if let Some(f) = fix {
                    let name_span = if simple {
                        te.span
                    } else {
                        Span::new(te.span.start + 1, te.span.start + 1 + n.len())
                    };
                    d = d.with_fix_replace(name_span, &f, &format!("use '{f}'"));
                }
                self.err(d);
                return Ty::Err;
            }
        };
        if !simple {
            self.err(Diag::error(
                "E0108",
                te.span,
                &format!("type {} takes no arguments", te.name),
            ));
        }
        t
    }

    // ---------- scopes ----------

    fn bind(&mut self, b: Binding) -> BindId {
        let id = self.bindings.len();
        let name = b.name.clone();
        self.bindings.push(b);
        self.scopes.last_mut().unwrap().insert(name, id);
        id
    }

    fn lookup(&self, name: &str) -> Option<BindId> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn visible_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.scopes.iter().flat_map(|s| s.keys().cloned()).collect();
        v.extend(self.info.consts.keys().cloned());
        v
    }

    fn local(&self, name: &str, ty: Ty, kind: BindKind) -> Binding {
        Binding {
            name: name.to_string(),
            ty,
            kind,
            loop_depth: self.loop_depth,
            decl_kw: None,
            param_ty_span: None,
        }
    }

    // ---------- fns ----------

    fn check_fn(&mut self, f: &FnDef) {
        let sig = self.info.fns.get(&f.name).cloned().unwrap();
        self.scopes = vec![HashMap::new()];
        self.loop_depth = 0;
        self.ret = sig.ret.clone();
        self.in_test = false;
        self.cur_fn = Some(FnCtx {
            name: f.name.clone(),
            name_span: f.name_span,
            param_insert_at: f.param_insert_at,
            has_params_clause: f.has_params_clause,
        });
        let mut ids = Vec::new();
        for (p, (pname, pty, pmut)) in f.params.iter().zip(sig.params.iter()) {
            if self.lookup(pname).is_some() {
                self.err(Diag::error(
                    "E0104",
                    p.span,
                    &format!("duplicate parameter '{pname}'"),
                ));
            }
            if KEYWORDS.contains(&pname.as_str()) {
                self.err(Diag::error(
                    "E0105",
                    p.span,
                    &format!("'{pname}' is a keyword"),
                ));
            }
            if *pmut && matches!(pty, Ty::Cap(_)) {
                self.err(Diag::error(
                    "E0305",
                    p.span,
                    "capabilities are never mutable parameters",
                ));
            }
            let mut b = self.local(pname, pty.clone(), BindKind::Param { mutable: *pmut });
            b.param_ty_span = Some(p.ty.span);
            ids.push(self.bind(b));
        }
        self.fn_params.insert(f.name.clone(), ids);
        if f.name == "main" {
            for (p, (_, t, _)) in f.params.iter().zip(sig.params.iter()) {
                if !matches!(t, Ty::Cap(_) | Ty::Err) {
                    self.err(Diag::error(
                        "E0306",
                        p.span,
                        "main only receives capabilities: Console, Fs, Clock, Rand, Env, Net (use (args env) for command-line arguments)",
                    ));
                }
            }
            let ok_ret = matches!(&sig.ret, Ty::Unit | Ty::Err)
                || sig.ret == Ty::Res(Box::new(Ty::Unit), Box::new(Ty::Str));
            if !ok_ret {
                self.err(Diag::error(
                    "E0307",
                    f.ret.as_ref().map(|t| t.span).unwrap_or(f.name_span),
                    "main returns Unit or (Result Unit Str)",
                ));
            }
        }
        self.in_contract = true;
        for r in &f.requires {
            self.check_expect(r, &Ty::Bool);
        }
        if !f.ensures.is_empty() {
            self.scopes.push(HashMap::new());
            let rb = self.local("result", sig.ret.clone(), BindKind::Borrow);
            let id = self.bind(rb);
            self.result_binding.insert(f.name.clone(), id);
            for e in &f.ensures {
                self.check_expect(e, &Ty::Bool);
            }
            self.scopes.pop();
        }
        self.in_contract = false;
        let ret = sig.ret.clone();
        self.check_block_expect(&f.body, &ret, f.span);
        self.cur_fn = None;
    }

    fn check_test(&mut self, t: &TestDef) {
        self.scopes = vec![HashMap::new()];
        self.loop_depth = 0;
        self.ret = Ty::Unit;
        self.in_test = true;
        self.cur_fn = None;
        self.check_block(&t.body, None);
        self.in_test = false;
    }

    // ---------- expressions ----------

    fn set_ty(&mut self, e: &Expr, t: Ty) -> Ty {
        self.ty[e.id as usize] = t.clone();
        t
    }

    fn compat(found: &Ty, expected: &Ty) -> bool {
        found == expected
            || matches!(found, Ty::Never | Ty::Err)
            || expected.is_err()
            || Self::compat_inner(found, expected)
    }

    fn compat_inner(a: &Ty, b: &Ty) -> bool {
        match (a, b) {
            (Ty::Vec(x), Ty::Vec(y)) | (Ty::Opt(x), Ty::Opt(y)) => Self::compat(x, y),
            (Ty::Map(x1, x2), Ty::Map(y1, y2)) | (Ty::Res(x1, x2), Ty::Res(y1, y2)) => {
                Self::compat(x1, y1) && Self::compat(x2, y2)
            }
            _ => false,
        }
    }

    fn check_expect(&mut self, e: &Expr, expected: &Ty) -> Ty {
        let t = self.check(e, Some(expected));
        if !Self::compat(&t, expected) {
            self.mismatch(e, expected, &t);
        }
        t
    }

    fn mismatch(&mut self, e: &Expr, expected: &Ty, found: &Ty) {
        let mut d = Diag::error(
            "E0201",
            e.span,
            &format!("type mismatch: expected {expected}, found {found}"),
        )
        .with_expected(expected.to_string(), found.to_string());
        if found.is_numeric() && expected.is_numeric() {
            d = d.with_fix_replace(
                e.span,
                &format!("(as {expected} {})", "\u{0}"),
                "convert with (as ...)",
            );
            // Fill in the original source text later in apply: we cannot see the source here, so use a sentinel.
        } else if *expected == Ty::Str && found.is_displayable() {
            d = d.with_fix_replace(
                e.span,
                &format!("(str {})", "\u{0}"),
                "convert to Str with (str ...)",
            );
        } else if let Ty::Opt(inner) = expected {
            if Self::compat(found, inner) {
                d = d.with_fix_replace(
                    e.span,
                    &format!("(some {})", "\u{0}"),
                    "wrap in (some ...)",
                );
            }
        } else if let Ty::Res(ok, _) = expected {
            if Self::compat(found, ok) {
                d = d.with_fix_replace(e.span, &format!("(ok {})", "\u{0}"), "wrap in (ok ...)");
            }
        }
        if let (Ty::Res(ok, _), true) = (found, !matches!(expected, Ty::Res(..))) {
            if Self::compat(ok, expected) && matches!(self.ret, Ty::Res(..)) {
                d = d.with_fix_replace(
                    e.span,
                    &format!("(try {})", "\u{0}"),
                    "propagate the error with (try ...)",
                );
            }
        }
        self.err(d);
    }

    fn check_block(&mut self, es: &[Expr], expected: Option<&Ty>) -> Ty {
        self.scopes.push(HashMap::new());
        let mut t = Ty::Unit;
        for (i, e) in es.iter().enumerate() {
            let last = i + 1 == es.len();
            if !last || expected == Some(&Ty::Unit) {
                self.stmt_ids.insert(e.id);
            }
            t = self.check(e, if last { expected } else { None });
            if !last
                && matches!(
                    e.kind,
                    ExprKind::Var(_) | ExprKind::Int(_) | ExprKind::Str(_) | ExprKind::Float(_)
                )
            {
                self.err(
                    Diag::warning("W0001", e.span, "this value is computed and discarded")
                        .with_fix_replace(e.span, "", "remove it"),
                );
            }
        }
        if let Some(last) = es.last() {
            if matches!(last.kind, ExprKind::Let { .. }) {
                t = Ty::Unit;
            }
        }
        self.scopes.pop();
        t
    }

    fn check_block_expect(&mut self, es: &[Expr], expected: &Ty, span: Span) -> Ty {
        let t = self.check_block(es, Some(expected));
        if *expected == Ty::Unit {
            return Ty::Unit;
        }
        if !Self::compat(&t, expected) {
            match es.last() {
                Some(last) if !matches!(last.kind, ExprKind::Let { .. }) => {
                    self.mismatch(last, expected, &t)
                }
                _ => {
                    let at = es.last().map(|l| l.span).unwrap_or(span);
                    self.err(
                        Diag::error(
                            "E0202",
                            at,
                            &format!("block must end with a value of type {expected}"),
                        )
                        .with_expected(expected.to_string(), t.to_string()),
                    )
                }
            }
        }
        t
    }

    fn check(&mut self, e: &Expr, exp: Option<&Ty>) -> Ty {
        let t = self.check_inner(e, exp);
        self.set_ty(e, t)
    }

    fn int_lit(&mut self, e: &Expr, v: i128, exp: Option<&Ty>) -> Ty {
        let t = match exp {
            Some(Ty::Int(k)) => Ty::Int(*k),
            Some(Ty::F64) => return Ty::F64,
            Some(Ty::F32) => return Ty::F32,
            _ => Ty::Int(IntK::I64),
        };
        if let Ty::Int(k) = t {
            let (lo, hi) = k.range();
            if v < lo || v > hi {
                self.err(Diag::error(
                    "E0203",
                    e.span,
                    &format!("literal {v} is out of range for {}", k.name()),
                ));
            }
        }
        t
    }

    fn check_inner(&mut self, e: &Expr, exp: Option<&Ty>) -> Ty {
        let id = e.id as usize;
        match &e.kind {
            ExprKind::Int(v) => self.int_lit(e, *v, exp),
            ExprKind::Float(_) => match exp {
                Some(Ty::F32) => Ty::F32,
                _ => Ty::F64,
            },
            ExprKind::Str(_) => Ty::Str,
            ExprKind::Var(name) => self.check_var(e, name, exp),
            ExprKind::Let {
                name,
                name_span,
                kw_span,
                mutable,
                ty,
                value,
            } => {
                let declared = ty.as_ref().map(|t| self.resolve_type(t));
                let vt = match &declared {
                    Some(d) => {
                        self.check_expect(value, d);
                        d.clone()
                    }
                    None => self.check(value, None),
                };
                if KEYWORDS.contains(&name.as_str()) {
                    self.err(Diag::error(
                        "E0105",
                        *name_span,
                        &format!("'{name}' is a keyword and cannot be a variable name"),
                    ));
                }
                if name.contains('.') {
                    self.err(Diag::error(
                        "E0105",
                        *name_span,
                        "variable names cannot contain '.'",
                    ));
                }
                if matches!(vt, Ty::Cap(_)) {
                    self.err(Diag::error(
                        "E0304",
                        e.span,
                        "capabilities cannot be rebound; use the parameter directly",
                    ));
                }
                let vt = if vt == Ty::Never { Ty::Err } else { vt };
                let mut b = self.local(name, vt, BindKind::Local { mutable: *mutable });
                b.decl_kw = Some(*kw_span);
                let bid = self.bind(b);
                self.decl.insert(id, bid);
                Ty::Unit
            }
            ExprKind::Set(place, value) => {
                let pt = self.check(place, None);
                self.require_mutable_place(place, "assign to");
                self.check_expect(value, &pt);
                Ty::Unit
            }
            ExprKind::If(c, t, f) => {
                self.check_expect(c, &Ty::Bool);
                let discard = self.stmt_ids.contains(&e.id) || exp == Some(&Ty::Unit);
                if discard {
                    self.stmt_ids.insert(t.id);
                    self.check(t, None);
                    if let Some(f) = f {
                        self.stmt_ids.insert(f.id);
                        self.check(f, None);
                    }
                    return Ty::Unit;
                }
                match f {
                    Some(f) => {
                        let tt = self.check(t, exp);
                        let want = if tt == Ty::Never || tt.is_err() {
                            exp.cloned()
                        } else {
                            Some(tt.clone())
                        };
                        let ft = match &want {
                            Some(w) => self.check_expect(f, w),
                            None => self.check(f, None),
                        };
                        if tt == Ty::Never { ft } else { tt }
                    }
                    None => {
                        let tt = self.check(t, None);
                        if !matches!(tt, Ty::Unit | Ty::Never | Ty::Err)
                            && exp.is_some_and(|x| *x != Ty::Unit)
                        {
                            self.err(
                                Diag::error("E0204", e.span, "(if c then) without else has type Unit; add an else branch to produce a value")
                                    .with_fix_insert(e.span.end - 1, " ELSE-VALUE", "add an else branch"),
                            );
                        }
                        Ty::Unit
                    }
                }
            }
            ExprKind::Do(es) => {
                if self.stmt_ids.contains(&e.id) || exp == Some(&Ty::Unit) {
                    self.check_block(es, Some(&Ty::Unit));
                    Ty::Unit
                } else {
                    self.check_block(es, exp)
                }
            }
            ExprKind::While(c, body) => {
                self.check_expect(c, &Ty::Bool);
                self.loop_depth += 1;
                self.check_block(body, None);
                self.loop_depth -= 1;
                Ty::Unit
            }
            ExprKind::For(var, _vspan, iter, body) => {
                self.scopes.push(HashMap::new());
                let (elem, kind, root) = match iter {
                    ForIter::Range(a, b) => {
                        let at = self.check_same(&[&**a, &**b], Some(&Ty::Int(IntK::I64)));
                        if !at.is_int() && !at.is_err() {
                            self.err(Diag::error(
                                "E0205",
                                a.span,
                                &format!("range bounds must be integers, found {at}"),
                            ));
                        }
                        (at, BindKind::Local { mutable: false }, None)
                    }
                    ForIter::Each(v) => {
                        let vt = self.check(v, None);
                        let elem = match &vt {
                            Ty::Vec(t) => (**t).clone(),
                            Ty::Err => Ty::Err,
                            Ty::Map(..) => {
                                self.err(
                                    Diag::error(
                                        "E0206",
                                        v.span,
                                        "iterate a map with (for k (keys m) ...)",
                                    )
                                    .with_fix_replace(
                                        v.span,
                                        &format!("(keys {})", "\u{0}"),
                                        "iterate the sorted keys",
                                    ),
                                );
                                Ty::Err
                            }
                            Ty::Str => {
                                self.err(
                                    Diag::error(
                                        "E0206",
                                        v.span,
                                        "iterate a string with (for c (chars s) ...)",
                                    )
                                    .with_fix_replace(
                                        v.span,
                                        &format!("(chars {})", "\u{0}"),
                                        "iterate characters",
                                    ),
                                );
                                Ty::Err
                            }
                            other => {
                                self.err(Diag::error(
                                    "E0206",
                                    v.span,
                                    &format!(
                                        "cannot iterate over {other}; expected a Vec or (range a b)"
                                    ),
                                ));
                                Ty::Err
                            }
                        };
                        (elem, BindKind::Borrow, self.place_path(v))
                    }
                };
                self.loop_depth += 1;
                let b = self.local(var, elem, kind);
                let bid = self.bind(b);
                self.decl.insert(id, bid);
                let pushed = root.is_some();
                if let Some(r) = root {
                    self.iter_roots.push(r);
                }
                self.check_block(body, None);
                if pushed {
                    self.iter_roots.pop();
                }
                self.loop_depth -= 1;
                self.scopes.pop();
                Ty::Unit
            }
            ExprKind::Break | ExprKind::Continue => {
                if self.loop_depth == 0 {
                    self.err(Diag::error(
                        "E0207",
                        e.span,
                        "break/continue outside of a loop",
                    ));
                }
                Ty::Never
            }
            ExprKind::Return(v) => {
                if self.in_contract {
                    self.err(Diag::error(
                        "E0601",
                        e.span,
                        "return is not allowed in contracts",
                    ));
                }
                let ret = self.ret.clone();
                match v {
                    Some(v) => {
                        self.check_expect(v, &ret);
                    }
                    None => {
                        if !matches!(ret, Ty::Unit | Ty::Err) {
                            self.err(Diag::error(
                                "E0201",
                                e.span,
                                &format!("return needs a value of type {ret}"),
                            ));
                        }
                    }
                }
                Ty::Never
            }
            ExprKind::Match(scrut, arms) => self.check_match(e, scrut, arms, exp),
            ExprKind::New(name, nspan, fields) => {
                let Some(si) = self.info.structs.get(name).cloned() else {
                    let mut d = Diag::error("E0110", *nspan, &format!("unknown struct '{name}'"));
                    if let Some(c) = closest(name, self.info.structs.keys().map(|s| s.as_str())) {
                        d = d.with_fix_replace(*nspan, &c, &format!("use '{c}'"));
                    }
                    if self.info.enums.contains_key(name) {
                        d = d.note(format!(
                            "'{name}' is an enum; build it with (EnumName.Variant args...)"
                        ));
                    }
                    self.err(d);
                    for (_, _, v) in fields {
                        self.check(v, None);
                    }
                    return Ty::Err;
                };
                self.check_fields(e, name, &si, fields, true);
                Ty::User(name.clone())
            }
            ExprKind::With(base, fields) => {
                let bt = self.check(base, exp);
                let Ty::User(name) = &bt else {
                    if !bt.is_err() {
                        self.err(Diag::error(
                            "E0111",
                            base.span,
                            &format!("(with ...) needs a struct value, found {bt}"),
                        ));
                    }
                    return Ty::Err;
                };
                let Some(si) = self.info.structs.get(name).cloned() else {
                    self.err(Diag::error(
                        "E0111",
                        base.span,
                        &format!("(with ...) needs a struct value, found enum {bt}"),
                    ));
                    return Ty::Err;
                };
                self.check_fields(e, name, &si, fields, false);
                bt
            }
            ExprKind::Field(base, fname, fspan) => {
                let bt = self.check(base, None);
                match &bt {
                    Ty::User(n) if self.info.structs.contains_key(n) => {
                        let si = &self.info.structs[n];
                        match si.fields.iter().find(|f| &f.0 == fname) {
                            Some((_, t)) => t.clone(),
                            None => {
                                let mut d = Diag::error(
                                    "E0112",
                                    *fspan,
                                    &format!("struct {n} has no field '{fname}'"),
                                );
                                let names: Vec<&str> =
                                    si.fields.iter().map(|f| f.0.as_str()).collect();
                                if let Some(c) = closest(fname, names.iter().copied()) {
                                    d = d.with_fix_replace(*fspan, &c, &format!("use '{c}'"));
                                }
                                d = d.note(format!("fields of {n}: {}", names.join(", ")));
                                self.err(d);
                                Ty::Err
                            }
                        }
                    }
                    Ty::Err => Ty::Err,
                    other => {
                        self.err(Diag::error(
                            "E0112",
                            e.span,
                            &format!("field access on {other}, which is not a struct"),
                        ));
                        Ty::Err
                    }
                }
            }
            ExprKind::Try(inner) => {
                let it = self.check(inner, None);
                let ret = self.ret.clone();
                match (&it, &ret) {
                    (Ty::Res(ok, er), Ty::Res(_, rer)) => {
                        if !Self::compat(er, rer) {
                            self.err(
                                Diag::error("E0208", e.span, &format!("try propagates error type {er} but the function returns errors of type {rer}"))
                                    .with_expected(rer.to_string(), er.to_string())
                                    .note("convert the error explicitly with match"),
                            );
                        }
                        (**ok).clone()
                    }
                    (Ty::Opt(t), Ty::Opt(_)) => (**t).clone(),
                    (Ty::Err, _) => Ty::Err,
                    (Ty::Res(..), _) | (Ty::Opt(_), _) => {
                        self.err(
                            Diag::error("E0208", e.span, &format!("try needs the function to return a Result/Option, but it returns {ret}"))
                                .note("handle the value with (match ...) or (unwrap-or ...) instead"),
                        );
                        Ty::Err
                    }
                    (other, _) => {
                        self.err(Diag::error(
                            "E0208",
                            e.span,
                            &format!("try needs a Result or Option, found {other}"),
                        ));
                        Ty::Err
                    }
                }
            }
            ExprKind::As(te, inner) => {
                let target = self.resolve_type(te);
                // The operand keeps its own type: (as F64 (% n 1000)) is integer % then a conversion.
                let it = self.check(inner, None);
                if !(it.is_numeric() || it.is_err()) || !(target.is_numeric() || target.is_err()) {
                    if !(it.is_err() || target.is_err()) {
                        let mut d = Diag::error(
                            "E0209",
                            e.span,
                            &format!(
                                "(as ...) converts between numeric types; cannot convert {it} to {target}"
                            ),
                        );
                        if target == Ty::Str {
                            d = d.with_fix_replace(
                                e.span,
                                "(str VALUE)",
                                "use (str x) to make a string",
                            );
                        }
                        if it == Ty::Str && target.is_numeric() {
                            d = d.note("parse strings with (parse-i64 s) or (parse-f64 s), which return Option");
                        }
                        self.err(d);
                    }
                }
                target
            }
            ExprKind::Copy(inner) => self.check(inner, exp),
            ExprKind::Call(name, nspan, args) => self.check_call(e, name, *nspan, args, exp),
        }
    }

    fn check_fields(
        &mut self,
        e: &Expr,
        name: &str,
        si: &StructInfo,
        fields: &[(String, Span, Expr)],
        all_required: bool,
    ) {
        let mut seen = Vec::new();
        for (fname, fspan, v) in fields {
            if seen.contains(fname) {
                self.err(Diag::error(
                    "E0113",
                    *fspan,
                    &format!("field '{fname}' given twice"),
                ));
            }
            seen.push(fname.clone());
            match si.fields.iter().find(|f| &f.0 == fname) {
                Some((_, t)) => {
                    let t = t.clone();
                    self.check_expect(v, &t);
                }
                None => {
                    let mut d = Diag::error(
                        "E0112",
                        *fspan,
                        &format!("struct {name} has no field '{fname}'"),
                    );
                    if let Some(c) = closest(fname, si.fields.iter().map(|f| f.0.as_str())) {
                        d = d.with_fix_replace(*fspan, &c, &format!("use '{c}'"));
                    }
                    self.err(d);
                    self.check(v, None);
                }
            }
        }
        if all_required {
            let missing: Vec<&(String, Ty)> =
                si.fields.iter().filter(|f| !seen.contains(&f.0)).collect();
            if !missing.is_empty() {
                let names: Vec<String> =
                    missing.iter().map(|f| format!("{} {}", f.0, f.1)).collect();
                let ins: String = missing
                    .iter()
                    .map(|f| format!(" ({} {})", f.0, placeholder(&f.1)))
                    .collect();
                self.err(
                    Diag::error(
                        "E0114",
                        e.span,
                        &format!("missing fields of {name}: {}", names.join(", ")),
                    )
                    .with_fix_insert(
                        e.span.end - 1,
                        &ins,
                        "add the missing fields with placeholder values",
                    ),
                );
            }
        }
    }

    fn check_var(&mut self, e: &Expr, name: &str, exp: Option<&Ty>) -> Ty {
        let id = e.id as usize;
        if let Some(b) = self.lookup(name) {
            self.res[id] = Res::Local(b);
            let t = self.bindings[b].ty.clone();
            if self.in_contract && matches!(t, Ty::Cap(_)) {
                self.err(Diag::error(
                    "E0303",
                    e.span,
                    "contracts must be pure: capabilities cannot be used here",
                ));
            }
            return t;
        }
        match name {
            "true" | "false" => {
                self.res[id] = Res::Bool(name == "true");
                return Ty::Bool;
            }
            "unit" => {
                self.res[id] = Res::Unit;
                return Ty::Unit;
            }
            "none" => {
                self.res[id] = Res::None;
                return match exp {
                    Some(t @ Ty::Opt(_)) => t.clone(),
                    Some(Ty::Err) | None => {
                        self.err(
                            Diag::error("E0210", e.span, "cannot infer the type of 'none' here")
                                .note("annotate the binding: (let x (Option I64) none)"),
                        );
                        Ty::Err
                    }
                    Some(other) => {
                        self.err(
                            Diag::error(
                                "E0201",
                                e.span,
                                &format!("'none' is an Option, but {other} is expected"),
                            )
                            .with_expected(other.to_string(), "(Option ?)"),
                        );
                        Ty::Err
                    }
                };
            }
            _ => {}
        }
        if let Some(t) = self.info.consts.get(name) {
            self.res[id] = Res::Const(name.to_string());
            return t.clone();
        }
        if let Some((en, vn)) = name.split_once('.') {
            if let Some(ei) = self.info.enums.get(en) {
                if let Some(vi) = ei.variants.iter().position(|v| v.0 == vn) {
                    if !ei.variants[vi].1.is_empty() {
                        self.err(Diag::error(
                            "E0115",
                            e.span,
                            &format!(
                                "variant {name} has {} field(s); construct it as ({name} ...)",
                                ei.variants[vi].1.len()
                            ),
                        ));
                        return Ty::Err;
                    }
                    self.res[id] = Res::UnitVariant(en.to_string(), vi);
                    return Ty::User(en.to_string());
                }
                return self.unknown_variant(e.span, en, vn);
            }
        }
        if self.info.fns.contains_key(name) || BUILTINS.contains(&name) {
            self.err(
                Diag::error(
                    "E0116",
                    e.span,
                    &format!("'{name}' is a function; call it as ({name} ...)"),
                )
                .note("functions are not values"),
            );
            return Ty::Err;
        }
        let mut d = Diag::error("E0101", e.span, &format!("unknown name '{name}'"));
        let visible = self.visible_names();
        let fix = known_alias(name)
            .map(|s| s.to_string())
            .or_else(|| closest(name, visible.iter().map(|s| s.as_str())));
        if let Some(f) = fix {
            d = d.with_fix_replace(e.span, &f, &format!("use '{f}'"));
        }
        if let Some(Ty::User(en)) = exp {
            if let Some(ei) = self.info.enums.get(en) {
                if let Some(v) = ei.variants.iter().find(|v| v.0 == name) {
                    d = Diag::error(
                        "E0101",
                        e.span,
                        &format!("unknown name '{name}'; enum variants are qualified"),
                    )
                    .with_fix_replace(
                        e.span,
                        &format!("{en}.{}", v.0),
                        &format!("use '{en}.{}'", v.0),
                    );
                }
            }
        }
        self.err(d);
        Ty::Err
    }

    fn unknown_variant(&mut self, span: Span, en: &str, vn: &str) -> Ty {
        let ei = &self.info.enums[en];
        let names: Vec<&str> = ei.variants.iter().map(|v| v.0.as_str()).collect();
        let mut d = Diag::error("E0117", span, &format!("enum {en} has no variant '{vn}'"))
            .note(format!("variants of {en}: {}", names.join(", ")));
        if let Some(c) = closest(vn, names.iter().copied()) {
            d = d.with_fix_replace(span, &format!("{en}.{c}"), &format!("use '{en}.{c}'"));
        }
        self.err(d);
        Ty::Err
    }

    // ---------- places, mutability, aliasing ----------

    /// Root binding of a place expression (variable, field chain, (at v i) chain). None if not a place.
    fn place_root(&self, e: &Expr) -> Option<BindId> {
        match &e.kind {
            ExprKind::Var(_) => match self.res[e.id as usize] {
                Res::Local(b) => Some(b),
                _ => None,
            },
            ExprKind::Field(b, _, _) => self.place_root(b),
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 => {
                self.place_root(args[0].expr())
            }
            _ => None,
        }
    }

    /// Root binding plus field path of a place; `[]` stands for any element of a Vec.
    fn place_path(&self, e: &Expr) -> Option<(BindId, Vec<String>)> {
        match &e.kind {
            ExprKind::Var(_) => match self.res[e.id as usize] {
                Res::Local(b) => Some((b, Vec::new())),
                _ => None,
            },
            ExprKind::Field(b, f, _) => {
                let (r, mut p) = self.place_path(b)?;
                p.push(f.clone());
                Some((r, p))
            }
            ExprKind::Call(n, _, args) if n == "at" && args.len() == 2 => {
                let (r, mut p) = self.place_path(args[0].expr())?;
                p.push("[]".into());
                Some((r, p))
            }
            _ => None,
        }
    }

    fn require_mutable_place(&mut self, place: &Expr, what: &str) -> Option<BindId> {
        let Some(root) = self.place_root(place) else {
            if !self.ty[place.id as usize].is_err() {
                self.err(Diag::error(
                    "E0402",
                    place.span,
                    &format!("cannot {what} this expression: only variables, fields (. x f) and elements (at v i) are places"),
                ));
            }
            return None;
        };
        let b = self.bindings[root].clone();
        match b.kind {
            BindKind::Local { mutable: true } | BindKind::Param { mutable: true } => {}
            BindKind::Local { mutable: false } => {
                let mut d = Diag::error(
                    "E0401",
                    place.span,
                    &format!("cannot {what} '{}': it was declared with let", b.name),
                );
                if let Some(kw) = b.decl_kw {
                    d = d.with_fix_replace(kw, "var", "declare it with var");
                }
                self.err(d);
            }
            BindKind::Param { mutable: false } => {
                let mut d = Diag::error(
                    "E0401",
                    place.span,
                    &format!("cannot {what} parameter '{}': it is read-only", b.name),
                );
                if let Some(ts) = b.param_ty_span {
                    d = d.with_fix(
                        "make the parameter mutable (callers must then pass (mut x))",
                        vec![
                            Edit {
                                span: Span::new(ts.start, ts.start),
                                text: "(mut ".into(),
                            },
                            Edit {
                                span: Span::new(ts.end, ts.end),
                                text: ")".into(),
                            },
                        ],
                    );
                }
                d = d.note("or work on a local copy: (var tmp (copy x))");
                self.err(d);
            }
            BindKind::Borrow | BindKind::BoxBorrow => {
                self.err(
                    Diag::error(
                        "E0401",
                        place.span,
                        &format!(
                            "cannot {what} '{}': loop and pattern bindings are read-only",
                            b.name
                        ),
                    )
                    .note("copy it into a var first: (var tmp (copy x))"),
                );
            }
        }
        let path = self.place_path(place).map(|p| p.1).unwrap_or_default();
        if self
            .iter_roots
            .iter()
            .any(|(r, p)| *r == root && paths_overlap(p, &path))
        {
            self.err(
                Diag::error(
                    "E0403",
                    place.span,
                    &format!("cannot {what} '{}' while iterating over it", b.name),
                )
                .note("iterate by index instead: (for i (range 0 (len v)) ...)"),
            );
        }
        Some(root)
    }

    fn mentions(&self, e: &Expr, root: BindId) -> bool {
        let mut found = false;
        walk(e, &mut |x| {
            if let ExprKind::Var(_) = x.kind {
                if let Res::Local(b) = self.res[x.id as usize] {
                    if b == root {
                        found = true;
                    }
                }
            }
        });
        found
    }

    // ---------- calls ----------

    fn arg_val<'a>(&mut self, a: &'a Arg) -> &'a Expr {
        if let Arg::Mut(e, span) = a {
            let inner_span = e.span;
            self.err(
                Diag::error(
                    "E0405",
                    *span,
                    "this argument is not mutated; pass it without (mut ...)",
                )
                .with_fix(
                    "remove (mut ...)",
                    vec![
                        Edit {
                            span: Span::new(span.start, inner_span.start),
                            text: String::new(),
                        },
                        Edit {
                            span: Span::new(inner_span.end, span.end),
                            text: String::new(),
                        },
                    ],
                ),
            );
        }
        a.expr()
    }

    fn arg_mut(&mut self, a: &Arg) -> (Ty, Option<BindId>) {
        match a {
            Arg::Mut(e, _) => {
                let t = self.check(e, None);
                let r = self.require_mutable_place(e, "mutate");
                (t, r)
            }
            Arg::Val(e) => {
                let t = self.check(e, None);
                let mut d = Diag::error(
                    "E0406",
                    e.span,
                    "this argument is mutated by the call; mark it with (mut ...)",
                );
                d = d.with_fix(
                    "wrap it in (mut ...)",
                    vec![
                        Edit {
                            span: Span::new(e.span.start, e.span.start),
                            text: "(mut ".into(),
                        },
                        Edit {
                            span: Span::new(e.span.end, e.span.end),
                            text: ")".into(),
                        },
                    ],
                );
                self.err(d);
                let r = self.place_root(e);
                (t, r)
            }
        }
    }

    fn cap_fix(&mut self, cap: Cap, head_span: Span, call_span: Span) -> Diag {
        let mut d = Diag::error(
            "E0301",
            call_span,
            &format!(
                "this call needs a {} capability as its first argument",
                cap.name()
            ),
        );
        let existing = self
            .scopes
            .iter()
            .rev()
            .flat_map(|s| s.iter())
            .find(|(_, b)| self.bindings[**b].ty == Ty::Cap(cap))
            .map(|(n, _)| n.clone());
        if let Some(n) = existing {
            return d.with_fix_insert(head_span.end, &format!(" {n}"), &format!("pass '{n}'"));
        }
        if self.in_test {
            return d
                .note("tests are pure and receive no capabilities; test the pure logic instead");
        }
        let pname = cap.default_param();
        if let Some(f) = &self.cur_fn {
            let param_edit = if f.has_params_clause {
                Edit {
                    span: Span::new(f.param_insert_at, f.param_insert_at),
                    text: format!(" ({pname} {})", cap.name()),
                }
            } else {
                Edit {
                    span: Span::new(f.name_span.end, f.name_span.end),
                    text: format!(" (params ({pname} {}))", cap.name()),
                }
            };
            d = d
                .note(format!("function '{}' has no {} parameter; effects are explicit capabilities passed down from main", f.name, cap.name()))
                .with_fix(
                    &format!("add a ({pname} {}) parameter and pass it", cap.name()),
                    vec![param_edit, Edit { span: Span::new(head_span.end, head_span.end), text: format!(" {pname}") }],
                );
        }
        d
    }

    /// Returns remaining args after the capability, or None if the cap is missing (error reported).
    fn need_cap<'a>(
        &mut self,
        cap: Cap,
        args: &'a [Arg],
        head_span: Span,
        call_span: Span,
    ) -> Option<&'a [Arg]> {
        if let Some(first) = args.first() {
            let e = first.expr();
            let is_cap_var = match &e.kind {
                ExprKind::Var(n) => {
                    self.lookup(n).map(|b| self.bindings[b].ty.clone()) == Some(Ty::Cap(cap))
                }
                _ => false,
            };
            if is_cap_var {
                let t = self.check(e, None);
                debug_assert!(t == Ty::Cap(cap));
                return Some(&args[1..]);
            }
            if let ExprKind::Var(n) = &e.kind {
                if let Some(b) = self.lookup(n) {
                    if let Ty::Cap(other) = self.bindings[b].ty {
                        self.check(e, None);
                        self.err(Diag::error(
                            "E0302",
                            e.span,
                            &format!(
                                "expected a {} capability, '{n}' is a {}",
                                cap.name(),
                                other.name()
                            ),
                        ));
                        return None;
                    }
                }
            }
        }
        let d = self.cap_fix(cap, head_span, call_span);
        self.err(d);
        for a in args {
            self.check(a.expr(), None);
        }
        None
    }

    fn arity(&mut self, name: &str, args: &[Arg], n: usize, span: Span, shape: &str) -> bool {
        if args.len() != n {
            self.err(Diag::error(
                "E0211",
                span,
                &format!(
                    "'{name}' takes {n} argument(s): {shape}, got {}",
                    args.len()
                ),
            ));
            for a in args {
                self.check(a.expr(), None);
            }
            return false;
        }
        true
    }

    fn is_literal(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) => true,
            ExprKind::Call(n, _, a) if n == "-" && a.len() == 1 => Self::is_literal(a[0].expr()),
            _ => false,
        }
    }

    /// Check expressions that must share one type; non-literals decide the type for literals.
    fn check_same(&mut self, es: &[&Expr], exp: Option<&Ty>) -> Ty {
        let mut anchor: Option<Ty> = None;
        let mut done = vec![false; es.len()];
        if let Some(t) = exp {
            if t.is_numeric() {
                anchor = Some(t.clone());
            }
        }
        if anchor.is_none() {
            for (i, e) in es.iter().enumerate() {
                if !Self::is_literal(e) {
                    let t = self.check(e, None);
                    done[i] = true;
                    if !t.is_err() && t != Ty::Never {
                        anchor = Some(t);
                        break;
                    }
                }
            }
        }
        if let Some(a) = &anchor {
            if a.is_int()
                && exp.map_or(true, |t| !t.is_numeric())
                && es.iter().any(|e| has_float_lit(e))
            {
                if let Some((i, x)) = es
                    .iter()
                    .enumerate()
                    .find(|(i, x)| done[*i] && !Self::is_literal(x))
                {
                    let _ = i;
                    self.err(
                        Diag::error(
                            "E0201",
                            x.span,
                            &format!("this is {a} but it is combined with a float literal"),
                        )
                        .with_expected("F64", a.to_string())
                        .with_fix_replace(
                            x.span,
                            &format!("(as F64 {})", "\u{0}"),
                            "convert it with (as F64 ...)",
                        ),
                    );
                    for (j, y) in es.iter().enumerate() {
                        if !done[j] {
                            self.check(y, None);
                        }
                    }
                    return Ty::Err;
                }
            }
        }
        let anchor = anchor.unwrap_or_else(|| {
            if es.iter().any(|e| has_float_lit(e)) {
                Ty::F64
            } else {
                Ty::Int(IntK::I64)
            }
        });
        for (i, e) in es.iter().enumerate() {
            if !done[i] {
                self.check_expect(e, &anchor);
            } else {
                let t = self.ty[e.id as usize].clone();
                if !Self::compat(&t, &anchor) {
                    self.mismatch(e, &anchor, &t);
                }
            }
        }
        anchor
    }

    fn vals<'a>(&mut self, args: &'a [Arg]) -> Vec<&'a Expr> {
        args.iter().map(|a| self.arg_val(a)).collect()
    }

    fn check_call(
        &mut self,
        e: &Expr,
        name: &str,
        nspan: Span,
        args: &[Arg],
        exp: Option<&Ty>,
    ) -> Ty {
        let id = e.id as usize;
        if self.info.fns.contains_key(name) {
            if let Some(v) = self.clash_uses.get_mut(name) {
                v.push(nspan);
            }
            self.res[id] = Res::UserFn(name.to_string());
            return self.check_user_call(e, name, nspan, args);
        }
        if let Some((en, vn)) = name.split_once('.') {
            if let Some(ei) = self.info.enums.get(en).cloned() {
                let Some(vi) = ei.variants.iter().position(|v| v.0 == vn) else {
                    for a in args {
                        self.check(a.expr(), None);
                    }
                    return self.unknown_variant(nspan, en, vn);
                };
                self.res[id] = Res::Variant(en.to_string(), vi);
                let fts = ei.variants[vi].1.clone();
                if fts.is_empty() {
                    self.err(
                        Diag::error(
                            "E0115",
                            e.span,
                            &format!("variant {name} has no fields; write it without parens"),
                        )
                        .with_fix_replace(
                            e.span,
                            name,
                            "remove the parens",
                        ),
                    );
                    return Ty::User(en.to_string());
                }
                if args.len() != fts.len() {
                    let shape: Vec<String> = fts.iter().map(|t| t.to_string()).collect();
                    self.err(Diag::error(
                        "E0211",
                        e.span,
                        &format!(
                            "{name} takes {} field(s): ({name} {})",
                            fts.len(),
                            shape.join(" ")
                        ),
                    ));
                }
                for (a, t) in args.iter().zip(fts.iter()) {
                    let ex = self.arg_val(a);
                    self.check_expect(ex, t);
                }
                return Ty::User(en.to_string());
            }
        }
        if BUILTINS.contains(&name) {
            self.res[id] = Res::Builtin(name.to_string());
            return self.check_builtin(e, name, nspan, args, exp);
        }
        for a in args {
            self.check(a.expr(), None);
        }
        if matches!(name, "Vec" | "Map" | "Option" | "Result" | "Str" | "Bool" | "Unit" | "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64" | "F32" | "F64") {
            self.err(
                Diag::error("E0102", nspan, &format!("'{name}' is a type, not a value or function"))
                    .note("types never appear inside expressions; put the type on the binding: (var grid (Vec (Vec I64)) (fill n (vec)))"),
            );
            return Ty::Err;
        }
        let mut d = Diag::error("E0102", nspan, &format!("unknown function '{name}'"));
        let mut cands: Vec<&str> = BUILTINS.to_vec();
        cands.extend(self.info.fns.keys().map(|s| s.as_str()));
        let fix = known_alias(name)
            .map(|s| s.to_string())
            .or_else(|| closest(name, cands.into_iter()));
        if let Some(f) = fix {
            d = d.with_fix_replace(nspan, &f, &format!("use '{f}'"));
        }
        if self.info.structs.contains_key(name) {
            d = d.note(format!(
                "to build a struct write (new {name} (field value)...)"
            ));
        } else if d.fixes.is_empty() && !name.contains('.') {
            // Probably a helper the program forgot to define: show the signature its call implies.
            let params: Vec<String> = args
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let t = self.ty[a.expr().id as usize].clone();
                    let t = if t.is_err() {
                        "TYPE".to_string()
                    } else {
                        t.to_string()
                    };
                    match a {
                        Arg::Mut(..) => format!("(p{} (mut {t}))", i + 1),
                        Arg::Val(_) => format!("(p{} {t})", i + 1),
                    }
                })
                .collect();
            let ret = match exp {
                Some(t) if !t.is_err() => format!(" (returns {t})"),
                _ => String::new(),
            };
            let params = if params.is_empty() {
                String::new()
            } else {
                format!(" (params {})", params.join(" "))
            };
            d = d.note(format!("if it is a helper you meant to write, define it: (fn {name}{params}{ret} (body ...))"));
        }
        if let Some(ei) = self
            .info
            .enums
            .values()
            .find(|ei| ei.variants.iter().any(|v| v.0 == name))
        {
            let en = self
                .info
                .enums
                .iter()
                .find(|(_, v)| std::ptr::eq(*v, ei))
                .map(|(k, _)| k.clone())
                .unwrap();
            d = Diag::error(
                "E0102",
                nspan,
                &format!("unknown function '{name}'; enum variants are qualified"),
            )
            .with_fix_replace(
                nspan,
                &format!("{en}.{name}"),
                &format!("use '{en}.{name}'"),
            );
        }
        self.err(d);
        Ty::Err
    }

    fn check_user_call(&mut self, e: &Expr, name: &str, nspan: Span, args: &[Arg]) -> Ty {
        let sig = self.info.fns[name].clone();
        // Missing capability arguments: caller passed exactly the non-capability params.
        let ncap = sig
            .params
            .iter()
            .filter(|p| matches!(p.1, Ty::Cap(_)))
            .count();
        let arg_is_cap = |c: &Checker, a: &Arg| match &a.expr().kind {
            ExprKind::Var(n) => c
                .lookup(n)
                .is_some_and(|b| matches!(c.bindings[b].ty, Ty::Cap(_))),
            _ => false,
        };
        if ncap > 0
            && args.len() == sig.params.len() - ncap
            && !args.iter().any(|a| arg_is_cap(self, a))
        {
            let missing: Vec<Cap> = sig
                .params
                .iter()
                .filter_map(|p| if let Ty::Cap(c) = p.1 { Some(c) } else { None })
                .collect();
            let mut edits = Vec::new();
            let mut ai = 0;
            let mut add_params = Vec::new();
            for p in &sig.params {
                if let Ty::Cap(c) = p.1 {
                    let at = if ai < args.len() {
                        args[ai].expr().span.start
                    } else {
                        e.span.end - 1
                    };
                    let have = self
                        .scopes
                        .iter()
                        .rev()
                        .flat_map(|s| s.iter())
                        .find(|(_, b)| self.bindings[**b].ty == Ty::Cap(c))
                        .map(|(n, _)| n.clone());
                    let n = have
                        .clone()
                        .unwrap_or_else(|| c.default_param().to_string());
                    if have.is_none() {
                        add_params.push(c);
                    }
                    let text = if ai < args.len() {
                        format!("{n} ")
                    } else {
                        format!(" {n}")
                    };
                    edits.push(Edit {
                        span: Span::new(at, at),
                        text,
                    });
                } else {
                    ai += 1;
                }
            }
            let mut d = Diag::error(
                "E0301",
                e.span,
                &format!(
                    "'{name}' needs capabilities {} passed explicitly",
                    missing
                        .iter()
                        .map(|c| c.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
            if self.in_test && !add_params.is_empty() {
                d = d.note("tests are pure and receive no capabilities");
            } else if let Some(f) = &self.cur_fn {
                if !add_params.is_empty() {
                    let text: String = add_params
                        .iter()
                        .map(|c| format!(" ({} {})", c.default_param(), c.name()))
                        .collect();
                    let pe = if f.has_params_clause {
                        Edit {
                            span: Span::new(f.param_insert_at, f.param_insert_at),
                            text,
                        }
                    } else {
                        Edit {
                            span: Span::new(f.name_span.end, f.name_span.end),
                            text: format!(" (params{text})"),
                        }
                    };
                    edits.push(pe);
                }
                d = d.with_fix(
                    "pass the capabilities (adding parameters to this function if needed)",
                    edits,
                );
            }
            self.err(d);
            for a in args {
                self.check(a.expr(), None);
            }
            return sig.ret;
        }
        if args.len() != sig.params.len() {
            let shape: Vec<String> = sig
                .params
                .iter()
                .map(|(n, t, m)| {
                    if *m {
                        format!("(mut {n}: {t})")
                    } else {
                        format!("{n}: {t}")
                    }
                })
                .collect();
            self.err(
                Diag::error(
                    "E0211",
                    nspan,
                    &format!(
                        "'{name}' takes {} argument(s), got {}",
                        sig.params.len(),
                        args.len()
                    ),
                )
                .note(format!("signature: ({name} {})", shape.join(" "))),
            );
            for a in args {
                self.check(a.expr(), None);
            }
            return sig.ret;
        }
        let mut mut_roots: Vec<(BindId, Span)> = Vec::new();
        let path_of = |c: &Checker, x: &Expr| c.place_path(x).map(|p| p.1).unwrap_or_default();
        for (a, (pname, pt, pmut)) in args.iter().zip(sig.params.iter()) {
            if *pmut {
                match a {
                    Arg::Mut(x, _) => {
                        let t = self.check(x, Some(pt));
                        if !Self::compat(&t, pt) {
                            self.mismatch(x, pt, &t);
                        }
                        if let Some(r) = self.require_mutable_place(x, "mutate") {
                            mut_roots.push((r, x.span));
                        }
                    }
                    Arg::Val(x) => {
                        let (_t, r) = self.arg_mut(a);
                        let _ = (x, pname);
                        if let Some(r) = r {
                            mut_roots.push((r, a.expr().span));
                        }
                    }
                }
            } else {
                let x = self.arg_val(a);
                if let Ty::Cap(c) = pt {
                    let t = self.check(x, Some(pt));
                    if t != Ty::Cap(*c) && !t.is_err() {
                        self.err(Diag::error(
                            "E0302",
                            x.span,
                            &format!(
                                "parameter '{pname}' expects a {} capability, found {t}",
                                c.name()
                            ),
                        ));
                    }
                } else {
                    self.check_expect(x, pt);
                }
            }
        }
        // Exclusivity: a mutated root cannot be visible through any other argument passed by reference.
        for (i, (root, rspan)) in mut_roots.iter().enumerate() {
            for (j, (a, (_, pt, pmut))) in args.iter().zip(sig.params.iter()).enumerate() {
                let x = a.expr();
                if x.span == *rspan {
                    continue;
                }
                let by_ref = !self.info.is_copy(pt) && self.place_root(x).is_some();
                let mpath = args
                    .iter()
                    .find(|a| a.expr().span == *rspan)
                    .map(|a| path_of(self, a.expr()))
                    .unwrap_or_default();
                let xpath = path_of(self, x);
                let conflict = if *pmut {
                    mut_roots
                        .iter()
                        .enumerate()
                        .any(|(k, (r2, s2))| k != i && r2 == root && *s2 == x.span)
                        && paths_overlap(&mpath, &xpath)
                } else {
                    by_ref && self.place_root(x) == Some(*root) && paths_overlap(&mpath, &xpath)
                };
                let _ = j;
                if conflict {
                    let bname = self.bindings[*root].name.clone();
                    self.err(
                        Diag::error(
                            "E0404",
                            x.span,
                            &format!("'{bname}' is mutated by this call, so it cannot also be passed as another argument (aliasing)"),
                        )
                        .note("values never alias in ail; pass an explicit copy if you need the old value")
                        .with_fix(
                            "pass an explicit copy",
                            vec![Edit { span: Span::new(x.span.start, x.span.start), text: "(copy ".into() }, Edit { span: Span::new(x.span.end, x.span.end), text: ")".into() }],
                        ),
                    );
                }
            }
        }
        if self.in_contract && sig.params.iter().any(|p| matches!(p.1, Ty::Cap(_))) {
            self.err(Diag::error(
                "E0303",
                e.span,
                "contracts must be pure: this function takes capabilities",
            ));
        }
        sig.ret
    }

    fn check_builtin(
        &mut self,
        e: &Expr,
        name: &str,
        nspan: Span,
        args: &[Arg],
        exp: Option<&Ty>,
    ) -> Ty {
        let span = e.span;
        let n = args.len();
        macro_rules! ar {
            ($k:expr, $shape:expr) => {
                if !self.arity(name, args, $k, span, $shape) {
                    return Ty::Err;
                }
            };
        }
        match name {
            "+" | "*" | "-" | "/" | "%" => {
                if n == 0 || (name != "-" && n < 2) || ((name == "/" || name == "%") && n != 2) {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        &format!(
                            "'{name}' takes {} arguments",
                            if name == "+" || name == "*" {
                                "2 or more"
                            } else if name == "-" {
                                "1 or 2"
                            } else {
                                "2"
                            }
                        ),
                    ));
                    for a in args {
                        self.check(a.expr(), None);
                    }
                    return Ty::Err;
                }
                if name == "-" && n > 2 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "'-' takes 1 or 2 arguments; nest them: (- (- a b) c)",
                    ));
                }
                let es = self.vals(args);
                let t = self.check_same(&es, exp);
                if t == Ty::Str && name == "+" {
                    self.err(
                        Diag::error(
                            "E0212",
                            nspan,
                            "strings are joined with (str a b ...), not +",
                        )
                        .with_fix_replace(nspan, "str", "use str"),
                    );
                    return Ty::Str;
                }
                if !t.is_numeric() && !t.is_err() {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        &format!("'{name}' needs numbers, found {t}"),
                    ));
                    return Ty::Err;
                }
                t
            }
            "==" | "!=" => {
                ar!(2, "(== a b)");
                let es = self.vals(args);
                let t = self.check_same(&es, None);
                if matches!(t, Ty::Cap(_)) {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        "capabilities cannot be compared",
                    ));
                }
                Ty::Bool
            }
            "<" | "<=" | ">" | ">=" => {
                ar!(2, "(< a b)");
                let es = self.vals(args);
                let t = self.check_same(&es, None);
                if !self.info.orderable(&t) {
                    self.err(Diag::error("E0212", span, &format!("'{name}' cannot order values of type {t} (maps and capabilities have no order)")));
                }
                Ty::Bool
            }
            "and" | "or" => {
                if n < 2 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        &format!("'{name}' takes 2 or more arguments"),
                    ));
                }
                for a in args {
                    let x = self.arg_val(a);
                    self.check_expect(x, &Ty::Bool);
                }
                Ty::Bool
            }
            "not" => {
                ar!(1, "(not b)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Bool);
                Ty::Bool
            }
            "bit-and" | "bit-or" | "bit-xor" | "shl" | "shr" => {
                ar!(2, "(bit-and a b)");
                let es = self.vals(args);
                let t = if name == "shl" || name == "shr" {
                    let t = self.check(es[0], exp);
                    self.check_expect(es[1], &t.clone());
                    t
                } else {
                    self.check_same(&es, exp)
                };
                if !t.is_int() && !t.is_err() {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        &format!("'{name}' needs integers, found {t}"),
                    ));
                }
                t
            }
            "abs" => {
                ar!(1, "(abs x)");
                let x = self.arg_val(&args[0]);
                let t = self.check(x, exp);
                if !t.is_numeric() && !t.is_err() {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        &format!("abs needs a number, found {t}"),
                    ));
                }
                t
            }
            "min" | "max" => {
                ar!(2, "(min a b)");
                let es = self.vals(args);
                let t = self.check_same(&es, exp);
                if !self.info.orderable(&t) {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        &format!("'{name}' needs comparable values, found {t}"),
                    ));
                }
                t
            }
            "pow" => {
                ar!(2, "(pow base exponent)");
                let es = self.vals(args);
                let t = self.check(es[0], exp);
                if t.is_int() {
                    self.check_expect(es[1], &Ty::Int(IntK::U32));
                } else if t.is_float() {
                    self.check_expect(es[1], &t.clone());
                } else if !t.is_err() {
                    self.err(Diag::error(
                        "E0212",
                        span,
                        &format!("pow needs a number, found {t}"),
                    ));
                }
                t
            }
            "sqrt" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" | "exp" | "ln" => {
                ar!(1, "(sqrt x)");
                let x = self.arg_val(&args[0]);
                let want = if matches!(exp, Some(Ty::F32)) {
                    Ty::F32
                } else {
                    Ty::F64
                };
                let t = self.check(x, Some(&want));
                if !t.is_float() && !t.is_err() {
                    self.err(
                        Diag::error(
                            "E0212",
                            x.span,
                            &format!("'{name}' needs a float, found {t}"),
                        )
                        .with_fix_replace(
                            x.span,
                            &format!("(as F64 {})", "\u{0}"),
                            "convert with (as F64 ...)",
                        ),
                    );
                    return Ty::F64;
                }
                t
            }
            "atan2" => {
                ar!(2, "(atan2 y x)");
                let es = self.vals(args);
                self.check_same(&es, Some(&Ty::F64))
            }
            "str" | "print" | "println" | "eprintln" => {
                let rest = if name == "str" {
                    args
                } else {
                    match self.need_cap(Cap::Console, args, nspan, span) {
                        Some(r) => r,
                        None => return if name == "str" { Ty::Str } else { Ty::Unit },
                    }
                };
                for a in rest {
                    let x = self.arg_val(a);
                    let t = self.check(x, None);
                    if !t.is_displayable() {
                        let hint = match &t {
                            Ty::Opt(_) | Ty::Res(..) => {
                                "unwrap it first with (unwrap-or x default) or (match x ...)"
                            }
                            Ty::Vec(_) => {
                                "join the elements, e.g. (join strings \", \") after converting them with str"
                            }
                            Ty::User(_) => {
                                "format the fields one by one: (str (. p x) \",\" (. p y))"
                            }
                            _ => "convert it to text explicitly",
                        };
                        self.err(Diag::error("E0213", x.span, &format!("cannot turn {t} into text; only numbers, Bool and Str are displayable")).note(hint));
                    }
                }
                if name == "str" { Ty::Str } else { Ty::Unit }
            }
            "len" | "is-empty" => {
                ar!(1, "(len x)");
                let x = self.arg_val(&args[0]);
                let t = self.check(x, None);
                if !matches!(t, Ty::Vec(_) | Ty::Str | Ty::Map(..) | Ty::Err) {
                    self.err(Diag::error(
                        "E0214",
                        x.span,
                        &format!("'{name}' works on Vec, Str and Map, found {t}"),
                    ));
                }
                if name == "len" {
                    Ty::Int(IntK::I64)
                } else {
                    Ty::Bool
                }
            }
            "at" => {
                ar!(2, "(at vec index)");
                let es = self.vals(args);
                let t = self.check(es[0], None);
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                match t {
                    Ty::Vec(el) => *el,
                    Ty::Err => Ty::Err,
                    Ty::Map(..) => {
                        self.err(
                            Diag::error(
                                "E0214",
                                nspan,
                                "use (get m k) on maps; it returns an Option",
                            )
                            .with_fix_replace(nspan, "get", "use get"),
                        );
                        Ty::Err
                    }
                    Ty::Str => {
                        self.err(Diag::error("E0214", span, "strings are not indexed by (at ...); use (byte-at s i) or (substr s a b)"));
                        Ty::Err
                    }
                    other => {
                        self.err(Diag::error(
                            "E0214",
                            es[0].span,
                            &format!("(at ...) needs a Vec, found {other}"),
                        ));
                        Ty::Err
                    }
                }
            }
            "vec" => {
                let el = match exp {
                    Some(Ty::Vec(t)) => Some((**t).clone()),
                    _ => None,
                };
                if n == 0 {
                    return match el {
                        Some(t) => Ty::Vec(Box::new(t)),
                        None => {
                            self.err(
                                Diag::error(
                                    "E0210",
                                    span,
                                    "cannot infer the element type of an empty (vec)",
                                )
                                .note("annotate the binding: (var xs (Vec I64) (vec))"),
                            );
                            Ty::Err
                        }
                    };
                }
                let es = self.vals(args);
                let t = match el {
                    Some(t) => {
                        for x in &es {
                            self.check_expect(x, &t);
                        }
                        t
                    }
                    None => self.check_same(&es, None),
                };
                Ty::Vec(Box::new(t))
            }
            "fill" => {
                ar!(2, "(fill count value)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Int(IntK::I64));
                let el = match exp {
                    Some(Ty::Vec(t)) => Some((**t).clone()),
                    _ => None,
                };
                let t = match el {
                    Some(t) => self.check_expect(es[1], &t),
                    None => self.check(es[1], None),
                };
                Ty::Vec(Box::new(t))
            }
            "push" | "insert-at" => {
                if name == "push" {
                    ar!(2, "(push (mut vec) value)");
                } else {
                    ar!(3, "(insert-at (mut vec) index value)");
                }
                let (vt, _) = self.arg_mut(&args[0]);
                let el = self.vec_elem(&vt, args[0].expr().span, name);
                if name == "insert-at" {
                    let i = self.arg_val(&args[1]);
                    self.check_expect(i, &Ty::Int(IntK::I64));
                }
                let x = self.arg_val(args.last().unwrap());
                self.check_expect(x, &el);
                Ty::Unit
            }
            "pop" => {
                ar!(1, "(pop (mut vec))");
                let (vt, _) = self.arg_mut(&args[0]);
                Ty::Opt(Box::new(self.vec_elem(&vt, args[0].expr().span, name)))
            }
            "last" => {
                ar!(1, "(last vec)");
                let x = self.arg_val(&args[0]);
                let vt = self.check(x, None);
                Ty::Opt(Box::new(self.vec_elem(&vt, x.span, name)))
            }
            "remove-at" => {
                ar!(2, "(remove-at (mut vec) index)");
                let (vt, _) = self.arg_mut(&args[0]);
                let i = self.arg_val(&args[1]);
                self.check_expect(i, &Ty::Int(IntK::I64));
                self.vec_elem(&vt, args[0].expr().span, name)
            }
            "swap" => {
                ar!(3, "(swap (mut vec) i j)");
                let (vt, _) = self.arg_mut(&args[0]);
                self.vec_elem(&vt, args[0].expr().span, name);
                for a in &args[1..] {
                    let x = self.arg_val(a);
                    self.check_expect(x, &Ty::Int(IntK::I64));
                }
                Ty::Unit
            }
            "clear" => {
                ar!(1, "(clear (mut x))");
                let (t, _) = self.arg_mut(&args[0]);
                if !matches!(t, Ty::Vec(_) | Ty::Map(..) | Ty::Str | Ty::Err) {
                    self.err(Diag::error(
                        "E0214",
                        args[0].expr().span,
                        &format!("clear works on Vec, Map and Str, found {t}"),
                    ));
                }
                Ty::Unit
            }
            "extend" => {
                ar!(2, "(extend (mut vec) other-vec)");
                let (vt, _) = self.arg_mut(&args[0]);
                let x = self.arg_val(&args[1]);
                if !vt.is_err() {
                    self.check_expect(x, &vt);
                } else {
                    self.check(x, None);
                }
                Ty::Unit
            }
            "slice" => {
                ar!(3, "(slice vec start end)");
                let es = self.vals(args);
                let t = self.check(es[0], exp);
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                self.check_expect(es[2], &Ty::Int(IntK::I64));
                if !matches!(t, Ty::Vec(_) | Ty::Err) {
                    let mut d = Diag::error(
                        "E0214",
                        es[0].span,
                        &format!("slice needs a Vec, found {t}"),
                    );
                    if t == Ty::Str {
                        d = d.with_fix_replace(nspan, "substr", "use substr for strings");
                    }
                    self.err(d);
                }
                t
            }
            "concat" => {
                ar!(2, "(concat vec-a vec-b)");
                let es = self.vals(args);
                let t = self.check(es[0], exp);
                self.check_expect(es[1], &t.clone());
                if !matches!(t, Ty::Vec(_) | Ty::Err) {
                    let mut d = Diag::error(
                        "E0214",
                        es[0].span,
                        &format!("concat joins two Vecs, found {t}"),
                    );
                    if t == Ty::Str {
                        d = d.with_fix_replace(nspan, "str", "use (str a b) for strings");
                    }
                    self.err(d);
                }
                t
            }
            "contains" | "index-of" => {
                ar!(2, "(contains collection item)");
                let es = self.vals(args);
                let t = self.check(es[0], None);
                match &t {
                    Ty::Vec(el) => {
                        let el = (**el).clone();
                        self.check_expect(es[1], &el);
                    }
                    Ty::Str => {
                        self.check_expect(es[1], &Ty::Str);
                    }
                    Ty::Map(..) => {
                        self.err(
                            Diag::error("E0214", nspan, "use (has m k) for maps")
                                .with_fix_replace(nspan, "has", "use has"),
                        );
                    }
                    Ty::Err => {
                        self.check(es[1], None);
                    }
                    other => {
                        self.err(Diag::error(
                            "E0214",
                            es[0].span,
                            &format!("'{name}' works on Vec or Str, found {other}"),
                        ));
                    }
                }
                if name == "contains" {
                    Ty::Bool
                } else {
                    Ty::Opt(Box::new(Ty::Int(IntK::I64)))
                }
            }
            "sort" | "sort-desc" | "reverse" => {
                ar!(1, "(sort (mut vec))");
                let (vt, _) = self.arg_mut(&args[0]);
                let el = self.vec_elem(&vt, args[0].expr().span, name);
                if name != "reverse" && !self.info.orderable(&el) {
                    self.err(Diag::error(
                        "E0214",
                        args[0].expr().span,
                        &format!("sort cannot order elements of type {el} (maps have no order)"),
                    ));
                }
                Ty::Unit
            }
            "sum" => {
                ar!(1, "(sum vec)");
                let x = self.arg_val(&args[0]);
                let vt = self.check(x, exp.map(|t| Ty::Vec(Box::new(t.clone()))).as_ref());
                let el = self.vec_elem(&vt, x.span, name);
                if !el.is_numeric() && !el.is_err() {
                    self.err(Diag::error(
                        "E0214",
                        x.span,
                        &format!("sum needs numeric elements, found {el}"),
                    ));
                }
                el
            }
            "map" => {
                if n != 0 {
                    self.err(Diag::error("E0211", span, "(map) creates an empty map and takes no arguments; add entries with (put (mut m) k v)").note("ail has no higher-order map; use a for loop"));
                }
                match exp {
                    Some(t @ Ty::Map(..)) => t.clone(),
                    _ => {
                        self.err(
                            Diag::error("E0210", span, "cannot infer the key/value types of (map)")
                                .note("annotate the binding: (var m (Map Str I64) (map))"),
                        );
                        Ty::Err
                    }
                }
            }
            "put" => {
                ar!(3, "(put (mut map) key value)");
                let (mt, _) = self.arg_mut(&args[0]);
                let (k, v) = self.map_kv(&mt, args[0].expr().span);
                let kx = self.arg_val(&args[1]);
                self.check_expect(kx, &k);
                let vx = self.arg_val(&args[2]);
                self.check_expect(vx, &v);
                Ty::Unit
            }
            "get" | "has" => {
                ar!(2, "(get map key)");
                let es = self.vals(args);
                let mt = self.check(es[0], None);
                if let Ty::Vec(el) = &mt {
                    let el = (**el).clone();
                    self.check(es[1], None);
                    self.err(
                        Diag::error("E0214", nspan, "use (at v i) on vectors")
                            .with_fix_replace(nspan, "at", "use at"),
                    );
                    return el;
                }
                let (k, v) = self.map_kv(&mt, es[0].span);
                self.check_expect(es[1], &k);
                if name == "get" {
                    Ty::Opt(Box::new(v))
                } else {
                    Ty::Bool
                }
            }
            "del" => {
                ar!(2, "(del (mut map) key)");
                let (mt, _) = self.arg_mut(&args[0]);
                let (k, v) = self.map_kv(&mt, args[0].expr().span);
                let kx = self.arg_val(&args[1]);
                self.check_expect(kx, &k);
                Ty::Opt(Box::new(v))
            }
            "keys" | "values" => {
                ar!(1, "(keys map)");
                let x = self.arg_val(&args[0]);
                let mt = self.check(x, None);
                let (k, v) = self.map_kv(&mt, x.span);
                Ty::Vec(Box::new(if name == "keys" { k } else { v }))
            }
            "some" => {
                ar!(1, "(some value)");
                let x = self.arg_val(&args[0]);
                let inner = match exp {
                    Some(Ty::Opt(t)) => Some((**t).clone()),
                    _ => None,
                };
                let t = match inner {
                    Some(t) => self.check_expect(x, &t),
                    None => self.check(x, None),
                };
                Ty::Opt(Box::new(t))
            }
            "ok" | "err" => {
                ar!(1, "(ok value)");
                let x = self.arg_val(&args[0]);
                match exp {
                    Some(Ty::Res(a, b)) => {
                        let want = if name == "ok" {
                            (**a).clone()
                        } else {
                            (**b).clone()
                        };
                        self.check_expect(x, &want);
                        Ty::Res(a.clone(), b.clone())
                    }
                    Some(Ty::Err) => {
                        self.check(x, None);
                        Ty::Err
                    }
                    _ => {
                        self.check(x, None);
                        let mut d = Diag::error(
                            "E0210",
                            span,
                            &format!("cannot infer the full Result type of ({name} ...) here"),
                        )
                        .note("use it where a Result is expected (return value, annotated let)");
                        if let Some(other) = exp {
                            d = Diag::error(
                                "E0201",
                                span,
                                &format!("({name} ...) builds a Result, but {other} is expected"),
                            )
                            .with_expected(other.to_string(), "(Result ? ?)");
                        }
                        self.err(d);
                        Ty::Err
                    }
                }
            }
            "unwrap-or" => {
                ar!(2, "(unwrap-or option-or-result default)");
                let es = self.vals(args);
                let t = self.check(es[0], exp.map(|t| Ty::Opt(Box::new(t.clone()))).as_ref());
                let inner = match &t {
                    Ty::Opt(i) | Ty::Res(i, _) => (**i).clone(),
                    Ty::Err => Ty::Err,
                    other => {
                        self.err(Diag::error(
                            "E0214",
                            es[0].span,
                            &format!("unwrap-or needs an Option or Result, found {other}"),
                        ));
                        Ty::Err
                    }
                };
                self.check_expect(es[1], &inner);
                inner
            }
            "expect" => {
                ar!(2, "(expect option-or-result \"message\")");
                let es = self.vals(args);
                let t = self.check(es[0], exp.map(|t| Ty::Opt(Box::new(t.clone()))).as_ref());
                self.check_expect(es[1], &Ty::Str);
                match &t {
                    Ty::Opt(i) | Ty::Res(i, _) => (**i).clone(),
                    Ty::Err => Ty::Err,
                    other => {
                        self.err(Diag::error(
                            "E0214",
                            es[0].span,
                            &format!("expect needs an Option or Result, found {other}"),
                        ));
                        Ty::Err
                    }
                }
            }
            "is-some" | "is-none" | "is-ok" | "is-err" => {
                ar!(1, "(is-some x)");
                let x = self.arg_val(&args[0]);
                let t = self.check(x, None);
                let ok = match name {
                    "is-some" | "is-none" => matches!(t, Ty::Opt(_) | Ty::Err),
                    _ => matches!(t, Ty::Res(..) | Ty::Err),
                };
                if !ok {
                    self.err(Diag::error(
                        "E0214",
                        x.span,
                        &format!("'{name}' does not apply to {t}"),
                    ));
                }
                Ty::Bool
            }
            "substr" => {
                ar!(3, "(substr s start end)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Str);
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                self.check_expect(es[2], &Ty::Int(IntK::I64));
                Ty::Str
            }
            "split" => {
                ar!(2, "(split s separator)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Str);
                self.check_expect(es[1], &Ty::Str);
                Ty::Vec(Box::new(Ty::Str))
            }
            "trim" | "upper" | "lower" => {
                ar!(1, "(trim s)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Str
            }
            "starts-with" | "ends-with" => {
                ar!(2, "(starts-with s prefix)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Str);
                self.check_expect(es[1], &Ty::Str);
                Ty::Bool
            }
            "replace" => {
                ar!(3, "(replace s from to)");
                for x in self.vals(args) {
                    self.check_expect(x, &Ty::Str);
                }
                Ty::Str
            }
            "join" => {
                ar!(2, "(join strings separator)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Vec(Box::new(Ty::Str)));
                self.check_expect(es[1], &Ty::Str);
                Ty::Str
            }
            "parse-i64" | "parse-f64" => {
                ar!(1, "(parse-i64 s)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Opt(Box::new(if name == "parse-i64" {
                    Ty::Int(IntK::I64)
                } else {
                    Ty::F64
                }))
            }
            "chars" => {
                ar!(1, "(chars s)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Vec(Box::new(Ty::Str))
            }
            "byte-at" => {
                ar!(2, "(byte-at s index)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Str);
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                Ty::Int(IntK::I64)
            }
            "fmt-float" => {
                ar!(2, "(fmt-float x decimals)");
                let es = self.vals(args);
                let t = self.check(es[0], Some(&Ty::F64));
                if !t.is_float() && !t.is_err() {
                    self.err(
                        Diag::error(
                            "E0212",
                            es[0].span,
                            &format!("fmt-float needs a float, found {t}"),
                        )
                        .with_fix_replace(
                            es[0].span,
                            &format!("(as F64 {})", "\u{0}"),
                            "convert with (as F64 ...)",
                        ),
                    );
                }
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                Ty::Str
            }
            "from-code" => {
                ar!(1, "(from-code code-point)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Int(IntK::I64));
                Ty::Str
            }
            "repeat" => {
                ar!(2, "(repeat s count)");
                let es = self.vals(args);
                self.check_expect(es[0], &Ty::Str);
                self.check_expect(es[1], &Ty::Int(IntK::I64));
                Ty::Str
            }
            "read-line" => {
                if self.need_cap(Cap::Console, args, nspan, span).is_some() && n != 1 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "(read-line con) takes only the capability",
                    ));
                }
                Ty::Opt(Box::new(Ty::Str))
            }
            "read-file" => {
                if let Some(rest) = self.need_cap(Cap::Fs, args, nspan, span) {
                    if rest.len() != 2 {
                        self.err(
                            Diag::error(
                                "E0211",
                                span,
                                "(read-file fs path max-bytes): the size limit is mandatory",
                            )
                            .note("unbounded reads are not representable in ail"),
                        );
                    } else {
                        let p = self.arg_val(&rest[0]);
                        self.check_expect(p, &Ty::Str);
                        let m = self.arg_val(&rest[1]);
                        self.check_expect(m, &Ty::Int(IntK::I64));
                    }
                }
                Ty::Res(Box::new(Ty::Str), Box::new(Ty::Str))
            }
            "write-file" => {
                if let Some(rest) = self.need_cap(Cap::Fs, args, nspan, span) {
                    if rest.len() != 2 {
                        self.err(Diag::error("E0211", span, "(write-file fs path content)"));
                    } else {
                        for a in rest {
                            let x = self.arg_val(a);
                            self.check_expect(x, &Ty::Str);
                        }
                    }
                }
                Ty::Res(Box::new(Ty::Unit), Box::new(Ty::Str))
            }
            "now-ms" | "now-ns" => {
                if self.need_cap(Cap::Clock, args, nspan, span).is_some() && n != 1 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        &format!("({name} clock) takes only the capability"),
                    ));
                }
                Ty::Int(IntK::I64)
            }
            "rand-int" => {
                if let Some(rest) = self.need_cap(Cap::Rand, args, nspan, span) {
                    if rest.len() != 2 {
                        self.err(Diag::error(
                            "E0211",
                            span,
                            "(rand-int rng low high): high is exclusive",
                        ));
                    } else {
                        for a in rest {
                            let x = self.arg_val(a);
                            self.check_expect(x, &Ty::Int(IntK::I64));
                        }
                    }
                }
                Ty::Int(IntK::I64)
            }
            "rand-f64" => {
                if self.need_cap(Cap::Rand, args, nspan, span).is_some() && n != 1 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "(rand-f64 rng) takes only the capability; returns [0, 1)",
                    ));
                }
                Ty::F64
            }
            "args" => {
                if self.need_cap(Cap::Env, args, nspan, span).is_some() && n != 1 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "(args env) takes only the capability",
                    ));
                }
                Ty::Vec(Box::new(Ty::Str))
            }
            "draw-clear" | "draw-rect" | "draw-circle" | "draw-line" | "draw-text" => {
                let (nums, strs, shape) = match name {
                    "draw-clear" => (0, 1, "(draw-clear g color)"),
                    "draw-rect" => (4, 1, "(draw-rect g x y width height color)"),
                    "draw-circle" => (3, 1, "(draw-circle g x y radius color)"),
                    "draw-line" => (5, 1, "(draw-line g x1 y1 x2 y2 width color)"),
                    _ => (3, 2, "(draw-text g x y size color text)"),
                };
                if let Some(rest) = self.need_cap(Cap::Gfx, args, nspan, span) {
                    if rest.len() != nums + strs {
                        self.err(Diag::error("E0211", span, &format!("'{name}' is {shape}; numbers are F64 pixels, colors are CSS strings like \"#ff8800\"")));
                    } else {
                        for (i, a) in rest.iter().enumerate() {
                            let x = self.arg_val(a);
                            let want = if i < nums { Ty::F64 } else { Ty::Str };
                            self.check_expect(x, &want);
                        }
                    }
                }
                Ty::Unit
            }
            "read-all" => {
                if self.need_cap(Cap::Console, args, nspan, span).is_some() && n != 1 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "(read-all con) takes only the capability",
                    ));
                }
                Ty::Str
            }
            "lines" => {
                ar!(1, "(lines text)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Vec(Box::new(Ty::Str))
            }
            "is-digit" | "is-alpha" | "is-alnum" | "is-space" | "is-upper" | "is-lower" => {
                ar!(1, "(is-digit s)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Bool
            }
            "code-of" => {
                ar!(1, "(code-of s)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Int(IntK::I64)
            }
            "json-parse" => {
                ar!(1, "(json-parse text)");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Res(Box::new(Ty::User("Json".into())), Box::new(Ty::Str))
            }
            "json-str" => {
                ar!(1, "(json-str value)");
                let x = self.arg_val(&args[0]);
                let t = self.check(x, None);
                if !t.is_err() && !json_encodable(&self.info, &t, &mut Vec::new()) {
                    self.err(
                        Diag::error("E0212", x.span, &format!("json-str cannot encode {t}"))
                            .note("it encodes Json and any value built from numbers, Bool, Str, Vec, Option, (Map Str V) and structs of those"),
                    );
                }
                Ty::Str
            }
            "http-next" => {
                if let Some(rest) = self.need_cap(Cap::Net, args, nspan, span) {
                    if rest.len() != 2 {
                        self.err(Diag::error("E0211", span, "(http-next net port max-body-bytes): waits for the next request; the body limit is mandatory"));
                    } else {
                        for a in rest {
                            let x = self.arg_val(a);
                            self.check_expect(x, &Ty::Int(IntK::I64));
                        }
                    }
                }
                Ty::Res(Box::new(Ty::User("Request".into())), Box::new(Ty::Str))
            }
            "http-respond" => {
                if let Some(rest) = self.need_cap(Cap::Net, args, nspan, span) {
                    if rest.len() != 4 {
                        self.err(Diag::error(
                            "E0211",
                            span,
                            "(http-respond net request status content-type body)",
                        ));
                    } else {
                        let r = self.arg_val(&rest[0]);
                        self.check_expect(r, &Ty::User("Request".into()));
                        let st = self.arg_val(&rest[1]);
                        self.check_expect(st, &Ty::Int(IntK::I64));
                        for a in &rest[2..] {
                            let x = self.arg_val(a);
                            self.check_expect(x, &Ty::Str);
                        }
                    }
                }
                Ty::Res(Box::new(Ty::Unit), Box::new(Ty::Str))
            }
            "fail" => {
                ar!(1, "(fail \"message\")");
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Str);
                Ty::Never
            }
            "assert" => {
                if n != 1 && n != 2 {
                    self.err(Diag::error(
                        "E0211",
                        span,
                        "(assert condition) or (assert condition \"message\")",
                    ));
                    return Ty::Unit;
                }
                let x = self.arg_val(&args[0]);
                self.check_expect(x, &Ty::Bool);
                if n == 2 {
                    let m = self.arg_val(&args[1]);
                    self.check_expect(m, &Ty::Str);
                }
                Ty::Unit
            }
            _ => unreachable!("builtin {name} not handled"),
        }
    }

    fn vec_elem(&mut self, t: &Ty, span: Span, what: &str) -> Ty {
        match t {
            Ty::Vec(el) => (**el).clone(),
            Ty::Err => Ty::Err,
            other => {
                self.err(Diag::error(
                    "E0214",
                    span,
                    &format!("'{what}' needs a Vec, found {other}"),
                ));
                Ty::Err
            }
        }
    }

    fn map_kv(&mut self, t: &Ty, span: Span) -> (Ty, Ty) {
        match t {
            Ty::Map(k, v) => ((**k).clone(), (**v).clone()),
            Ty::Err => (Ty::Err, Ty::Err),
            other => {
                self.err(Diag::error(
                    "E0214",
                    span,
                    &format!("expected a Map, found {other}"),
                ));
                (Ty::Err, Ty::Err)
            }
        }
    }

    // ---------- match ----------

    fn check_match(&mut self, e: &Expr, scrut: &Expr, arms: &[Arm], exp: Option<&Ty>) -> Ty {
        let st = self.check(scrut, None);
        let discard = self.stmt_ids.contains(&e.id) || exp == Some(&Ty::Unit);
        let exp = if discard { Some(&Ty::Unit) } else { exp };
        let mut result: Option<Ty> = exp.cloned();
        let mut covered: Vec<String> = Vec::new();
        let mut total = false;
        let mut first_ty: Option<Ty> = None;
        for arm in arms {
            if total {
                self.err(
                    Diag::warning(
                        "W0002",
                        arm.span,
                        "unreachable arm: an earlier arm already matches everything",
                    )
                    .with_fix_replace(arm.span, "", "remove it"),
                );
            }
            self.scopes.push(HashMap::new());
            let refutable = self.check_pat(&arm.pat, &st, false);
            match (&arm.pat, refutable) {
                (Pat::Wild(_) | Pat::Bind(..), _) => total = true,
                (Pat::Ctor(name, _, _), false) => covered.push(ctor_key(name)),
                _ => {}
            }
            let t = match &result {
                Some(r) => {
                    let r = r.clone();
                    let t = if discard {
                        self.check_block(&arm.body, None)
                    } else {
                        self.check_block(&arm.body, Some(&r))
                    };
                    if !discard && !Self::compat(&t, &r) && !(r == Ty::Unit && exp.is_none()) {
                        match arm.body.last() {
                            Some(l) => self.mismatch(l, &r, &t),
                            None => {}
                        }
                    }
                    t
                }
                None => self.check_block(&arm.body, None),
            };
            if result.is_none() && t != Ty::Never && !t.is_err() {
                result = Some(t.clone());
            }
            if first_ty.is_none() {
                first_ty = Some(t);
            }
            self.scopes.pop();
        }
        if !total {
            let needed: Option<Vec<(String, String)>> = match &st {
                Ty::Bool => Some(vec![
                    ("true".into(), "true".into()),
                    ("false".into(), "false".into()),
                ]),
                Ty::Opt(_) => Some(vec![
                    ("none".into(), "none".into()),
                    ("some".into(), "(some _)".into()),
                ]),
                Ty::Res(..) => Some(vec![
                    ("ok".into(), "(ok _)".into()),
                    ("err".into(), "(err _)".into()),
                ]),
                Ty::User(n) if self.info.enums.contains_key(n) => Some(
                    self.info.enums[n]
                        .variants
                        .iter()
                        .map(|(v, fs)| {
                            let p = if fs.is_empty() {
                                format!("{n}.{v}")
                            } else {
                                format!("({n}.{v}{})", " _".repeat(fs.len()))
                            };
                            (v.clone(), p)
                        })
                        .collect(),
                ),
                Ty::Err => None,
                _ => Some(Vec::new()),
            };
            if let Some(needed) = needed {
                let missing: Vec<&(String, String)> = needed
                    .iter()
                    .filter(|(k, _)| !covered.contains(k))
                    .collect();
                if needed.is_empty() {
                    self.err(
                        Diag::error(
                            "E0502",
                            e.span,
                            &format!("match on {st} must end with a catch-all arm (case _ ...)"),
                        )
                        .with_fix_insert(
                            e.span.end - 1,
                            " (case _ (fail \"unhandled case\"))",
                            "add a catch-all arm (replace the body)",
                        ),
                    );
                } else if !missing.is_empty() {
                    let pats: Vec<&str> = missing.iter().map(|m| m.1.as_str()).collect();
                    let ins: String = missing
                        .iter()
                        .map(|m| format!(" (case {} (fail \"TODO: handle {}\"))", m.1, m.0))
                        .collect();
                    self.err(
                        Diag::error(
                            "E0501",
                            e.span,
                            &format!("match is not exhaustive: missing {}", pats.join(", ")),
                        )
                        .with_fix_insert(
                            e.span.end - 1,
                            &ins,
                            "add the missing arms (replace each placeholder body)",
                        ),
                    );
                }
            }
        }
        if arms.is_empty() {
            return Ty::Never;
        }
        if discard {
            return Ty::Unit;
        }
        result.unwrap_or(first_ty.unwrap_or(Ty::Unit))
    }

    /// Returns true if the pattern is refutable.
    fn check_pat(&mut self, p: &Pat, t: &Ty, boxed: bool) -> bool {
        let kind = if boxed {
            BindKind::BoxBorrow
        } else {
            BindKind::Borrow
        };
        match p {
            Pat::Wild(_) => false,
            Pat::Bind(name, span) => {
                // A bare name that matches a variant is almost certainly a mistake.
                if let Ty::User(en) = t {
                    if let Some(ei) = self.info.enums.get(en) {
                        if ei.variants.iter().any(|v| &v.0 == name) {
                            self.err(
                                Diag::error(
                                    "E0504",
                                    *span,
                                    &format!(
                                        "variants are qualified in patterns: write {en}.{name}"
                                    ),
                                )
                                .with_fix_replace(
                                    *span,
                                    &format!("{en}.{name}"),
                                    "qualify the variant",
                                ),
                            );
                        }
                    }
                }
                if let Some(fixed) = known_alias(name)
                    .filter(|a| ["none", "some", "ok", "err", "true", "false"].contains(a))
                {
                    self.err(
                        Diag::error(
                            "E0504",
                            *span,
                            &format!("'{name}' is not a constructor in ail; use '{fixed}'"),
                        )
                        .with_fix_replace(
                            *span,
                            fixed,
                            &format!("use {fixed}"),
                        ),
                    );
                }
                let b = self.local(name, t.clone(), kind);
                let bid = self.bind(b);
                self.decl.insert(pat_key(*span), bid);
                false
            }
            Pat::Int(v, span) => {
                if boxed {
                    self.err(Diag::error("E0503", *span, "fields of recursive types can only be bound by name in patterns; match them in a nested match"));
                }
                match t {
                    Ty::Int(k) => {
                        let (lo, hi) = k.range();
                        if *v < lo || *v > hi {
                            self.err(Diag::error(
                                "E0203",
                                *span,
                                &format!("literal {v} is out of range for {}", k.name()),
                            ));
                        }
                    }
                    Ty::Err => {}
                    other => self.err(Diag::error(
                        "E0505",
                        *span,
                        &format!("integer pattern on a value of type {other}"),
                    )),
                }
                true
            }
            Pat::Str(_, span) => {
                if !matches!(t, Ty::Str | Ty::Err) {
                    self.err(Diag::error(
                        "E0505",
                        *span,
                        &format!("string pattern on a value of type {t}"),
                    ));
                }
                true
            }
            Pat::Ctor(name, subs, span) => {
                if boxed {
                    self.err(Diag::error("E0503", *span, "fields of recursive types can only be bound by name in patterns; match them in a nested match"));
                }
                let mut refutable_sub = false;
                let mut sub = |c: &mut Checker, pats: &[Pat], tys: &[Ty], boxes: &[bool]| {
                    for ((sp, st), bx) in pats.iter().zip(tys.iter()).zip(boxes.iter()) {
                        if c.check_pat(sp, st, *bx) {
                            refutable_sub = true;
                        }
                    }
                };
                let arity_err = |c: &mut Checker, want: usize, shape: &str| {
                    c.err(Diag::error(
                        "E0506",
                        *span,
                        &format!("pattern {name} takes {want} sub-pattern(s): {shape}"),
                    ));
                };
                match (name.as_str(), t) {
                    ("true" | "false", Ty::Bool) => {
                        if !subs.is_empty() {
                            arity_err(self, 0, name);
                        }
                    }
                    ("none", Ty::Opt(_)) => {
                        if !subs.is_empty() {
                            arity_err(self, 0, "none");
                        }
                    }
                    ("some", Ty::Opt(inner)) => {
                        if subs.len() != 1 {
                            arity_err(self, 1, "(some x)");
                        } else {
                            sub(self, subs, &[(**inner).clone()], &[false]);
                        }
                    }
                    ("ok" | "err", Ty::Res(a, b)) => {
                        if subs.len() != 1 {
                            arity_err(self, 1, "(ok x)");
                        } else {
                            let inner = if name == "ok" {
                                (**a).clone()
                            } else {
                                (**b).clone()
                            };
                            sub(self, subs, &[inner], &[false]);
                        }
                    }
                    (_, Ty::Err) => {
                        for s in subs {
                            self.check_pat(s, &Ty::Err, false);
                        }
                    }
                    (_, Ty::User(en)) if name.contains('.') && self.info.enums.contains_key(en) => {
                        let (pen, vn) = name.split_once('.').unwrap();
                        if pen != en {
                            self.err(Diag::error(
                                "E0505",
                                *span,
                                &format!("pattern of enum {pen} on a value of type {en}"),
                            ));
                            return true;
                        }
                        let ei = self.info.enums[en].clone();
                        let Some(vi) = ei.variants.iter().position(|v| v.0 == vn) else {
                            self.unknown_variant(*span, en, vn);
                            return true;
                        };
                        let fts = &ei.variants[vi].1;
                        if subs.len() != fts.len() {
                            let shape = if fts.is_empty() {
                                name.clone()
                            } else {
                                format!("({name}{})", " _".repeat(fts.len()))
                            };
                            let d = Diag::error(
                                "E0506",
                                *span,
                                &format!(
                                    "pattern {name} takes {} sub-pattern(s): {shape}",
                                    fts.len()
                                ),
                            )
                            .with_fix_replace(
                                *span,
                                &shape,
                                "use the right number of sub-patterns",
                            );
                            self.err(d);
                        } else {
                            let boxes: Vec<bool> = (0..fts.len())
                                .map(|i| self.info.is_boxed(en, vi, i))
                                .collect();
                            sub(self, subs, fts, &boxes);
                        }
                    }
                    _ => {
                        self.err(Diag::error(
                            "E0505",
                            *span,
                            &format!("pattern '{name}' does not match a value of type {t}"),
                        ));
                        return true;
                    }
                }
                refutable_sub
            }
        }
    }
}

/// Two places overlap unless they diverge into different fields.
fn paths_overlap(a: &[String], b: &[String]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x == y || x == "[]" || y == "[]")
}

fn ctor_key(name: &str) -> String {
    match name.split_once('.') {
        Some((_, v)) => v.to_string(),
        None => name.to_string(),
    }
}

fn has_float_lit(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Float(_) => true,
        ExprKind::Call(n, _, a) if n == "-" && a.len() == 1 => has_float_lit(a[0].expr()),
        _ => false,
    }
}

fn placeholder(t: &Ty) -> String {
    match t {
        Ty::Int(_) => "0".into(),
        Ty::F32 | Ty::F64 => "0.0".into(),
        Ty::Bool => "false".into(),
        Ty::Str => "\"\"".into(),
        Ty::Unit => "unit".into(),
        Ty::Vec(_) => "(vec)".into(),
        Ty::Map(..) => "(map)".into(),
        Ty::Opt(_) => "none".into(),
        _ => "TODO".into(),
    }
}

/// True if `(json-str x)` can encode a value of type `t` directly: Json itself, or numbers, Bool,
/// Str, Vec, Option (none is null), maps with Str keys, and structs made of those (an object
/// with its fields in name order).
pub fn json_encodable(info: &TypeInfo, t: &Ty, seen: &mut Vec<String>) -> bool {
    match t {
        Ty::Int(_) | Ty::F32 | Ty::F64 | Ty::Bool | Ty::Str => true,
        Ty::Vec(x) | Ty::Opt(x) => json_encodable(info, x, seen),
        Ty::Map(k, v) => **k == Ty::Str && json_encodable(info, v, seen),
        Ty::User(n) if n == "Json" => true,
        Ty::User(n) => match info.structs.get(n) {
            Some(si) => {
                if seen.contains(n) {
                    return true;
                }
                seen.push(n.clone());
                let ok = si.fields.iter().all(|(_, ft)| json_encodable(info, ft, seen));
                seen.pop();
                ok
            }
            None => false,
        },
        _ => false,
    }
}

/// Pre-order walk over an expression tree.
pub fn walk(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Var(_)
        | ExprKind::Break
        | ExprKind::Continue => {}
        ExprKind::Call(_, _, args) => args.iter().for_each(|a| walk(a.expr(), f)),
        ExprKind::Let { value, .. } => walk(value, f),
        ExprKind::Set(a, b) => {
            walk(a, f);
            walk(b, f)
        }
        ExprKind::If(c, t, e2) => {
            walk(c, f);
            walk(t, f);
            if let Some(x) = e2 {
                walk(x, f)
            }
        }
        ExprKind::Do(es) => es.iter().for_each(|x| walk(x, f)),
        ExprKind::While(c, b) => {
            walk(c, f);
            b.iter().for_each(|x| walk(x, f))
        }
        ExprKind::For(_, _, it, b) => {
            match it {
                ForIter::Range(a, c) => {
                    walk(a, f);
                    walk(c, f)
                }
                ForIter::Each(v) => walk(v, f),
            }
            b.iter().for_each(|x| walk(x, f))
        }
        ExprKind::Return(v) => {
            if let Some(v) = v {
                walk(v, f)
            }
        }
        ExprKind::Match(s, arms) => {
            walk(s, f);
            for a in arms {
                a.body.iter().for_each(|x| walk(x, f))
            }
        }
        ExprKind::New(_, _, fs) => fs.iter().for_each(|x| walk(&x.2, f)),
        ExprKind::With(b, fs) => {
            walk(b, f);
            fs.iter().for_each(|x| walk(&x.2, f))
        }
        ExprKind::Field(b, _, _) => walk(b, f),
        ExprKind::Try(x) | ExprKind::As(_, x) | ExprKind::Copy(x) => walk(x, f),
    }
}

#[allow(dead_code)]
fn unused(c: &Checker, e: &Expr, r: BindId) -> bool {
    c.mentions(e, r)
}
