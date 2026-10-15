//! AST and lowering from s-expressions. Every syntax error carries a fix when one is obvious.

use crate::diag::{Diag, Span};
use crate::sexpr::{SExpr, SKind};

#[derive(Clone, Debug)]
pub struct TypeExpr {
    pub name: String,
    pub args: Vec<TypeExpr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub mutable: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    /// Byte offset where a new parameter can be inserted (before the closing paren of `params`,
    /// or right after the name if there is no `params` clause).
    pub param_insert_at: usize,
    pub has_params_clause: bool,
    pub ret: Option<TypeExpr>,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub body: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, TypeExpr)>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<TypeExpr>, Span)>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ConstDef {
    pub name: String,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TestDef {
    pub name: String,
    pub body: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Item {
    Fn(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Const(ConstDef),
    Test(TestDef),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub items: Vec<Item>,
    pub expr_count: u32,
}

#[derive(Clone, Debug)]
pub enum Arg {
    Val(Expr),
    Mut(Expr, Span),
}

impl Arg {
    pub fn expr(&self) -> &Expr {
        match self {
            Arg::Val(e) | Arg::Mut(e, _) => e,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Pat {
    Wild(Span),
    Bind(String, Span),
    Int(i128, Span),
    Str(String, Span),
    /// `none`, `(some p)`, `(ok p)`, `(err p)`, `true`, `false`, `Enum.Variant`, `(Enum.Variant p...)`
    Ctor(String, Vec<Pat>, Span),
}

impl Pat {
    pub fn span(&self) -> Span {
        match self {
            Pat::Wild(s) | Pat::Bind(_, s) | Pat::Int(_, s) | Pat::Str(_, s) | Pat::Ctor(_, _, s) => *s,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub pat: Pat,
    pub body: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ForIter {
    Range(Box<Expr>, Box<Expr>),
    Each(Box<Expr>),
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Int(i128),
    Float(f64),
    Str(String),
    Var(String),
    Call(String, Span, Vec<Arg>),
    Let { name: String, name_span: Span, kw_span: Span, mutable: bool, ty: Option<TypeExpr>, value: Box<Expr> },
    Set(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Do(Vec<Expr>),
    While(Box<Expr>, Vec<Expr>),
    For(String, Span, ForIter, Vec<Expr>),
    Break,
    Continue,
    Return(Option<Box<Expr>>),
    Match(Box<Expr>, Vec<Arm>),
    New(String, Span, Vec<(String, Span, Expr)>),
    With(Box<Expr>, Vec<(String, Span, Expr)>),
    Field(Box<Expr>, String, Span),
    Try(Box<Expr>),
    As(TypeExpr, Box<Expr>),
    Copy(Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub id: u32,
    pub kind: ExprKind,
    pub span: Span,
}

pub struct Lower {
    next_id: u32,
    pub diags: Vec<Diag>,
}

pub fn lower(forms: &[SExpr]) -> (Program, Vec<Diag>) {
    let mut l = Lower { next_id: 0, diags: Vec::new() };
    let mut items = Vec::new();
    for f in forms {
        if let Some(it) = l.item(f) {
            items.push(it);
        }
    }
    (Program { items, expr_count: l.next_id }, l.diags)
}

const ITEM_FORMS: &str = "fn, struct, enum, const, test";

impl Lower {
    fn err(&mut self, code: &'static str, span: Span, msg: &str) {
        self.diags.push(Diag::error(code, span, msg));
    }

    fn mk(&mut self, kind: ExprKind, span: Span) -> Expr {
        let id = self.next_id;
        self.next_id += 1;
        Expr { id, kind, span }
    }

    fn item(&mut self, f: &SExpr) -> Option<Item> {
        let Some(list) = f.list() else {
            self.err("E0010", f.span, &format!("top-level forms must be lists: ({ITEM_FORMS} ...)"));
            return None;
        };
        let head = f.head().unwrap_or("");
        match head {
            "fn" => self.fn_def(list, f.span).map(Item::Fn),
            "struct" => self.struct_def(list, f.span).map(Item::Struct),
            "enum" => self.enum_def(list, f.span).map(Item::Enum),
            "const" => {
                if list.len() != 4 {
                    self.err("E0011", f.span, "const form is (const NAME Type value)");
                    return None;
                }
                let name = self.ident(&list[1])?;
                let ty = self.type_expr(&list[2])?;
                let value = self.expr(&list[3])?;
                Some(Item::Const(ConstDef { name, ty, value, span: f.span }))
            }
            "test" => {
                let name = match list.get(1).map(|x| &x.kind) {
                    Some(SKind::Str(s)) => s.clone(),
                    _ => {
                        self.err("E0012", f.span, "test form is (test \"name\" body...)");
                        return None;
                    }
                };
                let body = self.exprs(&list[2..]);
                Some(Item::Test(TestDef { name, body, span: f.span }))
            }
            _ => {
                self.err("E0010", f.span, &format!("unknown top-level form '{head}'; expected one of: {ITEM_FORMS}"));
                None
            }
        }
    }

    fn ident(&mut self, s: &SExpr) -> Option<String> {
        match s.sym() {
            Some(n) if valid_ident(n) => Some(n.to_string()),
            _ => {
                self.err("E0013", s.span, "expected an identifier (letters, digits, '-', '_', '?')");
                None
            }
        }
    }

    fn fn_def(&mut self, list: &[SExpr], span: Span) -> Option<FnDef> {
        let name_s = list.get(1)?;
        let name = self.ident(name_s)?;
        let mut def = FnDef {
            name,
            name_span: name_s.span,
            params: Vec::new(),
            param_insert_at: name_s.span.end,
            has_params_clause: false,
            ret: None,
            requires: Vec::new(),
            ensures: Vec::new(),
            body: Vec::new(),
            span,
        };
        let mut has_body = false;
        for clause in &list[2..] {
            let Some(cl) = clause.list() else {
                self.err("E0014", clause.span, "fn clauses are lists: (params ...) (returns T) (requires e) (ensures e) (body ...)");
                continue;
            };
            match clause.head().unwrap_or("") {
                "params" => {
                    def.has_params_clause = true;
                    def.param_insert_at = clause.span.end - 1;
                    for p in &cl[1..] {
                        if let Some(p) = self.param(p) {
                            def.params.push(p);
                        }
                    }
                }
                "returns" => {
                    if cl.len() != 2 {
                        self.err("E0015", clause.span, "returns clause is (returns Type)");
                        continue;
                    }
                    def.ret = self.type_expr(&cl[1]);
                }
                "requires" => {
                    for e in &cl[1..] {
                        if let Some(e) = self.expr(e) {
                            def.requires.push(e);
                        }
                    }
                }
                "ensures" => {
                    for e in &cl[1..] {
                        if let Some(e) = self.expr(e) {
                            def.ensures.push(e);
                        }
                    }
                }
                "body" => {
                    has_body = true;
                    def.body = self.exprs(&cl[1..]);
                }
                other => {
                    self.diags.push(
                        Diag::error(
                            "E0014",
                            clause.span,
                            &format!("unknown fn clause '{other}'; valid clauses: params, returns, requires, ensures, body"),
                        )
                        .note("expressions go inside (body ...)"),
                    );
                }
            }
        }
        if !has_body {
            self.diags.push(
                Diag::error("E0016", span, &format!("fn '{}' has no (body ...) clause", def.name))
                    .with_fix_insert(span.end - 1, " (body unit)", "add an empty body"),
            );
        }
        Some(def)
    }

    fn param(&mut self, p: &SExpr) -> Option<Param> {
        let bad = |l: &mut Lower| {
            l.err("E0017", p.span, "parameter is (name Type) or (name (mut Type))");
            None
        };
        let Some(pl) = p.list() else { return bad(self) };
        if pl.len() != 2 {
            return bad(self);
        }
        let name = self.ident(&pl[0])?;
        let (ty_s, mutable) = match pl[1].head() {
            Some("mut") if pl[1].list().unwrap().len() == 2 => (&pl[1].list().unwrap()[1], true),
            _ => (&pl[1], false),
        };
        let ty = self.type_expr(ty_s)?;
        Some(Param { name, ty, mutable, span: p.span })
    }

    pub fn type_expr(&mut self, s: &SExpr) -> Option<TypeExpr> {
        match &s.kind {
            SKind::Sym(n) => Some(TypeExpr { name: n.clone(), args: Vec::new(), span: s.span }),
            SKind::List(l) if !l.is_empty() && l[0].sym().is_some() => {
                let mut args = Vec::new();
                for a in &l[1..] {
                    args.push(self.type_expr(a)?);
                }
                Some(TypeExpr { name: l[0].sym().unwrap().to_string(), args, span: s.span })
            }
            _ => {
                self.err("E0018", s.span, "expected a type, e.g. I64, Str, (Vec I64), (Option Str), Point");
                None
            }
        }
    }

    fn struct_def(&mut self, list: &[SExpr], span: Span) -> Option<StructDef> {
        let name = self.ident(list.get(1)?)?;
        let mut fields = Vec::new();
        for f in &list[2..] {
            match f.list() {
                Some(fl) if fl.len() == 2 => {
                    let fname = self.ident(&fl[0])?;
                    let ty = self.type_expr(&fl[1])?;
                    fields.push((fname, ty));
                }
                _ => self.err("E0019", f.span, "struct field is (name Type)"),
            }
        }
        Some(StructDef { name, fields, span })
    }

    fn enum_def(&mut self, list: &[SExpr], span: Span) -> Option<EnumDef> {
        let name = self.ident(list.get(1)?)?;
        let mut variants = Vec::new();
        for v in &list[2..] {
            match v.list() {
                Some(vl) if !vl.is_empty() => {
                    let vname = self.ident(&vl[0])?;
                    let mut tys = Vec::new();
                    for t in &vl[1..] {
                        tys.push(self.type_expr(t)?);
                    }
                    variants.push((vname, tys, v.span));
                }
                _ => {
                    let mut d = Diag::error("E0020", v.span, "enum variant is (Name FieldType...)");
                    if let Some(s) = v.sym() {
                        d = d.with_fix_replace(v.span, &format!("({s})"), "wrap the variant in parens");
                    }
                    self.diags.push(d);
                }
            }
        }
        Some(EnumDef { name, variants, span })
    }

    pub fn exprs(&mut self, xs: &[SExpr]) -> Vec<Expr> {
        xs.iter().filter_map(|x| self.expr(x)).collect()
    }

    fn fields(&mut self, xs: &[SExpr]) -> Vec<(String, Span, Expr)> {
        let mut out = Vec::new();
        for f in xs {
            match f.list() {
                Some(fl) if fl.len() == 2 && fl[0].sym().is_some() => {
                    if let Some(e) = self.expr(&fl[1]) {
                        out.push((fl[0].sym().unwrap().to_string(), fl[0].span, e));
                    }
                }
                _ => self.err("E0021", f.span, "field initializer is (field-name value)"),
            }
        }
        out
    }

    pub fn expr(&mut self, s: &SExpr) -> Option<Expr> {
        let span = s.span;
        let kind = match &s.kind {
            SKind::Int(v) => ExprKind::Int(*v),
            SKind::Float(v) => ExprKind::Float(*v),
            SKind::Str(v) => ExprKind::Str(v.clone()),
            SKind::Sym(n) => match n.as_str() {
                "break" => ExprKind::Break,
                "continue" => ExprKind::Continue,
                _ => ExprKind::Var(n.clone()),
            },
            SKind::List(l) => {
                if l.is_empty() {
                    self.diags.push(
                        Diag::error("E0022", span, "empty list '()' is not an expression")
                            .with_fix_replace(span, "unit", "use 'unit' for the unit value"),
                    );
                    return None;
                }
                let Some(head) = l[0].sym() else {
                    self.err("E0023", l[0].span, "the first element of a list form must be a name (function or keyword)");
                    return None;
                };
                let args = &l[1..];
                return self.form(head, l[0].span, args, span);
            }
        };
        Some(self.mk(kind, span))
    }

    fn arity(&mut self, form: &str, args: &[SExpr], n: usize, span: Span, shape: &str) -> bool {
        if args.len() != n {
            self.err("E0024", span, &format!("'{form}' takes {n} argument(s): {shape}"));
            return false;
        }
        true
    }

    fn form(&mut self, head: &str, head_span: Span, args: &[SExpr], span: Span) -> Option<Expr> {
        let kind = match head {
            "let" | "var" => {
                let mutable = head == "var";
                let shape = format!("({head} name value) or ({head} name Type value)");
                if args.len() != 2 && args.len() != 3 {
                    self.err("E0024", span, &format!("'{head}' form is {shape}"));
                    return None;
                }
                let name = self.ident(&args[0])?;
                let ty = if args.len() == 3 { Some(self.type_expr(&args[1])?) } else { None };
                let value = self.expr(args.last().unwrap())?;
                ExprKind::Let { name, name_span: args[0].span, kw_span: head_span, mutable, ty, value: Box::new(value) }
            }
            "set" => {
                if !self.arity("set", args, 2, span, "(set place value)") {
                    return None;
                }
                ExprKind::Set(Box::new(self.expr(&args[0])?), Box::new(self.expr(&args[1])?))
            }
            "if" => {
                if args.len() != 2 && args.len() != 3 {
                    self.err("E0024", span, "'if' form is (if cond then else) or (if cond then); use (do ...) for several expressions");
                    return None;
                }
                let c = self.expr(&args[0])?;
                let t = self.expr(&args[1])?;
                let e = if args.len() == 3 { Some(Box::new(self.expr(&args[2])?)) } else { None };
                ExprKind::If(Box::new(c), Box::new(t), e)
            }
            "do" => ExprKind::Do(self.exprs(args)),
            "while" => {
                if args.is_empty() {
                    self.err("E0024", span, "'while' form is (while cond body...)");
                    return None;
                }
                let c = self.expr(&args[0])?;
                ExprKind::While(Box::new(c), self.exprs(&args[1..]))
            }
            "for" => {
                if args.len() < 2 {
                    self.err("E0024", span, "'for' form is (for x (range a b) body...) or (for x vec body...)");
                    return None;
                }
                let var = self.ident(&args[0])?;
                let iter = if args[1].head() == Some("range") {
                    let r = args[1].list().unwrap();
                    if r.len() != 3 {
                        self.err("E0024", args[1].span, "(range start end) takes 2 arguments; end is exclusive");
                        return None;
                    }
                    ForIter::Range(Box::new(self.expr(&r[1])?), Box::new(self.expr(&r[2])?))
                } else {
                    ForIter::Each(Box::new(self.expr(&args[1])?))
                };
                ExprKind::For(var, args[0].span, iter, self.exprs(&args[2..]))
            }
            "return" => match args.len() {
                0 => ExprKind::Return(None),
                1 => ExprKind::Return(Some(Box::new(self.expr(&args[0])?))),
                _ => {
                    self.err("E0024", span, "'return' takes at most one value");
                    return None;
                }
            },
            "match" => {
                if args.is_empty() {
                    self.err("E0024", span, "'match' form is (match value (case pattern body...)...)");
                    return None;
                }
                let scrut = self.expr(&args[0])?;
                let mut arms = Vec::new();
                for a in &args[1..] {
                    let Some(al) = a.list().filter(|al| a.head() == Some("case") && al.len() >= 3) else {
                        self.err("E0025", a.span, "match arm is (case pattern body...)");
                        continue;
                    };
                    if let Some(pat) = self.pat(&al[1]) {
                        let body = self.exprs(&al[2..]);
                        arms.push(Arm { pat, body, span: a.span });
                    }
                }
                ExprKind::Match(Box::new(scrut), arms)
            }
            "new" => {
                if args.is_empty() {
                    self.err("E0024", span, "'new' form is (new TypeName (field value)...)");
                    return None;
                }
                let name = self.ident(&args[0])?;
                ExprKind::New(name, args[0].span, self.fields(&args[1..]))
            }
            "with" => {
                if args.is_empty() {
                    self.err("E0024", span, "'with' form is (with value (field new-value)...)");
                    return None;
                }
                let base = self.expr(&args[0])?;
                ExprKind::With(Box::new(base), self.fields(&args[1..]))
            }
            "." => {
                if args.len() < 2 {
                    self.err("E0024", span, "field access is (. value field) or (. value field subfield...)");
                    return None;
                }
                let mut e = self.expr(&args[0])?;
                for f in &args[1..] {
                    let Some(fname) = f.sym() else {
                        self.err("E0013", f.span, "field name expected");
                        return None;
                    };
                    let fspan = Span::new(span.start, f.span.end);
                    e = self.mk(ExprKind::Field(Box::new(e), fname.to_string(), f.span), fspan);
                }
                return Some(e);
            }
            "try" => {
                if !self.arity("try", args, 1, span, "(try result-or-option)") {
                    return None;
                }
                ExprKind::Try(Box::new(self.expr(&args[0])?))
            }
            "as" => {
                if !self.arity("as", args, 2, span, "(as Type value)") {
                    return None;
                }
                let t = self.type_expr(&args[0])?;
                ExprKind::As(t, Box::new(self.expr(&args[1])?))
            }
            "copy" => {
                if !self.arity("copy", args, 1, span, "(copy value)") {
                    return None;
                }
                ExprKind::Copy(Box::new(self.expr(&args[0])?))
            }
            "mut" => {
                self.err("E0026", span, "(mut x) is only allowed as a call argument, e.g. (push (mut v) 1)");
                return None;
            }
            "case" => {
                self.err("E0025", span, "(case ...) is only allowed inside (match ...)");
                return None;
            }
            "range" => {
                self.err("E0027", span, "(range a b) is only allowed as the iterable of (for ...)");
                return None;
            }
            _ => {
                let mut cargs = Vec::new();
                for a in args {
                    if a.head() == Some("mut") {
                        let al = a.list().unwrap();
                        if al.len() != 2 {
                            self.err("E0026", a.span, "(mut place) takes exactly one place");
                            return None;
                        }
                        cargs.push(Arg::Mut(self.expr(&al[1])?, a.span));
                    } else {
                        cargs.push(Arg::Val(self.expr(a)?));
                    }
                }
                ExprKind::Call(head.to_string(), head_span, cargs)
            }
        };
        Some(self.mk(kind, span))
    }

    fn pat(&mut self, s: &SExpr) -> Option<Pat> {
        match &s.kind {
            SKind::Int(v) => Some(Pat::Int(*v, s.span)),
            SKind::Str(v) => Some(Pat::Str(v.clone(), s.span)),
            SKind::Float(_) => {
                self.err("E0028", s.span, "float literals cannot be patterns; use (if (== x 1.5) ...)");
                None
            }
            SKind::Sym(n) => Some(match n.as_str() {
                "_" => Pat::Wild(s.span),
                "none" | "true" | "false" => Pat::Ctor(n.clone(), Vec::new(), s.span),
                _ if n.contains('.') => Pat::Ctor(n.clone(), Vec::new(), s.span),
                _ if valid_ident(n) => Pat::Bind(n.clone(), s.span),
                _ => {
                    self.err("E0028", s.span, "invalid pattern");
                    return None;
                }
            }),
            SKind::List(l) => {
                let Some(h) = l.first().and_then(|h| h.sym()) else {
                    self.err("E0028", s.span, "pattern list must start with a constructor: (some x), (ok x), (err e), (Enum.Variant a b)");
                    return None;
                };
                let mut subs = Vec::new();
                for p in &l[1..] {
                    subs.push(self.pat(p)?);
                }
                Some(Pat::Ctor(h.to_string(), subs, s.span))
            }
        }
    }
}

pub fn valid_ident(n: &str) -> bool {
    let mut cs = n.chars();
    match cs.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '?' || c == '.')
}
