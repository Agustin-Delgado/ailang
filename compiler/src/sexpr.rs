//! Reader: source text -> s-expressions with byte spans.

use crate::diag::{Diag, Span};

#[derive(Clone, Debug)]
pub enum SKind {
    List(Vec<SExpr>),
    Sym(String),
    Int(i128),
    Float(f64),
    Str(String),
}

#[derive(Clone, Debug)]
pub struct SExpr {
    pub kind: SKind,
    pub span: Span,
}

impl SExpr {
    pub fn sym(&self) -> Option<&str> {
        match &self.kind {
            SKind::Sym(s) => Some(s),
            _ => None,
        }
    }
    pub fn list(&self) -> Option<&[SExpr]> {
        match &self.kind {
            SKind::List(l) => Some(l),
            _ => None,
        }
    }
    /// Head symbol of a list form: `(head ...)`.
    pub fn head(&self) -> Option<&str> {
        self.list().and_then(|l| l.first()).and_then(|h| h.sym())
    }
}

pub fn read(src: &str) -> Result<Vec<SExpr>, Diag> {
    let mut r = Reader {
        b: src.as_bytes(),
        src,
        pos: 0,
    };
    let mut out = Vec::new();
    loop {
        r.skip_ws();
        if r.pos >= r.b.len() {
            return Ok(out);
        }
        out.push(r.expr()?);
    }
}

struct Reader<'a> {
    b: &'a [u8],
    src: &'a str,
    pos: usize,
}

fn is_delim(c: u8) -> bool {
    c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b'"' || c == b';'
}

impl<'a> Reader<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.b.len() {
            let c = self.b[self.pos];
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else if c == b';' {
                while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn expr(&mut self) -> Result<SExpr, Diag> {
        let start = self.pos;
        let c = self.b[self.pos];
        match c {
            b'(' => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    if self.pos >= self.b.len() {
                        return Err(Diag::error(
                            "E0001",
                            Span::new(start, start + 1),
                            "unclosed '(': reached end of file",
                        )
                        .with_fix_insert(
                            self.b.len(),
                            ")",
                            "close the list",
                        ));
                    }
                    if self.b[self.pos] == b')' {
                        self.pos += 1;
                        return Ok(SExpr {
                            kind: SKind::List(items),
                            span: Span::new(start, self.pos),
                        });
                    }
                    items.push(self.expr()?);
                }
            }
            b')' => Err(
                Diag::error("E0002", Span::new(start, start + 1), "unexpected ')'")
                    .with_fix_replace(Span::new(start, start + 1), "", "remove the extra ')'"),
            ),
            b'"' => self.string(),
            _ => {
                while self.pos < self.b.len() && !is_delim(self.b[self.pos]) {
                    self.pos += 1;
                }
                let text = &self.src[start..self.pos];
                let span = Span::new(start, self.pos);
                Ok(SExpr {
                    kind: atom(text, span)?,
                    span,
                })
            }
        }
    }

    fn string(&mut self) -> Result<SExpr, Diag> {
        let start = self.pos;
        self.pos += 1;
        let mut s = String::new();
        loop {
            if self.pos >= self.b.len() {
                return Err(Diag::error(
                    "E0003",
                    Span::new(start, start + 1),
                    "unterminated string literal",
                )
                .with_fix_insert(self.b.len(), "\"", "close the string"));
            }
            let ch = self.src[self.pos..].chars().next().unwrap();
            self.pos += ch.len_utf8();
            match ch {
                '"' => break,
                '\\' => {
                    let esc_start = self.pos - 1;
                    let e = self.src[self.pos..].chars().next();
                    let Some(e) = e else { continue };
                    self.pos += e.len_utf8();
                    match e {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '0' => s.push('\0'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        'u' => {
                            // \u{XXXX}
                            let rest = &self.src[self.pos..];
                            let close = rest.find('}');
                            let ok = rest.starts_with('{') && close.is_some();
                            let parsed = if ok {
                                u32::from_str_radix(&rest[1..close.unwrap()], 16)
                                    .ok()
                                    .and_then(char::from_u32)
                            } else {
                                None
                            };
                            match parsed {
                                Some(c) => {
                                    s.push(c);
                                    self.pos += close.unwrap() + 1;
                                }
                                None => {
                                    return Err(Diag::error(
                                        "E0004",
                                        Span::new(esc_start, self.pos),
                                        "invalid unicode escape; expected \\u{HEX}",
                                    ));
                                }
                            }
                        }
                        other => {
                            return Err(Diag::error(
                                "E0004",
                                Span::new(esc_start, self.pos),
                                &format!(
                                    "unknown escape '\\{other}'; valid: \\n \\t \\r \\0 \\\\ \\\" \\u{{HEX}}"
                                ),
                            ));
                        }
                    }
                }
                c => s.push(c),
            }
        }
        Ok(SExpr {
            kind: SKind::Str(s),
            span: Span::new(start, self.pos),
        })
    }
}

fn atom(text: &str, span: Span) -> Result<SKind, Diag> {
    let first = text.as_bytes()[0];
    let numeric_start = first.is_ascii_digit()
        || ((first == b'-' || first == b'+')
            && text.len() > 1
            && text.as_bytes()[1].is_ascii_digit());
    if !numeric_start {
        return Ok(SKind::Sym(text.to_string()));
    }
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    let (neg, body) = match clean.strip_prefix('-') {
        Some(b) => (true, b.to_string()),
        None => (false, clean.trim_start_matches('+').to_string()),
    };
    let bad = || Diag::error("E0005", span, &format!("malformed number literal '{text}'"));
    if let Some(hex) = body.strip_prefix("0x") {
        let v = i128::from_str_radix(hex, 16).map_err(|_| bad())?;
        return Ok(SKind::Int(if neg { -v } else { v }));
    }
    if let Some(bin) = body.strip_prefix("0b") {
        let v = i128::from_str_radix(bin, 2).map_err(|_| bad())?;
        return Ok(SKind::Int(if neg { -v } else { v }));
    }
    if body.contains('.') || body.contains('e') || body.contains('E') {
        let v: f64 = body.parse().map_err(|_| bad())?;
        return Ok(SKind::Float(if neg { -v } else { v }));
    }
    let v: i128 = body.parse().map_err(|_| bad())?;
    Ok(SKind::Int(if neg { -v } else { v }))
}
