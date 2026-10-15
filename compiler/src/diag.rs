//! Diagnostics: stable codes, exact spans, machine-applicable fixes.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
    }
}

#[derive(Clone, Debug)]
pub struct Edit {
    pub span: Span,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Fix {
    pub description: String,
    pub edits: Vec<Edit>,
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub code: &'static str,
    pub severity: &'static str,
    pub span: Span,
    pub message: String,
    pub path: String,
    pub expected: Option<String>,
    pub found: Option<String>,
    pub notes: Vec<String>,
    pub fixes: Vec<Fix>,
}

impl Diag {
    pub fn error(code: &'static str, span: Span, message: &str) -> Diag {
        Diag {
            code,
            severity: "error",
            span,
            message: message.to_string(),
            path: String::new(),
            expected: None,
            found: None,
            notes: Vec::new(),
            fixes: Vec::new(),
        }
    }
    pub fn warning(code: &'static str, span: Span, message: &str) -> Diag {
        let mut d = Diag::error(code, span, message);
        d.severity = "warning";
        d
    }
    pub fn with_expected(mut self, expected: impl Into<String>, found: impl Into<String>) -> Diag {
        self.expected = Some(expected.into());
        self.found = Some(found.into());
        self
    }
    pub fn note(mut self, n: impl Into<String>) -> Diag {
        self.notes.push(n.into());
        self
    }
    pub fn with_fix(mut self, description: &str, edits: Vec<Edit>) -> Diag {
        self.fixes.push(Fix {
            description: description.to_string(),
            edits,
        });
        self
    }
    pub fn with_fix_replace(self, span: Span, text: &str, description: &str) -> Diag {
        self.with_fix(
            description,
            vec![Edit {
                span,
                text: text.to_string(),
            }],
        )
    }
    pub fn with_fix_insert(self, at: usize, text: &str, description: &str) -> Diag {
        self.with_fix(
            description,
            vec![Edit {
                span: Span::new(at, at),
                text: text.to_string(),
            }],
        )
    }
}

pub struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(src: &str) -> LineIndex {
        let mut starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        LineIndex { starts }
    }
    /// Byte offset where 1-based `line` starts.
    pub fn line_start(&self, line: usize) -> usize {
        self.starts.get(line - 1).copied().unwrap_or(*self.starts.last().unwrap())
    }
    /// 1-based line and column (column counted in bytes).
    pub fn pos(&self, offset: usize) -> (usize, usize) {
        let line = match self.starts.binary_search(&offset) {
            Ok(l) => l,
            Err(l) => l - 1,
        };
        (line + 1, offset - self.starts[line] + 1)
    }
}

/// The diagnostic with every span moved by `-base` (into the coordinates of one file).
/// Fix edits that fall outside the file are dropped.
pub fn shifted(d: &Diag, base: usize, len: usize) -> Diag {
    let mv = |s: Span| Span::new(s.start.saturating_sub(base), s.end.saturating_sub(base));
    let mut o = d.clone();
    o.span = mv(d.span);
    for f in o.fixes.iter_mut() {
        f.edits.retain(|e| e.span.start >= base && e.span.end <= base + len);
        for e in f.edits.iter_mut() {
            e.span = mv(e.span);
        }
    }
    o.fixes.retain(|f| !f.edits.is_empty());
    o
}

pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn span_json(li: &LineIndex, file: &str, s: Span) -> String {
    let (l1, c1) = li.pos(s.start);
    let (l2, c2) = li.pos(s.end);
    format!(
        "{{\"file\":{},\"start\":{},\"end\":{},\"line\":{},\"col\":{},\"end_line\":{},\"end_col\":{}}}",
        json_str(file),
        s.start,
        s.end,
        l1,
        c1,
        l2,
        c2
    )
}

pub fn to_json(d: &Diag, src: &str, file: &str) -> String {
    let li = LineIndex::new(src);
    let mut o = format!(
        "{{\"code\":{},\"severity\":{},\"message\":{},\"span\":{}",
        json_str(d.code),
        json_str(d.severity),
        json_str(&d.message),
        span_json(&li, file, d.span)
    );
    if !d.path.is_empty() {
        o.push_str(&format!(",\"node\":{}", json_str(&d.path)));
    }
    let snippet = src
        .get(d.span.start..d.span.end.min(d.span.start + 120))
        .unwrap_or("");
    o.push_str(&format!(",\"source\":{}", json_str(snippet)));
    if let Some(e) = &d.expected {
        o.push_str(&format!(",\"expected\":{}", json_str(e)));
    }
    if let Some(f) = &d.found {
        o.push_str(&format!(",\"found\":{}", json_str(f)));
    }
    if !d.notes.is_empty() {
        let n: Vec<String> = d.notes.iter().map(|n| json_str(n)).collect();
        o.push_str(&format!(",\"notes\":[{}]", n.join(",")));
    }
    let fixes: Vec<String> = d
        .fixes
        .iter()
        .map(|f| {
            let edits: Vec<String> = f
                .edits
                .iter()
                .map(|e| {
                    format!(
                        "{{\"span\":{},\"text\":{}}}",
                        span_json(&li, file, e.span),
                        json_str(&e.text)
                    )
                })
                .collect();
            format!(
                "{{\"description\":{},\"edits\":[{}]}}",
                json_str(&f.description),
                edits.join(",")
            )
        })
        .collect();
    o.push_str(&format!(",\"fixes\":[{}]}}", fixes.join(",")));
    o
}

pub fn render_human(d: &Diag, src: &str, file: &str) -> String {
    let li = LineIndex::new(src);
    let (line, col) = li.pos(d.span.start);
    let mut o = format!(
        "{}[{}]: {}\n  --> {}:{}:{}\n",
        d.severity, d.code, d.message, file, line, col
    );
    let line_text = src.lines().nth(line - 1).unwrap_or("");
    o.push_str(&format!("   | {}\n", line_text));
    let width = (d.span.end.saturating_sub(d.span.start))
        .clamp(1, line_text.len().saturating_sub(col - 1).max(1));
    o.push_str(&format!(
        "   | {}{}\n",
        " ".repeat(col - 1),
        "^".repeat(width)
    ));
    if let (Some(e), Some(f)) = (&d.expected, &d.found) {
        o.push_str(&format!("   = expected: {}\n   = found:    {}\n", e, f));
    }
    for n in &d.notes {
        o.push_str(&format!("   = note: {}\n", n));
    }
    for f in &d.fixes {
        o.push_str(&format!("   = fix: {}", f.description));
        for e in &f.edits {
            let (l, c) = li.pos(e.span.start);
            if e.span.start == e.span.end {
                o.push_str(&format!(" [insert {:?} at {}:{}]", e.text, l, c));
            } else {
                o.push_str(&format!(
                    " [replace {:?} at {}:{} with {:?}]",
                    &src[e.span.start..e.span.end],
                    l,
                    c,
                    e.text
                ));
            }
        }
        o.push('\n');
    }
    o
}

/// Apply the first fix of each diagnostic, skipping overlapping edits. Returns new source and count applied.
pub fn apply_fixes(src: &str, diags: &[Diag]) -> (String, usize) {
    let mut edits: Vec<&Edit> = Vec::new();
    let mut applied = 0;
    'outer: for d in diags {
        if let Some(f) = d.fixes.first() {
            for e in &f.edits {
                for x in &edits {
                    let overlap = e.span.start < x.span.end && x.span.start < e.span.end;
                    let same_point = e.span.start == x.span.start;
                    if overlap || same_point {
                        continue 'outer;
                    }
                }
            }
            edits.extend(f.edits.iter());
            applied += 1;
        }
    }
    edits.sort_by(|a, b| b.span.start.cmp(&a.span.start));
    let mut out = src.to_string();
    for e in edits {
        out.replace_range(e.span.start..e.span.end, &e.text);
    }
    (out, applied)
}
