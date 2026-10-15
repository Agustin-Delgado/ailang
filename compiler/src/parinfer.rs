//! Rebuilds closing parentheses from indentation.
//!
//! Models indent code correctly far more reliably than they count closing parens, so when the
//! parens disagree with the indentation, the indentation is the better witness of intent.
//! A line indented at or left of an open paren's column closes that paren; the closers at the
//! end of each line (its "paren trail") are recomputed, closers in the middle of a line are kept.

struct Line {
    code: String,
    comment: String,
}

/// Candidate repairs, most likely first. The first reads indentation against the column of each
/// open paren. The second against the indentation of the line the paren was opened on, which is
/// right for hanging forms such as `(let e (match x` followed by `(case ...)` lines indented
/// under `let`: those cases belong to `match`, the innermost form still open on that line.
pub fn candidates(src: &str) -> Vec<String> {
    let a = infer_with(src, false);
    let b = infer_with(src, true);
    if a == b { vec![a] } else { vec![a, b] }
}

pub fn infer(src: &str) -> String {
    infer_with(src, false)
}

fn infer_with(src: &str, by_line: bool) -> String {
    let mut out: Vec<Line> = Vec::new();
    // (column, indentation of the line it was opened on) per open paren.
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut in_str = false;
    let mut last_code: Option<usize> = None;
    for raw in src.split('\n') {
        let starts_in_str = in_str;
        // Split into code and comment, tracking string state.
        let chars: Vec<char> = raw.chars().collect();
        let mut i = 0;
        let mut comment_at = None;
        let mut s = in_str;
        while i < chars.len() {
            let c = chars[i];
            if s {
                if c == '\\' {
                    i += 1;
                } else if c == '"' {
                    s = false;
                }
            } else if c == '"' {
                s = true;
            } else if c == ';' {
                comment_at = Some(i);
                break;
            }
            i += 1;
        }
        let ends_in_str = s;
        let code_chars: Vec<char> = match comment_at {
            Some(k) => chars[..k].to_vec(),
            None => chars.clone(),
        };
        let comment: String = match comment_at {
            Some(k) => chars[k..].iter().collect(),
            None => String::new(),
        };
        let trimmed_start = code_chars.iter().position(|c| !c.is_whitespace());
        let is_code = trimmed_start.is_some() && !starts_in_str;
        if is_code {
            let indent = trimmed_start.unwrap();
            let mut closers = 0;
            while stack
                .last()
                .is_some_and(|&(col, line)| (if by_line { line } else { col }) >= indent)
            {
                stack.pop();
                closers += 1;
            }
            if closers > 0 {
                if let Some(lc) = last_code {
                    out[lc].code.push_str(&")".repeat(closers));
                }
            }
        }
        // Parens opened on this line that the original text leaves open at its end.
        let mut open_at_eol: Vec<usize> = Vec::new();
        {
            let mut s = starts_in_str;
            let mut j = 0;
            while j < code_chars.len() {
                let c = code_chars[j];
                if s {
                    if c == '\\' {
                        j += 1;
                    } else if c == '"' {
                        s = false;
                    }
                } else if c == '"' {
                    s = true;
                } else if c == '(' {
                    open_at_eol.push(j);
                } else if c == ')' {
                    open_at_eol.pop();
                }
                j += 1;
            }
        }
        // Remove the paren trail (unless the line ends inside a string).
        let mut end = code_chars.len();
        if !ends_in_str {
            while end > 0 && (code_chars[end - 1] == ')' || code_chars[end - 1].is_whitespace()) {
                end -= 1;
            }
        }
        let kept: Vec<char> = code_chars[..end].to_vec();
        // Scan kept code: opens push, middle closers pop.
        let mut s = starts_in_str;
        let mut j = 0;
        let mut kept_out = String::new();
        while j < kept.len() {
            let c = kept[j];
            if s {
                if c == '\\' && j + 1 < kept.len() {
                    kept_out.push(c);
                    j += 1;
                    kept_out.push(kept[j]);
                    j += 1;
                    continue;
                }
                if c == '"' {
                    s = false;
                }
                kept_out.push(c);
            } else if c == '"' {
                s = true;
                kept_out.push(c);
            } else if c == '(' {
                let line = if open_at_eol.contains(&j) { trimmed_start.unwrap_or(0) } else { j };
                stack.push((j, line));
                kept_out.push(c);
            } else if c == ')' {
                if stack.pop().is_some() {
                    kept_out.push(c);
                }
            } else {
                kept_out.push(c);
            }
            j += 1;
        }
        in_str = ends_in_str;
        let has_code = is_code || (!kept_out.trim().is_empty());
        out.push(Line {
            code: kept_out,
            comment,
        });
        if has_code {
            last_code = Some(out.len() - 1);
        }
    }
    if let Some(lc) = last_code {
        out[lc].code.push_str(&")".repeat(stack.len()));
    }
    let mut res = Vec::with_capacity(out.len());
    for l in out {
        if l.comment.is_empty() {
            res.push(l.code);
        } else if l.code.trim().is_empty() {
            res.push(format!("{}{}", l.code, l.comment));
        } else {
            res.push(format!("{} {}", l.code.trim_end(), l.comment));
        }
    }
    res.join("\n")
}

#[cfg(test)]
mod tests {
    use super::infer;

    #[test]
    fn adds_missing_closer() {
        let src = "(fn f\n  (body\n    (+ 1 2)\n(fn g (body 1))";
        assert_eq!(infer(src), "(fn f\n  (body\n    (+ 1 2)))\n(fn g (body 1))");
    }

    #[test]
    fn removes_extra_closer() {
        let src = "(match x\n  (case 1 \"a\")))\n  (case _ \"b\"))";
        assert_eq!(infer(src), "(match x\n  (case 1 \"a\")\n  (case _ \"b\"))");
    }

    #[test]
    fn hanging_form_candidate() {
        let src = "(a\n  (let e (match x\n    (case 1 2)))\n    (case _ 3)))\n  (b))";
        let c = super::candidates(src);
        assert_eq!(
            c.last().unwrap(),
            "(a\n  (let e (match x\n    (case 1 2)\n    (case _ 3)))\n  (b))"
        );
    }

    #[test]
    fn keeps_strings_and_comments() {
        let src = "(a \")(\" ; comment )\n  (b))";
        assert_eq!(infer(src), "(a \")(\" ; comment )\n  (b))");
    }
}

