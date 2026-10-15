mod ast;
mod check;
mod diag;
mod emit_rust;
mod parinfer;
mod runtime;
mod runtime_net;
mod sexpr;
mod types;

use diag::Diag;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const SPEC: &str = include_str!("../../SPEC.md");

const USAGE: &str = "ail - an AI-native programming language

usage:
  ail check <file.ail> [--json]     type, capability and aliasing check (instant)
  ail run   <file.ail> [args...]    compile (cached) and run
  ail test  <file.ail> [--json]     run (test ...) blocks
  ail build <file.ail> [-o out]     native optimized binary
  ail fix   <file.ail>              apply every machine-applicable fix, repeatedly
  ail emit  <file.ail>              print the generated Rust
  ail spec                          print the language specification

flags:
  --json      diagnostics as JSON (one object with a \"diagnostics\" array)
  --fast      faster compile, slower program (opt-level 0)
  --native    optimize for this CPU (target-cpu=native)
";

struct Front {
    prog: ast::Program,
    ck: check::Checked,
    warnings: Vec<Diag>,
}

fn fill_sentinels(src: &str, diags: &mut [Diag]) {
    for d in diags.iter_mut() {
        for f in d.fixes.iter_mut() {
            for e in f.edits.iter_mut() {
                if e.text.contains('\u{0}') {
                    e.text = e.text.replace('\u{0}', &src[e.span.start..e.span.end]);
                }
            }
        }
    }
}

/// Types every program can use. Declared in ail itself; never produces diagnostics.
const PRELUDE: &str = "
(enum Json (Null) (Bool Bool) (Num F64) (Str Str) (Arr (Vec Json)) (Obj (Map Str Json)))
(struct Request (id I64) (method Str) (path Str) (query (Map Str Str)) (headers (Map Str Str)) (body Str))
";

fn frontend(src: &str) -> Result<Front, Vec<Diag>> {
    match frontend_raw(src) {
        Ok(f) => Ok(f),
        Err(mut diags) => {
            if let Some(d) = paren_fix(src, &diags) {
                diags.insert(0, d);
            }
            Err(diags)
        }
    }
}

/// Like `frontend`, but when the parens are broken and the indentation-implied structure is a
/// valid program, use that program (with a warning). A valid program never changes meaning:
/// the repair only applies when the original does not compile.
fn frontend_lenient(src: &str) -> Result<(Front, String), Vec<Diag>> {
    match frontend(src) {
        Ok(f) => Ok((f, src.to_string())),
        Err(diags) => {
            if diags.first().is_some_and(|d| d.code == "E0006") {
                let fixed = parinfer::infer(src);
                if let Ok(mut f) = frontend_raw(&fixed) {
                    let mut w = diags[0].clone();
                    w.severity = "warning";
                    w.code = "W0004";
                    w.message = "closing parentheses were repaired from the indentation; run `ail fix` to write the repair into the file".into();
                    f.warnings.insert(0, w);
                    return Ok((f, fixed));
                }
            }
            Err(diags)
        }
    }
}

fn error_count(r: &Result<Front, Vec<Diag>>) -> usize {
    match r {
        Ok(_) => 0,
        Err(d) => d.iter().filter(|d| d.severity == "error").count(),
    }
}

/// When parens disagree with indentation and the indentation-implied structure is better,
/// offer it as a fix: one edit per changed line.
fn paren_fix(src: &str, diags: &[Diag]) -> Option<Diag> {
    let fixed = parinfer::infer(src);
    if fixed == src {
        return None;
    }
    let before = diags.iter().filter(|d| d.severity == "error").count();
    let after = error_count(&frontend_raw(&fixed));
    if after >= before {
        return None;
    }
    let old: Vec<&str> = src.split('\n').collect();
    let new: Vec<&str> = fixed.split('\n').collect();
    if old.len() != new.len() {
        return None;
    }
    let mut edits = Vec::new();
    let mut offset = 0;
    let mut first = None;
    for (o, n) in old.iter().zip(new.iter()) {
        if o != n {
            first.get_or_insert(offset);
            edits.push(diag::Edit { span: diag::Span::new(offset, offset + o.len()), text: n.to_string() });
        }
        offset += o.len() + 1;
    }
    let at = first.unwrap_or(0);
    let lines = edits.len();
    Some(
        Diag::error("E0006", diag::Span::new(at, at + old.iter().map(|l| l.len()).next().unwrap_or(0).min(1)), "parentheses do not match the indentation")
            .note(format!("rebuilding the closing parens from the indentation changes {lines} line(s) and fixes {} of {before} error(s)", before - after))
            .with_fix("rebalance the parentheses according to the indentation", edits),
    )
}

fn frontend_raw(src: &str) -> Result<Front, Vec<Diag>> {
    let mut forms = sexpr::read(PRELUDE).expect("prelude parses");
    forms.extend(sexpr::read(src).map_err(|d| vec![d])?);
    let (prog, mut diags) = ast::lower(&forms);
    if diags.iter().any(|d| d.severity == "error") {
        return Err(diags);
    }
    let (ck, cdiags) = check::check(&prog);
    diags.extend(cdiags);
    fill_sentinels(src, &mut diags);
    diags.sort_by_key(|d| d.span.start);
    if diags.iter().any(|d| d.severity == "error") {
        return Err(diags);
    }
    Ok(Front { prog, ck, warnings: diags })
}

fn report(diags: &[Diag], src: &str, file: &str, json: bool) {
    if json {
        let items: Vec<String> = diags.iter().map(|d| diag::to_json(d, src, file)).collect();
        let errors = diags.iter().filter(|d| d.severity == "error").count();
        println!("{{\"ok\":{},\"errors\":{},\"diagnostics\":[{}]}}", errors == 0, errors, items.join(","));
    } else {
        for d in diags {
            eprint!("{}", diag::render_human(d, src, file));
        }
    }
}

struct Opts {
    json: bool,
    fast: bool,
    native: bool,
    out: Option<String>,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "help" || argv[0] == "--help" {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let cmd = argv[0].as_str();
    if cmd == "spec" {
        print!("{SPEC}");
        return ExitCode::SUCCESS;
    }
    let Some(file) = argv.get(1) else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let mut opts = Opts { json: false, fast: false, native: false, out: None };
    let mut prog_args = Vec::new();
    let mut i = 2;
    while i < argv.len() {
        match argv[i].as_str() {
            "--json" if cmd != "run" => opts.json = true,
            "--fast" if cmd != "run" || prog_args.is_empty() => opts.fast = true,
            "--native" if cmd != "run" || prog_args.is_empty() => opts.native = true,
            "-o" if cmd == "build" => {
                i += 1;
                opts.out = argv.get(i).cloned();
            }
            "--" if cmd == "run" => {
                prog_args.extend(argv[i + 1..].iter().cloned());
                break;
            }
            a => prog_args.push(a.to_string()),
        }
        i += 1;
    }
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s.replace("\r\n", "\n"),
        Err(e) => {
            eprintln!("cannot read {file}: {e}");
            return ExitCode::from(2);
        }
    };
    let fname = Path::new(file).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or(file.clone());
    match cmd {
        "check" => match frontend_lenient(&src) {
            Ok((f, eff)) => {
                report(&f.warnings, &eff, &fname, opts.json);
                if !opts.json {
                    eprintln!("ok");
                }
                ExitCode::SUCCESS
            }
            Err(d) => {
                report(&d, &src, &fname, opts.json);
                ExitCode::from(1)
            }
        },
        "fix" => {
            let mut cur = src.clone();
            let mut total = 0;
            for _ in 0..20 {
                let diags = match frontend(&cur) {
                    Ok(f) => f.warnings,
                    Err(d) => d,
                };
                let errors: Vec<Diag> = diags.into_iter().filter(|d| d.severity == "error").collect();
                let (next, n) = diag::apply_fixes(&cur, &errors);
                if n == 0 || next == cur {
                    break;
                }
                total += n;
                cur = next;
            }
            if total > 0 {
                std::fs::write(file, &cur).expect("write");
            }
            eprintln!("applied {total} fix(es)");
            match frontend(&cur) {
                Ok(f) => {
                    report(&f.warnings, &cur, &fname, opts.json);
                    ExitCode::SUCCESS
                }
                Err(d) => {
                    report(&d, &cur, &fname, opts.json);
                    ExitCode::from(1)
                }
            }
        }
        "emit" | "build" | "run" | "test" => {
            let (f, src) = match frontend_lenient(&src) {
                Ok(x) => x,
                Err(d) => {
                    report(&d, &src, &fname, opts.json);
                    return ExitCode::from(1);
                }
            };
            if !opts.json {
                report(&f.warnings, &src, &fname, false);
            }
            let mode = if cmd == "test" { emit_rust::Mode::Test } else { emit_rust::Mode::Run };
            if cmd == "test" && !f.prog.items.iter().any(|i| matches!(i, ast::Item::Test(_))) {
                eprintln!("no (test ...) blocks in {fname}");
                return ExitCode::from(1);
            }
            let rust = emit_rust::emit(&f.prog, &f.ck, &src, &fname, mode);
            if cmd == "emit" {
                print!("{rust}");
                return ExitCode::SUCCESS;
            }
            let bin = match compile(&rust, cmd == "test", &opts) {
                Ok(b) => b,
                Err(e) => {
                    let d = Diag::error("E9000", diag::Span::new(0, 0), "internal compiler error: generated Rust did not compile")
                        .note(e.lines().take(40).collect::<Vec<_>>().join("\n"));
                    report(&[d], &src, &fname, opts.json);
                    return ExitCode::from(3);
                }
            };
            if cmd == "build" {
                let out = opts.out.clone().unwrap_or_else(|| {
                    let stem = Path::new(file).file_stem().unwrap().to_string_lossy().to_string();
                    if cfg!(windows) { format!("{stem}.exe") } else { stem }
                });
                std::fs::copy(&bin, &out).expect("copy binary");
                eprintln!("built {out}");
                return ExitCode::SUCCESS;
            }
            let status = Command::new(&bin).args(&prog_args).status().expect("run binary");
            ExitCode::from(status.code().unwrap_or(1) as u8)
        }
        _ => {
            eprint!("unknown command '{cmd}'\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn compile(rust: &str, test: bool, o: &Opts) -> Result<PathBuf, String> {
    let mut flags = vec!["--edition", "2021", "-C", "overflow-checks=on", "-C", "debuginfo=0"];
    if test {
        flags.extend(["-C", "opt-level=1", "-C", "panic=unwind"]);
    } else if o.fast {
        flags.extend(["-C", "opt-level=0", "-C", "panic=abort"]);
    } else {
        flags.extend(["-C", "opt-level=3", "-C", "panic=abort", "-C", "codegen-units=1"]);
    }
    if o.native {
        flags.extend(["-C", "target-cpu=native"]);
    }
    let key = format!("{:016x}", fnv(&format!("{}{}", rust, flags.join(" "))));
    let dir = std::env::temp_dir().join("ail-cache").join(&key);
    let bin = dir.join(if cfg!(windows) { "prog.exe" } else { "prog" });
    if bin.exists() {
        return Ok(bin);
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let src = dir.join("prog.rs");
    std::fs::write(&src, rust).map_err(|e| e.to_string())?;
    let out = Command::new("rustc")
        .args(&flags)
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .current_dir(&dir)
        .output()
        .map_err(|e| format!("cannot run rustc: {e}"))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&bin);
        return Err(format!("{}\n(generated source: {})", String::from_utf8_lossy(&out.stderr), src.display()));
    }
    Ok(bin)
}
