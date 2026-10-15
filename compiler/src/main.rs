mod ast;
mod check;
mod diag;
mod emit_rust;
mod runtime;
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

fn frontend(src: &str) -> Result<Front, Vec<Diag>> {
    let forms = sexpr::read(src).map_err(|d| vec![d])?;
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
        "check" => match frontend(&src) {
            Ok(f) => {
                report(&f.warnings, &src, &fname, opts.json);
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
                let (next, n) = diag::apply_fixes(&cur, &diags);
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
            let f = match frontend(&src) {
                Ok(f) => f,
                Err(d) => {
                    report(&d, &src, &fname, opts.json);
                    return ExitCode::from(1);
                }
            };
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
