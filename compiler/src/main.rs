mod ast;
mod check;
mod diag;
mod emit_rust;
mod gpu;
mod parinfer;
mod runtime;
mod runtime_gpu;
mod runtime_net;
mod runtime_web;
mod sexpr;
mod spirv;
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
  ail build <file.ail> --target web [-o dir]   browser app: dir/index.html + dir/app.wasm
  ail web   <file.ail> [--port 8000]           build for the browser and serve it locally
  ail fix   <file.ail>              apply every machine-applicable fix, repeatedly
  ail emit  <file.ail>              print the generated Rust
  ail spec                          print the language specification

flags:
  --json      diagnostics as JSON (one object with a \"diagnostics\" array)
  --fast      faster compile, slower program (opt-level 0)
  --native    optimize for this CPU (target-cpu=native)
";

/// A program's files: the entry file and every file it uses, read into one text so that spans
/// are global. Each file starts on a known line of that text.
struct Sources {
    text: String,
    files: Vec<SrcFile>,
    /// Problems found while loading (a used file that does not exist).
    diags: Vec<Diag>,
}

struct SrcFile {
    /// As shown in diagnostics: relative to the entry file's directory.
    name: String,
    path: PathBuf,
    start_line: usize,
}

impl Sources {
    fn file_lines(&self) -> Vec<usize> {
        self.files.iter().map(|f| f.start_line).collect()
    }
    fn names(&self) -> String {
        self.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join("\n")
    }
    /// The file holding `offset` of `text` (which may be a repaired copy with the same lines):
    /// index, byte range.
    fn locate(&self, text: &str, offset: usize) -> (usize, usize, usize) {
        let li = diag::LineIndex::new(text);
        let (line, _) = li.pos(offset.min(text.len()));
        let i = self.files.partition_point(|f| f.start_line <= line).saturating_sub(1);
        let start = li.line_start(self.files[i].start_line);
        let end = match self.files.get(i + 1) {
            Some(n) => li.line_start(n.start_line).saturating_sub(1),
            None => text.len(),
        };
        (i, start, end)
    }
}

/// Read `entry` and, transitively, every `(use name)` it mentions (`name.ail` next to the file
/// that uses it). `texts` overrides what is on disk (used by `ail fix`).
fn load_sources(entry: &Path, texts: &std::collections::HashMap<PathBuf, String>) -> Result<Sources, String> {
    let root = entry.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut queue = vec![entry.to_path_buf()];
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut out = Sources { text: String::new(), files: Vec::new(), diags: Vec::new() };
    while let Some(path) = queue.first().cloned() {
        queue.remove(0);
        let key = std::fs::canonicalize(&path).unwrap_or(path.clone());
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let text = match texts.get(&path) {
            Some(t) => t.clone(),
            None => std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?
                .replace("\r\n", "\n"),
        };
        let base = out.text.len();
        let start_line = out.text.matches('\n').count() + 1;
        let name = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if let Ok(forms) = sexpr::read(&text) {
            let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            for f in &forms {
                if f.head() != Some("use") {
                    continue;
                }
                let Some(l) = f.list() else { continue };
                let Some(m) = l.get(1).and_then(|x| x.sym()) else { continue };
                let p = dir.join(format!("{m}.ail"));
                if p.exists() {
                    queue.push(p);
                } else {
                    out.diags.push(Diag::error(
                        "E0031",
                        diag::Span::new(base + l[1].span.start, base + l[1].span.end),
                        &format!("cannot find {m}.ail next to {name}"),
                    ));
                }
            }
        }
        out.files.push(SrcFile { name, path, start_line });
        out.text.push_str(&text);
        if !text.ends_with('\n') {
            out.text.push('\n');
        }
    }
    Ok(out)
}

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
(struct Reply (status I64) (content-type Str) (body Str))
(enum Event (Tick F64) (KeyDown Str) (KeyUp Str) (MouseDown F64 F64) (MouseUp F64 F64) (MouseMove F64 F64)
  (Click Str) (Input Str Str) (Resize F64 F64))
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
                for fixed in parinfer::candidates(src) {
                    let Ok(mut f) = frontend_raw(&fixed) else {
                        continue;
                    };
                    {
                        let mut w = diags[0].clone();
                        w.severity = "warning";
                        w.code = "W0004";
                        w.message = "closing parentheses were repaired from the indentation; run `ail fix` to write the repair into the file".into();
                        f.warnings.insert(0, w);
                        return Ok((f, fixed));
                    }
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
    let before = diags.iter().filter(|d| d.severity == "error").count();
    let (after, fixed) = parinfer::candidates(src)
        .into_iter()
        .filter(|c| c != src)
        .map(|c| (error_count(&frontend_raw(&c)), c))
        .min_by_key(|x| x.0)?;
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
            edits.push(diag::Edit {
                span: diag::Span::new(offset, offset + o.len()),
                text: n.to_string(),
            });
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
    Ok(Front {
        prog,
        ck,
        warnings: diags,
    })
}

/// Report diagnostics of a (possibly multi-file) program, each in the coordinates of its file.
fn report(diags: &[Diag], src: &str, srcs: &Sources, json: bool) {
    let local: Vec<(Diag, String, String)> = diags
        .iter()
        .map(|d| {
            let (i, a, b) = srcs.locate(src, d.span.start);
            (diag::shifted(d, a, b - a), src[a..b].to_string(), srcs.files[i].name.clone())
        })
        .collect();
    if json {
        let items: Vec<String> = local.iter().map(|(d, s, f)| diag::to_json(d, s, f)).collect();
        let errors = diags.iter().filter(|d| d.severity == "error").count();
        println!(
            "{{\"ok\":{},\"errors\":{},\"diagnostics\":[{}]}}",
            errors == 0,
            errors,
            items.join(",")
        );
    } else {
        for (d, s, f) in &local {
            eprint!("{}", diag::render_human(d, s, f));
        }
    }
}

struct Opts {
    json: bool,
    fast: bool,
    native: bool,
    web: bool,
    port: u16,
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
    let mut opts = Opts {
        json: false,
        fast: false,
        native: false,
        web: cmd == "web",
        port: 8000,
        out: None,
    };
    let mut prog_args = Vec::new();
    let mut i = 2;
    while i < argv.len() {
        match argv[i].as_str() {
            "--json" if cmd != "run" => opts.json = true,
            "--fast" if cmd != "run" || prog_args.is_empty() => opts.fast = true,
            "--native" if cmd != "run" || prog_args.is_empty() => opts.native = true,
            "-o" if cmd == "build" || cmd == "web" => {
                i += 1;
                opts.out = argv.get(i).cloned();
            }
            "--target" if cmd == "build" || cmd == "emit" => {
                i += 1;
                match argv.get(i).map(|s| s.as_str()) {
                    Some("web") => opts.web = true,
                    Some("native") => opts.web = false,
                    other => {
                        eprintln!(
                            "unknown target {:?}; use native or web",
                            other.unwrap_or("")
                        );
                        return ExitCode::from(2);
                    }
                }
            }
            "--port" if cmd == "web" => {
                i += 1;
                opts.port = argv.get(i).and_then(|p| p.parse().ok()).unwrap_or(8000);
            }
            "--" if cmd == "run" => {
                prog_args.extend(argv[i + 1..].iter().cloned());
                break;
            }
            a => prog_args.push(a.to_string()),
        }
        i += 1;
    }
    let entry = Path::new(file);
    let srcs = match load_sources(entry, &std::collections::HashMap::new()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let src = srcs.text.clone();
    let fname = srcs.names();
    if !srcs.diags.is_empty() && cmd != "fix" {
        report(&srcs.diags, &src, &srcs, opts.json);
        return ExitCode::from(1);
    }
    match cmd {
        "check" => match frontend_lenient(&src) {
            Ok((f, eff)) => {
                report(&f.warnings, &eff, &srcs, opts.json);
                if !opts.json {
                    eprintln!("ok");
                }
                ExitCode::SUCCESS
            }
            Err(d) => {
                report(&d, &src, &srcs, opts.json);
                ExitCode::from(1)
            }
        },
        "fix" => {
            // Fixes are applied file by file; the program is re-read after each round.
            let mut texts: std::collections::HashMap<PathBuf, String> = std::collections::HashMap::new();
            let mut total = 0;
            for _ in 0..20 {
                let Ok(cur) = load_sources(entry, &texts) else { break };
                let diags = match frontend(&cur.text) {
                    Ok(f) => f.warnings,
                    Err(d) => d,
                };
                let mut n_round = 0;
                for (i, sf) in cur.files.iter().enumerate() {
                    let errors: Vec<Diag> = diags
                        .iter()
                        .filter(|d| d.severity == "error" && cur.locate(&cur.text, d.span.start).0 == i)
                        .map(|d| {
                            let (_, a, b) = cur.locate(&cur.text, d.span.start);
                            diag::shifted(d, a, b - a)
                        })
                        .collect();
                    let (_, a, b) = cur.locate(&cur.text, cur.text.len().min(diag::LineIndex::new(&cur.text).line_start(sf.start_line)));
                    let old = cur.text[a..b].to_string();
                    let (next, n) = diag::apply_fixes(&old, &errors);
                    if n > 0 && next != old {
                        n_round += n;
                        texts.insert(sf.path.clone(), next);
                    }
                }
                if n_round == 0 {
                    break;
                }
                total += n_round;
            }
            for (p, t) in &texts {
                std::fs::write(p, t).expect("write");
            }
            eprintln!("applied {total} fix(es)");
            let cur = match load_sources(entry, &texts) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            if !cur.diags.is_empty() {
                report(&cur.diags, &cur.text, &cur, opts.json);
                return ExitCode::from(1);
            }
            match frontend(&cur.text) {
                Ok(f) => {
                    report(&f.warnings, &cur.text, &cur, opts.json);
                    ExitCode::SUCCESS
                }
                Err(d) => {
                    report(&d, &cur.text, &cur, opts.json);
                    ExitCode::from(1)
                }
            }
        }
        "emit" | "build" | "run" | "test" | "web" => {
            let (f, src) = match frontend_lenient(&src) {
                Ok(x) => x,
                Err(d) => {
                    report(&d, &src, &srcs, opts.json);
                    return ExitCode::from(1);
                }
            };
            if !opts.json {
                report(&f.warnings, &src, &srcs, false);
            }
            // `emit` of a file with tests and no main shows the test build.
            let test_file = !f.ck.info.fns.contains_key("main") && f.prog.items.iter().any(|i| matches!(i, ast::Item::Test(_)));
            let mode = if cmd == "test" || (cmd == "emit" && test_file) {
                emit_rust::Mode::Test
            } else if opts.web {
                if !f.ck.info.fns.contains_key("init") {
                    eprintln!(
                        "the web target needs an app: (fn init ...), (fn update ...) and (fn view ...) or (fn page ...)"
                    );
                    return ExitCode::from(1);
                }
                emit_rust::Mode::Web
            } else {
                if !f.ck.info.fns.contains_key("main") && cmd != "emit" {
                    eprintln!(
                        "{fname} is an app (init/update/view); build it with --target web or run it with `ail web`"
                    );
                    return ExitCode::from(1);
                }
                emit_rust::Mode::Run
            };
            if cmd == "test" && !f.prog.items.iter().any(|i| matches!(i, ast::Item::Test(_))) {
                eprintln!("no (test ...) blocks in {fname}");
                return ExitCode::from(1);
            }
            let rust = emit_rust::emit(&f.prog, &f.ck, &src, &fname, &srcs.file_lines(), mode);
            if cmd == "emit" {
                print!("{rust}");
                return ExitCode::SUCCESS;
            }
            let bin = match compile(&rust, cmd == "test", &opts) {
                Ok(b) => b,
                Err(e) => {
                    let d = Diag::error(
                        "E9000",
                        diag::Span::new(0, 0),
                        "internal compiler error: generated Rust did not compile",
                    )
                    .note(e.lines().take(40).collect::<Vec<_>>().join("\n"));
                    report(&[d], &src, &srcs, opts.json);
                    return ExitCode::from(3);
                }
            };
            if opts.web {
                let dir = PathBuf::from(opts.out.clone().unwrap_or_else(|| {
                    format!(
                        "{}-web",
                        Path::new(file).file_stem().unwrap().to_string_lossy()
                    )
                }));
                std::fs::create_dir_all(&dir).expect("create output dir");
                std::fs::copy(&bin, dir.join("app.wasm")).expect("copy wasm");
                let title = Path::new(file)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                std::fs::write(
                    dir.join("index.html"),
                    runtime_web::HOST_HTML.replace("{TITLE}", &title),
                )
                .expect("write html");
                eprintln!("built {}", dir.display());
                if cmd == "web" {
                    return serve(&dir, opts.port);
                }
                return ExitCode::SUCCESS;
            }
            if cmd == "build" {
                let out = opts.out.clone().unwrap_or_else(|| {
                    let stem = Path::new(file)
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    if cfg!(windows) {
                        format!("{stem}.exe")
                    } else {
                        stem
                    }
                });
                std::fs::copy(&bin, &out).expect("copy binary");
                eprintln!("built {out}");
                return ExitCode::SUCCESS;
            }
            let status = Command::new(&bin)
                .args(&prog_args)
                .status()
                .expect("run binary");
            ExitCode::from(status.code().unwrap_or(1) as u8)
        }
        _ => {
            eprint!("unknown command '{cmd}'\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Minimal static file server for `ail web` (wasm must be served over http).
fn serve(dir: &Path, port: u16) -> ExitCode {
    use std::io::{BufRead, BufReader, Write};
    let listener = match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot listen on port {port}: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!("serving http://localhost:{port}/  (Ctrl+C to stop)");
    for stream in listener.incoming().flatten() {
        let mut s = stream;
        let mut line = String::new();
        if BufReader::new(&s).read_line(&mut line).is_err() {
            continue;
        }
        let path = line
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .split('?')
            .next()
            .unwrap_or("/")
            .to_string();
        let name = if path == "/" {
            "index.html".to_string()
        } else {
            path.trim_start_matches('/').to_string()
        };
        let safe = !name.contains("..") && !name.contains('\\');
        match std::fs::read(dir.join(&name)) {
            Ok(body) if safe => {
                let ct = if name.ends_with(".wasm") {
                    "application/wasm"
                } else if name.ends_with(".html") {
                    "text/html; charset=utf-8"
                } else {
                    "application/octet-stream"
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(&body);
            }
            _ => {
                let _ = write!(
                    s,
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        }
    }
    ExitCode::SUCCESS
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
    // A trap in a read handler of http-serve answers 500 and the server keeps going: that needs
    // unwinding. Everything else stops at the first trap, so it can abort.
    let unwinds = rust.contains("ail_http::serve(");
    let mut flags = vec![
        "--edition",
        "2021",
        "-C",
        "overflow-checks=on",
        "-C",
        "debuginfo=0",
    ];
    if test {
        flags.extend(["-C", "opt-level=1", "-C", "panic=unwind"]);
    } else if o.fast {
        flags.extend(["-C", "opt-level=0", "-C", if unwinds { "panic=unwind" } else { "panic=abort" }]);
    } else {
        flags.extend([
            "-C",
            "opt-level=3",
            "-C",
            if unwinds { "panic=unwind" } else { "panic=abort" },
            "-C",
            "codegen-units=1",
            "-C",
            "lto=fat",
            "-C",
            "strip=symbols",
        ]);
    }
    if o.native {
        flags.extend(["-C", "target-cpu=native"]);
    }
    if o.web {
        flags.extend([
            "--target",
            "wasm32-unknown-unknown",
            "--crate-type",
            "cdylib",
        ]);
    }
    let key = format!("{:016x}", fnv(&format!("{}{}", rust, flags.join(" "))));
    let dir = std::env::temp_dir().join("ail-cache").join(&key);
    let bin = dir.join(if o.web {
        "prog.wasm"
    } else if cfg!(windows) {
        "prog.exe"
    } else {
        "prog"
    });
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
        return Err(format!(
            "{}\n(generated source: {})",
            String::from_utf8_lossy(&out.stderr),
            src.display()
        ));
    }
    Ok(bin)
}
