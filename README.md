# ail

**A programming language designed to be written by AI agents.** ail aims to be fast enough to
replace Rust and C++ and safe enough that common bad practices cannot be written at all.

ail started from a blank page. It is not a dialect or a wrapper of an existing language. Every
choice starts from one question: what does a model need in order to write correct, fast code on
the first attempt?

- **The whole language fits in a prompt.** `SPEC.md` is about 4,000 words, and `ail spec`
  prints it. A model that has never seen ail learns it from that page.
- **Syntax is typed s-expressions.** A model cannot misread operator precedence or layout.
  If the parentheses are off, the compiler uses the indentation as a second source of the
  intended structure and repairs them (warning `W0004`).
- **Values, not references.** Values have no shared mutable aliases. Mutation is marked where
  the call happens, as `(mut x)`, so data races and aliasing bugs cannot be written.
- **Effects are capabilities.** A function can print, read files or open sockets only if it was
  given that capability, and capabilities come down from `main`.
- **No undefined behavior.** Integer overflow, out-of-bounds access and failed `expect` stop the
  program with a clear message. Nothing continues silently with corrupted state.
- **Compiler errors are data.** Each diagnostic is JSON with an exact span and fixes a machine
  can apply. `ail fix` applies them until the program compiles.
- **One language for native programs, servers and the browser.** The compiler emits Rust and
  compiles it with LLVM, for native targets and WebAssembly.

## Benchmarks

Each task is specified exactly in [`bench/SPECS.md`](bench/SPECS.md). Each language version was
written by its own AI agent, working without seeing the other: one idiomatic Rust program (using
`serde` where natural, built with fat LTO and overflow checks off) and one idiomatic ail program
(overflow and bounds checks always on). **Every row produces byte-identical output.** Times are
the best of 5 whole-process runs on a Ryzen 5 2600 under Windows 11.

### Native: 13 tasks

![ail vs Rust on 13 native tasks](docs/charts/native.svg)

| task | what it exercises | ail | Rust | ail/Rust time |
|---|---|---|---|---|
| trees | allocation-heavy binary trees | 6.96 s | 41.85 s | **0.17** |
| strings | string building and search | 0.32 s | 0.63 s | **0.50** |
| hashmap | integer hash map | 1.28 s | 2.51 s | **0.51** |
| json | build, serialize, parse 40k-record documents | 1.90 s | 3.22 s | **0.59** |
| pathfind | A* search | 0.063 s | 0.089 s | 0.70 * |
| fannkuch | permutations, array indexing | 2.61 s | 3.16 s | 0.82 |
| api | REST handlers: routing, JSON in and out, store | 2.09 s | 2.32 s | 0.90 |
| raytrace | f64 vector math, enum shapes | 0.034 s | 0.034 s | 1.00 * |
| sieve | bounds-checked array loops | 5.56 s | 5.26 s | 1.06 |
| nbody | f64 struct math | 3.35 s | 3.13 s | 1.07 |
| game | 4,000-entity arena with a spatial grid | 1.38 s | 1.28 s | 1.08 |
| interp | bytecode VM, enum dispatch | 3.74 s | 3.25 s | 1.15 |
| wordfreq | tokenize 3M words, count, sort | 0.52 s | 0.33 s | 1.58 |

\* These runs take under 0.1 s, so process start-up dominates them.

### Real HTTP server

![HTTP requests per second](docs/charts/http.svg)

The same todo REST API runs over TCP on localhost, against a hand-written Rust `std` server and
against `axum` on `tokio`. Every request opens a new connection. The mix is 30% `POST` and 70%
`GET`, measured for 10 s after a 2 s warmup, with zero errors. The ail program itself is
single-threaded: the runtime accepts and parses requests on a pool of I/O threads. With that,
ail beats the `std` server by 16 to 21% under load and stays within 2 to 4% of axum.

### Browser game (WebAssembly)

![ms per frame in the browser](docs/charts/web.svg)

[`bench/web/arena.ail`](bench/web/arena.ail) is the `game` world written as a browser app.
It is compared with the same game in Rust compiled to wasm without wasm-bindgen. A headless
harness on Node drives both versions for 1,500 frames, and both end with the same checksum as the
native run. The ail module is 78 KB.

![the arena running in the browser](docs/arena.png)

### Can a model write a language it has never seen?

![first-attempt correctness](docs/charts/experiment.svg)

Fresh agents got 16 console tasks with hidden tests: 10 seen while the spec was being tuned and
6 held out. Each agent had nothing but the task and `SPEC.md`, and wrote the program once, with
no compiling and no testing. ail compiled as often as Rust and TypeScript. Its one remaining
miss with each model was a logic slip or an undefined helper, and one repair round fixed it.
These tasks are too easy to separate the three languages; a harder run is next. Details are in
[`experiment/RESULTS.md`](experiment/RESULTS.md).

### Where ail still loses, and what comes next

- **wordfreq (1.58×):** every substring is an owned string, while Rust borrows slices. The next
  step is borrowed string views.
- **interp and game (~1.1×):** overflow checks are always on in the hot loop. The plan is to
  prove ranges at compile time and drop the checks that can never fire.
- **Browser view:** string literals are owned values, so the program allocates one string per
  entity per frame.

Full write-up: [`bench/RESULTS.md`](bench/RESULTS.md). Charts are generated by
[`bench/charts.py`](bench/charts.py).

## A taste

```lisp
(struct Todo (id I64) (title Str) (done Bool))

(fn page-size (params (req Request)) (returns I64)
  (body
    (match (get (. req query) "limit")
      (case (some s) (min 100 (max 1 (unwrap-or (parse-i64 s) 20))))
      (case none 20))))

(fn add (params (todos (mut (Vec Todo))) (title Str))
  (body (push (mut todos) (new Todo (id (len todos)) (title title) (done false)))))

(fn main (params (con Console))
  (body
    (var todos (Vec Todo) (vec))
    (add (mut todos) "write the spec")
    (println con (json-str todos))))
```

When something is wrong, the compiler answers with data that another program can act on:

```json
{"code":"E0501","message":"match is not exhaustive: missing Color.Blue",
 "span":{"line":5,"col":5,"end_line":7,"end_col":34},
 "fixes":[{"description":"add the missing arms (replace each placeholder body)",
           "edits":[{"span":{"line":7,"col":33},
                     "text":" (case Color.Blue (fail \"TODO: handle Blue\"))"}]}]}
```

## Quick start

You need Rust (stable). For browser builds, also run `rustup target add wasm32-unknown-unknown`.

```
cd compiler && cargo build --release
target/release/ail run ../examples/hello.ail
target/release/ail check ../examples/broken.ail --json
target/release/ail fix ../examples/broken.ail
target/release/ail web ../examples/dodge.ail          # build for the browser and serve it
```

| command | what it does |
|---|---|
| `ail check <file> [--json]` | type, capability and aliasing check (instant) |
| `ail run <file>` | compile (cached) and run |
| `ail test <file>` | run the `(test ...)` blocks |
| `ail build <file> [--target web]` | optimized native binary, or `index.html` + `app.wasm` |
| `ail fix <file>` | apply every machine-applicable fix, repeatedly |
| `ail emit <file>` | print the generated Rust |
| `ail spec` | print the language specification |

## Repository

| path | contents |
|---|---|
| [`SPEC.md`](SPEC.md) | the whole language |
| [`DESIGN.md`](DESIGN.md) | why each decision was made |
| [`compiler/`](compiler) | the compiler (Rust): parser, checker, Rust emitter, runtime |
| [`bench/`](bench) | ail vs Rust: native tasks, HTTP server, browser game |
| [`experiment/`](experiment) | AI agents writing ail vs Rust vs TypeScript |
| [`examples/`](examples) | programs, including a JSON REST API and a browser game |

## Status

ail is at v0.3, an early research language. The pieces above work and are measured. Generics
for user types, lambdas, concurrency and modules are not built yet.
