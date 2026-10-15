# ail

A programming language designed to be written by AI agents: typed s-expressions, value
semantics without references, effects as capabilities, no undefined behavior, and compiler
diagnostics that are data (JSON with exact, machine-applicable fixes).

- `SPEC.md`: the whole language, short enough to fit in a model's context. `ail spec` prints it.
- `DESIGN.md`: why each decision was made.
- `compiler/`: the compiler (Rust). It type-checks, verifies capabilities and aliasing, and emits
  Rust that `rustc` compiles with LLVM.
- `bench/`: ail vs idiomatic Rust on 13 native tasks, a browser game (wasm) and a real HTTP server; see `bench/RESULTS.md`.
- `experiment/`: AI agents writing ail vs Rust vs TypeScript on the same tasks, first attempt only.
- `examples/`: programs, including a JSON REST API (`todo_api.ail`).

## Quick start

```
cd compiler && cargo build --release
target/release/ail.exe run ../examples/hello.ail
target/release/ail.exe check ../examples/broken.ail --json
target/release/ail.exe fix ../examples/broken.ail
```

## Hello

```
(fn main (params (con Console))
  (body (println con "hello from ail")))
```
