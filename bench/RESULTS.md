# ail vs Rust: results (2026-10-15)

Machine: Ryzen 5 2600, Windows 11. Each task is specified in `SPECS.md` and was written twice by
independent agents: idiomatic Rust (serde where natural; cargo release profile with fat LTO and
overflow checks off) and idiomatic ail (overflow and bounds checks always on). Outputs are
byte-identical in every row. Times are best of 5 whole-process runs; lower ratio is better for ail.

## Native

| benchmark | what it exercises | ail s | Rust s | ail/Rust |
|---|---|---|---|---|
| trees | allocation-heavy binary trees | 6.96 | 41.85 | **0.17** |
| strings | string building and search | 0.32 | 0.63 | **0.50** |
| hashmap | integer hash map | 1.28 | 2.51 | **0.51** |
| json | build, serialize, parse 40k-record docs (Rust: serde_json::Value) | 1.90 | 3.22 | **0.59** |
| pathfind | A* with a hand-written heap vs BinaryHeap | 0.063 | 0.089 | 0.70 * |
| fannkuch | permutations, array indexing | 2.61 | 3.16 | 0.82 |
| api | REST handlers: routing, JSON in/out, store (Rust: serde derive) | 2.09 | 2.32 | 0.90 |
| raytrace | f64 vector math, enum shapes | 0.034 | 0.034 | 1.00 * |
| sieve | bounds-checked array loops | 5.56 | 5.26 | 1.06 |
| nbody | f64 struct math | 3.35 | 3.13 | 1.07 |
| game | 4000-entity arena, spatial grid collisions | 1.38 | 1.28 | 1.08 |
| interp | bytecode VM, enum dispatch | 3.74 | 3.25 | 1.15 |
| wordfreq | tokenize 3M words, count, sort | 0.52 | 0.33 | 1.58 |

\* runs under 0.1 s are dominated by process start-up.
`trees` and `wordfreq` were measured in a second run with the same compiler: the system stopped the
first full run for lack of memory after the other 11 rows.

## Browser game (WebAssembly, `web/`)

The `game` world as an app (`web/arena.ail`) vs the same game in Rust compiled to wasm without
wasm-bindgen (`web/rust`), driven headless by `web/harness.mjs` (Node): per frame one update and
one view, plus the host decoding the draw commands. 1500 frames, same final checksum (and equal to
the native `game` checksum).

| | update ms/frame | view ms/frame | host ms/frame | total | wasm size |
|---|---|---|---|---|---|
| ail | 1.13 | 0.14 | 0.12 | 1.39 | 78 KB |
| Rust | 0.95 | 0.04 | 0.08 | 1.07 | 36 KB |

ail/Rust 1.30. Before this round it was 2.57 with a 1.6 MB module. Interning draw strings,
decoding each once per frame on the host, and building with LTO and strip closed most of the gap.

## Real HTTP server (`http/`)

The todo API over TCP on localhost, one connection per request, 30% POST / 70% GET, 10 s after a
2 s warmup, 0 errors.

| clients | ail req/s | Rust std (1 thread) | Rust axum + tokio | ail/std | ail/axum |
|---|---|---|---|---|---|
| 1 | 1242 | 1448 | 1274 | 0.86 | 0.97 |
| 8 | 5602 | 4635 | 5865 | 1.21 | 0.96 |
| 64 | 5324 | 4595 | 5421 | 1.16 | 0.98 |

Before this round ail was at 0.45–0.60 of Rust std. The program is still single-threaded. The
runtime now accepts and parses requests on a pool of I/O threads, so ail beats a hand-written
Rust std server and matches axum without the program having to deal with concurrency.

## What changed in the compiler because of these benchmarks

- `(set s (str s ...))` appends in place. Building a string in a loop used to be quadratic.
- `for` over `split` streams the pieces, without a Vec or an allocation per piece.
  `lower`/`upper` take an ASCII fast path and convert temporaries in place.
- `json-str` encodes typed values (structs, Vec, Option, maps) directly, without building a Json
  tree. The writer is faster: no per-number allocation, byte-scanned escaping, and small objects
  sort their keys on the stack. The parser has an integer fast path and rejects control characters.
- `(match (get m k) ...)` borrows instead of copying the subtree. A match whose arms mutate the
  scrutinee's variable now works (it used to fail to compile in Rust).
- `expect`, `fail` and `assert` no longer allocate their message on the success path.
- HTTP: an I/O thread pool, the response sent in one write, no socket clones per request, and
  `TCP_NODELAY`. Bodies that are not valid UTF-8 are rejected with 400.
- The allocator has per-thread free lists, which threads need.
- Web: draw strings are interned per frame. Builds use fat LTO and strip symbols (wasm 1.6 MB to
  78 KB).
- Bugs that agents reported: `(as F64 (% n 1000))` was type-checked as float. Paren repair failed
  on hanging forms such as `(let e (match x` followed by indented cases. A type written in an
  expression got a wrong "use vec" fix. `ail test` failed on apps.

## Where ail still loses, and why

- **wordfreq (1.58)**: every substring is an owned `Str` (`substr` and `lower` allocate). Rust
  borrows slices. Borrowed string views are the next step.
- **interp / game (~1.1)**: always-on overflow checks in the hot loop. The ail game also uses a
  `Vec` of `Vec`s as its grid, while the Rust agent used a flat counting-sort grid.
- **Browser view (3.4x on 0.1 ms)**: `(let color (match ... "literal"))` allocates a String per
  entity, because string literals are owned values.
