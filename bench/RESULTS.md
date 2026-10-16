# ail vs Rust: results (2026-10-15, ail 0.4 dev)

Machine: Ryzen 5 2600, Windows 11. Each task is specified in `SPECS.md` and was written twice by
independent agents: idiomatic Rust (serde where natural; cargo release profile with fat LTO and
overflow checks off) and idiomatic ail (overflow and bounds checks always on). Outputs are
byte-identical in every row. Times are best of 5 whole-process runs (ail and Rust interleaved);
lower ratio is better for ail.

## Native

| benchmark | what it exercises | ail s | Rust s | ail/Rust |
|---|---|---|---|---|
| trees | allocation-heavy binary trees | 7.79 | 50.48 | **0.15** |
| strings | string building and search | 0.18 | 0.56 | **0.32** |
| hashmap | integer hash map | 1.54 | 2.72 | **0.57** |
| json | build, serialize, parse 40k-record docs (Rust: serde_json::Value) | 2.06 | 3.06 | **0.67** |
| pathfind | A* with a hand-written heap vs BinaryHeap | 0.051 | 0.069 | **0.75** * |
| fannkuch | permutations, array indexing | 2.69 | 3.45 | **0.78** |
| interp | bytecode VM, enum dispatch | 2.30 | 2.87 | **0.80** |
| wordfreq | tokenize 3M words, count, sort | 0.31 | 0.34 | **0.92** |
| api | REST handlers: routing, JSON in/out, store (Rust: serde derive) | 2.78 | 3.00 | **0.93** |
| nbody | f64 struct math | 2.92 | 2.85 | 1.02 |
| raytrace | f64 vector math, enum shapes | 0.024 | 0.024 | 1.04 * |
| mandel | Mandelbrot 2000x2000, rows in parallel (ail par-map, Rust rayon) | 1.02 | 0.95 | 1.08 |
| sieve | bounds-checked array loops | 4.14 | 3.71 | 1.12 |
| game | 4000-entity arena, spatial grid collisions | 1.57 | 1.38 | 1.14 |

\* runs under 0.1 s are dominated by process start-up.
The machine had other heavy programs running during this run (about 80% CPU load before the
benchmarks started), so ail and Rust runs alternate and each number is the best of 5 to 9 runs.
Ratios within about ±0.06 of 1.0 are noise at this load.

## Browser game (WebAssembly, `web/`)

The `game` world as an app (`web/arena.ail`) vs the same game in Rust compiled to wasm without
wasm-bindgen (`web/rust`), driven headless by `web/harness.mjs` (Node): per frame one update and
one view, plus the host decoding the draw commands. 1500 frames, same final checksum (and equal to
the native `game` checksum).

| | update ms/frame | view ms/frame | host ms/frame | total | wasm size |
|---|---|---|---|---|---|
| ail | 0.91 | 0.11 | 0.09 | 1.11 | 78 KB |
| Rust | 0.82 | 0.04 | 0.05 | 0.90 | 36 KB |

ail/Rust 1.22 (1.30 in ail 0.3, 2.57 before that with a 1.6 MB module). Interning draw strings,
decoding each once per frame on the host, building with LTO and strip, and string views (the color
of each entity is no longer a fresh String) closed most of the gap.

## Real HTTP server (`http/`)

The todo API over TCP on localhost, one connection per request, 100,000 seeded todos. Each cell is
the best of 2 rounds of 10 s after a 2 s warmup (the machine is shared). The servers:

- **ail loop:** `http-next`, one handler at a time.
- **ail serve:** `http-serve`, GET in parallel, writes one at a time.
- **std:** Rust std::net, single-threaded.
- **axum:** Rust axum + tokio, multi-threaded, with an RwLock store.

I/O-bound mix (30% POST, 70% GET /todos?limit=20), in requests/s:

| clients | ail loop | ail serve | Rust std | Rust axum | serve/axum |
|---|---|---|---|---|---|
| 1 | 1411 | 1504 | 1598 | 1202 | 1.25 |
| 8 | 6555 | 9530 | 4736 | 6426 | 1.48 |
| 64 | 6486 | 10829 | 4672 | 6148 | 1.76 |

CPU-bound mix (10% POST, 90% GET /search, which scans the whole store), in requests/s:

| clients | ail loop | ail serve | Rust std | Rust axum | serve/axum |
|---|---|---|---|---|---|
| 1 | 388 | 396 | 198 | 184 | 2.15 |
| 8 | 479 | 1712 | 191 | 905 | 1.89 |
| 64 | 467 | 1903 | 186 | 1040 | 1.83 |

- **Every server gives the same counts.** I checked this on several queries.
- **One thread, twice Rust's speed:** ail's size-class allocator puts the 200,000 small strings of
  the store next to each other, while the system allocator that idiomatic Rust uses scatters them.
- **The parallel gain:** `http-serve` comes from the 12 cores, minus the 10% of POSTs, which take
  the state exclusively.
- **Errors:** two requests failed, out of all rounds. Both failed in the load generator's
  `connect` with Windows error 10055 (out of socket buffers). Neither was a server error.

## Fan-out client (`http/fanout.py`)

3,000 GET requests from one client to the Rust axum server, 64 in flight at a time. Rounds alternate
ail and Rust, and the table keeps the median of 5. Both clients get the same sum, with 0 failed requests.

| mode | ail (`par-run` + `http-request`) | Rust (tokio + reqwest) | ail/Rust |
|---|---|---|---|
| todos (cheap answers) | 106 ms | 165 ms | 0.64 |
| search (backend scans 100,000 todos) | 1961 ms | 2115 ms | 0.93 |

The first version of `http-request` opened a new connection for every request. It was 5 times slower
than reqwest and ran Windows out of ephemeral ports. Keeping connections alive and reusing them
closed the gap and then some.

## What changed in the compiler because of these benchmarks

ail 0.4 (string views):

- Read-only Str parameters are passed as `&str`. An immutable local bound to a piece of another
  string (`substr`, `trim`, a literal, `if`/`match` picking one) is a borrowed view when its source
  is not changed or moved while the view is used; `lower`/`upper` copy only if a character changes.
- A fn whose Str result is always a piece of one parameter returns a view (`strip-punct` in
  wordfreq returns `&str`).
- `(put (mut m) k (+ 1 (unwrap-or (get m k) 0)))` does one lookup, and a view key is copied only
  when it is new.
- `for` over `split` with a one-character literal separator iterates without dynamic dispatch and
  without copying each piece. `(str ...)` and appends write strings and integers directly instead
  of going through `format!`.
- Result: wordfreq went from 1.58 to 0.92, strings from 0.50 to 0.32, the wasm view got 25% faster.

- `par-map` spreads a map over all cores with dynamic block scheduling (threads take small blocks
  from a shared counter), so uneven rows like Mandelbrot's still keep every core busy. mandel is
  the parallel benchmark: ail `par-map` against Rust with rayon, 1.08.

ail 0.3:

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

- **game / sieve (~1.1)**: bounds checks in the inner loops, and the ail game uses a `Vec` of
  `Vec`s as its grid while the Rust agent used a flat counting-sort grid.
- **Overflow checks are not the cost.** Building interp and game with overflow checks off changes
  their time by 0 to 4%, within noise, so proving ranges to remove them is not worth it yet.
- **Browser view (about 3x on 0.1 ms)**: the Rust version sends palette indices; ail sends interned
  color strings.
