# Application benchmarks: task specifications

Each benchmark is written twice, independently: once in ail (`bench/ail/<name>.ail`) and once in
idiomatic Rust (`bench/rust/src/bin/<name>.rs`). Both programs must print exactly the same output.
The spec below is the only contract between them.

Shared rules:

- No input, no command-line arguments, no clock, no OS randomness. Everything comes from the RNG below.
- **RNG** (Park–Miller, fits in 63 bits, identical in both languages):
  state starts at the given seed (an integer in 1..2147483646);
  `next()` sets `state = state * 48271 % 2147483647` and returns the new state.
  `rand_below(n)` = `next() % n`. `rand_f64()` = `next() as f64 / 2147483647.0`.
  Every random draw in the spec is one call, in the order written.
- Floating point: use f64 and evaluate formulas in the order written (left to right, as parenthesized).
  Do not use fused multiply-add. `floor` means floor toward negative infinity.
- Rust: idiomatic, the way an experienced Rust developer writes production code. Safe Rust only (no `unsafe`).
  Use std; crates already in `bench/rust/Cargo.toml` (serde, serde_json, rayon) where a Rust developer would.
  Don't hand-tune for this benchmark beyond what is normal good practice; don't write deliberately slow code either.
- ail: idiomatic ail per `SPEC.md`.

---

## game — headless arena simulation (entity update + spatial grid collisions)

World `W = 1600.0` by `H = 1200.0`. `N = 4000` entities, `FRAMES = 1500`, `DT = 1.0 / 60.0`, `CELL = 32.0`.
Seed `12345`.

Entity fields: `x, y, vx, vy, r` (f64) and `hp` (integer).

Spawn (also used for respawn), draws in this order:
`r = 3.0 + rand_f64() * 9.0`; `x = r + rand_f64() * (W - 2.0 * r)`; `y = r + rand_f64() * (H - 2.0 * r)`;
`angle = rand_f64() * 6.283185307179586`; `speed = 40.0 + rand_f64() * 160.0`;
`vx = cos(angle) * speed`; `vy = sin(angle) * speed`; `hp = 3`.
Initially spawn entities 0..N-1 in index order.

Each frame:
1. **Move** every entity in index order: `x = x + vx * DT`, `y = y + vy * DT`.
   Then walls: if `x < r` then `x = r` and `vx = -vx`; else if `x > W - r` then `x = W - r` and `vx = -vx`.
   Same for `y` with `H` and `vy`.
2. **Grid**: `cols = floor(W / CELL)` = 50, `rows = floor(H / CELL)` = 37 (integers).
   Cell of an entity: `cx = min(floor(x / CELL) as int, cols - 1)`, `cy = min(floor(y / CELL) as int, rows - 1)`.
   Bucket every entity index into its cell.
3. **Collisions**: for each entity `i` (index order) and each neighbor cell `(cx+dx, cy+dy)` with
   `dx, dy` in `-1..=1` that is inside the grid, for each entity `j` in that cell with `j > i`:
   with `ddx = x[j] - x[i]`, `ddy = y[j] - y[i]`, `rr = r[i] + r[j]`,
   if `ddx * ddx + ddy * ddy < rr * rr` it is a collision: add 1 to `collisions` (a 64-bit total
   over the whole run), subtract 1 from `hp` of both i and j. (Velocities do not change.)
   A pair is counted at most once per frame by construction (j > i, and each j lives in exactly one cell).
4. **Respawn**: for each entity in index order whose `hp <= 0`, replace it with a freshly spawned entity
   (draws as above) and add 1 to `respawns`.

Output, after all frames (three lines):
```
collisions <total>
respawns <total>
checksum <sum over entities, in index order, of floor(x) + floor(y), as an integer>
```

## api — in-process REST API (routing + JSON parse/serialize + store)

A todo store driven by a generated stream of `REQUESTS = 200000` requests. Seed `777`.
No sockets: each request is a (method, path, body) triple built as strings, then handled by a
`handle(store, method, path, body) -> (status, response_body)` function that parses the path
and the JSON body, exactly as a server would. The point is the request-handling code.

Todo: `id` (integer), `title` (string), `done` (bool), `tags` (list of strings).
Store: todos in insertion order plus `next_id` starting at 1.
Response JSON for a todo is exactly `{"done":<bool>,"id":<int>,"tags":[...],"title":"<title>"}`
(keys in this order, no spaces, standard JSON string escaping).

Request generation, for `k` in `0..REQUESTS`: `op = rand_below(100)`.
- `op < 45`: `POST /todos` with body `{"title":"task <k>","tags":["t<a>","t<b>"]}` where
  `a = rand_below(10)`, `b = rand_below(10)` (drawn in that order).
- `op < 75`: `GET /todos?limit=<l>&offset=<o>` with `l = 1 + rand_below(50)`, `o = rand_below(next_id_hint)`
  where `next_id_hint` is the store's current `next_id` (read before handling this request).
- `op < 90`: `PATCH /todos/<id>` with body `{"done":true}`, `id = 1 + rand_below(next_id_hint)`.
- otherwise: `DELETE /todos/<id>`, `id = 1 + rand_below(next_id_hint)`.

Handling:
- Path parsing: split the path at `?`; query is `&`-separated `key=value`. Parse `/todos/<id>` with an
  integer id. Missing or non-integer `limit` defaults to 20, missing `offset` to 0.
- `POST /todos`: parse body as JSON; `title` must be a string and `tags` an array of strings, else 400
  with body `{"error":"invalid body"}`. Creates the todo with `done=false`, id = next_id (then next_id += 1).
  Respond 201 with the todo JSON.
- `GET /todos`: skip the first `offset` todos (in insertion order), take up to `limit`; respond 200 with
  a JSON array of todo JSON objects.
- `PATCH /todos/<id>`: parse body; if it has a bool `done`, set it. 200 with the todo JSON; 404 with
  `{"error":"not found"}` if no todo has that id.
- `DELETE /todos/<id>`: remove it (keep the order of the rest); 204 with an empty body; 404 as above.

Output (five lines):
```
status 200 <count>
status 201 <count>
status 204 <count>
status 404 <count>
bytes <sum of response body lengths in bytes>
```
(400 never happens with this stream; the code path must still exist.)

## json — build, serialize, parse and aggregate dynamic JSON documents

`ROUNDS = 8`, `RECORDS = 40000`. Seed `4242`.

Per round:
1. Build a JSON array of `RECORDS` objects; record `i` draws, in order: `n = rand_below(1000000)`,
   `s = rand_below(10000)`, `t = rand_below(5)`. The object is
   `{"id": i, "name": "user<n>", "score": s as f64 / 4.0 + 0.125, "active": (n % 2 == 0),
     "tags": ["tag0", ..., "tag<t>"] (t+1 strings), "pos": {"x": n % 1000, "y": n / 1000}}`
   (`id`, `x`, `y` are integers). Build it as a dynamic JSON value (ail `Json`, Rust `serde_json::Value`).
2. Serialize it to a compact string (no spaces). Add its length to `total_bytes`.
   Note: object key order in the text may differ between implementations; only the length is compared,
   and it is the same for any key order.
3. Parse the string back into a dynamic value.
4. From the parsed value: for each record add `score` to `score_sum` (f64), count `active == true`
   into `actives`, add the number of tags to `tag_count`, and add `x + y` to `pos_sum` (integers).

Output (four lines; the float with exactly 3 decimals):
```
bytes <total_bytes>
score <score_sum with 3 decimals>
actives <actives>
tags <tag_count> pos <pos_sum>
```

## wordfreq — text processing (tokenize, count in a hash map, sort)

Seed `99`. Build a vocabulary of `VOCAB = 5000` words: word `w` has `2 + rand_below(3)` syllables, each
syllable is `CONS[rand_below(16)] + VOW[rand_below(5)]` (draws: count first, then for each syllable the
consonant then the vowel) with `CONS = "bcdfghjklmnprstv"` (one char each) and `VOW = "aeiou"`.
The first letter of each word is uppercased if `rand_below(4) == 0` (drawn after its syllables).

Build one text of `WORDS = 3000000` words: word index `rand_below(VOCAB)` (skewed: draw
`a = rand_below(VOCAB)` then `b = rand_below(VOCAB)` and use `min(a, b)`), followed by a separator:
`", "` if `rand_below(10) == 0`, `". "` if (else) it is `1`, otherwise `" "` (one draw per word).

Then process the text as text (this is what is measured together with the build):
split on whitespace, strip leading/trailing `,` and `.` from each token, lowercase it, skip empty tokens,
count occurrences in a hash map.

Output: `distinct <number of distinct words>`, then the top 15 words by count descending, ties by word
ascending (bytewise), each line `<word> <count>`.

## pathfind — A* on a grid (game AI)

Grid `640 x 640`, seed `2024`. Cell `(x, y)` (row-major, `y` outer loop then `x`) is a wall if
`rand_below(100) < 28`. Then cells `(0,0)` and `(639,639)` are forced free.

`QUERIES = 40`. For each query draw a start then a goal, each as `x = rand_below(640)`, `y = rand_below(640)`
(start x, start y, goal x, goal y). If either is a wall, the query result is `-2` (still drawn).
Otherwise run A* with 4-neighborhood, unit step cost, Manhattan heuristic, and a binary heap
priority queue (ail has no built-in heap: write one). Result: the length (number of steps) of the
shortest path, or `-1` if unreachable.

Output: one line per query `q<i> <result>`, then `total <sum of results that are >= 0>`.

## raytrace — small ray tracer (f64 vector math, structs, enums)

Image `480 x 360`. No randomness. Camera at `(0, 1, -4)`, looking toward +z; for pixel `(px, py)`:
`u = (px as f64 + 0.5) / 480.0 * 2.0 - 1.0`, `v = 1.0 - (py as f64 + 0.5) / 360.0 * 2.0`,
direction `normalize((u * 1.3333333333333333, v, 1.5))`.

Vectors: `dot(a,b) = a.x*b.x + a.y*b.y + a.z*b.z` (left to right). `len(a) = sqrt(dot(a,a))`.
`normalize(a) = (a.x / l, a.y / l, a.z / l)` with `l = len(a)`. `reflect(d, n) = d - n * (2.0 * dot(d, n))`
(i.e. componentwise `d.c - n.c * k`). Color = vector (r, g, b).

Scene: a list of objects, each a sphere or a plane, plus its color and `reflectivity`:
- Sphere center `(0, 1, 3)`, radius `1`, color `(1.0, 0.2, 0.2)`, reflectivity `0.3`.
- Sphere center `(-2.2, 0.7, 4)`, radius `0.7`, color `(0.2, 1.0, 0.2)`, reflectivity `0.5`.
- Sphere center `(2.0, 0.5, 2.5)`, radius `0.5`, color `(0.2, 0.4, 1.0)`, reflectivity `0.0`.
- Sphere center `(0.8, 0.3, 1.2)`, radius `0.3`, color `(1.0, 1.0, 0.2)`, reflectivity `0.6`.
- Plane `y = 0` (normal `(0,1,0)`, `d = 0`), color `(0.8, 0.8, 0.8)`, reflectivity `0.4`.
Model objects as an enum (Sphere | Plane) — this is part of what is measured.

Intersections (return the nearest `t > 0.0001`):
- Sphere: `oc = o - c`; `b = dot(oc, d)`; `cc = dot(oc, oc) - r*r`; `disc = b*b - cc`; if `disc < 0` miss;
  `s = sqrt(disc)`; `t = -b - s`; if `t <= 0.0001` then `t = -b + s`; if `t <= 0.0001` miss.
  Normal at hit `p`: `normalize(p - c)`.
- Plane `y = 0`: `denom = d.y`; if `abs(denom) < 1e-9` miss; `t = -(o.y) / denom`; miss if `t <= 0.0001`.
  Normal `(0, 1, 0)`.
Nearest hit: iterate objects in order, keep strictly smaller `t`.

Shading `trace(o, d, depth)`: no hit → sky `(0.5 + 0.5 * d.y) * (0.6, 0.8, 1.0)` componentwise as
`(0.6 * k, 0.8 * k, 1.0 * k)` with `k = 0.5 + 0.5 * d.y`.
Hit at `p = o + d * t` with normal `n`; light direction `l = normalize((-4.0, 6.0, -3.0))`;
shadow: if any object hits the ray `(p + n * 0.001, l)` then `diff = 0.0` else `diff = max(dot(n, l), 0.0)`.
`base = color * (0.1 + 0.9 * diff)` componentwise.
If `reflectivity > 0` and `depth < 3`: `rc = trace(p + n * 0.001, reflect(d, n), depth + 1)` and the
result is `base * (1 - refl) + rc * refl` componentwise (`b.c * (1.0 - refl) + rc.c * refl`), else `base`.

Pixel value per channel: `floor(min(max(c, 0.0), 1.0) * 255.0)` as integer. Start each pixel at depth 0.

Output: `checksum <sum of all channel values of all pixels>` and `center <r> <g> <b>` for pixel (240, 180).

## http — real HTTP server under load (lives in `bench/http/`)

The todo API from `api`, but only `GET /todos?limit=<l>&offset=<o>` and `POST /todos` (body
`{"title":"...","tags":[...]}`), served over real TCP on localhost, `Connection: close` (one request per
connection), max body 65536 bytes. Every server starts with 100,000 seeded todos: todo `i` (0-based)
has id `i + 1`, title `task <i>` and tags `t<i mod 10>`, `t<(i / 10) mod 10>`. A third route,
`GET /search?q=<q>&tag=<tag>`, answers `{"count":N}`, where N is the number of todos that have the tag
and contain `q` in the title: it scans the whole store, so it is CPU-bound. Implementations:
- ail: `bench/http/server.ail` (`http-next` / `http-respond` loop: handlers one at a time).
- ail: `bench/http/server_par.ail` (`http-serve`: GET handlers in parallel, POST one at a time).
- Rust std: `std::net::TcpListener`, single-threaded.
- Rust axum + tokio, multi-threaded, store in an `RwLock` (the reference a Rust developer would deploy).
Load generator (Rust, `bench/http/loadgen`): C concurrent clients for a fixed duration; reports
requests/s, p50 and p99 latency. Mix `todos`: POST with probability 0.3, else GET /todos with limit 20
and offset 0. Mix `search`: POST with probability 0.1, else `GET /search?q=<n mod 1000>&tag=t<n mod 10>`.

## fanout — many requests to one service (lives in `bench/http/`)

One client sends N GET requests (`q = 0..N-1`) to the axum server above, 64 in flight at a time, and
adds up the answers. Mode `todos`: `GET /todos?limit=1&offset=<q>`, adds the body lengths (cheap
answers: the client dominates). Mode `search`: `GET /search?q=<q>&tag=t<q mod 10>`, adds the counts
(the backend dominates). ail: `bench/http/fanout.ail` (`par-run` + `http-request`). Rust: tokio +
reqwest + `buffered(64)` (`bench/http/fanout`). Output `sum <S> failed <F> ms <T>`.

## interp — bytecode interpreter (enum dispatch in a hot loop)

A stack VM. Instructions (an enum): `Push(i64)`, `Load(slot)`, `Store(slot)`, `Add`, `Sub`, `Mul`, `Div`,
`Mod`, `Lt` (pushes 1/0), `Eq` (pushes 1/0), `Jmp(target)`, `Jz(target)` (pops; jumps if 0), `Halt`.
Binary ops pop `b` then `a` and push `a op b`. Locals: 8 slots, all initially 0. Integers are 64-bit.

The program (hand-assembled into a vector of instructions — write it as data in the source) computes
`total = sum over n in 1..LIMIT of collatz_steps(n)` with `LIMIT = 300000`, where `collatz_steps(n)`
counts steps until reaching 1 (`n even → n/2`, odd → `3n+1`). Use slots: 0 = n, 1 = x, 2 = steps total.

The program, exactly (index: instruction):
```
 0 Push 1        1 Store 0                                   ; n = 1
 2 Load 0        3 Push 300000   4 Lt      5 Jz 38           ; outer: if !(n < LIMIT) goto end
 6 Load 0        7 Store 1                                   ; x = n
 8 Load 1        9 Push 1       10 Eq     11 Jz 13  12 Jmp 33 ; inner: if x == 1 goto next
13 Load 1       14 Push 2       15 Mod    16 Jz 24            ; if x % 2 == 0 goto even
17 Push 3       18 Load 1       19 Mul    20 Push 1  21 Add  22 Store 1  23 Jmp 28   ; x = 3*x + 1
24 Load 1       25 Push 2       26 Div    27 Store 1         ; even: x = x / 2
28 Load 2       29 Push 1       30 Add    31 Store 2  32 Jmp 8   ; total += 1; goto inner
33 Load 0       34 Push 1       35 Add    36 Store 0  37 Jmp 2   ; next: n += 1; goto outer
38 Halt
```
After `Halt`, print `total <slot 2>` and `executed <number of instructions executed, including Halt>`.

---

## mandel — Mandelbrot set, parallel over rows (the parallel benchmark)

Image `W = 2000` by `H = 2000` pixels, `MAX_ITER = 1000`. Rows are independent: **compute them in
parallel on all cores** (ail: `par-map`; Rust: rayon, the idiomatic choice), then combine in row order.

For pixel `(x, y)` with `0 <= x < W`, `0 <= y < H`:
`cr = 3.0 * x / W - 2.2` and `ci = 2.4 * y / H - 1.2` (f64; `x`, `W`, `y`, `H` converted to f64, and
evaluated exactly as written: `((3.0 * x) / W) - 2.2`).
Iterate from `zr = 0.0, zi = 0.0, i = 0`: while `i < MAX_ITER` and `zr*zr + zi*zi <= 4.0`:
`t = zr*zr - zi*zi + cr` (as `((zr*zr) - (zi*zi)) + cr`), `zi = 2.0*zr*zi + ci` (as `((2.0*zr)*zi) + ci`),
`zr = t`, `i += 1`. The pixel's value is the final `i`.

For each row compute `row_sum` = sum of the row's pixel values (64-bit integer) and `row_inside` = number
of pixels whose value is `MAX_ITER`. Print:
```
total <sum of all row_sum>
inside <sum of all row_inside>
checksum <sum over rows y of (row_sum * (y + 1)) mod 1000000007, computed as a running value: c = (c + row_sum * (y + 1)) % 1000000007>
```
