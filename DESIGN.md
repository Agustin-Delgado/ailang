# ail — design decisions

ail is a programming language whose primary author is an AI agent. Every decision below is
judged by two questions: does it make a model produce correct programs with fewer attempts, and
does it let the compiler produce code at least as fast as Rust?

## 1. Mutable value semantics, no references

User code never sees a reference. Every value has one owner; assigning or passing it behaves
like a copy. A function changes caller data only through a `(mut T)` parameter, and the caller
writes `(mut place)` at the call site.

Why it matters for agents: there are no lifetimes, no borrow annotations and no
"cannot borrow as mutable" fights. The agent writes plain values.

Why it matters for speed: since user code cannot create aliases, the compiler knows that every
`mut` parameter is the only path to its data. The backend lowers read-only parameters to `&T`
and `mut` parameters to `&mut T`, which LLVM marks `noalias`. Copies are emitted only where a
value really is duplicated, and a local's final use moves instead of copying.

Aliasing check: a place passed as `(mut x)` cannot appear in any other argument of the same call.
That is the only aliasing rule there is, and it is checked statically.

## 2. Effects are capabilities

There are no ambient effects. `main` receives capability values (`Console`, `Fs`, `Clock`,
`Rand`, `Env`) and passes them down as ordinary parameters. A function without capability
parameters is pure. Reading a signature therefore tells you everything a function can do, and
"which code touches the file system?" is answered by grep. Capabilities cannot be stored in data,
returned, or used in contracts.

When a function needs an effect it does not have, the diagnostic carries a fix that adds the
parameter and passes it. `ail fix` repeats this until the capability reaches `main`.

## 3. No undefined behavior, ever

Integer overflow, out-of-bounds access, division by zero, invalid casts, failed contracts and
`(fail ...)` are *traps*: the program stops and reports the ail source line. There is no wrapping
arithmetic by accident and no unchecked indexing. Errors that a caller should handle are values
(`Result`, `Option`).

## 4. One way to write each thing

The grammar is a small set of s-expression forms with labeled clauses. There is no operator
precedence, no statement/expression split, no implicit conversion, no inheritance, no
overloading and no macros. Every construct is self-describing (`(fn name (params ...) (returns ...) (body ...))`),
so a model rarely has to guess.

## 5. The compiler talks to agents

Every diagnostic has a stable code, an exact span, `expected`/`found` types and, whenever the
fix is mechanical, a list of exact text edits. Patterns that models import from other
languages (`String`, `i64`, `None`, `Some`, `println!`, `+` on strings, `null`...) are recognized
and corrected. `ail fix` applies all fixes and iterates to a fixed point.

Traps report the ail source line at zero runtime cost: the generated code carries a table from
machine lines to source lines that is read only when a trap happens.

## 6. Performance by default

- `mut` parameters are `noalias` (see 1).
- Maps use a fast word-at-a-time hasher instead of a DoS-resistant one.
- Small allocations come from a size-class allocator with per-class free lists.
- Output is buffered and flushed at exit, before reading input, and on traps.
- Overflow and bounds checks stay on; they are the price of having no UB and are measured in the
  benchmarks as such. An index is an I64, and one unsigned comparison against the length covers
  both a negative index and one past the end (a negative value is above any length as unsigned);
  only the trap message tells the two apart. In an innermost loop whose vector indices are `j + c`, one test before the
  loop finds the iterations where every access is in range; those run without bounds checks and
  the edges run checked. If that loop only computes, its integer operations record overflow in a
  flag instead of branching, so LLVM can vectorize it; on overflow that part runs again checked
  and traps where it would have.
- A loop whose inner loops index vectors with values read from other vectors (a graph walk over
  offsets and adjacency lists) is bound by memory. It runs on 32-bit copies of its I64 vectors,
  made on entry, and a vector whose elements serve as indices of another keeps them within that
  one's length (checked on entry and at every store), so those reads need no bounds check. A
  store that does not fit stops the fast run: what it printed was held and is dropped, the
  variables it changed are put back, and the loop runs again as written. The fast run computes
  the same values in the same order, so a trap inside it is the program's own trap.
- `(pow 2 e)` is a shift while the result fits, the ordinary checked pow beyond.
- A vector index is checked with one unsigned comparison: a negative index is a huge one and
  fails the same test, and the trap still says "negative index". This matters most in pointer
  chasing such as a union-find `find`, where every step is an index read from the vector itself.
- `split` on a one-byte separator scans bytes directly instead of going through the general
  string searcher, which costs more than the scan on the short fields of typical input lines.

## 7. Strings are values, read in place

A `Str` behaves like any other value: it is never shared and never changes behind your back. The
compiler still avoids copying it whenever it can prove that nobody could tell the difference:

- read-only `Str` parameters are passed as `&str`;
- an immutable local bound to a piece of another string (`substr`, `trim`, a literal, a branch
  that picks one of several, an element `(at v i)` of a vector) is a borrowed view, as long as
  the string it comes from is neither changed nor moved while the view is used; `lower`/`upper`
  give a view that is copied only when a character actually changes;
- a fn whose result is always a piece of one of its `Str` parameters returns a view of it;
- a `(Vec Str)` that is only read by element (`len`, `at`, `for`, passing it on) holds views
  instead of copies: `(let f (split line " "))` and `(let ls (lines (read-all con)))` are
  vectors of pieces of the string, and a `(var parts (Vec Str) (vec))` that is only pushed
  pieces of strings that outlive it (and literals) is joined with a single copy. A read-only
  `(Vec Str)` parameter used that way accepts either kind of vector;
- `(put (mut m) k (... (get m k) ...))` looks `k` up once, and copies `k` only if it is new; so
  does `(let c (... (get m k) ...))` right before `(put (mut m) k (... c ...))`, and `k` may be a
  field `(. r name)`.

None of this is visible in the language: there are no lifetimes, and the program means the same
with or without it. It is what lets idiomatic ail match hand-tuned Rust on text processing.

## 8. Functions are arguments, never values

A parameter can be a function `(Fn A B R)`, and the caller passes a `lambda`, a fn name or a
function it received. That is all: functions cannot be bound with `let`, stored in data or
returned, and a lambda only reads its surroundings. So there are no closures that outlive the
call, nothing a closure can mutate, no capture rules to learn and nothing to allocate: every
function argument compiles to a direct, inlinable call. `map`, `filter`, `fold`, `sort-by` and
friends cover the common loops; a `for` loop is still the way to change variables or exit
early.

## 9. Generics without bounds

`(type-params T)` on a fn, struct or enum. There is no trait or bound syntax to learn: every use
with concrete types is type-checked as a concrete copy, and an operation the type does not support
is reported at the line inside the generic body, naming the copy (`while checking total<Str>`).
The copies are also what is compiled, so generic code is exactly as fast as code written for
one type. The price is that a generic fn nobody calls is not checked; the compiler says so.

## 10. Parallelism that cannot race

Values are never shared and lambdas only read, so running a lambda on many threads at once is
safe by construction: `(par-map xs f)` splits the Vec across the cores and returns the same Vec,
in the same order, as `map`. The checker also forbids capabilities inside it, so parallel code is
deterministic: no interleaved output, no timing-dependent results. There are no locks, atomics or
threads in the language to get wrong.

Work that waits on the network or the disk uses `(par-run xs f)`: the same shape, up to 64 elements
in flight, results in order. Its function may use Net, Fs, Clock and Env, because their effects do
not depend on which thread runs first. Console and Rand stay out: output would interleave and the
random sequence would depend on timing. `http-request` keeps connections alive and reuses them,
so fanning out to a service costs one handshake per connection, not per request.

### Loops that parallelize themselves

Models rarely ask for parallelism unless the task mentions cores; they write a plain `for`. Since
values are never shared and effects are capabilities, the compiler can see what an iteration
touches outside itself, and when the iterations are independent it runs the loop in chunks on the
thread pool, with no change to the program:

- variables declared in the body are private to the iteration;
- `(set x (+ x e))` on an outside integer is summed privately per chunk; the total is taken only
  if the magnitudes of the terms prove that no partial sum, in program order, could have
  overflowed (otherwise the loop runs sequentially); the same for elements of a vector (a
  histogram) and for sums modulo a constant;
- a running maximum or minimum with its position (`(if (> e best) (do (set best e) (set at i)))`)
  combines chunk by chunk in order, so ties resolve as in the sequential loop; the comparison may
  come with a condition that does not read the maximum (`(if (and ok (< e best)) ...)`);
- an outside variable that the body only assigns takes the value of the last assignment;
- `(set (at v k) x)` is allowed when each iteration writes its own elements: `k` is the loop index,
  or `i * w + j` with `j` from an inner range no wider than `w` (checked before the loop); the
  index may also go down as the loop goes up (`c - i`);
- in-place dynamic programming over a row walked backwards (the 0/1 knapsack): when the last
  statement writes `v` at `c ± i` and the other reads of `v` hit elements only later iterations
  write (checked before the loop), those reads come from a copy of `v` taken before the loop
  (on the GPU, a device copy made at launch);
- a scratch vector that every iteration overwrites before reading gets one copy per chunk;
- `push` and `print` collect per chunk and are appended in iteration order.

A loop of a few heavy iterations that do not split evenly over the cores (16 on 12) gives each
one a thread of its own, so the cores share the last ones instead of letting them run alone.

A parallel loop inside another one (in the body, or in a function it calls) runs in parallel too,
one level deep, when the outer loop has only a few iterations or some thread has run out of work:
the thread that reaches it posts its chunks, and every thread that becomes idle joins in until
they run out. A loop over a handful of queries of unequal size, each scanning a large range, then
keeps every core busy to the end instead of leaving the biggest query to one thread.

If any chunk traps, the parallel run is thrown away and the loop runs again sequentially, so the
output and the trap are exactly the sequential ones (for a nested loop, the chunk of the outer loop
that contains it stops too). `tools/partraps.py` checks the programs in `tests/par_traps/` against
a run on one thread. A loop that already uses `par-map` keeps that parallelism.

### Loops that run on the GPU

A parallel loop whose only effects are sums and writes to its own elements, over vectors of
64- or 32-bit integers or F64, is also compiled to a SPIR-V compute kernel and run through
Vulkan, loaded at run time (no driver needed to build or run the program). Nested loops whose
inner range does not depend on the outer index become one work item per (i, j), also when the
inner loop carries a sliding window (`(set a b) (set b e)` at the end of each step, with `e` a
value computed from j and values fixed for the row): each item computes `e` again for the
columns before it that the window holds, so the items do not depend on each other. Every operation
that traps on the CPU marks the invocation instead; a trapped launch is thrown away and the loop
runs on the CPU, which traps where the program would. Vectors keep a device copy between loops
(a mirror beside each vector in the fns that use the GPU), and a statement that touches a vector
on the CPU first brings its current contents back. `AIL_GPU=0` turns this off, `AIL_GPU=force`
sends every eligible loop to the GPU (used to check that every program gives the same output).

A device copy of a vector of integers is stored as narrow as its values allow (1, 2, 4 or 8 bytes
per element, chosen when it is uploaded); the kernel reads and writes it at that width. A store
that does not fit stops the launch like a trap, but instead of running the loop on the CPU the
launch is undone, the copy goes back to full width for good, and the launch runs again.
`AIL_GPU_NARROW=0` keeps every copy at full width.

Waiting for each launch costs more than most launches, so an outer loop that only computes (it
uses no capability, does not leave early, and changes outside itself only plain values and
vectors with a device copy) runs in growing groups of iterations whose launches are queued and
checked together. The runtime keeps a copy of every vector the group changes, the generated code
keeps the plain values; if anything traps inside the group, on the GPU or the CPU, all of it is
put back and those iterations run again one launch at a time, which traps exactly where the
program does. `tools/gputraps.py` runs the programs in `tests/gpu_traps/` with the GPU off and
forced and checks that output, trap message and line are the same.

`sort` and `sort-desc` on a long vector also use every core: chunks are sorted in parallel and
merged pairwise, each merge cut among the threads at points found by binary search. On ties an
element of the left run goes first, so the result is the one a stable sequential sort gives.

## 11. Servers: reads in parallel, writes alone

A server is a state plus two handlers: `(http-serve net port max-body state read write)`. GET and
HEAD go to `read`, which gets the state read-only and runs on many threads at once; every other
method goes to `write`, a fn that takes the state as `mut` and runs one request at a time. The
runtime holds the state behind a reader-writer lock, but the program never sees it. Two bad
practices become unrepresentable: a GET that changes data does not type-check (E0330), and a data
race between handlers cannot be written. A read handler may call other services (Net, Fs, Clock,
Env), but the write handler gets no capabilities at all: slow I/O while holding the state
exclusively, which stalls every other request, cannot be written either. A trap in `read` answers 500 and the server keeps going, since a reader
could not have changed anything; a trap in `write` stops the program, because the state may be
half-changed.

## 12. Tasks that cannot deadlock

`(tasks ...)` is structured: the block starts its tasks and ends when every one has finished, so
no task outlives the code that started it. Tasks talk only through channels, and the checker
holds them to three rules. Every channel has exactly one sender and one receiver. The graph of
tasks has no cycle. An outer variable is changed by at most one task, its owner, and no other task
may read it. With unbounded channels, the only way to wait forever is a cycle of waits, and a
cycle cannot be written, so a deadlock cannot be written either. Shared mutable state cannot be
written, so there are no data races. Each channel keeps its order, and Console and Rand belong to
one task at most, so a program's output does not depend on timing.

## 13. Toolchain

`ail check` (instant), `ail fix`, `ail run`, `ail test`, `ail build`, `ail emit` (generated Rust),
`ail spec` (the specification an agent should read, embedded in the binary).
The current backend emits Rust and drives `rustc`, which brings LLVM and the WASM target.
