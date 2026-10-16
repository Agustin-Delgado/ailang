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
  benchmarks as such.

## 7. Strings are values, read in place

A `Str` behaves like any other value: it is never shared and never changes behind your back. The
compiler still avoids copying it whenever it can prove that nobody could tell the difference:

- read-only `Str` parameters are passed as `&str`;
- an immutable local bound to a piece of another string (`substr`, `trim`, a literal, a branch
  that picks one of several) is a borrowed view, as long as the string it comes from is neither
  changed nor moved while the view is used; `lower`/`upper` give a view that is copied only when a
  character actually changes;
- a fn whose result is always a piece of one of its `Str` parameters returns a view of it;
- `(put (mut m) k (... (get m k) ...))` looks `k` up once, and copies `k` only if it is new.

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

## 11. Servers: reads in parallel, writes alone

A server is a state plus two handlers: `(http-serve net port max-body state read write)`. GET and
HEAD go to `read`, which gets the state read-only and runs on many threads at once; every other
method goes to `write`, a fn that takes the state as `mut` and runs one request at a time. The
runtime holds the state behind a reader-writer lock, but the program never sees it. Two bad
practices become unrepresentable: a GET that changes data does not type-check (E0330), and a data
race between handlers cannot be written. Handlers get no capabilities: they compute a `Reply` and
the runtime sends it. A trap in `read` answers 500 and the server keeps going, since a reader
could not have changed anything; a trap in `write` stops the program, because the state may be
half-changed.

## 12. Toolchain

`ail check` (instant), `ail fix`, `ail run`, `ail test`, `ail build`, `ail emit` (generated Rust),
`ail spec` (the specification an agent should read, embedded in the binary).
The current backend emits Rust and drives `rustc`, which brings LLVM and the WASM target.
