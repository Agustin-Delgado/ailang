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

## 7. Toolchain

`ail check` (instant), `ail fix`, `ail run`, `ail test`, `ail build`, `ail emit` (generated Rust),
`ail spec` (the specification an agent should read, embedded in the binary).
The current backend emits Rust and drives `rustc`, which brings LLVM and the WASM target.
