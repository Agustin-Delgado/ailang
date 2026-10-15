# ail language specification (v0.1)

ail is a language written by AI agents. A program is a list of typed s-expressions.
There is exactly one way to write each thing. Values never alias, effects are explicit
capabilities, errors are values, and there is no undefined behavior.

Toolchain: `ail check f.ail --json` (instant, JSON diagnostics with fixes), `ail fix f.ail`
(applies every fix), `ail run f.ail`, `ail test f.ail`, `ail build f.ail -o out`.

## Syntax

- Indent nested forms. If the closing parens do not compile but the indentation implies a valid
  program, the compiler uses the indentation (warning W0004) and `ail fix` rewrites the parens.
- Comments start with `;`. Names: letters, digits, `-`, `_`, `?` (e.g. `max-depth`, `empty?`).
- Literals: `42`, `-7`, `1_000`, `0xff`, `3.14`, `1e-9`, `"text\n"`, `true`, `false`, `unit`, `none`.
- Integer literals adapt to the expected type (`I32`, `U8`, `F64`...); otherwise they are `I64`.
  Float literals are `F64`. A literal also adapts to the other operand: `(* 2 x)` works for any numeric
  `x`, but a float literal never converts an integer *variable*: `(+ n 2.0)` with `n : I64` is an error,
  write `(+ (as F64 n) 2.0)`.

## Types

`I8 I16 I32 I64 U8 U16 U32 U64 F32 F64 Bool Str Unit`, `(Vec T)`, `(Map K V)` (K: integer, Str, Bool),
`(Option T)`, `(Result T E)`, user structs and enums, and capabilities `Console Fs Clock Rand Env Net Gfx`.
Indexes and lengths are always `I64`.

## Top-level items

```
(struct Point (x F64) (y F64))
(enum Shape (Circle Point F64) (Rect Point Point) (Empty))   ; variants have positional fields
(const MAX I64 100)                                           ; numeric or Bool only
(fn name (params (a I64) (v (mut (Vec I64))))                ; params is optional
  (returns I64)                                               ; optional, default Unit
  (requires (> a 0))                                          ; optional contracts, checked at runtime
  (ensures (>= result 0))                                     ; `result` is the return value
  (body expr...))                                             ; value of the last expression
(test "name" expr...)                                         ; run with `ail test`
(use geometry)                                                ; load geometry.ail (see Files)
```

Structs, enums and fns can be generic: `(struct Box (type-params T) (value T))` (see Generics).

Functions may call each other in any order (mutual recursion needs no declarations).
A `let`/`var` may shadow an earlier binding of the same name. Defining a function with the name of a
builtin replaces the builtin for that program. Type and variant names start with an uppercase letter. Enum variants are always qualified:
`Shape.Circle`. Recursive types are allowed (`(enum Tree (Leaf) (Node Tree Tree))`).

## Expressions

```
(let x 5)  (let x I32 5)            ; immutable binding (optional type)
(var n 0)  (var xs (Vec I64) (vec)) ; mutable binding
(set n (+ n 1))                     ; assign to a var, a field (. p x), or an element (at v i)
(set (. p x) 1.0)  (set (at v 0) 9)  (set (at (at grid y) x) 1)  ; places nest
(if cond then else)                 ; (if cond then) when then is Unit
(do e1 e2 ... en)                   ; sequence, value of en
(while cond body...)
(for i (range 0 n) body...)         ; 0..n-1, n exclusive
(for x xs body...)                  ; elements of a Vec (read-only)
break  continue  (return v)  (return)   ; these have type Never: they fit anywhere
(match v (case pattern body...) ...) ; must be exhaustive
(new Point (x 1.0) (y 2.0))         ; all fields required
(with p (x 5.0))                    ; copy of p with fields replaced
(. p x)  (. a b c)                  ; field access, chained
(Shape.Circle p 2.0)  Shape.Empty   ; enum construction
(try e)                             ; unwrap Result/Option or return the err/none early
(as F64 n)                          ; numeric conversion (traps if the value does not fit)
(copy x)                            ; explicit copy (see aliasing)
(f a b)                             ; call; mutated args are marked: (push (mut v) 1)
```

Patterns: `_`, a name (binds), integer and string literals, `true`, `false`, `none`, `(some p)`,
`(ok p)`, `(err p)`, `Enum.Variant`, `(Enum.Variant p1 p2 ...)`. Pattern bindings are read-only.

## Values, mutation and aliasing

Every value has one owner; assignment and argument passing behave like copies.
A function can only change caller data through a `(mut T)` parameter, and the caller must
write `(mut place)` at the call site. A place that is passed as `(mut ...)` cannot appear in
any other argument of the same call (no aliasing). Bindings made with `let`, loop variables
and pattern bindings are read-only; use `var` to mutate. You cannot mutate a Vec while
iterating it with `for x xs`; iterate by index instead.
Reading never copies: `for x xs`, `(match (get m k) ...)`, `(match (at v i) ...)` and passing a
value to a non-`mut` parameter all look at the data in place. Copies happen only when a value is
stored somewhere new (`let`/`var`, `push`, `put`, a struct field).

## Effects are capabilities

A function without capability parameters is pure: no I/O, no clock, no randomness, no globals.
`main` receives the capabilities the program needs and passes them down explicitly:

```
(fn main (params (con Console) (fs Fs) (clock Clock) (rng Rand) (env Env) (net Net)) ...)  ; any subset
```

`main` returns `Unit` or `(Result Unit Str)` (an `err` prints the message and exits 1).
Capabilities cannot be stored in structs, returned, or used in contracts. Tests get none.

## Discarded values

A value computed in statement position is discarded, so the branches of an `if`/`match` used as a
statement may have different types, and a function returning Unit may end with any expression
(e.g. `(if found (del (mut m) k) (put (mut m) k 1))`).

## Errors

No exceptions and no null. Fallible operations return `Result`/`Option`; handle them with
`match`, `try`, `unwrap-or`, or `expect`. Overflow, out-of-bounds, division by zero, failed
`expect`/`assert`/contracts and `(fail msg)` stop the program with a *trap* that reports the
source line. Arithmetic is checked: overflow never wraps silently.

## Builtin functions

Arithmetic and logic: `(+ a b ...)` `(- a b)` `(- a)` `(* a b ...)` `(/ a b)` `(% a b)`
`(== a b)` `(!= a b)` `(< a b)` `(<= a b)` `(> a b)` `(>= a b)` `(and a b ...)` `(or a b ...)` `(not a)`
`(bit-and a b)` `(bit-or a b)` `(bit-xor a b)` `(shl a n)` `(shr a n)` `(abs x)` `(min a b)` `(max a b)`
`(pow b e)` (int: e is U32) `(sqrt x)` `(floor x)` `(ceil x)` `(round x)` `(sin x)` `(cos x)` `(tan x)`
`(atan2 y x)` `(exp x)` `(ln x)`. Operands of one operator must have the same type; convert with `as`.
Explicit overflow handling (integers): `(wrapping-add a b)` `(wrapping-sub a b)` `(wrapping-mul a b)`
wrap around (for hashes and RNGs); `(checked-add a b)` `(checked-sub a b)` `(checked-mul a b)` ->
`(Option T)`, `none` on overflow. Plain `+ - *` trap on overflow.
`(+ a b c)` evaluates left to right as `((a + b) + c)`. Floats are IEEE-754 and never fused
(no multiply-add contraction), so float results are the same on every platform.
`(as T x)` checks `x` with its own type first, then converts: `(as F64 (% n 1000))` is integer `%`.

Text: `(str a b ...)` concatenates numbers, Bool and Str into a Str. `(len s)` (bytes) `(substr s start end)`
`(split s sep)` -> `(Vec Str)` `(join xs sep)` `(trim s)` `(upper s)` `(lower s)` `(starts-with s p)`
`(ends-with s p)` `(contains s part)` `(index-of s part)` -> `(Option I64)` `(replace s from to)`
`(chars s)` -> `(Vec Str)` `(byte-at s i)` -> `I64` `(from-code n)` -> Str of one code point `(repeat s n)`
`(parse-i64 s)` `(parse-f64 s)` -> Option (surrounding whitespace and a leading `+`/`-` accepted) `(fmt-float x decimals)` -> Str (`(fmt-float 2.5 2)` is `"2.50"`;
rounds to nearest, ties to even on the exact binary value, so use integer cents for money).
Characters: `(is-digit s)` `(is-alpha s)` `(is-alnum s)` `(is-space s)` `(is-upper s)` `(is-lower s)`
(true when s is non-empty and every character qualifies) `(code-of s)` -> code point of the first character.
`(lines text)` -> `(Vec Str)` splits on `
` (a trailing newline does not add an empty line; `
` is removed).
Floats print in full precision with `str`/`println` (`2.5`, `0.1`, `1.0`).
To build a long string, append in place: `(set s (str s part ...))` grows `s` without copying it
(linear time), or collect the parts in a `(Vec Str)` and `join` them.

Vec: `(vec a b ...)` `(fill n x)` `(len v)` `(is-empty v)` `(at v i)` `(push (mut v) x)` `(pop (mut v))` -> Option
`(last v)` -> Option `(insert-at (mut v) i x)` `(remove-at (mut v) i)` -> T `(swap (mut v) i j)` `(clear (mut v))`
`(extend (mut v) other)` `(slice v start end)` `(concat a b)` `(contains v x)` `(index-of v x)` -> Option
`(sort (mut v))` ascending `(sort-desc (mut v))` descending `(reverse (mut v))` `(sum v)`.

Ordering: numbers, Str (bytewise), Bool, and Vec/Option/structs/enums built from them compare with
`< <= > >= min max` and `sort`. Structs compare field by field in declaration order and enums by variant
order. To sort records by one field use `(sort-by (mut v) (lambda (r) (. r field)))`. Maps have no order.

Map: `(map)` empty (needs a type annotation) `(put (mut m) k v)` `(get m k)` -> Option `(has m k)`
`(del (mut m) k)` -> Option `(keys m)` (sorted) `(values m)` (sorted by key) `(len m)` `(is-empty m)` `(clear (mut m))`.

Vec with a function (see below): `(map v f)` -> Vec of the results `(filter v f)` `(any v f)` `(all v f)`
`(count v f)` -> I64 `(find v f)` -> Option `(find-index v f)` -> `(Option I64)`
`(fold v init (lambda (acc x) new-acc))` `(sort-by (mut v) (lambda (x) key))` (stable, ascending; negate a
number key for descending) `(min-by v key-fn)` `(max-by v key-fn)` -> Option (first element with the
smallest/largest key). The Vec always comes first: `(map xs (lambda (x) (* x 2)))`.
`(par-map v f)` is `map` spread over all CPU cores, with the same result in the same order. Its
function cannot use capabilities (no printing, clock or randomness inside), so the result never
depends on timing; lambdas cannot change variables, so there are no data races to write.

Option/Result: `(some x)` `none` `(ok x)` `(err e)` `(unwrap-or o default)` `(expect o "msg")`
`(is-some o)` `(is-none o)` `(is-ok r)` `(is-err r)`.

Effects (first argument is the capability):
`(print con a b ...)` `(println con a b ...)` `(eprintln con ...)` `(read-line con)` -> `(Option Str)` (none at end of input)
`(read-all con)` -> Str (all of stdin; the usual way to read input: `(for line (lines (read-all con)) ...)`)
`(read-file fs path max-bytes)` -> `(Result Str Str)` `(write-file fs path text)` -> `(Result Unit Str)`
`(now-ms clock)` `(now-ns clock)` `(rand-int rng low high)` (high exclusive) `(rand-f64 rng)` `(args env)` -> `(Vec Str)`.
`(http-next net port max-body-bytes)` -> `(Result Request Str)` waits for the next HTTP request (larger bodies
are rejected with 400 automatically). `(http-respond net req status content-type body)` -> `(Result Unit Str)`.

## Apps for the browser (games and pages)

An app replaces `main` with three functions; extra capability parameters (e.g. `(rng Rand)`) may follow:

```
(fn init (returns Model))
(fn update (params (m (mut Model)) (e Event)))      ; Event is a prelude enum
(fn view (params (m Model) (g Gfx)))                 ; draw on a canvas, every frame
; or (fn page (params (m Model)) (returns Str))      ; HTML; elements with data-on="name" send Click/Input
```

Drawing (pixels as F64, colors as CSS strings): `(draw-clear g color)` `(draw-rect g x y w h color)`
`(draw-circle g x y r color)` `(draw-line g x1 y1 x2 y2 width color)` `(draw-text g x y size color text)`.
`ail web app.ail` builds to WebAssembly and serves it; `ail build app.ail --target web -o dir` writes the files.

## Prelude types (always available)

```
(enum Json (Null) (Bool Bool) (Num F64) (Str Str) (Arr (Vec Json)) (Obj (Map Str Json)))
(struct Request (id I64) (method Str) (path Str) (query (Map Str Str)) (headers (Map Str Str)) (body Str))
(enum Event (Tick F64) (KeyDown Str) (KeyUp Str) (MouseDown F64 F64) (MouseUp F64 F64) (MouseMove F64 F64)
  (Click Str) (Input Str Str) (Resize F64 F64))      ; Tick carries seconds since the last frame
```

`(json-parse text)` -> `(Result Json Str)`, `(json-str j)` -> Str (compact, object keys sorted;
numbers in shortest round-trip form, integral values without a fraction: `3`, `2.5`, `-0.0025`).
`json-str` also encodes typed values directly, which is the fast way to write responses: numbers,
Bool, Str, Vec, Option (`none` is `null`), `(Map Str V)` and structs of those (an object with the
fields in name order): `(json-str (new Point (x 1) (y 2)))` is `{"x":1,"y":2}`.
Request `path` and `query` are URL-decoded; header names are lowercase. A server is a loop:

```
(while true
  (let req (try (http-next net 8080 65536)))
  (match (str (. req method) " " (. req path))
    (case "GET /health" (try (http-respond net req 200 "text/plain" "ok")))
    (case _ (try (http-respond net req 404 "text/plain" "not found")))))
```

Other: `(fail "msg")` stops the program (type Never, fits anywhere). `(assert cond)` `(assert cond "msg")`.

## Functions as arguments

A parameter can be a function. `(Fn A B R)` lists the argument types, then the result:

```
(fn twice (params (f (Fn I64 I64)) (x I64)) (returns I64) (body (f (f x))))
(twice (lambda (x) (* x 3)) 2)   ; 18
(twice inc 5)                    ; the name of a fn works too
```

`(lambda (x y) body...)` gets its parameter types from the `(Fn ...)` it is passed to; write
`(lambda ((x I64)) ...)` to state one. Its value is its last expression. A lambda reads the variables
around it but never changes them, and `return`, `break`, `continue` and `try` cannot leave it.
Functions are only passed as arguments (to a `(Fn ...)` parameter or a builtin like `map`): they
cannot be bound with `let`, stored in data or returned.

## Files

`(use name)` at the top level loads `name.ail` from the same directory as the file that says it
(`(use lib/geometry)` loads `lib/geometry.ail`). Every file is loaded once, wherever it is used,
and all files share one namespace: a name defined in two files is an error. Only the file you run
should define `main`; `ail test main.ail` runs the tests of every file.

## Generics

Fns, structs and enums can take type parameters with a `(type-params ...)` clause:

```
(struct Pair (type-params A B) (first A) (second B))
(enum Tree (type-params T) (Leaf) (Node (Tree T) T (Tree T)))
(fn largest (type-params T) (params (xs (Vec T))) (returns (Option T))
  (body ...))
```

Write the type with its arguments: `(Pair I64 Str)`, `(Tree Str)`. Type arguments are worked out
from the call or the values (`(largest (vec 3 9 2))`, `(new Pair (first 1) (second "a"))`); when
nothing says what they are, put the type on the binding: `(let t (Tree I64) Tree.Leaf)`.
There are no bounds to declare. Each use with concrete types is checked as if written for them,
so `(+ a b)` on a `T` works for numbers and is an error, at the line inside the generic fn, when
`T` is Str. A generic fn that is never called is not checked (warning W0005).

There are no globals except `const`, no inheritance, no exceptions and no implicit conversions.

## Example

```
(struct Account (owner Str) (balance I64))

(fn deposit (params (a (mut Account)) (amount I64))
  (requires (> amount 0))
  (body (set (. a balance) (+ (. a balance) amount))))

(fn parse-amounts (params (line Str)) (returns (Result (Vec I64) Str))
  (body
    (var out (Vec I64) (vec))
    (for part (split line ",")
      (match (parse-i64 part)
        (case (some n) (push (mut out) n))
        (case none (return (err (str "not a number: " part))))))
    (ok out)))

(fn main (params (con Console)) (returns (Result Unit Str))
  (body
    (var acc (new Account (owner "ana") (balance 0)))
    (for n (try (parse-amounts "10,20,30")) (deposit (mut acc) n))
    (println con (. acc owner) " has " (. acc balance))
    (ok unit)))

(test "parse" (assert (is-err (parse-amounts "1,x"))))
```

## Diagnostics

`ail check --json` prints `{"ok":bool,"errors":n,"diagnostics":[...]}`. Each diagnostic has a stable
`code`, `message`, `span` (line/col), `expected`/`found` when relevant, `notes`, and `fixes`: each fix is a
list of exact text edits. `ail fix` applies them. Missing capabilities, missing `(mut ...)`, `let` that
should be `var`, non-exhaustive matches and misspelled names are all fixed automatically.
