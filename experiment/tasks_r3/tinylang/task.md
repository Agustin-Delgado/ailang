# tinylang

Interpret a tiny programming language. Standard input is the program, one statement per line
(blank lines are ignored; tokens may be separated by any number of spaces):

- `let NAME = EXPR` sets variable NAME (creating it if needed).
- `set NAME = EXPR` changes an existing variable; using `set` on an undefined name is an error.
- `print EXPR` prints the value.
- `if EXPR` ... `else` ... `end` runs the first block when the value is not 0, else the second
  (`else` and its block are optional).
- `while EXPR` ... `end` repeats the block while the value is not 0. Blocks nest.

Values are signed 64-bit integers. NAME is letters, digits and `_`, starting with a letter. EXPR is
built from integer literals (decimal, no sign), names, parentheses, unary `-`, binary `* / %`
(highest), `+ -`, and comparisons `< <= > >= == !=` (lowest, yielding 1 or 0). Operators of the same
level associate to the left (`a - b - c` is `(a - b) - c`; `a < b < c` is `(a < b) < c`).
`/` truncates toward zero and `%` takes the sign of the left operand (`-7 / 2` is -3, `-7 % 2` is -1).

Errors stop the program: print `error: division by zero at line L`, `error: undefined NAME at line L`
or `error: overflow at line L` (any result outside the signed 64-bit range, including literals),
where L is the 1-based line number of the statement in the input. Output printed before the error
stays. The input is always well-formed apart from these runtime errors, and every program ends.
