# rpn

Each line of standard input is a reverse Polish notation expression: tokens separated by one or
more spaces; a token is an integer (optionally negative, like `-3`) or one of `+ - * /`.
`/` is integer division truncating toward zero. For each non-empty line print the result, or:

- `error: division by zero` when dividing by zero,
- `error: stack underflow` when an operator finds fewer than 2 values,
- `error: bad token TOKEN` for anything that is not an integer or operator,
- `error: leftover values` when the line ends with more than one value on the stack.

Report the first error found scanning left to right. An empty or blank line prints nothing.
