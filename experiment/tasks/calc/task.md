# calc

Each non-empty line of standard input is an arithmetic expression over integers with `+ - * /`,
parentheses and unary minus, with arbitrary spaces. `*` and `/` bind tighter than `+` and `-`; all
binary operators are left-associative; unary minus binds tighter than everything (`-2*3` is -6,
`--4` is 4). `/` is integer division truncating toward zero. Print the value of each expression,
or `error` if the expression is malformed (bad character, unbalanced parentheses, missing operand)
or divides by zero. Values fit in a signed 64-bit integer. Blank lines print nothing.
