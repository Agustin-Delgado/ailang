# jsonq

Query a JSON document. The first line of standard input is a JSON value on one line: objects, arrays,
strings, integers (no fractions or exponents, |n| < 10^9), true, false and null, with any spaces between
tokens. Strings use only ASCII and the escapes `\"`, `\\` and `\n`; object keys are unique within
each object. Every following non-empty line is a query: a PATH, `keys PATH` or `len PATH`.

A PATH is `.` alone (the whole document) or a sequence of steps, each `.name` (an object member; names
are letters, digits and `_`) or `[i]` (an array element, 0-based), e.g. `.users[2].name`.

- PATH: print the value at PATH as compact JSON: no spaces, object members sorted by key, strings
  escaped the same way as the input (`\"`, `\\`, `\n`).
- `keys PATH`: the keys of the object at PATH, sorted, separated by single spaces (an empty line for
  an empty object).
- `len PATH`: the number of elements of an array, members of an object, or characters of a string.

If the path does not exist (missing member, index out of range, or stepping into the wrong kind of
value), or `keys`/`len` is used on a value they do not apply to, print `none`.
