# matmul

Multiply two square matrices modulo 1000000007.

Input: the first line has N (1 <= N <= 800). Then N lines with N integers: matrix A. Then N lines
with N integers: matrix B. Every entry is between 0 and 1000000006.

Let C = A x B with every entry taken modulo 1000000007 (C[i][j] = sum over k of A[i][k] * B[k][j],
mod 1000000007). Output three lines: `sum S` (the sum of all entries of C, modulo 1000000007),
`first X` (C[0][0]) and `last Y` (C[N-1][N-1]).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here, on a machine
with 12 CPU cores; it must finish within 60 seconds. Any feature of the language and its standard
library may be used, including threads.
