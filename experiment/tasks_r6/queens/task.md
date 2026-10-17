# queens

Count placements of queens on a board with forbidden squares.

Input: the first line has `N F` (1 <= N <= 16, 0 <= F <= 256). Then F lines `r c`: a forbidden square
in row r, column c (0 <= r, c < N; the same square can be listed more than once).

Count the ways to place N queens on the N x N board, one in each row, so that no two queens share a
column or a diagonal and no queen stands on a forbidden square. Print the count.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
