# heat

Simulate heat spreading on a square grid of integers.

Input: the first line has `N T`. Then N lines with N integers each (separated by single spaces):
the starting grid. 1 <= N <= 1500, 0 <= T <= 400, every value is between 0 and 10000.

Run T steps. In each step every cell that is not on the border (row 0, row N-1, column 0, column
N-1) gets the value `(4*c + up + down + left + right) / 8`, rounded down, where c is its own value
and up/down/left/right are its four neighbours, all taken from the grid as it was before the step.
Border cells never change.

Output three lines: `sum S` (the sum of all cells after T steps), `center V` (the value at row N/2,
column N/2, both rounded down, rows and columns numbered from 0) and `max M` (the largest value).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
