# life

Run Conway's Game of Life on a grid that wraps around.

Input: the first line has `N T` (1 <= N <= 2000, 0 <= T <= 1000). Then N lines of N characters each:
`#` is a live cell, `.` a dead one.

The neighbours of the cell in row r, column c are the eight cells ((r + dr) mod N, (c + dc) mod N)
for dr and dc in {-1, 0, 1}, not both 0 (rows and columns numbered from 0). When N is 1 or 2 some of
these eight are the same cell, or the cell itself; each of the eight positions counts. In each
generation a live cell with 2 or 3 live neighbours stays alive, a dead cell with exactly 3 live
neighbours becomes alive, and every other cell is dead in the next generation. All cells change at
the same time.

Output two lines after T generations: `alive A`, the number of live cells, and `sum S`, the sum of
r*N + c over the live cells.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
