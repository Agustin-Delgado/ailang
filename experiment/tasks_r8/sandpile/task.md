# sandpile

Let a sandpile settle.

Input: the first line has `W H` (1 <= W, H <= 1000). Then H lines with W integers each, separated by
single spaces: the number of grains of sand on each cell, between 0 and 30000. Row r (from 0) is
line r + 1 after the first, and column c (from 0) is its c-th value.

A cell with 4 or more grains is unstable. Toppling an unstable cell takes 4 grains from it and gives
one to each of its neighbors above, below, left and right; a grain given to a position outside the
grid is lost. Topple unstable cells, one toppling at a time, until no cell is unstable. Whatever
order the topplings are done in, the process ends, with the same final grid and the same total
number of topplings.

Output four lines:
- `topples T`: the total number of topplings;
- `grains G`: the number of grains left on the grid;
- `cells A B C D`: the number of cells that end with 0, 1, 2 and 3 grains;
- `check X`: the sum over all cells of grains * (r + 1) * (c + 1) in the final grid.

All of these fit in a signed 64-bit integer.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
