# life

Conway's Game of Life. The first line of standard input is the number of generations G (0 <= G).
The following lines are the grid, all of the same width, with `#` for a live cell and `.` for a dead
one. Cells outside the grid are always dead (no wrap-around). A live cell with 2 or 3 live
neighbours (of 8) survives; a dead cell with exactly 3 becomes alive; every other cell is dead.
Print the grid after G generations in the same format, followed by a final line `alive N` with the
number of live cells.
