# sparselife

Conway's Game of Life on an unbounded grid. The first line of standard input is G, the number of
generations to run (0 <= G <= 200). Every other non-empty line is `X Y`: a live cell at column X,
row Y (integers, possibly negative, |X|, |Y| <= 10^9; duplicates count once).

Each generation, a live cell with 2 or 3 live neighbours (of its 8) stays alive, a dead cell with
exactly 3 live neighbours becomes alive, and every other cell is dead.

After G generations print `alive N`. If N > 0 also print `box MINX MINY MAXX MAXY` (the smallest
rectangle containing every live cell) and then the first min(N, 10) live cells in order of Y then X,
one per line as `X Y`.
