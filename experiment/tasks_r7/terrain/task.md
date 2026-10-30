# terrain

Cheapest routes across a terrain map.

Input: the first line has `R C` (1 <= R, C <= 3000). Then R lines of C characters each: a digit `1` to
`9` is a cell that costs that much to enter, and `#` is a wall that cannot be entered.

You start in the top-left cell (row 0, column 0); standing there costs nothing. From a cell you can
move to the cell above, below, left or right of it, if it is inside the map and not a wall, paying the
cost of the cell you enter. If the top-left cell is a wall, no cell can be reached, not even that one.

The cost of a cell is the smallest total cost of a sequence of moves from the start to it (0 for the
start itself). Output three lines:
- `cost X`: the cost of the bottom-right cell (row R-1, column C-1), or `cost unreachable` if it
  cannot be reached;
- `reached N`: the number of cells that can be reached, the start included;
- `sum S`: the sum of the costs of all cells that can be reached.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
