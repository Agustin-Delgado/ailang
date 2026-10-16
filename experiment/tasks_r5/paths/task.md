# paths

Count monotone paths through a grid with walls.

Input: the first line has `R C` (1 <= R, C <= 4000). Then R lines of C characters each: `.` is free,
`#` is a wall.

Count the paths from the top-left cell to the bottom-right cell that only move right or down and
never enter a wall. If either corner is a wall there are 0 paths. Print the count modulo 1000000007.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
