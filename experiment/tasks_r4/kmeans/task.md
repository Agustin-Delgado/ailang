# kmeans

Run k-means clustering on integer points, with exact integer arithmetic.

Input: the first line has `N K I`. Then N lines `x y`, the points (integers). 1 <= K <= N <= 1000000,
K <= 64, 1 <= I <= 20, and -1000000 <= x, y <= 1000000.

The starting centers are the first K points, in order (center 0 is the first point). Then repeat I
times:

1. Assign every point to the center with the smallest squared distance `(x-cx)^2 + (y-cy)^2`; on a
   tie, to the center with the smallest index.
2. Move every center to the mean of the points assigned to it, each coordinate rounded down
   (toward negative infinity: the mean of -3 and 0 is -2). A center with no points stays where it is.

Output K lines, one per center in index order: `x y n`, the center's final position and the number of
points assigned to it in the last step 1.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here, on a machine
with 12 CPU cores; it must finish within 60 seconds. Any feature of the language and its standard
library may be used, including threads.
