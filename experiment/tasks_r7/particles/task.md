# particles

Simulate particles that pull on each other.

Input: the first line has `N T R` (1 <= N <= 3000, 0 <= T <= 200, 0 <= R <= 1000000000). Then N lines
`x y vx vy m`: the position, velocity and mass of particle i (numbered from 0 in input order), all
integers, with -1000000 <= x, y <= 1000000, -1000 <= vx, vy <= 1000 and 1 <= m <= 100.

Run T steps. In each step, first compute the acceleration of every particle from the positions at the
start of the step: the acceleration (ax, ay) of particle i is the sum, over every other particle j
with |xj - xi| <= R and |yj - yi| <= R, of (mj * sign(xj - xi), mj * sign(yj - yi)), where sign(d) is
1 if d > 0, -1 if d < 0 and 0 if d = 0. Then, for every particle, vx = vx + ax and vy = vy + ay, and
then x = x + vx and y = y + vy (using the new velocity). All values are exact integers and fit in a
signed 64-bit integer.

After T steps output four lines:
- `sumx SX` and `sumy SY`: the sums of the x and of the y coordinates;
- `momentum P`: the sum over all particles of m * (|vx| + |vy|);
- `check C`: the sum over all particles of (i + 1) * (x - y).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
