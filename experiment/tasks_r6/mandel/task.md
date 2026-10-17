# mandel

Count escape times of the Mandelbrot iteration on a grid of points.

Input: one line `W H M X Y S`: integers W and H (1 <= W, H <= 3000) and M (1 <= M <= 5000), then
decimal numbers X, Y and S (S > 0), such as `-2.25` or `0.00125`.

The point of the pixel in column i (0 <= i < W) and row j (0 <= j < H) is cr = X + i*S,
ci = Y + j*S. Starting from zr = zi = 0, repeat: if zr*zr + zi*zi > 4, stop; otherwise, if M steps
have been done, stop; otherwise do one step: the new zr is zr*zr - zi*zi + cr and the new zi is
2*zr*zi + ci, both computed from the old zr and zi. The pixel's count is the number of steps done
(between 0 and M).

All arithmetic is 64-bit IEEE floating point, evaluated in exactly the order written, left to right,
without fused multiply-add: zr*zr - zi*zi + cr is (zr*zr) - (zi*zi), then + cr; 2*zr*zi + ci is
(2*zr)*zi, then + ci; X + i*S is X + (i*S); zr*zr + zi*zi is (zr*zr) + (zi*zi).

Output two lines: `total T`, the sum of the counts of all pixels, and `inside K`, the number of
pixels whose count is M.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
