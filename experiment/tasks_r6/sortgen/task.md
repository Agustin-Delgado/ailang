# sortgen

Sort numbers made by a pseudo-random generator.

Input: one line `N X` (1 <= N <= 30000000, 0 <= X < 2^63).

Let x_0 = X and x_(i+1) = (x_i * 6364136223846793005 + 1442695040888963407) mod 2^64, computed on
unsigned 64-bit integers. The numbers are a_i = floor(x_i / 2^33) for i = 1..N (each is below 2^31).

Output three lines: `median M`, the number at position floor(N/2) when a_1..a_N are sorted in
ascending order (positions numbered from 0); `distinct D`, how many different numbers there are; and
`top T`, the sum of the 1000 largest numbers, counting repeats (the sum of all of them when N < 1000).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
