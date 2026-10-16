# collatz

Find the longest Collatz chains in ranges.

The Collatz step takes n to n/2 if n is even and to 3n+1 if n is odd. steps(n) is how many steps it
takes to reach 1 (steps(1) = 0, steps(6) = 8).

Input: the first line has Q (1 <= Q <= 10). Each of the next Q lines has `A B` with 1 <= A <= B <= 10000000.

For each query print `n s`: the n in [A, B] with the largest steps(n), and s = steps(n). On a tie,
the smallest such n.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
