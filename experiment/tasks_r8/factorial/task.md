# factorial

Compute a large factorial exactly.

Input: one line with N (0 <= N <= 50000).

N! is the product 1 * 2 * ... * N (and 0! = 1). Write N! in decimal, without leading zeros. Output
five lines:
- `digits D`: the number of digits;
- `sum S`: the sum of the digits;
- `zeros Z`: the number of trailing zeros (zero digits at the end);
- `head H`: the first 30 digits (all of them if there are fewer than 30);
- `tail T`: the last 30 digits of the number that remains after removing the trailing zeros (all of
  its digits if it has fewer than 30; T can start with zeros).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
