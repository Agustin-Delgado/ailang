# nonce

Find proof-of-work nonces.

The hash of a sequence of bytes is computed on unsigned 64-bit integers, where every multiplication
wraps around modulo 2^64, `^` is bitwise exclusive or and `>>` is a logical right shift:

    h = 0x243F6A8885A308D3
    for each byte b, in order:
        h = (h ^ b) * 0x9E3779B97F4A7C15
        h = h ^ (h >> 29)
    h = h ^ (h >> 32)
    h = h * 0xD6E8FEB86659FD93
    h = h ^ (h >> 32)

Input: the first line has Q (1 <= Q <= 20). Then Q lines `P D`: a prefix P of 1 to 20 characters,
each a lowercase letter or a digit, and a difficulty D (0 <= D <= 32).

For each query find the smallest integer n >= 0 such that the hash of the bytes of P followed by the
decimal digits of n (no leading zeros; n = 0 is the single digit `0`) has at least D leading zero
bits, that is, the hash is less than 2^(64-D). For example, for P = `ab` and n = 107 the bytes hashed
are those of `ab107`. The tests are chosen so that every answer is below 10^10.

For each query, in order, print one line `n HASH`: the number n and its hash as 16 lowercase
hexadecimal digits (with leading zeros).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
