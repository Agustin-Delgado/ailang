# spell

Suggest spelling corrections.

Input: the first line has `D Q` (1 <= D <= 300000, 0 <= Q <= 50000). Then D lines, the dictionary:
each a word of 1 to 15 lowercase letters `a`-`z` (a word can be listed more than once; it is still
one word). Then Q lines, the queries: each a word of 1 to 15 lowercase letters.

A dictionary word is a suggestion for a query when it is the query itself, or can be made from the
query by exactly one of these edits: deleting one letter, inserting one letter `a`-`z` anywhere
(before the first letter, between two, or after the last), or replacing one letter by a different
letter `a`-`z`.

For each query, in order, print one line `QUERY K BEST`: K is the number of different dictionary
words that are suggestions for it, and BEST is the smallest of them in alphabetical order, or `-`
if K is 0. After those lines print `total T`, the sum of all the K.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
