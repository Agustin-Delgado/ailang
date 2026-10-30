# segment

Split text without spaces into dictionary words.

Input: the first line has `D L` (1 <= D <= 50000, 0 <= L <= 1000000). Then D lines, the dictionary:
each is a word of 1 to 20 lowercase letters `a`-`z` (a word can be listed more than once; it is
still one word). Then L lines, each a string of lowercase letters, possibly empty. The L strings
have at most 12000000 letters in total.

A split of a string is a way to write it as a sequence of dictionary words placed one after the
other, with nothing left over; a word may be used any number of times. Two splits are different when
their sequences of words are different. The empty string has exactly one split, with no words.

For each string, in input order, print one line: `W K`, where W is the smallest number of words in
a split of it and K is the number of different splits modulo 1000000007; or `none` if it has no
split. After those lines print `split S`, the number of strings that have at least one split.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
