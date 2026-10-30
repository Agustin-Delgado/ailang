# patterns

Count how often each of many words occurs in a text.

Input: the first line has `K L` (1 <= K <= 5000, 0 <= L <= 1000000). Then K lines, the patterns: each
is 1 to 30 lowercase letters `a`-`z` (the same pattern can appear more than once). Then L lines, the
text: each line is made of lowercase letters and spaces and can be empty. The text has at most
40000000 characters.

For each pattern, count its occurrences in the text: the number of pairs (line, position) such that
the pattern appears in that line starting at that position. Occurrences may overlap (`aa` occurs twice
in `aaa`) and never span two lines.

Output K lines, one per pattern in input order: the pattern, a space and its count (a pattern listed
twice is printed twice with the same count). Then one line `total T`, the sum of those K counts.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
