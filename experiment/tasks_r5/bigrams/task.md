# bigrams

Find the most common pairs of consecutive words in a text.

Input: the first line has K (1 <= K <= 1000). The rest of the input is the text, at most 50 MB of
ASCII. A word is a maximal run of letters A-Z or a-z, compared in lowercase; everything else
(digits, punctuation, spaces, line breaks) separates words. Words are consecutive across line breaks.

A pair is two consecutive words. Print the K most frequent pairs, one per line, as `count first
second`, most frequent first; pairs with the same count in ascending order of the text `first second`
(compared byte by byte). If there are fewer than K distinct pairs, print all of them.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
