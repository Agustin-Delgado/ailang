# median

Clean up a grayscale image with a repeated median filter.

Input: the first line has `W H T` (1 <= W, H <= 2500, 0 <= T <= 50). Then H lines with W integers
each, separated by single spaces: the pixel values, between 0 and 255. Row r (from 0, top to bottom)
is line r + 1 after the first, and column c (from 0) is its c-th value.

Apply the filter T times. One pass computes a new image in which the pixel at row r, column c is the
median of the 9 values old(clamp(r + dr, 0, H-1), clamp(c + dc, 0, W-1)) for dr and dc in {-1, 0, 1},
where old is the image before the pass and clamp(v, lo, hi) is v limited to the range [lo, hi] (so
pixels outside the image take the value of the nearest edge pixel; at a corner of a 1 x 1 image all 9
values are the same pixel). The median of 9 values is the 5th smallest, counting repeats.

Output three lines about the final image:
- `sum S`: the sum of all pixel values;
- `check C`: the sum over all pixels of value * (r + 1) * (c + 1);
- `changed D`: the number of pixels whose final value differs from their value in the input.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
