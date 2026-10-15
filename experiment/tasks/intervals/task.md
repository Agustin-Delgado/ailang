# intervals

Each non-empty line of standard input is a closed integer interval `A B` with A <= B (values may be
negative). Merge all intervals that overlap or touch (that is, [1,3] and [4,6] merge into [1,6] because
they cover consecutive integers; [1,3] and [5,6] do not). Print the merged intervals sorted by start,
one per line as `A B`, followed by a line `covered N` where N is the number of integers covered.
