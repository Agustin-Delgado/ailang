# reach

Breadth-first search from many sources.

Input: the first line has `N M S` (1 <= N <= 500000, 0 <= M <= 2000000, 1 <= S <= 100). Then M lines
`u v`: an undirected edge between u and v (0 <= u, v < N; the same edge can appear more than once,
and an edge can join a vertex to itself). Then one line with S vertices, the sources.

For each source s, in the order given, print one line `R D`: R is the number of vertices that can be
reached from s (s included) and D is the sum of the distances from s to them, the distance being the
number of edges on a shortest path.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
