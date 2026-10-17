# floyd

Shortest paths between all pairs of vertices.

Input: the first line has `N M` (1 <= N <= 1500, 0 <= M <= 200000). Then M lines `u v w`: a directed
edge from u to v with weight w (0 <= u, v < N, 1 <= w <= 1000000). There can be several edges
between the same two vertices, and edges from a vertex to itself.

Let d(u, v) be the length of a shortest path from u to v. Output two lines: `sum S`, the sum of
d(u, v) over all ordered pairs u != v such that v can be reached from u, and `unreachable U`, the
number of ordered pairs u != v such that v cannot be reached from u.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
