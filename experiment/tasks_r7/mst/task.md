# mst

Find a minimum spanning forest of a large weighted graph.

Input: the first line has `N M` (1 <= N <= 1000000, 0 <= M <= 3000000). Then M lines `u v w`: an
undirected edge between vertices u and v with weight w (0 <= u, v < N, 0 <= w <= 1000000000). There
can be several edges between the same two vertices, and edges from a vertex to itself.

A minimum spanning forest is a set of edges with no cycle that connects every pair of vertices that
the whole graph connects, with the smallest possible total weight.

Output three lines:
- `weight W`: the total weight of a minimum spanning forest;
- `components C`: the number of connected components of the graph (a vertex with no edges is a
  component on its own);
- `heaviest H`: the largest weight of an edge in a minimum spanning forest, or `heaviest none` if the
  forest has no edges. (Every minimum spanning forest gives the same W and H.)

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
