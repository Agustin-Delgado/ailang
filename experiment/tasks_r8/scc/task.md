# scc

Find the strongly connected components of a large directed graph.

Input: the first line has `N M` (1 <= N <= 1000000, 0 <= M <= 3000000). Then M lines `u v`: a
directed edge from vertex u to vertex v (0 <= u, v < N). The same edge can appear more than once,
and an edge can go from a vertex to itself.

Two vertices are in the same strongly connected component when each can be reached from the other
by following edges in their direction (every vertex can reach itself, so each vertex is in exactly
one component; a vertex that is on no cycle is a component on its own).

Output five lines:
- `components C`: the number of strongly connected components;
- `largest L`: the number of vertices in the largest component;
- `sources S`: the number of components that no edge enters from another component;
- `sinks T`: the number of components that no edge leaves to another component;
- `cyclic K`: the number of vertices that lie on at least one cycle, that is, vertices in a
  component of two or more vertices, or with an edge to themselves.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
