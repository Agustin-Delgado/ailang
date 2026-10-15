# roads

A road network with very long roads. The first line of standard input is `N M`: towns 1..N and M roads.
The next M lines are `U V W`: a two-way road between towns U and V of length W (1 <= W <= 10^15;
several roads may join the same pair of towns). Every remaining non-empty line is a command:

- `PATH A B`: print the length of the shortest route from A to B (0 when A = B), or `UNREACHABLE`.
- `CLOSE U V`: remove every road between U and V (in either direction); print nothing.
- `OPEN U V W`: add a two-way road of length W between U and V; print nothing.

All lengths fit in a signed 64-bit integer, but they can exceed 2^53: compute them exactly.
