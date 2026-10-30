# logstats

Summarize a web server access log.

Input: the whole input is the log, one request per line; every line ends with a line break (`\n`).
Empty lines (no characters at all) are ignored. Every other line is a record of seven fields separated
by single spaces:

    TIMESTAMP IP METHOD PATH STATUS BYTES MS

for example `2026-10-29T13:05:44 10.0.7.31 GET /api/users/42?full=1 200 5123 37`.

A non-empty line is valid when all of the following hold, and malformed otherwise:
- splitting it at every space gives exactly 7 fields, none of them empty;
- TIMESTAMP has exactly 19 characters: a digit at every position except `-` at positions 4 and 7, `T`
  at 10 and `:` at 13 and 16 (positions numbered from 0); the values themselves are not checked;
- IP is any field;
- METHOD is one of `GET`, `POST`, `PUT`, `DELETE`, `PATCH`;
- PATH starts with `/`;
- STATUS is exactly 3 digits and its first digit is 1 to 5;
- BYTES and MS are each 1 to 9 digits (leading zeros allowed: `007` is 7).

The route of a valid line is METHOD, a space, and its PATH normalized: cut PATH at its first `?` (that
character and everything after it are dropped), then split it at every `/` and replace every piece
that is non-empty and made only of digits by `{id}`; the other pieces, empty ones included, stay as
they are. So `/api/users/42/?x=1` becomes `/api/users/{id}/`, `/v2/items//007` becomes
`/v2/items//{id}` and `/?q=1` becomes `/`.

Output, in this order:
- `lines L`: the number of non-empty lines;
- `malformed M`: the number of malformed lines;
- `ips U`: the number of different IP fields among the valid lines (compared as text);
- `peak MINUTE K`: MINUTE is the first 16 characters of a TIMESTAMP (like `2026-10-29T13:05`), chosen
  as the one that the most valid lines have, K being that number of lines; on a tie, the smallest
  MINUTE in text order. If there are no valid lines print `peak none`;
- one line `status CODE K` for every STATUS value that appears in valid lines, in ascending order,
  with K the number of valid lines that have it;
- the 10 routes with the most valid lines (all of them if there are fewer), one line each:
  `route METHOD NORMALIZEDPATH COUNT BYTES AVGMS`, where COUNT is its number of valid lines, BYTES the
  sum of their BYTES, and AVGMS the sum of their MS divided by COUNT, rounded down. Order by COUNT
  descending, then BYTES descending, then the route text (METHOD, space, normalized path) in
  ascending byte order.

The input has at most 3000000 lines and at most 150 MB; the sums fit in a signed 64-bit integer.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
