# leaderboard

Keep a game leaderboard and answer rank queries.

Input: the first line has `P Q` (1 <= P <= 200000, 0 <= Q <= 2000000). There are P players,
numbered 1 to P, and every player starts with score 0. Then Q lines, each one operation:
- `add p x`: player p's score grows by x (1 <= p <= P, 1 <= x <= 1000000);
- `rank p`: print the rank of player p, which is 1 plus the number of players whose score is
  strictly higher than p's (players with equal scores share a rank);
- `top k`: print the score of the player in position k (1 <= k <= P) when all P players are listed
  by score from highest to lowest (players with equal scores are equivalent here, since only the
  score is printed).

Print one line per `rank` and `top` operation, in input order, with just the number. After all
operations print one more line `leader S C`: the highest score S and the number C of players who
have it. Scores fit in a signed 64-bit integer.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
