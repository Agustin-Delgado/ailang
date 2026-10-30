# sessions

Group a website's page views into user sessions.

Input: the first line has N (1 <= N <= 3000000). Then N lines `USER T`: user USER viewed a page at
time T, in seconds. USER is 1 to 16 lowercase letters or digits and 0 <= T <= 1000000000. The lines
are in no particular order, and a user can have several views at the same time.

Take each user's views sorted by time. They form sessions: a view belongs to the same session as the
user's previous view when it comes at most 1800 seconds after it, and starts a new session
otherwise (so views at times 0, 1800 and 3600 are one session, and views at 0 and 1801 are two). The
duration of a session is the time of its last view minus the time of its first view (0 for a session
of one view).

Output five lines:
- `users U`: the number of different users;
- `sessions S`: the total number of sessions;
- `average A`: the sum of the durations of all sessions divided by S, rounded down;
- `longest USER START DURATION`: the session with the largest duration, given by its user, the time
  of its first view and its duration; on a tie, the smallest USER in text order, then the earliest
  START;
- `busiest USER K`: the user with the most sessions and that number; on a tie, the smallest USER in
  text order.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
