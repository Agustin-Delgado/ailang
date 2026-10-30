# rooms

Assign meetings to rooms.

Input: the first line has N (0 <= N <= 3000000). Then N lines `s e`: meeting i (numbered from 0 in
input order) starts at time s and ends at time e (0 <= s < e <= 1000000000).

Rooms are numbered 1, 2, 3, ... and all start empty. Take the meetings in order of start time; meetings
with the same start time are taken in input order. Each meeting goes to the lowest-numbered room that
is free at its start time s. A room is free at time s if every meeting already put in it has ended by
then, that is, has an end time <= s (a meeting ending at 10 frees its room for one starting at 10). If
no room that has been used so far is free, the meeting goes to a new room, numbered one more than the
highest room used so far.

Output three lines:
- `rooms R`: the number of rooms used;
- `busiest B K`: the room B that got the most meetings and that number of meetings K; on a tie, the
  lowest-numbered such room. When N = 0 print `busiest 0 0`;
- `check X`: X is the sum over all meetings of (i + 1) * room_i, modulo 1000000007, where i is the
  meeting's input number and room_i is its room.

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
