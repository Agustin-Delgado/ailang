# scheduler

Simulate a CPU scheduler. Each non-empty line of standard input is a process:
`NAME ARRIVAL PRIORITY C1 I1 C2 I2 ... Ck` (k >= 1): it arrives at time ARRIVAL, then alternates CPU
bursts C1..Ck with I/O waits I1..I(k-1) (all integers >= 1; ARRIVAL >= 0). Names are unique words.
A lower PRIORITY number is more important.

Time advances in whole units. There is one CPU; I/O needs no CPU and any number of processes can be
in I/O at once. At each time T, in this order:
1. If the running process has finished its current CPU burst, it leaves the CPU: it starts its next
   I/O wait (ready again at T + I) or, if that was its last burst, it finishes at T.
2. Processes arriving at T, and processes whose I/O ends at T, become ready.
3. If the CPU is free, the best ready process starts running. If a process is running and some ready
   process has a strictly smaller PRIORITY number, the running one goes back to the ready processes
   (keeping what is left of its burst) and the best ready process runs instead.
"Best" means smallest PRIORITY, then smallest ARRIVAL, then NAME in alphabetical order.
Then the running process (if any) works on its burst for one unit, from T to T + 1.

A process is waiting whenever it is ready but not running. For every process print
`NAME FINISH WAITING` (finish time and total waiting time), ordered by finish time and then by name.
Finally print `idle T`: the number of time units in which the CPU ran nothing, counted from time 0
until the last process finishes.
