# bank

Process banking commands from standard input, one per line (tokens separated by single spaces):

- `open NAME` creates an account with balance 0. If it already exists print `error: NAME exists`.
- `deposit NAME AMOUNT` adds AMOUNT (a positive integer).
- `withdraw NAME AMOUNT` subtracts AMOUNT. If the balance is insufficient print
  `error: insufficient funds in NAME` and change nothing.
- `transfer FROM TO AMOUNT` moves AMOUNT. If FROM lacks funds print `error: insufficient funds in FROM`.

Any command that names an account that does not exist prints `error: no account NAME` (check
accounts left to right, report the first missing one) and changes nothing.
At the end print every account as `NAME BALANCE`, sorted by name ascending.
