# kvstore

An in-memory key-value store with nested transactions. Read commands from standard input, one per
line, tokens separated by single spaces; keys and values are words without spaces.

- `SET key value` sets the key.
- `GET key` prints the value, or `NULL` if the key is not set.
- `DELETE key` removes the key (no output; deleting a missing key is fine).
- `COUNT value` prints how many keys currently have exactly that value.
- `BEGIN` opens a new transaction (transactions nest).
- `ROLLBACK` undoes every change made since the most recent open `BEGIN` and closes that
  transaction. If there is no open transaction print `NO TRANSACTION`.
- `COMMIT` makes all changes permanent and closes all open transactions. If there is no open
  transaction print `NO TRANSACTION`.
