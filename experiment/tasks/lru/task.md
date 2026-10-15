# lru

Simulate an LRU cache. The first line of standard input is the capacity C (C >= 1). Every following
non-empty line is `GET key` or `PUT key value` (keys and values are words without spaces).

- `GET key`: if present, print its value and mark it most recently used; otherwise print `MISS`.
- `PUT key value`: insert or update the key and mark it most recently used. If inserting a new key
  makes the cache exceed C entries, evict the least recently used key and print `EVICT key`.

At the end print `hits H misses M` (counting only GETs) and then the keys currently in the cache from
most to least recently used, separated by single spaces (print an empty line if the cache is empty).
