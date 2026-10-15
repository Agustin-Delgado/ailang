# hashes

Fingerprint lines with 64-bit FNV-1a. For each line of standard input (without its line ending; the
input has no empty lines), compute the hash: start with h = 14695981039346656037; for each byte b of
the line (UTF-8), h = h XOR b, then h = h * 1099511628211, both on unsigned 64-bit integers where
the multiplication wraps around modulo 2^64.

Print one line per distinct hash, in the order the hash first appears: the hash as 16 lowercase
hexadecimal digits (with leading zeros), a space, and how many input lines had that hash.
Then print `xor X` where X is the XOR of the hashes of all lines (16 hex digits), and `sum S` where S
is the sum of the hashes of all lines modulo 2^64, in decimal.
