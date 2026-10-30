# csvstats

Read a CSV file and describe its columns.

Input: the whole input is a CSV document of ASCII characters, at most 60 MB. It is a sequence of
records; a record is a sequence of fields separated by commas, and records are separated by line
breaks (`\n`). There is no `\r` anywhere. A field is either
- unquoted: zero or more characters, none of them a comma, a double quote `"` or a line break; or
- quoted: a `"`, then any characters, then a closing `"`. Inside, a pair of double quotes `""`
  stands for one `"` in the value, and every other character, commas and line breaks included,
  stands for itself. The closing quote is followed by a comma, a line break or the end of the
  input.

The value of a field is its text, or for a quoted field the text between the quotes with every
`""` turned into `"`. If the input ends with a line break, that line break ends the last record
and no empty record follows it; an empty line anywhere else is a record with a single empty field.
The input is well formed and has at least one record.

The first record is the header: its F fields are the column names, each 1 to 20 letters, digits or
underscores (possibly quoted). Every following record is a data record. A data record is good when
it has exactly F fields and bad otherwise; bad records are counted and otherwise ignored.

Output:
- `records R`: the number of data records;
- `bad B`: the number of bad records;
- then, for each column in header order, one line `column NAME filled N distinct D longest L`:
  over the good records, N is the number whose value in that column is not empty, D the number of
  different non-empty values in it (compared exactly, as text), and L the length in characters of
  the longest value in it (0 when there are no good records).

## Performance

Besides the tests above, the program is timed on one input at the limits stated here; it must
finish within 60 seconds.
