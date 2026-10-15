# brackets

For each line of standard input (including empty lines), check that the brackets `()`, `[]` and
`{}` are balanced and properly nested; all other characters are ignored. Print `ok` if they are.
Otherwise print `error at N`, where N is the 1-based position (in characters) of the first closing
bracket that does not match, or, if every closing bracket matched but some remain open,
N = length of the line + 1.
