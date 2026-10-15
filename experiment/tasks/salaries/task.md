# salaries

Standard input is CSV without a header: each non-empty line is `department,name,salary` (salary
is a non-negative integer). Lines whose salary field is not a valid non-negative integer, or that do
not have exactly 3 fields, are invalid: skip them and count them.
For each department, sorted by name, print `DEPT count=N total=T avg=A max=NAME` where A is the
average salary with exactly 2 decimals (rounded half up on the cents, e.g. 1000.125 -> 1000.13) and
NAME is the person with the highest salary (ties: the alphabetically first name).
Finally print `invalid N`.
