# roman

Each non-empty line of standard input is either a decimal integer or a Roman numeral.
- For a decimal integer N with 1 <= N <= 3999 print its Roman numeral in canonical form (using the
  subtractive pairs IV, IX, XL, XC, CD, CM). For any other integer print `invalid`.
- For a Roman numeral (uppercase letters IVXLCDM) print its decimal value, but only if it is written
  in canonical form (exactly what the rule above would produce for that value); otherwise print
  `invalid`.
- Anything else prints `invalid`.
