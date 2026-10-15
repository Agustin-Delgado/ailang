# inventory

A shop's stock and orders. Each line of standard input is one of:

- `stock ITEM QTY PRICE` adds QTY units of ITEM (creating it if needed) and sets its unit price to
  PRICE (integer cents).
- `order CUSTOMER ITEM QTY` sells QTY units. If the item does not exist print
  `error: unknown item ITEM`; if there are not enough units print `error: only N ITEM left` (N =
  units in stock). Otherwise the sale happens at the current price.

At the end print a report:
1. For every item, sorted by name: `ITEM QTY` (units left).
2. For every customer with at least one successful order, sorted by total spent descending and then
   by name ascending: `CUSTOMER TOTAL` where TOTAL is in cents.
3. A last line `revenue TOTAL` with the sum of all sales in cents.
