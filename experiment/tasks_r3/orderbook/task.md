# orderbook

Run a limit order book. Each non-empty line of standard input is one command:

- `BUY ID QTY PRICE` or `SELL ID QTY PRICE`: a new order (ID is a word, QTY and PRICE positive integers).
- `CANCEL ID`: remove the resting order ID.

Matching: an incoming BUY trades against resting SELL orders whose price is <= the BUY price, best
(lowest) price first and, at equal price, the earliest resting order first. An incoming SELL trades
against resting BUY orders whose price is >= the SELL price, highest price first, then earliest. Each
trade prints `TRADE BUYID SELLID QTY PRICE`, where QTY is the amount traded and PRICE is the resting
order's price. Trading continues until the incoming order is filled or nothing else matches; whatever is
left of it rests in the book (keeping its arrival time; an order that was partly filled while resting
keeps its original place).

If a BUY or SELL uses an ID that any earlier order already used, print `REJECT ID` and ignore it.
If CANCEL names an ID that is not resting in the book (unknown, fully filled or already cancelled),
print `REJECT ID`.

At the end print `BOOK`, then every resting BUY as `BID ID QTY PRICE` from highest price to lowest
(earliest first at equal price), then every resting SELL as `ASK ID QTY PRICE` from lowest price to
highest (earliest first at equal price).
