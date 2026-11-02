# CTA Stack

A trend-following futures sleeve designed to sit on top of a 100% equity
position, from the first idea to an engine trading a demo account. This
repository is the code, the research notebooks with their outputs, and a
log of every bug found along the way.

## The strategy

* **Signal.** Time-series momentum on 12 futures markets across equity
  indices, commodities, currencies and US Treasuries: half the sign of the
  40-day return plus half the sign of the 252-day return, lagged one day.
* **Sizing.** Two tiers of volatility targeting: each position to 20%
  annualised volatility, then the whole book to 13%, with caps on both.
  Within each asset class, Hierarchical Risk Parity splits the risk so that
  correlated markets (gold and silver, the two oils) share an allocation.
* **Regime gate.** K-means clustering on five market-state features (volatility,
  correlation, trend efficiency, dispersion, signal agreement) scales the
  book's size by regime, with the multipliers learned from history only.
* **Use.** Held on top of the S&P 500, not instead of it, which is possible
  because futures need only a few percent of notional as margin.

**Result** (out of sample from September 2008, net of costs, Sharpe after
the risk-free rate): the sleeve alone has a Sharpe of about 0.44; stacked on
the S&P the combined book returns 17.8% a year against the S&P's 10.9%. The
headline carries a measured selection premium of about 0.19 Sharpe (Chapter
4), so a live expectation nearer 0.35 to 0.40 is the prudent one.

The research universe is called A-12 throughout: AUD, Brent, copper, GBP,
gold, JPY, Nasdaq, Nikkei, silver, S&P 500, US 10-year and WTI. The live book
is A-12 plus Bitcoin and Ether.

## What was tried, and what was kept

Every idea was tested against the same base, at matched volatility, and
re-chosen on training data only before it was scored on unseen years.

| Idea | Outcome |
|---|---|
| HRP within each asset class | Kept. The only risk layer that improved the book |
| Regime gate (k-means on market state) | Kept, with the multipliers learned, never tuned |
| Bitcoin and Ether as a fifth class | Kept, and trading live |
| ATR stop-loss | Not used in the backtest; at the tested width it costs return. A wide stop is kept live as a safety rail |
| Economic-trend overlay, efficiency-ratio gate, VIX gate, COT throttle, fast/slow sleeves | Rejected: no measurable gain, or a loss |
| A 56-market universe | Rejected: many of the extra markets are near-copies of each other (sixteen equity indices), so breadth is mostly apparent |
| Macro filter for the wide universe's bad years | Rejected |
| Class-level trend override | Passed its nested test, not deployed |

## How the data was built

Prices come from Bloomberg generic futures, which jump at every roll because
the series switches from one contract to the next without adjustment. Those
jumps would read as trend signals. The returns are rebuilt from the contract
chain instead: on a roll day the return is the new front contract against the
previous day's price of that same contract, so the roll gap never enters the
series. The roll calendar is checked against the chain before a market is
used. Live prices from the broker are spliced onto the history by ratio at the
last common date.

## How it trades

The research engine is Python. The live engine is a second implementation in
MQL5 running inside MetaTrader 5, so nothing outside the terminal is needed at
run time. The two are checked against each other day by day over the full
backtest and agree to within 1e-6. Before going live, an execution simulator
measured what trading costs on top of the research figures: spread and
slippage, a one-day execution lag (the largest single cost), financing, and
whole-lot rounding take the sleeve's Sharpe from 0.67 gross to 0.44. A daily
reconciliation compares the account's positions and fills with what the
engine asked for.

## Where to start

| If you want | Read |
|---|---|
| What went wrong and how it was found | [docs/BUGFIX_LOG.md](docs/BUGFIX_LOG.md) |
| The evidence | `notebooks/ch01` to `ch09`, outputs included |
| The engine | [cta/engine.py](cta/engine.py), `build_trend_nav` |

The nine chapters:

1. **The trend sleeve.** The bare engine, and why stacking is the point.
2. **Data.** Bloomberg generics, the unadjusted roll problem, and the
   restitching from the contract chain.
3. **Validation.** Walk-forward, costs, and the nested test every later
   decision is held to.
4. **The universe.** Choosing markets, and measuring how much the choice
   itself flatters the result.
5. **Layers.** HRP, macro trend, stops, filters: which survived and why.
6. **The regime gate.** Clustering, learned multipliers, auto-k, and the
   portable k-means that lets two languages agree.
7. **The final configuration,** against managed-futures benchmarks and a
   return-stacked ETF.
8. **Extensions.** Crypto (adopted), a 56-market universe (and why it loses),
   a macro filter and a class-level trend override.
9. **Live.** The execution simulator, verifying the MQL5 port, marking and
   reconciling the account.

## Layout

```
ctadata/    data: workbook loading, cleaning, the universe, roll restitching, rates
cta/        the research engine: signals, HRP, regime gate, metrics, walk-forward
live/       execution: broker specs, simulator, live price splice, signal file,
            reconciliation, account marking, the port check
mql5/       the live engine and executor (MetaTrader 5), a second
            implementation of cta/engine.py verified against it
notebooks/  src/chNN_*.py are the sources; build.py runs them into .ipynb
tests/      unit tests on synthetic data, plus fingerprints of the engine
docs/       bug-fix log
```

## Running it

```
pip install -r requirements.txt
pytest                      # synthetic-data tests run anywhere
```

The market data is licensed (Bloomberg) and is not included. The notebooks
are committed with their outputs so they can be read without it. With a
local copy, point the loaders at it and rebuild:

```
set CTADATA_DIR=path\to\workbooks
set CTADATA_RATES_DIR=path\to\rates
python notebooks/build.py notebooks/src/ch05_layers.py
```

The engine-fingerprint tests need the data and skip themselves when
`CTADATA_DIR` is unset; the rest run on synthetic prices. The `live/` package
needs a MetaTrader 5 terminal and runs on Windows only.

## Method, in five rules

1. **Never compare two books at different volatility.** Scaling a strategy
   does not change its return per unit of risk.
2. **State the configuration with every number:** universe, rebalancing,
   gross or net, risk-free subtracted or not, volatility.
3. **A full-sample result is a hypothesis.** Every choice is re-made on
   training data only, scored on the next unseen window, and chained.
4. **Target weights are not held positions.** Attribution is recorded inside
   the engine's loop.
5. **A result that looks too good is a bug until shown otherwise.** Several
   of the entries in the bug-fix log were found exactly that way.

## Status

Research is closed. The engine has run on a demo account since September
2026. It is not validated for live capital, and two items are open: the
port verification has to be redone for the live book's 14 markets in five
classes, and the regime gate's advantage on A-12 did not reproduce when all
the results were re-run in October (bug-fix log M7).
