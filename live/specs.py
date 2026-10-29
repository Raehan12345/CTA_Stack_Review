"""The live book's instruments at the broker: one row per research symbol.

The live columns were read from the MT5 terminal. Financing spreads are
DERIVED, not read: the broker quotes its overnight swap as an annual
percentage of notional (SYMBOL_SWAP_MODE 5), and the spread is that swap
less the reference rate on the day it was read, so it can be applied through
history on top of the historical rate:

    long pays    rf(t) + fin_long
    short earns  rf(t) - fin_short        (negative = the short pays)

For FX the swap is the pair's rate differential less a spread on each side.

Energy: the rolling-oil swap embeds the futures curve's roll yield (a long
was RECEIVING 8.2% on the read date, in backwardation). That is carry the
research data cannot see either way, so energy spreads are set to the
metals' values. The live book holds dated oil contracts, so this only
affects the all-CFD comparison in the simulator.

`kind`: "dated" markets are traded as the broker's dated futures contracts
and rolled; "cfd" markets are rolling CFDs that pay the swap.
`ccy` is the currency of the PRICE, except for FX, where it is the BASE
currency, because one FX lot is `contract` units of the base currency.
"""
import pandas as pd

SPECS = pd.DataFrame([
    # sym      broker        kind     contract  ccy    vmin  vstep spread_bps fin_long fin_short margin  cls
    ("spx",    "SPX500",    "dated",    50.0,   "USD", 0.01, 0.01,   0.66,    2.50,    3.50,    0.005, "eq"),
    ("nasdaq", "NDAQ100",   "dated",    20.0,   "USD", 0.01, 0.01,   0.31,    2.50,    3.50,    0.005, "eq"),
    ("nikkei", "JPN225",      "cfd",   500.0,   "JPY", 0.01, 0.01,   1.10,    2.98,    3.02,    0.005, "eq"),
    ("gold",   "GOLD",      "dated",   100.0,   "USD", 0.01, 0.01,   0.46,    1.30,    1.10,    0.005, "cm"),
    ("silver", "SILVER",    "dated",  5000.0,   "USD", 0.01, 0.01,   3.90,    1.50,    1.10,    0.010, "cm"),
    ("copper", "COPPER",    "dated",   250.0,   "USc", 0.01, 0.01,   4.80,    4.80,    1.20,    0.030, "cm"),
    ("wti",    "USOIL",     "dated",  1000.0,   "USD", 0.02, 0.01,   2.40,    1.50,    1.10,    0.030, "cm"),
    ("brent",  "UKOIL",     "dated",  1000.0,   "USD", 0.02, 0.01,   2.30,    1.50,    1.10,    0.030, "cm"),
    ("aud",    "AUDUSD.a",    "cfd", 100000.0,  "AUD", 0.01, 0.01,   0.50,    1.13,    1.13,    0.033, "fx"),
    ("gbp",    "GBPUSD.a",    "cfd", 100000.0,  "GBP", 0.01, 0.01,   0.50,    1.03,    1.03,    0.033, "fx"),
    ("jpy",    "USDJPY.a",    "cfd", 100000.0,  "USD", 0.01, 0.01,   0.50,    1.09,    1.09,    0.033, "fx"),
    ("us10y",  "TNOTE10YR",   "dated", 1000.0,  "USD", 0.01, 0.01,   1.50,    0.00,    0.00,    0.005, "fi"),
    # Crypto: rolling CFDs, so spot is the underlying and there is no roll.
    # Swaps were -15.0% long / +7.5% short on both. fin_short is negative
    # because a short EARNS more than the reference rate, which is unusual
    # and is the broker's quote, not an error.
    ("btc",    "BITCOIN",       "cfd",    3.0,   "USD", 0.01, 0.01,   1.80,   10.65,   -3.15,    0.200, "cy"),
    ("eth",    "ETHEREUM",      "cfd",   45.0,   "USD", 0.01, 0.01,   6.50,   10.65,   -3.15,    0.200, "cy"),
    # The Gilt was removed: the broker rejected every order server-side.
], columns=["sym", "broker", "kind", "contract", "ccy", "vmin", "vstep", "spread_bps",
            "fin_long", "fin_short", "margin", "cls"]).set_index("sym")

# The research jpy series is USD per JPY; the broker trades USDJPY. A long
# "jpy" in research is therefore a SHORT USDJPY of the same USD notional.
INVERTED = {"jpy"}

# Dated bond contracts roll quarterly, this many business days before the
# delivery month ends. Used by the simulator's roll-cost model.
ROLL_MONTHS = (3, 6, 9, 12)
ROLL_DAYS_BEFORE_MONTH_END = 8


def spec_buckets() -> dict:
    """{asset class: [symbols]} with members in SPEC order, not sorted.

    HRP's single-linkage clustering is sensitive to column order at the
    fourth decimal of Sharpe (0.4802 sorted against 0.4804 in spec order),
    and the EA iterates in spec order, so the Python twin must too.
    """
    from ctadata import asset_class
    out = {}
    for s in SPECS.index:
        out.setdefault(asset_class(s), []).append(s)
    return out
