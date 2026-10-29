"""Shared notebook setup: import path, data, the risk-free rate, the
standard out-of-sample start and plotting defaults. Every chapter starts with
`from _setup import *` so no chapter re-derives these on its own."""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402

from cta import build_trend_nav, build_trend_nav_fast, DEFAULT_PARAMS, CHOSEN, GATE  # noqa: E402
from cta.metrics import sharpe, vol, cagr, max_drawdown, stats, matched, table, returns  # noqa: E402
from ctadata import load_cta_data, A12, UNTRADEABLE, buckets           # noqa: E402
from ctadata.rates import rf_daily                                      # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams.update({"figure.figsize": (10, 4.2), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})

df = load_cta_data()
RF = rf_daily(df.index)
SPX = df["spx"].pct_change().fillna(0.0)

# Out of sample from 2008-09-04: the data starts 2006-09, and the first two
# years are the 252-day signal warm-up plus the first training window.
OOS = "2008-09-04"
MONTHLY = {**DEFAULT_PARAMS, "rebalance_freq": "M"}


def oos(r):
    return r.loc[OOS:]


def describe(series: dict, match_to: str = None, stacked: bool = False) -> pd.DataFrame:
    """The standard table: Sharpe after the risk-free rate, realised vol, CAGR
    and worst drawdown, out of sample. With `stacked`, each series is also
    shown on top of a 100% S&P position."""
    s = {k: oos(v) for k, v in series.items()}
    t = table(s, RF, match_to)
    if stacked:
        st = {k: oos(v + SPX.reindex(v.index).fillna(0.0)) for k, v in series.items()}
        ts = table(st, RF)[["sharpe", "cagr", "max_dd"]].add_prefix("stacked_")
        t = t.join(ts)
    return t


def chosen(symbols=None, **over):
    """Daily returns of the chosen configuration (A-12 unless told otherwise)."""
    syms = symbols or A12
    kw = {**DEFAULT_PARAMS, **CHOSEN, "universe": buckets(syms), **over}
    return returns(build_trend_nav(df, symbols=syms, **kw)[0])
