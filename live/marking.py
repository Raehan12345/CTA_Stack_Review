"""The live book's daily equity curve, rebuilt from the broker's deals.

    equity(t) = deposit + realised P&L to t + floating value of open positions at t

Two defects in an earlier version, both fixed here and both reproduced as
assertions in tests/test_live_marking.py:

1. LOOK-AHEAD. The entry price was one volume-weighted average per symbol
   over the WHOLE period. So the mark on any day used entries made after it,
   and a September long was blended with an October short into one price
   neither leg ever traded at. Replaced by average-cost accounting walked in
   deal order (position_book_daily).

2. FX NEVER MARKED. An FX lot is a fixed number of BASE currency units, so
   its USD notional does not depend on the price: marking by the change in
   notional gives exactly zero. Every open currency position, the book's
   largest risk allocation, was missing from the curve. mark_usd now prices
   FX as contract x (now - entry) in the quote currency, converted to USD.

Verified against the broker's own P&L to within a dollar.
"""
import datetime as dt

import numpy as np
import pandas as pd

from . import reconcile as rd

DEPOSIT = 100_000.0


def position_book_daily(deals: pd.DataFrame, end=None):
    """Signed lots AND the average entry price behind them, per symbol per
    calendar day.

    Average-cost accounting in deal order:
      adding to a position      blends the entry price
      reducing a position       leaves the entry alone: that P&L is already
                                realised in the deal's own profit
      flipping through flat     resets the entry to the price that flipped it
      closing to flat           no entry price at all
    """
    if deals.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = deals.sort_values("time")
    days = pd.date_range(d["time"].min().normalize(),
                         pd.Timestamp(end or dt.date.today()).normalize(), freq="D")
    syms = sorted(d["symbol"].unique())
    lots = pd.DataFrame(0.0, index=days, columns=syms)
    entry = pd.DataFrame(np.nan, index=days, columns=syms)
    pos = dict.fromkeys(syms, 0.0)
    avg = dict.fromkeys(syms, np.nan)
    d = d.assign(_day=d["time"].dt.normalize())
    for day in days:
        for _, r in d[d["_day"] == day].iterrows():
            s = r["symbol"]
            q = float(r["lots"]) if r["side"] == "BUY" else -float(r["lots"])
            p, cur = float(r["price"]), pos[s]
            if cur == 0.0:
                avg[s] = p
            elif (cur > 0) == (q > 0):
                avg[s] = (abs(cur) * avg[s] + abs(q) * p) / (abs(cur) + abs(q))
            elif abs(q) > abs(cur):
                avg[s] = p
            pos[s] = cur + q
            if abs(pos[s]) < 1e-9:
                pos[s], avg[s] = 0.0, np.nan
        for s in syms:
            lots.at[day, s] = pos[s]
            entry.at[day, s] = avg[s]
    return lots, entry


# The P&L of one FX lot lands in the QUOTE currency.
FX_QUOTE = {"aud": "USD", "gbp": "USD", "jpy": "JPY"}


def mark_usd(px, root: str, lots: float, entry_px: float, now_px: float) -> float:
    """USD profit on `lots` of `root` held from `entry_px` to `now_px`.

    `px` supplies fx_usd(ccy) and lot_value_usd(root, price) (a
    reconcile.Prices, or any object with those two methods). A missing price
    marks nothing rather than poisoning the whole day with NaN.
    """
    if abs(lots) < 1e-9 or not np.isfinite(entry_px) or not np.isfinite(now_px):
        return 0.0
    sym = rd.ROOT_TO_SYM.get(root)
    if sym in FX_QUOTE:
        q = px.fx_usd(FX_QUOTE[sym])
        if not np.isfinite(q):
            return 0.0
        return lots * float(rd.SPECS.loc[sym, "contract"]) * (now_px - entry_px) * q
    a, b = px.lot_value_usd(root, now_px), px.lot_value_usd(root, entry_px)
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return lots * (a - b)


def live_equity(start: str = rd.DEMO_START, end=None) -> pd.Series:
    """Daily account equity from the broker's deal history and daily bars.

    The last day marks against a daily bar that may still be forming: a run
    during the session can move once the day settles (one run gave -1.01%,
    then -2.44% once the same day closed). Run after the US close.
    """
    px = rd.Prices(True)
    if not px.ok:
        raise SystemExit("MT5 unavailable: open the terminal and log in")
    deals = rd.read_deals_mt5(dt.datetime.fromisoformat(start))
    days = pd.bdate_range(start, end or dt.date.today())
    realised = (deals.assign(d=deals["time"].dt.normalize())
                .groupby("d")[["profit", "swap", "commission"]].sum().sum(axis=1)
                .reindex(days, fill_value=0.0).cumsum())
    lots, entry = position_book_daily(deals, end or dt.date.today())
    lots = lots.reindex(days, method="ffill").fillna(0.0)
    entry = entry.reindex(days, method="ffill")
    floating = pd.Series(0.0, index=days)
    for sym in lots.columns:
        root = rd.root_of(sym)
        bars = px.d1(sym, days.min() - pd.Timedelta(days=10), days.max())
        if bars is None or bars.empty:
            continue
        close = bars["close"].reindex(days, method="ffill")
        for d in days:
            if pd.isna(close.get(d)):
                continue
            floating[d] += mark_usd(px, root, float(lots.at[d, sym]),
                                    float(entry.at[d, sym]), float(close[d]))
    eq = DEPOSIT + realised + floating
    eq.name = "live"
    return eq.dropna()
