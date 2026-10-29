"""Extend the research history to yesterday with the broker's daily closes.

The research frame ends whenever the Bloomberg workbooks were last copied
across. The engine needs prices through yesterday's close to produce today's
targets, so each market's broker closes are ratio-spliced onto the research
series at the last date both carry (see ctadata.restitch.ratio_splice for the
reasoning). Only the extension days come from the broker; nothing before the
splice date changes.

Two details that each caused a defect before they were handled:

  * The forming bar. MT5 returns TODAY's bar too, and it is not a close until
    the session ends. Only bars before today are used.
  * New contracts. A dated contract carries bars only from shortly before
    it lists, so a contract born after the research data ends never overlaps
    it and the market would splice to NaN at its first roll. The EA therefore
    writes the spliced days back into its history file every day, so the
    anchor is always yesterday, and this module reads that file when it is
    newer than the research frame.

Requires an initialised MT5 session.
"""
import datetime as dt
import os

import pandas as pd
import MetaTrader5 as mt5

from .specs import SPECS, INVERTED

ROLL_DAYS = 5      # must match the EA's InpRollDays
LIVE_HISTORY = "cta_port_prices.csv"


def _front_contract(root: str, roll_days: int = ROLL_DAYS) -> str:
    """Nearest listed contract for a dated root that expires more than
    `roll_days` from now, e.g. TNOTE10YR -> TNOTE10YRZ6. Same rule as
    CTA_ResolveContract in the EA."""
    horizon = dt.datetime.utcnow() + dt.timedelta(days=roll_days)
    cands = [(s.expiration_time, s.name) for s in mt5.symbols_get()
             if s.name.startswith(root) and s.expiration_time > 0
             and dt.datetime.utcfromtimestamp(s.expiration_time) > horizon]
    if not cands:
        raise RuntimeError(f"no listed contract for {root}")
    return sorted(cands)[0][1]


def broker_symbol(sym: str) -> str:
    row = SPECS.loc[sym]
    return _front_contract(row.broker) if row.kind == "dated" else row.broker


def daily_closes(symbol: str, start: pd.Timestamp) -> pd.Series:
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1,
                                start.to_pydatetime().replace(tzinfo=None), dt.datetime.utcnow())
    if bars is None or len(bars) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(bars["close"], index=pd.to_datetime(bars["time"], unit="s", utc=True).normalize())
    return s[~s.index.duplicated(keep="last")]


def extend_with_broker(df: pd.DataFrame, overlap_days: int = 10):
    """The research frame extended to the latest complete broker close, and a
    per-market report of what was spliced."""
    last = df.index.max()
    out = df.copy()
    report = {}
    for sym in SPECS.index:
        sym_broker = broker_symbol(sym)
        s = daily_closes(sym_broker, last - pd.Timedelta(days=overlap_days))
        if sym in INVERTED:
            s = 1.0 / s
        s = s[s.index.dayofweek < 5]
        common = s.index.intersection(df.index[df[sym].notna()])
        if len(common) == 0:
            report[sym] = f"{sym_broker}: no overlap, not extended"
            continue
        anchor = common.max()
        ratio = df.loc[anchor, sym] / s.loc[anchor]
        ext = s.loc[s.index > last] * ratio
        # the forming bar is not a close yet
        ext = ext[ext.index < pd.Timestamp.utcnow().normalize()]
        for d, v in ext.items():
            out.loc[d, sym] = v
        report[sym] = f"{sym_broker}: +{len(ext)} days, anchor {anchor.date()} ratio {ratio:.5f}"
    return out.sort_index(), report


def load_live_history(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """The EA's spliced history if it exists, has the same columns as `df`
    and ends later; otherwise `df` unchanged."""
    if not path or not os.path.exists(path):
        return df
    live = pd.read_csv(path, index_col=0)
    live.index = pd.to_datetime(live.index, format="%Y.%m.%d", utc=True)
    if list(live.columns) != list(df.columns) or live.index.max() <= df.index.max():
        return df
    return live.astype(float)
