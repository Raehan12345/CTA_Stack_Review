"""Restitching futures prices across contract rolls.

THE PROBLEM. A futures contract expires, so any long price history is a
chain of contracts. Bloomberg's generic tickers (ES1, CL1, ...) splice that
chain WITHOUT adjustment: on a roll day the series steps from the expiring
contract to the next one, and the price difference between the two (the
calendar spread, or basis) appears as a return that nobody earned. In
contango it is a spurious gain, in backwardation a spurious loss. Measured on
this data the distortion averages 3.5 to 4.3% a year depending on the set of
markets, and exceeds 15 points a year in the worst cases. The S&P's Monday after quarterly expiry averaged
-0.84% in 2010-2021 and +1.61% in 2023-2026: exactly the sign flip of the
calendar spread between the zero-rate and post-2022 regimes.

This module solves it three ways, for three different situations.

1. RESEARCH, single known splice: `cleaning.patch_known_data_issues`
   rebases the Gilt from its 2011 roll by the jump ratio. Hand-verified.

2. RESEARCH, every roll in every market: build a historical roll calendar
   from the full contract chain (19,034 contracts with expiry and delivery
   month), VALIDATE it against the price data, then stitch returns contract
   by contract:

       ordinary day   r_t = F1_t / F1_{t-1} - 1     same contract both days
       roll day       r_t = F1_t / F2_{t-1} - 1     F2 yesterday IS F1 today

   The second line is exact, not an approximation: on the day the front
   generic rolls, the contract it rolls INTO is the one the second generic
   was tracking the day before. No back-adjustment of levels is needed.

3. LIVE, broker contracts appended to the research history:
   `ratio_splice` anchors each live series on the last date it shares with
   the history and scales it by the price ratio there. Run daily with the
   result written back, the anchor is always yesterday, so when the broker's
   dated contract rolls the new contract is level-matched to yesterday and
   the switch never appears as a jump.
"""
import numpy as np
import pandas as pd

from .paths import data_dir
from .universe import TICKER_INFO, WIDE_FRONT, UNUSABLE

CHAIN_FILE = "FutChain_2026-10-01.xlsx"
DETAIL_FILE = "FutChain_Details_2026-10-01.xlsx"

# A market's chain-derived calendar is trusted only if roll days move at
# least this much more than ordinary days, on at least MIN_ROLLS rolls. Fixed
# before looking at which markets passed.
VALIDATE_MIN_RATIO = 1.2
MIN_ROLLS = 10

# Passed the ratio test by accident: the chain shows monthly rolls on a
# quarterly contract. Excluded by hand, and the reason is recorded here so the
# exclusion cannot be mistaken for a tuning choice.
EXCLUDE = frozenset({"euribor"})


def _month_num(v):
    """A contract's delivery month as an integer (year * 12 + month)."""
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return np.nan
    try:
        if "/" in s:
            m, y = s.split("/")[:2]
            return int(y) * 12 + int(m)
        d = pd.to_datetime(s)
        return d.year * 12 + d.month
    except Exception:
        return np.nan


def load_contracts(chain: pd.DataFrame = None, detail: pd.DataFrame = None) -> pd.DataFrame:
    """Every contract with its root, last tradeable date and delivery month,
    sorted by root then expiry. Pass frames directly to test without data."""
    if chain is None:
        chain = pd.read_excel(data_dir() / CHAIN_FILE)
    if detail is None:
        detail = pd.read_excel(data_dir() / DETAIL_FILE)
    detail = detail.copy()
    chain = chain.copy()
    detail["ticker"] = detail["ticker"].astype(str).str.strip()
    chain["contract"] = chain["contract"].astype(str).str.strip()
    m = chain.merge(detail, left_on="contract", right_on="ticker", how="left")
    m["ltd"] = pd.to_datetime(m["LAST_TRADEABLE_DT"], errors="coerce")
    m["cmon"] = m["FUT_CONTRACT_DT"].map(_month_num)
    return m.dropna(subset=["ltd", "cmon"]).sort_values(["root", "ltd"])


def root_to_name() -> dict:
    """Bloomberg root ticker -> the short market name used everywhere else."""
    return {t: n for t, n in list(TICKER_INFO.items()) + list(WIDE_FRONT.items())
            if t not in UNUSABLE}


def roll_calendar(index: pd.DatetimeIndex, contracts: pd.DataFrame, names: dict):
    """Per market, on every date: the gap in months from the front contract to
    the second, and whether the front changed that day.

    Convention: on date t the front is the nearest contract whose last
    tradeable date has not passed. Bloomberg may roll a day or two earlier for
    some markets; that shifts a roll date slightly without changing the gap,
    which is why the calendar is validated against the prices before use.
    """
    gaps, rolls = {}, {}
    naive = index.tz_localize(None).values if index.tz is not None else index.values
    for root, g in contracts.groupby("root"):
        name = names.get(root)
        if name is None:
            continue
        g = g.drop_duplicates("contract").sort_values("ltd")
        ltd, cmon = g["ltd"].values, g["cmon"].values
        pos = np.searchsorted(ltd, naive, side="left")     # first contract not yet expired
        ok = pos < len(ltd) - 1
        gap = np.full(len(index), np.nan)
        gap[ok] = cmon[pos[ok] + 1] - cmon[pos[ok]]
        s = pd.Series(gap, index=index)
        s[s <= 0] = np.nan
        gaps[name] = s
        rolls[name] = pd.Series(pos, index=index).diff().fillna(0) != 0
    return pd.DataFrame(gaps), pd.DataFrame(rolls).fillna(False)


def roll_ratio(px: pd.Series, rolls: pd.Series) -> float:
    """Mean absolute move on roll days over mean absolute move on other days.

    An unadjusted generic should JUMP on roll days, so a correct calendar
    gives a ratio well above 1. A calendar polluted by contract months that
    never actually trade (gold lists every month, the generic trades five)
    flags hundreds of ordinary days as rolls and the ratio collapses toward 1.
    """
    r = px.pct_change().abs()
    m = rolls.reindex(r.index).fillna(False).astype(bool)
    if m.sum() < MIN_ROLLS or not r[~m].mean():
        return np.nan
    return float(r[m].mean() / r[~m].mean())


def validated_rolls(px: pd.DataFrame, gaps: pd.DataFrame, rolls: pd.DataFrame):
    """Roll days only for markets whose calendar passes the price check.

    Returns (mask, report). Markets that fail keep no roll days, so their
    returns stay unadjusted: an unadjusted series is a known, measurable
    error, while adjusting on a wrong calendar would inject new false moves.
    """
    out = pd.DataFrame(False, index=px.index, columns=px.columns)
    report = {}
    for c in px.columns:
        if c not in rolls.columns or c in EXCLUDE:
            continue
        if not (c in gaps.columns and gaps[c].notna().any()):
            continue
        ratio = roll_ratio(px[c], rolls[c])
        passed = bool(ratio == ratio and ratio >= VALIDATE_MIN_RATIO)
        report[c] = (ratio, passed)
        if passed:
            out[c] = rolls[c].reindex(px.index).fillna(False).astype(bool)
    return out, pd.DataFrame(report, index=["ratio", "passed"]).T


def stitched_returns(front: pd.DataFrame, second: pd.DataFrame,
                     roll_mask: pd.DataFrame) -> pd.DataFrame:
    """Percentage returns stitched across rolls, one column per market."""
    cols = [c for c in front.columns if c in second.columns]
    f, s = front[cols], second[cols]
    ordinary = f / f.shift(1) - 1.0
    across = f / s.shift(1) - 1.0
    m = roll_mask.reindex(index=f.index, columns=cols).fillna(False).astype(bool)
    return ordinary.where(~m, across).replace([np.inf, -np.inf], np.nan)


def stitched_index(front, second, roll_mask) -> pd.DataFrame:
    """A continuous price index (starting at 1.0) from the stitched returns,
    so it can be fed to anything that expects prices, the engine included."""
    r = stitched_returns(front, second, roll_mask).fillna(0.0)
    return (1.0 + r).cumprod()


def ratio_splice(history: pd.Series, live: pd.Series, invert: bool = False):
    """Extend a research price series with a live broker series.

    The broker's instrument tracks the same market with a different basis
    and sometimes a different quote convention (it quotes USDJPY where the
    research series is USD per JPY, hence `invert`). On the last date both
    carry a value the ratio of the two prices is taken, and every NEWER live
    close is scaled by it. Nothing before the splice date changes.

    Returns (extended series, ratio, anchor date). Raises if the two never
    overlap: appending an unanchored series would put a level jump into the
    history, which is exactly the failure this function exists to prevent.
    """
    live = (1.0 / live) if invert else live
    both = history.dropna().index.intersection(live.dropna().index)
    if len(both) == 0:
        raise ValueError("no overlapping date to anchor the splice on")
    anchor = both.max()
    ratio = float(history.loc[anchor] / live.loc[anchor])
    new = live[live.index > history.dropna().index.max()] * ratio
    # Write into the history's own index rather than concatenating: the
    # history may already carry blank rows for those dates, and a concat
    # would duplicate them.
    out = history.reindex(history.index.union(new.index))
    out.loc[new.index] = new.values
    return out, ratio, anchor


def distortion(front, second, roll_mask, start=None) -> pd.DataFrame:
    """Raw CAGR, stitched CAGR and the difference ("fake" return), per market."""
    raw = front.pct_change()
    adj = stitched_returns(front, second, roll_mask)
    if start is not None:
        raw, adj = raw.loc[start:], adj.loc[start:]
    cols = [c for c in adj.columns if roll_mask[c].loc[adj.index].sum() >= MIN_ROLLS]
    yrs = len(adj) / 252
    out = pd.DataFrame({
        "raw_cagr": (1 + raw[cols].fillna(0)).prod() ** (1 / yrs) - 1,
        "stitched_cagr": (1 + adj[cols].fillna(0)).prod() ** (1 / yrs) - 1,
    })
    out["fake_pa"] = out["raw_cagr"] - out["stitched_cagr"]
    return out
