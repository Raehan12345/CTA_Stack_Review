"""Known defects in the raw data, and the fixes for them.

Every function here changes the price LEVEL from some date onward by a
constant factor. A constant factor cancels out of the ratio between two
prices that both carry it, so each fix corrects exactly one day's return and
leaves every other day's return untouched. That is why these are safe to
apply to a returns-based strategy, and also why a patched column must never
be compared by LEVEL with an unpatched series (see GILT_ROLL_FIX below).
"""
import pandas as pd

from .log import log

# The Gilt generic fell from 133.29 to 117.06 (-12.2%) on 2011-12-29 and
# stayed there. A 10-year gilt future does not move 12% in a session: the
# generic jumped from the expiring December contract onto March without
# adjustment.
#
# Found when an Efficiency Ratio gate concentrated the whole book into the
# Gilt on that date and produced a -108% day; it had been in every earlier
# backtest, diluted across the book. A scan of every market for daily moves
# above 8% found no other isolated anomaly: every other hit is a dated event
# (2008, March 2020, natural gas).
#
# Confirmed later against the second-month generic, which was not available
# when the fix was written: that day the front fell 12.18% while the second
# rose 0.42%, and the two closed at 117.06 and 117.00. The front had jumped
# onto the second contract's price.
#
# After the patch the column is an adjusted index, not a price. Comparing its
# LEVEL with the raw second month produced a spurious +10.8% basis that was
# once written up, wrongly, as a broken second-month ticker.
GILT_ROLL_FIX_DATE = pd.Timestamp("2011-12-29", tz="UTC")
GILT_ROLL_FIX_RATIO = 133.29 / 117.06


def patch_known_data_issues(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the hand-verified fixes. Only `gilt` is altered."""
    if "gilt" in data.columns:
        mask = data.index >= GILT_ROLL_FIX_DATE
        data.loc[mask, "gilt"] = data.loc[mask, "gilt"] * GILT_ROLL_FIX_RATIO
        log(f"gilt roll patched from {GILT_ROLL_FIX_DATE.date()} "
            f"(x{GILT_ROLL_FIX_RATIO:.6f}, {int(mask.sum())} rows)")
    return data


def fix_spike_and_revert(series: pd.Series, threshold: float = 0.07) -> pd.Series:
    """The Gilt fix, generalised: rebase from every day whose return exceeds
    `threshold`.

    Written for the SG Trend Indicator, which shows about 39 days of over 7%
    moves, nearly all in early January, each reversing within a day or two.
    A diversified trend benchmark does not do that every January.

    Applying it to several dates in order is safe: each correction is a
    constant multiplicative shift from its date onward, and a constant cancels
    from the ratio between two shifted levels.
    """
    s = series.astype(float).copy()
    rets = s.pct_change()
    flagged = rets[rets.abs() > threshold].index
    log(f"fix_spike_and_revert: {len(flagged)} days over {threshold:.0%}")
    for date in sorted(flagged):
        loc = s.index.get_loc(date)
        if loc == 0:
            continue
        ratio = s.iloc[loc - 1] / s.iloc[loc]
        s.iloc[loc:] = s.iloc[loc:] * ratio
    return s


MERGE_TOL = 1e-6


def overlap_check(old: pd.DataFrame, new: pd.DataFrame, tol: float = MERGE_TOL):
    """Compare the days both frames carry, treating each column's FINAL
    on-disk observation as provisional.

    Every workbook was pulled during a trading day, so its last row is an
    intraday tick rather than a settlement. The new pull returns the settled
    close for that date, so a difference there is expected and the new value
    is right. A difference on any EARLIER shared day means the pull is not
    the same series.

    Returns (hard, provisional): hard is fatal, provisional is informational.
    """
    both = old.index.intersection(new.index)
    hard, prov = [], []
    for c in old.columns:
        if c not in new.columns:
            continue
        a, b = old.loc[both, c], new.loc[both, c]
        m = a.notna() & b.notna()
        if not m.any():
            continue
        last_on_disk = old[c].dropna().index.max()
        rel = (a[m] - b[m]).abs() / a[m].abs().clip(lower=1e-12)
        for d, r in rel.items():
            if r <= tol:
                continue
            if d == last_on_disk:
                prov.append((c, d, float(a[d]), float(b[d]), float(r)))
            else:
                hard.append((c, d, float(r)))
    return hard, prov


def merge_pull(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge an incremental Bloomberg pull into an existing frame.

    Three properties, each learned from a real refresh:

    1. Refuse if the pull carries a column the existing data does not have:
       that means the wrong file.
    2. Refuse if any EARLIER shared day disagrees (see overlap_check).
    3. Otherwise the pull wins where both have a value (it is settled; the
       existing last row is an intraday tick) and the existing data fills
       everything else. `combine_first` is used rather than appending rows
       after the last date, because the files are ragged: one column can
       stop a day before its neighbours, and an append would leave that gap.
    """
    only_new = set(new.columns) - set(old.columns)
    if only_new:
        raise ValueError(f"pull has columns not in the existing data: {sorted(only_new)}")
    new = new.reindex(columns=list(old.columns))
    hard, _ = overlap_check(old, new)
    if hard:
        raise ValueError(f"pull disagrees on settled days: {hard[:5]}")
    return new.combine_first(old).sort_index()
