"""The class P&L override: a second trend layer, run on each asset class's
own profit and loss instead of on prices.

WHY. On a wide universe, sixteen highly correlated equity indices failed
together for 2018-2021. An instrument-level stop cannot see that, because
each index on its own looks like an ordinary losing trade. A class-level
rule can.

THE RULE. For each asset class, build the class's realised P&L curve from
the engine's own attribution. Run the book's 40/252 barbell on that curve.
Both legs down: stand the class down to zero. Both legs up: bring it back.
Two rebalances of confirmation each way, one day of implementation lag, and
the freed risk is left unused, so the test measures the stand-down alone.

WHY P&L AND NOT PRICE. A price trend asks "is this class going up". A P&L
trend asks "is this class working for us": a short book in a falling class
is working, and only the second question can see that. A price-based version
was specified first and did the opposite of what was predicted; it is
recorded as the first of two trials.

THE SELF-REFERENCE. Standing a class down freezes its own P&L, so the curve
driving re-entry would stop updating and the stand-down would be permanent.
The rule therefore reads a SHADOW curve that accrues as if the class were
always live. Here the shadow is exact and free, because the override is
applied after an unmasked engine run. A production version would have to
run a real shadow book alongside, so it would not reproduce these numbers
exactly.

Result (nested, eight unseen windows): helps the wide pools, hurts the
concentrated live book, as predicted before it was built. Not deployed.
"""
import numpy as np
import pandas as pd

from .config import TRANSACTION_COST_BPS

FAST, SLOW, CONFIRM = 40, 252, 2


def _legs(curve: pd.Series):
    """Barbell legs on an equity curve, lagged a day like every other signal."""
    return (np.sign(curve / curve.shift(FAST) - 1.0).shift(1),
            np.sign(curve / curve.shift(SLOW) - 1.0).shift(1))


def _mask(exit_raw: pd.Series, reentry_raw: pd.Series, is_rebal: pd.Series) -> pd.Series:
    """1 while the class is live, 0 while stood down. Decisions only on
    rebalance days, each needing CONFIRM consecutive confirmations."""
    m = pd.Series(1.0, index=exit_raw.index)
    live, sx, sr = True, 0, 0
    for i in range(len(m)):
        if bool(is_rebal.iloc[i]):
            if live:
                sx = sx + 1 if bool(exit_raw.iloc[i]) else 0
                if sx >= CONFIRM:
                    live, sx, sr = False, 0, 0
            else:
                sr = sr + 1 if bool(reentry_raw.iloc[i]) else 0
                if sr >= CONFIRM:
                    live, sx, sr = True, 0, 0
        m.iloc[i] = 1.0 if live else 0.0
    return m.shift(1).fillna(1.0)


def class_pnl_override(held: pd.DataFrame, contrib: pd.DataFrame, cost: pd.DataFrame,
                       universe: dict):
    """Apply the override to a finished engine run.

    held, contrib, cost: held weights, per-market contribution and cost
    (from build_trend_nav(..., collect_contrib=True) and
    live.simulator.held_weights), all on one index.

    Returns (base daily returns, overridden daily returns, share of days
    each class was stood down).
    """
    base = contrib.sum(axis=1) - cost.sum(axis=1)
    is_rebal = pd.Series(~pd.PeriodIndex(held.index.tz_localize(None), freq="M").duplicated(),
                         index=held.index)
    masked = pd.DataFrame(1.0, index=held.index, columns=held.columns)
    stood_down = {}
    for cls, names in sorted(universe.items()):
        names = [n for n in names if n in held.columns]
        if not names:
            continue
        shadow = (1 + contrib[names].sum(axis=1) - cost[names].sum(axis=1)).cumprod()
        f, s = _legs(shadow)
        m = _mask(((f < 0) & (s < 0)).fillna(False), ((f > 0) & (s > 0)).fillna(False), is_rebal)
        stood_down[cls] = float(1 - m.mean())
        for n in names:
            masked[n] = m
    over = (contrib * masked).sum(axis=1) - (cost * masked).sum(axis=1)
    # standing a class down or back up is a trade: charge it
    switch_cost = pd.Series(0.0, index=held.index)
    for cls, names in sorted(universe.items()):
        names = [n for n in names if n in held.columns]
        if not names:
            continue
        flip = masked[names[0]].diff().abs() > 0
        bps = np.mean([TRANSACTION_COST_BPS.get(n, 0.0005) for n in names])
        switch_cost += flip * held[names].abs().sum(axis=1) * bps
    return base, over - switch_cost, pd.Series(stood_down)
