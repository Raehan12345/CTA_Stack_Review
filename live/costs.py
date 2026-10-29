"""Crypto financing, the one cost the engine does not charge.

The engine charges no financing on purpose: for a futures contract the cost
of carry is already in the forward price, so charging it again would
double-count. A crypto CFD is not a future. A long pays 15% a year and a
short earns 7.5%, and neither is in a spot price, so it is charged on top.
It turned out to cost only 0.03 Sharpe: crypto is 3.3% of gross notional and
flat or short most of the time.
"""
import numpy as np
import pandas as pd

from ctadata.universe import CRYPTO_SWAP


def crypto_carry(held_w: pd.DataFrame, index) -> pd.Series:
    """Daily financing on the crypto legs as a fraction of NAV, POSITIVE where
    the book pays. Calendar-day accrual, capped at five days so a long gap in
    the data cannot charge a month in one step.

    held_w: held weights, fraction of NAV (simulator.held_weights).
    """
    days = pd.Series(index, index=index).diff().dt.days.fillna(1).clip(upper=5)
    cost = pd.Series(0.0, index=index)
    for sym, (lng, srt) in CRYPTO_SWAP.items():
        if sym not in held_w.columns:
            continue
        w = held_w[sym].reindex(index).fillna(0.0)
        rate = pd.Series(np.where(w >= 0, lng, srt), index=index)
        cost -= w.abs() * rate * days / 365.0
    return cost
