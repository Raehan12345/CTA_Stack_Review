"""Hierarchical Risk Parity, as a tilt WITHIN each asset class.

HRP (Lopez de Prado, 2016) clusters instruments by correlation and splits
risk down the resulting tree, so markets that move together share one
allocation instead of each taking a full one. Here it does not replace the
inverse-volatility sizing; it multiplies each instrument's volatility target
by a scale factor saying whether that instrument deserves more or less than
an equal share of its own class.

Measured on A-13, monthly, net: +0.108 Sharpe, better in 6 of 9 two-year
windows. Its benefit is concentrated in crisis windows, where correlations
spike: with 2008 outside the scored window it is roughly neutral.
"""
import numpy as np
import pandas as pd

# Memoises _hrp_scale for the life of the process. Keyed partly on
# id(prices), which is only valid because nothing mutates a price frame after
# passing it to the engine. A fresh process starts empty, so there is no
# stale-data risk across sessions.
_HRP_CACHE: dict = {}


def _hrp_scale(returns: pd.DataFrame, universe: dict, is_rebal: np.ndarray, hrp_window: int = 252,
               cache_key: tuple = None, lag: int = 0) -> pd.DataFrame:
    """Per-instrument scale factor on inst_target_vol, relative to an equal
    share of its own class: HRP weight times the number of instruments in the
    class. A class where HRP lands on equal weights is left exactly as it was
    (scale 1.0 for every member), so the layer composes with inverse-vol
    sizing rather than replacing it.

    Recomputed on each rebalance date from the trailing `hrp_window` days and
    carried forward between rebalances. Falls back to 1.0 where history is
    short, where a class has a single member (nothing to cluster), or where
    the window holds a gap or a flat column.

    lag=0 includes the rebalance day's own return in the window that sizes
    that day: a one-day peek, negligible in effect but impossible live.
    lag=1 ends the window the day before, and is what the live engine uses.
    """
    if cache_key is not None and cache_key in _HRP_CACHE:
        return _HRP_CACHE[cache_key]

    # Compatibility shim. pypfopt 1.6.0 validates its linkage method against
    # scipy.cluster.hierarchy._LINKAGE_METHODS, a private attribute newer
    # scipy no longer exposes. Without this every call raised, a broad except
    # swallowed it, and HRP silently returned equal weights: the first HRP
    # backtest was byte-identical to the baseline, which is how it was found.
    import scipy.cluster.hierarchy as _sch
    if not hasattr(_sch, "_LINKAGE_METHODS"):
        _sch._LINKAGE_METHODS = {"single", "complete", "average", "weighted",
                                 "centroid", "median", "ward"}
    from pypfopt.hierarchical_portfolio import HRPOpt

    symbols = [s for syms in universe.values() for s in syms]
    # NaN, not 1.0. The first version pre-filled with 1.0, which left ffill()
    # nothing to fill, so every non-rebalance day silently reverted to the
    # placeholder and HRP was live one day a month.
    scale = pd.DataFrame(np.nan, index=returns.index, columns=symbols)
    rebal_dates = returns.index[is_rebal]

    for cls, syms in universe.items():
        syms = [s for s in syms if s in returns.columns]
        if len(syms) < 2:
            scale[syms] = 1.0
            continue
        n = len(syms)
        bucket_returns = returns[syms]
        current = pd.Series(1.0, index=syms)
        for d in rebal_dates:
            window = bucket_returns.loc[:d].iloc[:len(bucket_returns.loc[:d]) - lag].iloc[-hrp_window:]
            long_enough = len(window) >= hrp_window // 2
            usable = long_enough and not window.isna().any().any() and (window.std() > 0).all()
            if usable:
                try:
                    w = HRPOpt(returns=window).optimize()
                    current = pd.Series({s: w.get(s, 1.0 / n) * n for s in syms})
                except Exception as e:
                    # Never silent: falling through keeps the PREVIOUS
                    # rebalance's split, and a book could run for months on a
                    # stale one with nothing to show for it.
                    print(f"[hrp] WARNING {cls} {d.date()}: optimise failed "
                          f"({type(e).__name__}: {e}), keeping the previous weights")
            scale.loc[d, syms] = current.values
        scale[syms] = scale[syms].ffill().fillna(1.0)

    if cache_key is not None:
        _HRP_CACHE[cache_key] = scale
    return scale
