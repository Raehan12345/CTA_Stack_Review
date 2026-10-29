"""The trend signal, and two per-market statistics built from prices.

Every function here is lagged one bar: the value used on day t is computed
from information available before day t. That single shift is the whole
look-ahead discipline of the signal path.
"""
import numpy as np
import pandas as pd


def _tsmom_score(prices: pd.DataFrame, short_lookback: int, long_lookback: int,
                 mode: str = "sign", vol_window: int = 60, clip: float = 2.0) -> pd.DataFrame:
    """Barbell time-series momentum: an equal blend of a short and a long
    lookback, each market judged only against its own history.

    mode="sign" is the strategy. Each leg is sign(return), so the score is
    +1 (both up), -1 (both down) or 0 (they disagree). Direction only: size
    comes entirely from volatility scaling and the regime gate.

    The other modes were tested against the chosen configuration and not
    adopted. Each expresses a leg in volatility units,
    z = return / (daily_vol * sqrt(lookback)):

      "continuous"  z clipped to [-clip, clip], divided by clip: a strong
                    trend gets a full position, a weak one a fraction.
                    +0.009 chained Sharpe: inside noise.
      "response"    z * exp(-z^2 / 4) normalised to peak at 1 (Baz et al.
                    2015): fades extended trends.
      "dominant"    sign mode, except a disagreement holds half size in the
                    direction of the leg that is larger in vol units, instead
                    of going flat. Worse: disagreement usually marks a
                    turning point, and going flat there is the useful part.
    """
    short_ret = prices.pct_change(short_lookback)
    long_ret = prices.pct_change(long_lookback)
    if mode == "sign":
        score = 0.5 * np.sign(short_ret) + 0.5 * np.sign(long_ret)
    elif mode == "dominant":
        daily_vol = prices.pct_change().fillna(0.0).rolling(vol_window).std()
        zs = short_ret / (daily_vol * np.sqrt(short_lookback))
        zl = long_ret / (daily_vol * np.sqrt(long_lookback))
        agree = np.sign(zs) == np.sign(zl)
        score = (0.5 * np.sign(zs) + 0.5 * np.sign(zl)).where(
            agree, 0.5 * np.sign(zs.where(zs.abs() >= zl.abs(), zl)))
        score = score.where(short_ret.notna() & long_ret.notna())
    elif mode in ("continuous", "response"):
        # fillna(0), as the engine's own volatility does. Without it a single
        # holiday gap anywhere in the 60-day window blanks the estimate; the
        # first run of this mode produced a near-empty book for that reason.
        daily_vol = prices.pct_change().fillna(0.0).rolling(vol_window).std()
        zs = short_ret / (daily_vol * np.sqrt(short_lookback))
        zl = long_ret / (daily_vol * np.sqrt(long_lookback))
        if mode == "continuous":
            f = lambda z: z.clip(-clip, clip) / clip
        else:
            peak = np.sqrt(2.0) * np.exp(-0.5)          # max of z*exp(-z^2/4), at z = sqrt(2)
            f = lambda z: z * np.exp(-z ** 2 / 4.0) / peak
        score = 0.5 * f(zs) + 0.5 * f(zl)
    else:
        raise ValueError(f"score_mode must be 'sign', 'dominant', 'continuous' or 'response', got {mode!r}")
    return score.shift(1)


def _efficiency_ratio(prices: pd.DataFrame, window: int = 40) -> pd.DataFrame:
    """Kaufman's Efficiency Ratio per market: |net move over the window|
    divided by the total distance travelled. Near 1 is a clean trend, near 0
    is chop. It is a PATH statistic: it depends on the order the returns
    arrived in, which is why a distribution-based method cannot see it.

    The window is 40, the barbell's fast leg: classifying on a much shorter
    window than the signal it judges lets the label whipsaw faster than the
    trend itself.
    """
    net_change = (prices - prices.shift(window)).abs()
    total_change = prices.diff().abs().rolling(window).sum()
    er = (net_change / total_change).fillna(0.0)
    return er.shift(1)


def _atr_proxy(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Close-only Average True Range: the rolling mean of the absolute daily
    price change, in price units. True ATR needs high and low prices and the
    research data is closing prices only; this answers the same question
    ("how far does this market usually move in a day") without the intraday
    range. 14 is Wilder's original window."""
    return prices.diff().abs().rolling(window).mean().shift(1)
