"""Performance statistics, and the one helper every comparison goes through.

One copy of each. The research code once had 37 separate definitions of a
Sharpe function, which is how two of them came to disagree on whether the
risk-free rate had been subtracted.
"""
import numpy as np
import pandas as pd

ANN = 252


def sharpe(r: pd.Series, rf: pd.Series = None) -> float:
    """Annualised Sharpe. With `rf` (a daily risk-free return series) it is
    the excess-return Sharpe; without it, the 0% convention. Every headline
    in this project is reported after the risk-free rate, which costs the
    sleeve 0.11 to 0.15 and changes no ranking."""
    x = r.dropna()
    if rf is not None:
        x = (x - rf.reindex(x.index)).dropna()
    if len(x) < 20 or not x.std():
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(ANN))


def vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(ANN))


def max_drawdown(r: pd.Series) -> float:
    nav = (1 + r.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def cagr(r: pd.Series) -> float:
    """Compound annual growth on CALENDAR years. An earlier convention,
    252 / number of rows, overstates the year count on a calendar with
    holidays removed."""
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float((1 + r).prod() ** (1 / years) - 1) if years > 0 else np.nan


def stats(r: pd.Series, rf: pd.Series = None) -> dict:
    """The four numbers every table reports. Realised volatility is always
    included so that matched-volatility comparisons can be checked rather
    than assumed."""
    return dict(sharpe=sharpe(r, rf), vol=vol(r), cagr=cagr(r), max_dd=max_drawdown(r))


def matched(r: pd.Series, target_vol: float) -> pd.Series:
    """Rescale a return series to a given realised volatility.

    Scaling a strategy does not change its return per unit of risk, so two
    books running at different volatility cannot be compared directly. This
    produced a wrong verdict on the regime gate that survived a full day: at
    a common target the gate ran the book smaller and looked like it cost
    return; at matched risk it added return and cut drawdown.
    """
    v = vol(r)
    return r * (target_vol / v) if v else r


def table(series: dict, rf: pd.Series = None, match_to: str = None) -> pd.DataFrame:
    """One row per series. With `match_to`, also report CAGR and drawdown
    with every series scaled to that series' realised volatility."""
    rows = {}
    base_vol = vol(series[match_to]) if match_to else None
    for name, r in series.items():
        row = stats(r, rf)
        if base_vol:
            m = matched(r, base_vol)
            row.update(cagr_vm=cagr(m), max_dd_vm=max_drawdown(m))
        rows[name] = row
    return pd.DataFrame(rows).T


def returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().fillna(0.0)
