"""Execution simulator, v1: the research book re-traded on the broker's instruments.

Takes the research engine's HELD weights (fraction of NAV per instrument,
already leverage-scaled) and re-trades them with:

  - a one-day execution lag: the research engine sizes at close t-1 and
    earns t; live, the signal is computed after close t-1 and the trade
    fills at close t (close-only data, so "next close" stands in for "next
    open")
  - lot rounding at the given capital, to the broker's volume_min / volume_step
  - spread (half per side) and slippage on every trade
  - nightly financing on CFD notional from the swap table in specs.py,
    accrued on calendar days
  - a round-trip spread at each quarterly roll of the dated bond contracts
  - no stop rail yet (v1)

P&L per instrument = lots x notional-per-lot(USD, previous close) x the
research return. That keeps the P&L on the same price series the research
used, so the only differences between the two books are execution effects.
"""
import numpy as np, pandas as pd
from .specs import SPECS, ROLL_MONTHS, ROLL_DAYS_BEFORE_MONTH_END


# The Bloomberg generics are quoted scaled: AD1 and BP1 are the rate x100,
# JY1 is USD-per-JPY x10000. Divide to get a real exchange rate.
_FX_SCALE = {"aud": 100.0, "gbp": 100.0, "jpy": 10000.0}


def _usd_per_unit(prices: pd.DataFrame, ccy: str):
    """USD value of one unit of `ccy`, as a series or a scalar."""
    if ccy == "USD":
        return 1.0
    if ccy == "USc":
        return 0.01
    src = {"JPY": "jpy", "GBP": "gbp", "AUD": "aud"}.get(ccy)
    if src is None:
        raise ValueError(f"no USD conversion for quote currency {ccy!r}")
    return prices[src] / _FX_SCALE[src]


def lot_value_usd(prices: pd.DataFrame) -> pd.DataFrame:
    """USD notional of one lot per instrument, per day.

    Driven by SPECS rather than a hardcoded list. The previous version named
    every instrument explicitly and went stale the moment the universe
    changed: it raised KeyError on `gilt` after the Gilt was dropped, and had
    no entry for crypto. Worse, bypassing it is how a lot-size analysis came
    out 158x wrong, by taking price x contract for JPY-quoted instruments
    without converting to USD. Anything that needs a lot value should call
    this, not recompute it.

    Two cases:
      FX pairs   one lot is `contract` units of the BASE currency, so there
                 is no price term. SPECS already encodes the base in `ccy`
                 (jpy's ccy is USD, because a USDJPY lot is 100k USD).
      everything else   contract x price x the quote currency in USD.
    """
    lv = pd.DataFrame(index=prices.index)
    for sym, row in SPECS.iterrows():
        fx = _usd_per_unit(prices, row.ccy)
        if row.cls == "fx":
            lv[sym] = row.contract * fx
        else:
            if sym not in prices.columns:
                continue                     # not in this frame; skip quietly
            lv[sym] = row.contract * prices[sym] * fx
    return lv


def financing_rate(rates: pd.DataFrame, index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annual financing earned per unit notional, long and short, per day.
    Positive = the position earns it. Reference rates from ctadata.rates."""
    r = rates.reindex(index, method="ffill").ffill().bfill()
    fl, fs = {}, {}
    for s, row in SPECS.iterrows():
        if row.kind == "dated":
            fl[s] = 0.0 * r["USD"]; fs[s] = 0.0 * r["USD"]
        elif row.cls == "fx":
            base, quote = {"aud": ("AUD", "USD"), "gbp": ("GBP", "USD"), "jpy": ("JPY", "USD")}[s]
            diff = r[base] - r[quote]           # research long earns the base rate
            fl[s] = diff - row.fin_long / 100; fs[s] = -diff - row.fin_short / 100
        else:
            ref = r[{"nikkei": "JPY"}.get(s, "USD")]
            fl[s] = -(ref + row.fin_long / 100); fs[s] = ref - row.fin_short / 100
    return pd.DataFrame(fl, index=index), pd.DataFrame(fs, index=index)


def roll_dates(index) -> pd.DatetimeIndex:
    out = []
    for (y, m), days in pd.Series(index, index=index).groupby([index.year, index.month]):
        if m in ROLL_MONTHS and len(days) > ROLL_DAYS_BEFORE_MONTH_END:
            out.append(days.iloc[-ROLL_DAYS_BEFORE_MONTH_END])
    return pd.DatetimeIndex(out)


def simulate(prices: pd.DataFrame, returns: pd.DataFrame, held_w: pd.DataFrame, is_rebal: np.ndarray,
             rates: pd.DataFrame, capital: float = 100_000.0, lag: int = 1,
             slippage_bps: float = 1.0, round_lots: bool = True, financing: bool = True,
             spreads: bool = True, financed: list = None, start=None) -> dict:
    """held_w: research held weight (fraction of NAV, leverage-scaled) per
    instrument per day. Trades happen on rebalance days, filled `lag` days
    later at that day's close."""
    syms = list(SPECS.index)
    idx = prices.index
    lv = lot_value_usd(prices)[syms].ffill()     # holiday gaps: carry the last close
    fl, fs = financing_rate(rates, idx)
    # `financed`: which symbols are held as CFDs and pay the swap. Default is
    # every CFD in specs; pass e.g. ["aud","gbp","jpy"] to model the other
    # nine on dated contracts (no financing, quarterly roll cost instead).
    if financed is None:
        financed = [s for s in syms if SPECS.loc[s, "kind"] == "cfd"]
    mask_fin = np.array([s in financed for s in syms], dtype=float)
    fl, fs = fl * mask_fin, fs * mask_fin
    rolls = set(roll_dates(idx))
    dated = [s for s in syms if s not in financed]
    half_spread = (SPECS["spread_bps"] / 2e4 if spreads else 0.0 * SPECS["spread_bps"]) + slippage_bps / 1e4
    vmin, vstep = SPECS["vmin"], SPECS["vstep"]
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(1).values

    n = len(idx)
    i0 = idx.get_indexer([pd.Timestamp(start, tz=idx.tz)], method="bfill")[0] if start is not None else 1
    nav = np.full(n, np.nan); nav[i0 - 1] = capital
    lots = pd.Series(0.0, index=syms)
    pnl = np.zeros((n, len(syms))); c_spread = np.zeros(n); c_fin = np.zeros(n); c_roll = np.zeros(n)
    trades = 0
    pending = None                       # (fill day, target weights) waiting for the lag
    for i in range(i0, n):
        t = idx[i]
        nav_prev = nav[i - 1]
        # 1. P&L on the book held into today
        r = returns.iloc[i][syms].fillna(0.0).values
        # An instrument with no history yet (crypto before 2015/2018) has a
        # NaN lot value. numpy makes 0 * NaN = NaN, so a flat position in a
        # not-yet-listed market poisoned the whole NAV from day one. Treat a
        # NaN lot value as "cannot hold this yet": no P&L, no financing, and
        # no position taken below.
        lv_prev = np.nan_to_num(lv.iloc[i - 1].values, nan=0.0)
        lv_now = lv.iloc[i].values
        tradeable = np.isfinite(lv_now) & (lv_now > 0)
        gross = lots.values * lv_prev * r
        pnl[i] = gross
        # 2. financing on the notional held overnight (calendar days)
        if financing:
            notional = lots.values * lv_prev
            rate = np.where(notional >= 0, fl.iloc[i - 1].values, fs.iloc[i - 1].values)
            c_fin[i] = float(np.sum(np.abs(notional) * rate * days[i] / 365.0))
        # 3. rolls on dated contracts
        if t in rolls:
            for s in dated:
                j = syms.index(s)
                if np.isfinite(lv.iloc[i, j]):
                    c_roll[i] += abs(lots[s]) * lv.iloc[i, j] * 2 * half_spread[s]
        # 4. fills. The research engine's weights for rebalance day k earn
        # day k, i.e. it is filled at close k-1. With lag L we fill at close
        # k-1+L and earn from k+L. lag=0 reproduces the research timing.
        k = i + 1 - lag
        if 0 <= k < n and is_rebal[k]:
            w = held_w.iloc[k][syms].fillna(0.0)
            nav_now = nav_prev + gross.sum() + c_fin[i] - c_roll[i]
            target_notional = np.where(tradeable, w.values * nav_now, 0.0)
            new_lots = np.divide(target_notional, lv_now,
                                 out=np.zeros_like(target_notional),
                                 where=tradeable)
            if round_lots:
                # shortcut: rounds to NEAREST while the live EA FLOORS
                # (CTA_NormalizeLots), so a target under one minimum lot is held
                # here and skipped live; backtests are slightly optimistic. Kept
                # deliberately: fixing it restates published net figures, and a
                # data backfill will force a full restatement anyway.
                new_lots = np.round(new_lots / vstep.values) * vstep.values
                new_lots = np.where(np.abs(new_lots) < vmin.values, 0.0, new_lots)
            delta = new_lots - lots.values
            c_spread[i] = float(np.sum(np.abs(delta) * np.nan_to_num(lv_now, nan=0.0)
                                       * half_spread.values))
            trades += int((delta != 0).sum())
            lots = pd.Series(new_lots, index=syms)
        nav[i] = nav_prev + gross.sum() + c_fin[i] - c_roll[i] - c_spread[i]
    nav = pd.Series(nav, index=idx, name="nav")
    return dict(nav=nav, ret=nav.pct_change(),
                pnl=pd.DataFrame(pnl, index=idx, columns=syms),
                cost_spread=pd.Series(c_spread, index=idx), cost_fin=pd.Series(c_fin, index=idx),
                cost_roll=pd.Series(c_roll, index=idx), trades=trades, lots_final=lots)


def held_weights(build_out, prices: pd.DataFrame, rebalance_freq: str = "M"):
    """Reconstruct the research engine's HELD weights from its outputs:
    target weights held from each calendar rebalance, times the leverage
    series it returns (which is the held leverage). Verified against the
    gross NAV in the runner."""
    nav, lev, inst_w = build_out[0], build_out[1], build_out[2]
    period = prices.index.tz_localize(None).to_period(rebalance_freq).values
    is_rebal = np.empty(len(prices), dtype=bool)
    is_rebal[1:] = period[1:] != period[:-1]; is_rebal[0] = False
    w_at = inst_w.where(pd.Series(is_rebal, index=prices.index), np.nan).ffill().fillna(0.0)
    return w_at.mul(lev, axis=0), is_rebal

