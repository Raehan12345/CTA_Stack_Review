"""Optional sizing layers that were tested and rejected, kept so the tests
that rejected them can be re-run.

  VIX gate          shrink the portfolio target when the VIX spikes.
                    Monotonically harmful: the sleeve's crisis alpha IS the
                    volatility spike, so it cut risk into the best regime.
  COT throttle      shrink a position when speculators are crowded on the
                    same side. Within noise; CFTC data covers few markets.
  Economic Trend    scale size up or down by agreement between the price
                    trend and macro fundamentals (AQR's "Economic Trend").
                    Helped on a 24-market pool, hurt on A-13 (-0.032): with
                    no per-country data for commodities, all of them received
                    an identical signal.
"""
import numpy as np
import pandas as pd


def _vix_target_mult(vix: pd.Series, index: pd.DatetimeIndex, stress_window: int = 252,
                      stress_scale: float = 1.0) -> pd.Series:
    """Shrinks the portfolio-level vol target when VIX is elevated relative
    to its own trailing mean/std (a z-score) -- same escalation mechanic as
    a reactive volatility-targeting rule used elsewhere, applied here to the
    *target* instead of the realised-vol denominator.
    Lagged one day, same lookahead discipline as the barbell score.

    TESTED AND REJECTED as a default, kept as an opt-in experiment (build_trend_nav's
    vix=None by default reproduces the unmodified baseline exactly). Unlike
    an FX book -- where a VIX spike is a generic,
    largely orthogonal risk-off signal worth hedging against -- this sleeve's
    crisis alpha *is* the VIX spike: 2008 and 2020 were the strategy's best
    sub-periods specifically because they were high-volatility trending
    episodes. Shrinking the target when VIX rises cuts exposure into the
    strategy's best-performing regime rather than ahead of a separate risk.
    Tested at stress_scale in {0.5, 1.0, 2.0, 3.0}: stacked Sharpe degrades
    monotonically at every level (0.73 baseline -> 0.62 at scale=1.0 -> 0.57
    at scale=3.0), and 2008's sub-period Sharpe alone falls from 0.76 to
    0.35. Do not enable this without a materially different mechanism (e.g.
    conditioning on trend *direction* concordance with the vol spike, not
    just the spike's magnitude)."""
    vix_aligned = vix.reindex(index).ffill()
    vix_z = (vix_aligned - vix_aligned.rolling(stress_window).mean()) / vix_aligned.rolling(stress_window).std()
    mult = 1 / (1 + np.maximum(0, vix_z.fillna(0)) * stress_scale)
    return mult.shift(1).fillna(1.0)


def _cot_throttle_mult(cot: pd.DataFrame, score: pd.DataFrame, index: pd.DatetimeIndex,
                        symbols: list[str], cot_window: int = 156, cot_threshold: float = 1.5,
                        cot_shrink: float = 0.5, release_lag_days: int = 3) -> pd.DataFrame:
    """Per-instrument, per-day multiplier on inst_target_vol: cot_shrink
    (<1, a throttle-down) whenever non-commercial positioning is at a
    historical extreme (|z| > cot_threshold, z-scored against its own
    trailing cot_window weeks -- COT's native cadence, not reindexed to
    daily first) AND the sleeve's own current score is aligned with that
    crowded side (long when the crowd is net-long-extreme, short when
    net-short-extreme) -- a size-down on a trade that's already crowded in
    the sleeve's own direction, not a standalone reversal signal and not a
    throttle on a contrarian position. Exactly 1.0 (no effect) otherwise,
    including for any symbol absent from cot (eurobund -- see
    ctadata.macro.COT_CONTRACT_CODES; no CFTC data for a Eurex product).

    LOOKAHEAD: cot's index is the report's 'as of' date (see
    ctadata.macro.load_cot's own warning); release_lag_days shifts each
    week's z-score forward by CFTC's actual ~3-day as-of-to-publication
    delay before it's forward-filled onto the daily index, so a given
    day only ever sees COT data that had genuinely been published by
    then."""
    cot_z = (cot - cot.rolling(cot_window).mean()) / cot.rolling(cot_window).std()
    cot_z = cot_z.copy()
    cot_z.index = cot_z.index + pd.Timedelta(days=release_lag_days)
    cot_z_daily = cot_z.reindex(index, method="ffill")

    mult = pd.DataFrame(1.0, index=index, columns=symbols)
    for sym in symbols:
        if sym not in cot_z_daily.columns:
            continue
        z = cot_z_daily[sym]
        crowd_extreme = z.abs() > cot_threshold
        aligned = np.sign(score[sym]) == np.sign(z)
        trigger = (crowd_extreme & aligned & score[sym].notna() & z.notna())
        mult[sym] = np.where(trigger, cot_shrink, 1.0)
    return mult


# ---------------------------------------------------------------------------
# LAYER 2 -- Economic Trend filter, per AQR's "Economic Trend" (Brooks,
# Feilbogen, Ooi, Akant, April 2023). Five macro themes, each with a
# theoretically- and empirically-motivated directional sign per asset
# class (paper's Exhibit 1). Scores each instrument's *trend in
# fundamentals* (12-month change, not level) and uses agreement/
# disagreement with the barbell price-trend score (Layer 1) as a
# continuous conviction multiplier -- see _econ_trend_mult()'s own
# docstring for why this is continuous, not a binary gate.
#
# KNOWN LIMITATION: Growth, Inflation, and Risk Aversion are single US-only series
# applied as one global backdrop to every instrument, not the per-country
# signals the paper builds -- no GDP/CPI forecasts were sourced for UK/
# Japan/Australia. Monetary Policy is proxied from the US 2-year note
# future's own price (no non-US short-rate series sourced either), also
# applied economy-wide. Only International Trade is genuinely
# country-specific, and only for the three currencies actually pulled
# (aud/jpy/gbp) -- it's excluded entirely (not faked) for every other
# instrument, since there's no sensible country to map e.g. gold or the
# S&P 500 to for this theme with the data in hand.
# "crypto" is 0 on every theme, which means the Economic Trend layer
# ABSTAINS on it: alignment is 0, so `mult = 1 + strength * 0` leaves the
# instrument at exactly its unmodified target vol, driven by price trend
# alone while the rest of the book gets the overlay. That is deliberate.
# AQR's Economic Trend framework has no theory for a digital asset, and the
# signs that would be needed (does tightening help or hurt it, is it an
# inflation hedge) are contested rather than established. Assigning them
# would be inventing evidence. Added 2026-09-25 with the crypto work; no
# existing configuration has a "crypto" class, so no published number moves
# (_verify_nochange identical before and after).
ECON_SIGN_TABLE = {
    "growth": {"equities": 1, "fixed_income": -1, "currencies": 1, "commodities": 1, "crypto": 0},
    "inflation": {"equities": -1, "fixed_income": -1, "currencies": 1, "commodities": 1, "crypto": 0},
    "trade": {"equities": 1, "fixed_income": -1, "currencies": -1, "commodities": -1, "crypto": 0},
    "monetary_policy": {"equities": -1, "fixed_income": -1, "currencies": 1, "commodities": -1, "crypto": 0},
    "risk_aversion": {"equities": 1, "fixed_income": -1, "currencies": 1, "commodities": 1, "crypto": 0},
}

# International Trade theme's input column, by instrument -- only the
# three currencies this was actually sourced for (ctadata.macro.EER_TICKER_INFO).
_TRADE_EER_COLUMN = {"aud": "aud_eer_nominal", "jpy": "jpy_eer_nominal", "gbp": "gbp_eer_nominal"}

# Growth/Inflation/Monetary-Policy "home country" per instrument -- every
# instrument that genuinely has one gets its own country's series;
# commodities (no sensible single home country -- see
# the paper's consumption-weighted treatment)
# fall back to "us" as a documented simplification, same as before this
# per-country upgrade.
_SYMBOL_COUNTRY = {
    "us10y": "us", "us2y": "us", "spx": "us", "nasdaq": "us",
    "gilt": "uk", "gbp": "uk",
    "jgb": "japan", "jpy": "japan", "nikkei": "japan",
    "aud": "australia",
    "gold": "us", "wti": "us", "copper": "us", "brent": "us", "silver": "us",
}

# GDP/CPI column names per country, in econ (US) / econ_intl (UK/Japan/
# Australia) -- lookback_periods is 4 for quarterly series, 12 for
# monthly, so the "12-month change" is computed correctly regardless of
# how often each country's statistics office actually publishes (US/UK/
# Japan CPI are monthly; Australia's is quarterly, matching the ABS's
# real-world release cadence).
_COUNTRY_GDP_COL = {"us": "gdp_qoq", "uk": "uk_gdp_qoq", "japan": "japan_gdp_qoq", "australia": "australia_gdp_qoq"}
_COUNTRY_CPI_COL = {"us": "cpi_yoy", "uk": "uk_cpi_yoy", "japan": "japan_cpi_yoy", "australia": "australia_cpi_yoy"}
_COUNTRY_CPI_LOOKBACK = {"us": 12, "uk": 12, "japan": 12, "australia": 4}


def _theme_growth(gdp: pd.Series, index: pd.DatetimeIndex, lag_days: int = 45) -> pd.Series:
    """12-month (4-quarter) change in a country's GDP QoQ annualized
    growth, forward-filled to daily with a ~45-day publication lag (a
    first GDP estimate is typically released roughly a month after
    quarter-end; 45 days is a deliberately conservative buffer against
    lookahead)."""
    gdp = gdp.dropna()
    chg = gdp - gdp.shift(4)
    chg = chg.copy()
    chg.index = chg.index + pd.Timedelta(days=lag_days)
    return chg.reindex(index, method="ffill")


def _theme_inflation(cpi: pd.Series, index: pd.DatetimeIndex, lookback_periods: int = 12,
                      lag_days: int = 30) -> pd.Series:
    """12-month change in a country's CPI YoY inflation, forward-filled
    with a ~30-day publication lag. lookback_periods matches that
    country's own release cadence (12 for monthly CPI, 4 for quarterly)
    so "12-month change" is computed over the right number of readings."""
    cpi = cpi.dropna()
    chg = cpi - cpi.shift(lookback_periods)
    chg = chg.copy()
    chg.index = chg.index + pd.Timedelta(days=lag_days)
    return chg.reindex(index, method="ffill")


def _theme_risk_aversion(econ: pd.DataFrame, spx: pd.Series, index: pd.DatetimeIndex,
                          z_window: int = 756) -> pd.Series:
    """Equal-weighted blend of 12-month trailing S&P 500 return and the
    negative of the 12-month change in US HY corporate OAS (narrowing
    spread = rising risk sentiment) -- each z-scored over its own
    trailing history before combining, so a percent-scale return and a
    points-scale spread change contribute on equal footing rather than
    whichever happens to have larger raw units."""
    eq_ret = spx.reindex(index).ffill().pct_change(252)
    oas = econ["hy_oas"].reindex(index).ffill()
    oas_chg = -(oas.diff(252))
    eq_z = (eq_ret - eq_ret.rolling(z_window).mean()) / eq_ret.rolling(z_window).std()
    oas_z = (oas_chg - oas_chg.rolling(z_window).mean()) / oas_chg.rolling(z_window).std()
    return 0.5 * eq_z.fillna(0.0) + 0.5 * oas_z.fillna(0.0)


def _theme_monetary_policy(us2y_price: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """12-month change in the (implied) US 2-year yield, proxied from
    the US 2Y note future's own price: a falling price implies a rising
    yield, so the signal is the *negative* of the future's 12-month
    price return. US-only, applied economy-wide -- see module note."""
    ret_12m = us2y_price.reindex(index).ffill().pct_change(252)
    return -ret_12m


def _theme_trade(eer: pd.DataFrame, ccy_col: str, index: pd.DatetimeIndex) -> pd.Series:
    """12-month change in a currency's own BIS nominal effective
    (trade-weighted) exchange rate, sign-flipped so a *depreciating*
    currency (falling EER) produces a *positive* signal -- matches the
    paper's "export-weighted FX depreciation" framing directly."""
    eer_s = eer[ccy_col].reindex(index).ffill()
    return -(eer_s.pct_change(252))


def economic_trend_score(prices: pd.DataFrame, symbols: list[str], universe: dict, econ: pd.DataFrame,
                          eer: pd.DataFrame, econ_intl: pd.DataFrame = None, z_window: int = 756) -> pd.DataFrame:
    """Composite Economic Trend score, one column per instrument, each
    theme z-scored before combining then averaged only over the themes
    actually available for that instrument's asset class (4 for
    everything except aud/jpy/gbp, which also get Trade). Positive =
    fundamentals argue bullish for that instrument; negative = bearish.
    Lagged one additional day on top of each theme's own publication lag
    -- the same generic one-bar lookahead discipline as the barbell
    score, layered on top of (not instead of) each series' own realistic
    publication delay.

    econ_intl: optional UK/Japan/Australia GDP+CPI (see
    ctadata.macro.load_econ_intl()). When given,
    Growth and Inflation are computed per-country (_SYMBOL_COUNTRY) for
    every instrument that has one; when None, every instrument falls back
    to the US-only series (this function's original behavior)."""
    sym_to_class = {s: cls for cls, syms in universe.items() for s in syms}
    index = prices.index

    country_data = {"us": econ}
    if econ_intl is not None:
        country_data["uk"] = econ_intl
        country_data["japan"] = econ_intl
        country_data["australia"] = econ_intl

    growth_z_by_country, inflation_z_by_country = {}, {}
    for country, src in country_data.items():
        gdp_col, cpi_col = _COUNTRY_GDP_COL[country], _COUNTRY_CPI_COL[country]
        if gdp_col not in src.columns or cpi_col not in src.columns:
            continue
        growth = _theme_growth(src[gdp_col], index)
        growth_z_by_country[country] = (growth - growth.rolling(z_window).mean()) / growth.rolling(z_window).std()
        inflation = _theme_inflation(src[cpi_col], index, _COUNTRY_CPI_LOOKBACK[country])
        inflation_z_by_country[country] = (
            (inflation - inflation.rolling(z_window).mean()) / inflation.rolling(z_window).std())

    risk_aversion_z = _theme_risk_aversion(econ, prices["spx"], index, z_window)

    monpol_z = None
    if "us2y" in prices.columns:
        monpol = _theme_monetary_policy(prices["us2y"], index)
        monpol_z = (monpol - monpol.rolling(z_window).mean()) / monpol.rolling(z_window).std()

    scores = pd.DataFrame(index=index, columns=symbols, dtype=float)
    for sym in symbols:
        cls = sym_to_class.get(sym)
        if cls is None:
            scores[sym] = 0.0
            continue
        country = _SYMBOL_COUNTRY.get(sym, "us")
        if country not in growth_z_by_country:
            country = "us"  # fall back if that country's data wasn't loaded
        components = [
            ECON_SIGN_TABLE["growth"][cls] * growth_z_by_country[country],
            ECON_SIGN_TABLE["inflation"][cls] * inflation_z_by_country[country],
            ECON_SIGN_TABLE["risk_aversion"][cls] * risk_aversion_z,
        ]
        if monpol_z is not None:
            components.append(ECON_SIGN_TABLE["monetary_policy"][cls] * monpol_z)
        if sym in _TRADE_EER_COLUMN:
            trade = _theme_trade(eer, _TRADE_EER_COLUMN[sym], index)
            trade_z = (trade - trade.rolling(z_window).mean()) / trade.rolling(z_window).std()
            components.append(ECON_SIGN_TABLE["trade"][cls] * trade_z)
        scores[sym] = pd.concat(components, axis=1).mean(axis=1, skipna=True)
    return scores.shift(1)


def _econ_trend_mult(econ_score: pd.DataFrame, price_score: pd.DataFrame, strength: float = 0.5,
                      floor: float = 0.5, ceiling: float = 1.5) -> pd.DataFrame:
    """Conviction multiplier on inst_target_vol: scales UP when the
    Economic Trend composite agrees in sign with the barbell price-trend
    score (fundamentals support the trend), scales DOWN when they
    disagree -- continuous and bounded [floor, ceiling], deliberately
    never a full flatten. The Efficiency-Ratio regime-gate experiment
    showed a binary trend-or-flat gate can
    concentrate the book into very few instruments at exactly the wrong
    moment and blow up leverage sizing; this stays continuous specifically
    to avoid that failure mode. `alignment` is clipped to [-1, 1] before
    scaling so one extreme econ_score reading can't blow past the bounds
    on its own."""
    alignment = (np.sign(price_score) * econ_score).clip(-1, 1)
    mult = 1 + strength * alignment
    return mult.clip(lower=floor, upper=ceiling).fillna(1.0)
