"""The risk-free rate, and the VIX.

Every Sharpe ratio in this project is reported after subtracting the real US
short rate, not the 0% convention many backtests use. The rate comes from
two sources:

  * Bloomberg money-market rates (long-format CSV exports), spliced from a
    discontinued panel rate into its replacement where one exists, in
    mapping-file order.
  * Central-bank rate decisions from the MT5 economic calendar, overlaid
    where they exist. A decision is a step function, not a continuous
    overnight rate, but the two track within a few basis points.
"""
import re

import pandas as pd

from .paths import data_dir, rates_dir

RATES_FILE = "rates_bbg.csv"
EXTRA_RATES_FILES = ("CADrates_bbg.csv", "soraindexrates_bbg.csv")
MAPPING_FILE = "ticker_mappings.csv"
STRESS_FILE = "stress_bbg.csv"
CALENDAR_FILE = "calendar_dump.csv"

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CHF", "NZD", "CAD",
              "SEK", "NOK", "MXN", "BRL", "ZAR", "SGD"]
COUNTRY_TO_CCY = {"US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY", "AU": "AUD",
                  "CH": "CHF", "NZ": "NZD", "CA": "CAD", "SE": "SEK", "NO": "NOK",
                  "MX": "MXN", "BR": "BRL", "ZA": "ZAR"}
RATE_DECISION = re.compile(r"interest rate decision", re.IGNORECASE)


def _read_long(paths) -> pd.DataFrame:
    """Concatenate long-format exports (ticker, date, field, value), dropping
    the duplicates a later export creates when it re-sends tickers."""
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["ticker", "date", "field"])


def load_rates_bloomberg() -> pd.DataFrame:
    """Wide per-currency short rates, decimal (0.0533, not 5.33)."""
    d = rates_dir()
    mapping = pd.read_csv(d / MAPPING_FILE)
    t2c = dict(zip(mapping["RATE_ticker"], mapping["RATE_currency"]))
    # USD_FED, GBP_SONIA and the like collapse to the plain currency code
    t2c = {t: (c.split("_")[0] if c.split("_")[0] in CURRENCIES else c) for t, c in t2c.items()}
    raw = _read_long([d / RATES_FILE] + [d / f for f in EXTRA_RATES_FILES])
    raw = raw[raw["field"] == "PX_LAST"]
    wide = raw.pivot_table(index="date", columns="ticker", values="value")
    out = pd.DataFrame(index=wide.index)
    for ccy in CURRENCIES:
        col = None
        # earlier tickers in the mapping fill first, later ones only fill gaps
        for t in [t for t, c in t2c.items() if c == ccy and t in wide.columns]:
            col = wide[t] if col is None else col.combine_first(wide[t])
        if col is not None:
            out[ccy] = col
    out = out.sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out[[c for c in CURRENCIES if c in out.columns]] / 100.0


def load_calendar_policy_rates() -> pd.DataFrame:
    """Policy-rate decisions from the MT5 calendar dump, forward-filled
    between meetings, decimal."""
    df = pd.read_csv(data_dir() / CALENDAR_FILE)
    df = df[df["event_name"].str.contains(RATE_DECISION.pattern, case=False,
                                          regex=True, na=False)]
    df = df[df["actual"].notna()]
    df["ccy"] = df["country"].map(COUNTRY_TO_CCY)
    df = df[df["ccy"].notna()]
    df["release_time"] = pd.to_datetime(df["release_time"], format="%Y.%m.%d %H:%M", utc=True)
    series = {}
    for ccy, grp in df.groupby("ccy"):
        s = grp.sort_values("release_time").set_index("release_time")["actual"]
        series[ccy] = s[~s.index.duplicated(keep="last")]
    if not series:
        return pd.DataFrame()
    idx = pd.date_range(min(s.index.min() for s in series.values()),
                        max(s.index.max() for s in series.values()), freq="D", tz="UTC")
    out = pd.DataFrame({c: s.reindex(idx, method="ffill") for c, s in series.items()}, index=idx)
    return (out / 100.0).dropna(how="all")


def load_rates() -> pd.DataFrame:
    """Bloomberg rates with the calendar decisions taking priority."""
    bbg = load_rates_bloomberg()
    try:
        cal = load_calendar_policy_rates()
    except FileNotFoundError:
        return bbg
    if cal.empty:
        return bbg
    out = cal.combine_first(bbg).reindex(columns=CURRENCIES).sort_index()
    return out.ffill()


def rf_daily(index: pd.DatetimeIndex, ccy: str = "USD") -> pd.Series:
    """The daily risk-free return on a trading calendar, from the annual rate."""
    annual = load_rates()[ccy].reindex(index, method="ffill").ffill().bfill()
    return (1 + annual) ** (1 / 252) - 1


def load_vix() -> pd.Series:
    raw = _read_long([rates_dir() / STRESS_FILE])
    raw = raw[raw["field"] == "PX_LAST"]
    wide = raw.pivot_table(index="date", columns="ticker", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide["VIX Index"]
