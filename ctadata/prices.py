"""Price frames: the research universe, the wider universe, crypto, and the
second-month contracts.

All frames share ONE calendar, the first workbook's, and every later source
is LEFT joined onto it. An outer join was the original choice and it was a
bug: the second pull's markets keep different national holidays, so it
injected about 33 dates on which every original market was NaN. The engine
fills missing returns with zero, so those became fabricated flat trading
days, and some rebalance dates moved. A left join keeps the calendar fixed.

NaN is left wherever a market's own history does not reach. Frames are
never truncated to the latest common start date.
"""
import pandas as pd

from .cleaning import patch_known_data_issues
from .log import log
from .paths import data_dir
from .universe import (TICKER_INFO, WIDE_FRONT, WIDE_SECOND, WIDE_SECOND_EXTRA,
                       CRYPTO_FRONT, CRYPTO_START)
from .workbooks import load_workbook, collect

PULLS = ("CTA_Data_Pull.xlsx", "CTA_Data_Pull2.xlsx",
         "FV1_OE1_XM1_CN1_DataPull.xlsx", "C1_KC1_JO1_GX1_HI1_DataPull.xlsx")
WIDE_FILES = ("DataPull_Part1.xlsx", "DataPull_Part2.xlsx",
              "DataPull_Part3.xlsx", "TierA_DataPull.xlsx")
SECOND_FILES = ("DataPull_Part1.xlsx", "DataPull_Part2.xlsx",
                "A13_SecondMonthContracts.xlsx", "G2_GC2_EC2_AD2_DataPull.xlsx")
SECOND_EXTRA_FILE = "WideSecondMonths.xlsx"
CRYPTO_FILE = "Crypto_DataPull.xlsx"


def load_cta_data() -> pd.DataFrame:
    """The 30-market research frame, short names, UTC index, 2006 onward.

    Deliberately limited to the original four pulls. Other code defines its
    pool as "every tradeable column of this frame", so widening it here
    would silently change every pool-based result.
    """
    d = data_dir()
    data = load_workbook(d / PULLS[0])
    for extra in PULLS[1:]:
        p = d / extra
        if p.exists():
            data = data.join(load_workbook(p), how="left")
        else:
            log(f"missing (skipped): {extra}")
    data.columns = [TICKER_INFO[t] for t in data.columns]
    data = data.astype(float)
    return patch_known_data_issues(data)


def load_wide_front(include_base: bool = True) -> pd.DataFrame:
    """Front months for the wider universe, on the research calendar."""
    base = load_cta_data()
    wide = collect(WIDE_FRONT, [data_dir() / f for f in WIDE_FILES]).reindex(base.index)
    if not include_base:
        return wide
    overlap = [c for c in wide.columns if c in base.columns]
    if overlap:
        wide = wide.drop(columns=overlap)
    return base.join(wide, how="left")


def load_crypto(apply_start: bool = True) -> pd.DataFrame:
    """BTC and ETH spot on the research calendar.

    The broker trades crypto as rolling CFDs, so spot is the right series,
    not the CME future. Crypto trades seven days a week and the calendar is
    business days, so weekend moves land in Monday's return, which is what a
    weekday-rebalanced book actually experiences.
    """
    base = load_cta_data()
    c = collect(CRYPTO_FRONT, [data_dir() / CRYPTO_FILE]).reindex(base.index)
    if apply_start:
        for sym, start in CRYPTO_START.items():
            if start and sym in c.columns:
                c.loc[c.index < pd.Timestamp(start, tz=c.index.tz), sym] = float("nan")
    return c


def load_live_universe() -> pd.DataFrame:
    """The frame the live book is built from: research markets plus crypto."""
    return load_cta_data().join(load_crypto(), how="left")


def frame(include_crypto: bool = False) -> pd.DataFrame:
    """The wide research frame, optionally with crypto."""
    px = load_wide_front()
    if include_crypto:
        px = px.join(load_crypto(), how="left")
    return px


def load_seconds(include_extra: bool = True) -> pd.DataFrame:
    """Second-month generics, named like the fronts, on the same calendar.

    Covers 63 of 64 fronts. The Gilt has no usable second month on disk.
    """
    d = data_dir()
    base_idx = load_cta_data().index
    s = collect(WIDE_SECOND, [d / f for f in SECOND_FILES]).reindex(base_idx)
    p = d / SECOND_EXTRA_FILE
    if include_extra and p.exists():
        w = load_workbook(p)
        w.columns = [str(c).strip() for c in w.columns]
        keep = {t: n for t, n in WIDE_SECOND_EXTRA.items() if t in w.columns}
        extra = w[list(keep)].rename(columns=keep).astype(float).reindex(base_idx)
        extra = extra.drop(columns=[c for c in extra.columns if c in s.columns])
        s = s.join(extra, how="left")
    return s


def tradeable_pool(px: pd.DataFrame, include_crypto: bool = False,
                   min_obs: int = 2000) -> list:
    """Markets with real history that the broker does not reject.

    Excludes the VIX future (not a directional market in this book) and every
    untradeable fixed-income contract. 56 markets, or 58 with crypto.
    """
    from .universe import UNTRADEABLE, CRYPTO
    out = sorted(c for c in px.columns
                 if c != "vix" and c not in CRYPTO
                 and px[c].notna().sum() > min_obs and c not in UNTRADEABLE)
    return out + (sorted(c for c in CRYPTO if c in px.columns) if include_crypto else [])
