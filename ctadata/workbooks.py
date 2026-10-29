"""Reader for Bloomberg Excel exports.

Every price workbook comes from Bloomberg's Data History wizard and has the
same shape: row 1 holds the tickers, row 2 the BQL formula (ignored), and
row 3 onwards a date followed by one PX_LAST column per ticker.
"""
from pathlib import Path

import pandas as pd

from .log import log


def load_workbook(path: Path) -> pd.DataFrame:
    """One workbook as a date-indexed frame, columns named by raw ticker."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    tickers = raw.iloc[0, 1:].tolist()
    data = raw.iloc[2:, :].copy()
    data.columns = ["date"] + tickers
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.dropna(subset=["date"])        # some exports end with a blank row
    data = data.set_index("date").sort_index()
    log(f"{path.name}: {len(data)} rows x {len(tickers)} tickers, "
        f"{data.index.min().date()} to {data.index.max().date()}")
    return data


def collect(mapping: dict, paths: list) -> pd.DataFrame:
    """Read several workbooks, keep the tickers in `mapping`, rename them.

    A ticker that comes back entirely empty is dropped rather than kept as
    an all-NaN column: one discontinued contract returned zero rows, and an
    empty column would otherwise look like a market with no signal.
    """
    out = {}
    for p in paths:
        if not Path(p).exists():
            continue
        d = load_workbook(Path(p))
        d.columns = [str(c).strip() for c in d.columns]
        for tk, name in mapping.items():
            if tk in d.columns:
                s = d[tk].astype(float)
                if s.dropna().empty:
                    continue
                out[name] = s if name not in out else out[name].fillna(s)
    return pd.DataFrame(out).sort_index()
