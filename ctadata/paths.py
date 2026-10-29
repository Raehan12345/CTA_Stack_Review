"""Where the data lives.

The price data is licensed Bloomberg data and is not distributed with this
repository. Point the package at a local copy with environment variables:

    CTADATA_DIR        the Bloomberg workbooks, contract chains, CFTC files,
                       benchmark files and the MT5 calendar dump
    CTADATA_RATES_DIR  the long-format rate exports used for the risk-free
                       rate (defaults to CTADATA_DIR/rates)

Nothing is read at import time, so the package (and its tests, which use
synthetic data) imports cleanly on a machine that has no data at all.
"""
import os
from pathlib import Path


class DataNotConfigured(RuntimeError):
    pass


def data_dir() -> Path:
    raw = os.environ.get("CTADATA_DIR")
    if not raw:
        raise DataNotConfigured(
            "Set CTADATA_DIR to the folder holding the Bloomberg workbooks. "
            "The data is licensed and is not part of this repository.")
    p = Path(raw)
    if not p.is_dir():
        raise DataNotConfigured(f"CTADATA_DIR={raw} is not a directory")
    return p


def rates_dir() -> Path:
    raw = os.environ.get("CTADATA_RATES_DIR")
    return Path(raw) if raw else data_dir() / "rates"
