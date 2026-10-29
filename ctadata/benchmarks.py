"""External benchmarks. None of these feed the strategy; they exist to
compare it against something real."""
import pandas as pd

from .cleaning import fix_spike_and_revert
from .paths import data_dir

SG_TREND_FILE = "SG_CTA_Trend_Indicator_Data.xlsx"
MBXAX_FILE = "DataPull_AUCPIYOY_MBXAXBenchmark.xlsx"
NH_FILE = "NHIndexDaily.csv"
RSST_FILE = "RSST_ETF.xlsx"


def load_sg_trend_index() -> pd.Series:
    """SG Trend Indicator, cleaned of its recurring January spike-and-revert
    artefacts (about 39 days of over 7% moves, each fully reversed within a
    day or two)."""
    raw = pd.read_excel(data_dir() / SG_TREND_FILE, sheet_name=0, header=None)
    data = raw.iloc[2:, [0, 1]].copy()
    data.columns = ["date", "neixtrnd"]
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    return fix_spike_and_revert(data["neixtrnd"])


def load_mbxax() -> pd.Series:
    """An investable managed-futures mutual fund's price level, from 2015."""
    raw = pd.read_excel(data_dir() / MBXAX_FILE, sheet_name="Sheet1")
    raw.columns = ["date", "mbxax", "aucpiyoy"]
    raw = raw.iloc[1:].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["mbxax"] = pd.to_numeric(raw["mbxax"], errors="coerce")
    return raw.dropna(subset=["mbxax"]).set_index("date")["mbxax"].sort_index()


def load_nilssonhedge_cta() -> pd.Series:
    """A composite of reported CTA fund returns, from 2019. Already returns."""
    raw = pd.read_csv(data_dir() / NH_FILE)
    cta = raw[raw["type"] == "CTA"].copy()
    cta["date"] = pd.to_datetime(cta["date"])
    return cta.sort_values("date").set_index("date")["ror"]


def load_rsst() -> pd.Series:
    """The return-stacked ETF this project set out to replicate, price only."""
    raw = pd.read_excel(data_dir() / RSST_FILE, sheet_name=0, header=None)
    data = raw.iloc[2:, [0, 1]].copy()
    data.columns = ["date", "rsst"]
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    data["rsst"] = pd.to_numeric(data["rsst"], errors="coerce")
    return data.dropna().set_index("date")["rsst"].sort_index()
