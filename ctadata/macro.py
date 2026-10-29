"""Non-price data: macro series for the Economic Trend layer, CFTC
positioning for the COT throttle.

Both layers were tested and rejected; the loaders stay so the tests that
rejected them can be re-run.
"""
import pandas as pd

from .log import log
from .paths import data_dir

ECON_FILE = "Economic_Trend_Data_Pull.xlsx"
EER_FILE = "Currencies_Data_Pull.xlsx"
INTL_FILE = "GDP_CPI_UKAUDJPY.xlsx"

ECON_TICKER_INFO = {
    "GDP CQOQ Index": "gdp_qoq",
    "CPI YOY Index": "cpi_yoy",
    "CPI CHNG Index": "cpi_mom",
    "LF98OAS Index": "hy_oas",
}
GDP_CPI_INTL_TICKER_INFO = {
    "UKGRABIQ Index": "uk_gdp_qoq",
    "JGDPNSAQ Index": "japan_gdp_qoq",
    "AUNAGDPC Index": "australia_gdp_qoq",
    "UKRPCJYR Index": "uk_cpi_yoy",
    "JNCPIYOY Index": "japan_cpi_yoy",
    "AUCPIYOY Index": "australia_cpi_yoy",
}
# Bank for International Settlements effective exchange rates. The first
# attempt guessed tickers and three came back empty; these were found by
# searching the series' plain-English names on the terminal.
EER_TICKER_INFO = {
    "BISBAUN Index": "aud_eer_nominal", "BISBAUR Index": "aud_eer_real",
    "BISBGBN Index": "gbp_eer_nominal", "BISBGBR Index": "gbp_eer_real",
    "BISBJPN Index": "jpy_eer_nominal", "BISBJPR Index": "jpy_eer_real",
}


def _load_macro(path, ticker_map: dict) -> pd.DataFrame:
    """Same workbook shape as the prices, keeping only mapped tickers.

    Quarterly GDP and monthly CPI stay sparse here. Forward-filling with a
    publication lag is the caller's job, because the lag differs per series.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)
    tickers = raw.iloc[0, 1:].tolist()
    data = raw.iloc[2:, :].copy()
    data.columns = ["date"] + tickers
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data[[t for t in tickers if t in ticker_map]]
    data.columns = [ticker_map[t] for t in data.columns]
    log(f"{path.name}: {list(data.columns)}")
    return data.astype(float)


def load_econ() -> pd.DataFrame:
    """US GDP, CPI and high-yield credit spread, at native frequency."""
    return _load_macro(data_dir() / ECON_FILE, ECON_TICKER_INFO)


def load_econ_intl() -> pd.DataFrame:
    """GDP and CPI for the UK, Japan and Australia, at native frequency."""
    return _load_macro(data_dir() / INTL_FILE, GDP_CPI_INTL_TICKER_INFO)


def load_eer() -> pd.DataFrame:
    """Effective exchange rates for AUD, GBP and JPY."""
    raw = pd.read_excel(data_dir() / EER_FILE, sheet_name=0, header=None)
    tickers = raw.iloc[0, 1:].tolist()
    data = raw.iloc[2:, :].copy()
    data.columns = ["date"] + tickers
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data.columns = [EER_TICKER_INFO[t] for t in data.columns]
    return data.astype(float)


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders: the free Legacy, Futures Only archive. Legacy
# rather than the finer Traders in Financial Futures report because TFF only
# starts in 2009, which does not cover the sample, let alone a z-score warm-up.
#
# Contract codes are the join key, not names: the CFTC has renamed several of
# these contracts over the decades.
# ---------------------------------------------------------------------------
COT_CONTRACT_CODES = {
    "043602": "us10y",
    "13874A": "spx",    # E-mini, from 1997
    "088691": "gold",
    "067651": "wti",
    "099741": "eur",
    "232741": "aud",
}
_COT_COLUMNS = ["CFTC_Contract_Market_Code", "Report_Date_as_MM_DD_YYYY",
                "Open_Interest_All", "NonComm_Positions_Long_All", "NonComm_Positions_Short_All"]


def load_cot() -> pd.DataFrame:
    """Non-commercial net positioning as a fraction of open interest.

    Indexed by the report's 'as of' Tuesday, NOT its publication date (the
    following Friday). Not shifted here because the right lag depends on how
    the caller aligns it; the COT layer applies a 3-day lag.
    """
    frames = []
    for path in sorted((data_dir() / "cot").glob("*.xls")):
        chunk = pd.read_excel(path, usecols=_COT_COLUMNS)
        chunk["CFTC_Contract_Market_Code"] = chunk["CFTC_Contract_Market_Code"].astype(str)
        frames.append(chunk[chunk["CFTC_Contract_Market_Code"].isin(COT_CONTRACT_CODES)])
    raw = pd.concat(frames, ignore_index=True)
    raw["symbol"] = raw["CFTC_Contract_Market_Code"].map(COT_CONTRACT_CODES)
    raw["date"] = pd.to_datetime(raw["Report_Date_as_MM_DD_YYYY"], utc=True)
    net = ((raw["NonComm_Positions_Long_All"] - raw["NonComm_Positions_Short_All"])
           / raw["Open_Interest_All"])
    wide = pd.DataFrame({"date": raw["date"], "symbol": raw["symbol"], "v": net})
    wide = wide.drop_duplicates(subset=["date", "symbol"], keep="first")
    return wide.pivot(index="date", columns="symbol", values="v").sort_index()
