"""Daily history of the listed managed-futures ETFs, from Yahoo.

Used as live peers for the demo (DBMF and KMLM) and as long-window
benchmarks at matched volatility. Yahoo is unofficial and rate limited, so
this is a cross-check rather than a primary source.

Both series are kept per fund because the difference is large and uneven:
one fund returned 6.2% on price and 10.3% on total return in a single year,
so a price-only comparison ranks the group wrong.

    close     price only
    adjclose  total return, distributions reinvested
"""
import argparse
import os
import time

import pandas as pd
import requests

TICKERS = ["KMLM", "DBMF", "CTA", "FMF", "WTMF", "SPY"]
NAMES = {"KMLM": "KraneShares Mount Lucas Managed Futures",
         "DBMF": "iMGP DBi Managed Futures Strategy",
         "CTA": "Simplify Managed Futures Strategy",
         "FMF": "First Trust Managed Futures Strategy",
         "WTMF": "WisdomTree Managed Futures Strategy",
         "SPY": "S&P 500 ETF (reference)"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def fetch(session, ticker, start="2010-01-01", tries=4) -> pd.DataFrame:
    """Explicit period1/period2, NOT range=max: Yahoo silently downsamples a
    max-range request to MONTHLY bars, which is how a first attempt returned
    305 rows for six years of daily data."""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    for k in range(tries):
        r = session.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                        params={"period1": p1, "period2": p2, "interval": "1d",
                                "events": "div,split"}, timeout=25)
        if r.status_code == 200:
            j = r.json()["chart"]["result"][0]
            idx = pd.to_datetime(j["timestamp"], unit="s").normalize()
            q = j["indicators"]["quote"][0]
            adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose")
            # Test the CONTENTS, not the list. Yahoo often returns a
            # full-length list of nulls, `if adj` passes that, and the result
            # was an all-NaN total-return series on the weekly report.
            usable = adj is not None and len(adj) == len(idx) and any(v is not None for v in adj)
            df = pd.DataFrame({"close": q["close"],
                               "adjclose": adj if usable else q["close"]}, index=idx)
            if not usable:
                print(f"   {ticker}: no adjusted close, using price only")
            return df.dropna(how="all")
        wait = 3 * (k + 1)
        print(f"   {ticker}: HTTP {r.status_code}, retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"{ticker}: gave up after {tries} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--out", default=os.path.join("live_output", "etfs.parquet"))
    a = ap.parse_args()
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    out = {}
    for t in TICKERS:
        out[t] = fetch(s, t, a.start)
        print(f"{t:5s} {len(out[t]):5d} bars   {NAMES[t]}")
        time.sleep(1.5)
    wide = pd.concat(out, axis=1)
    wide.columns.names = ["ticker", "field"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    wide.to_parquet(a.out)
    print(f"wrote {a.out} {wide.shape}")


if __name__ == "__main__":
    main()
