r"""Verify that the MQL5 engine reproduces the Python engine.

The live strategy runs inside MT5 with no Python at run time, so its engine
(mql5/CTAEngine.mqh) is a second implementation of build_trend_nav. This is
the harness that proves the two agree:

  1. export    run the chosen configuration in Python on the live universe
               and write two files:
                 cta_port_prices.csv   the price history the EA will load
                 cta_port_ref.csv      date, leverage, regime multiplier, then
                                       the held weight per market
  2. run mql5/CTAEngineTest.mq5 in the terminal: it loads the same prices,
     runs CTAEngine.mqh, and writes cta_port_mql5.csv
  3. compare   every day, every column, against the reference

Result when the port was built: 5,171 days by 13 markets, the regime
multiplier identical on every day, leverage and held weights within 2e-8.

    python -m live.port_check export --out DIR [--to-terminal]
    python -m live.port_check compare --dir DIR

`--to-terminal` writes cta_port_prices.csv into the terminal's MQL5\Files,
which REPLACES the live EA's price history and forces a re-splice. Never do
that on a running live account except deliberately; the default writes to
--out instead.
"""
import argparse
import os

import numpy as np
import pandas as pd

from cta import build_trend_nav, DEFAULT_PARAMS, CHOSEN
from cta.regime import _REGIME_CACHE
from ctadata import load_live_universe
from .simulator import held_weights
from .specs import SPECS, spec_buckets

# Bucket order must match CTASignal.mqh: equities 0, commodities 1,
# currencies 2, fixed income 3, crypto 4.
BUCKET_ORDER = ["equities", "commodities", "currencies", "fixed_income", "crypto"]


def export(out_dir: str) -> None:
    syms = list(SPECS.index)                 # engine column order = spec order
    df = load_live_universe()
    uni = spec_buckets()
    uni = {k: uni[k] for k in BUCKET_ORDER if k in uni}
    ref = build_trend_nav(df, symbols=syms, universe=uni, **{**DEFAULT_PARAMS, **CHOSEN})
    held, _ = held_weights(ref, df, "M")
    mult = list(_REGIME_CACHE.values())[-1]   # the multiplier this run produced
    px = df[syms].copy()
    px.index = px.index.tz_localize(None)
    os.makedirs(out_dir, exist_ok=True)
    px.to_csv(os.path.join(out_dir, "cta_port_prices.csv"), date_format="%Y.%m.%d",
              float_format="%.10g", na_rep="")
    out = pd.concat([ref[1].rename("leverage"), mult.rename("mult"), held[syms]], axis=1)
    out.index = out.index.tz_localize(None)
    out.to_csv(os.path.join(out_dir, "cta_port_ref.csv"), date_format="%Y.%m.%d",
               float_format="%.12g")
    print(f"wrote {out_dir}: prices {px.shape}, reference {out.shape}")


def compare(d: str, tol: float = 1e-9) -> bool:
    ref = pd.read_csv(os.path.join(d, "cta_port_ref.csv"), index_col=0)
    mq = pd.read_csv(os.path.join(d, "cta_port_mql5.csv"), index_col=0)
    ref.index = pd.to_datetime(ref.index, format="%Y.%m.%d")
    mq.index = pd.to_datetime(mq.index, format="%Y.%m.%d")
    assert list(ref.columns) == list(mq.columns), (ref.columns, mq.columns)
    assert len(ref) == len(mq)
    diff = (ref - mq).abs()
    print(f"{len(ref)} days, {len(ref.columns)} columns; max abs difference per column:")
    print(diff.max().to_string(float_format=lambda x: f"{x:.2e}"))
    bad = diff[(diff > tol).any(axis=1)]
    if len(bad):
        first = bad.index[0]
        print(f"\nfirst divergence {first.date()}:")
        print(pd.DataFrame({"python": ref.loc[first], "mql5": mq.loc[first]}))
        return False
    print(f"\nIDENTICAL to {tol:g} on every day and every column")
    return True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="port_check")
    e.add_argument("--to-terminal", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--dir", default="port_check")
    a = ap.parse_args()
    if a.cmd == "export":
        out = a.out
        if a.to_terminal:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                raise SystemExit("cannot reach the terminal")
            out = os.path.join(mt5.terminal_info().data_path, "MQL5", "Files")
            mt5.shutdown()
        export(out)
    else:
        compare(a.dir)


if __name__ == "__main__":
    main()
