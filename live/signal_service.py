r"""Python reference for the live engine: today's targets as a file.

    python -m live.signal_service [--equity-leg 1.0] [--capital 100000]
                                  [--stop-atr 10] [--out PATH] [--dry-run]

The live book normally runs entirely inside MT5 (mql5/CTAEngine.mqh), with
no Python at run time. This module is the Python twin of that engine. It is
used two ways:

  * as the REFERENCE the MQL5 port is checked against: both must produce the
    same targets from the same prices;
  * as a fallback ("file mode"): the EA can trade from this file instead of
    computing targets itself.

It runs the chosen configuration on the research frame extended to
yesterday, takes the held weights on the last row, folds the 100% equity leg
into the S&P row, converts weights to lots at live prices and account
equity, floors to the lot step, and writes:

    META,<written utc>,<rebalance id>,<equity>,<rows>,<engine fingerprint>
    symbol,kind,side,lots,weight,notional_usd,stop_atr_mult
    SPX500,dated,1,0.35,0.1834,18340,10

`rebalance id` is the month of the last calendar rebalance. The EA trades
only when it changes, so this can safely run every day.
"""
import argparse
import datetime as dt
import hashlib
import sys

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from cta import build_trend_nav, DEFAULT_PARAMS, CHOSEN
from ctadata import load_live_universe
from .live_prices import extend_with_broker, broker_symbol, load_live_history, LIVE_HISTORY
from .simulator import held_weights
from .specs import SPECS, INVERTED, spec_buckets

# The live universe IS the spec table. Buckets come from it too: a hand-kept
# class map once lacked the crypto rows and this module raised KeyError on
# import for three weeks without anyone noticing, because the EA was running
# in engine mode.
SYMBOLS = list(SPECS.index)


def usd_per_unit(currency: str) -> float:
    """USD per unit of `currency`, from live mid prices."""
    if currency == "USD":
        return 1.0
    if currency == "USc":
        return 0.01
    pair, inverted = {"GBP": ("GBPUSD.a", False), "AUD": ("AUDUSD.a", False),
                      "JPY": ("USDJPY.a", True)}[currency]
    mt5.symbol_select(pair, True)
    t = mt5.symbol_info_tick(pair)
    mid = (t.bid + t.ask) / 2
    return 1.0 / mid if inverted else mid


def live_lot_value_usd(symbol: str, sym: str) -> float:
    """USD notional of one lot, by the same rule as simulator.lot_value_usd
    and CTA_LotValueUsd in the EA.

    The currency comes from SPECS, never from MT5's profit currency: MT5
    reports copper in USD though it is quoted in US cents, which made copper's
    lot look 100x too large and floored its position to zero. Ad-hoc analysis
    that skipped the currency conversion entirely once made the two
    yen-quoted markets look 158x too large.
    """
    row = SPECS.loc[sym]
    mt5.symbol_select(symbol, True)
    if row.cls == "fx":
        # one FX lot is `contract` units of the BASE currency: no price term
        return row.contract * usd_per_unit(row.ccy)
    t = mt5.symbol_info_tick(symbol)
    return row.contract * (t.bid + t.ask) / 2 * usd_per_unit(row.ccy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-leg", type=float, default=1.0,
                    help="equity notional as a fraction of capital, folded into the S&P row")
    ap.add_argument("--capital", type=float, default=None, help="size on this instead of account equity")
    ap.add_argument("--stop-atr", type=float, default=10.0, help="server-side rail width in daily ATRs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not mt5.initialize():
        sys.exit(f"MT5 initialise failed: {mt5.last_error()}")
    acct = mt5.account_info()
    equity = a.capital or float(acct.equity)
    assert acct.currency == "USD", "lot sizing assumes a USD account"

    df = load_live_universe()[SYMBOLS]
    # Read the EA's own history if it is newer. Never WRITE it from here: the
    # EA is the only writer of its files, and a second writer can re-splice
    # the history underneath a running engine.
    df = load_live_history(df, mt5.terminal_info().data_path + "\\MQL5\\Files\\" + LIVE_HISTORY)
    df, report = extend_with_broker(df)
    for k, v in report.items():
        print(f"  {k:7s} {v}")

    asof = df.index.max()
    out = build_trend_nav(df, symbols=SYMBOLS, universe=spec_buckets(),
                          **{**DEFAULT_PARAMS, **CHOSEN})
    held, is_rebal = held_weights(out, df, "M")
    rebal_dates = df.index[is_rebal]
    rebalance_id = rebal_dates[rebal_dates <= asof].max().strftime("%Y-%m")
    fingerprint = hashlib.sha1(np.ascontiguousarray(out[0].values, dtype="float64")).hexdigest()[:12]
    w = held.iloc[-1].copy()
    w["spx"] += a.equity_leg                    # the equity leg rides on the S&P row

    rows = []
    for sym in SYMBOLS:
        row = SPECS.loc[sym]
        symbol = broker_symbol(sym)
        lv = live_lot_value_usd(symbol, sym)
        weight = float(w[sym])
        notional = weight * equity
        lots = notional / lv
        if sym in INVERTED:
            lots = -lots
        info = mt5.symbol_info(symbol)
        # FLOOR, never round up: rounding up a target smaller than one lot
        # would take a position up to 55% larger than asked for in a market
        # running at 70% volatility.
        lots = np.sign(lots) * max(0.0, np.floor(abs(lots) / info.volume_step) * info.volume_step)
        if abs(lots) < info.volume_min:
            lots = 0.0
        rows.append(dict(symbol=row.broker, kind=row.kind, side=int(np.sign(lots)),
                         lots=round(abs(lots), 2), weight=round(weight, 5),
                         notional_usd=round(notional, 0), stop_atr_mult=a.stop_atr,
                         resolved=symbol, lot_value_usd=round(lv, 0)))
    t = pd.DataFrame(rows)
    gross = (t.lots * t.lot_value_usd).sum()
    print(f"\nasof {asof.date()}  rebalance {rebalance_id}  equity {equity:,.0f}  "
          f"gross {gross:,.0f} ({gross / equity:.2f}x)  engine {fingerprint}")
    print(t.to_string(index=False))
    if a.dry_run:
        mt5.shutdown()
        return

    path = a.out or (mt5.terminal_info().data_path + r"\MQL5\Files\cta_targets.csv")
    stamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # newline="" matters on Windows: without it each "\n" is written as CRLF,
    # and an earlier writer that ALSO translated produced CR CR LF. MQL5 keeps
    # the stray CR on the last field, so a header check read "eth\r" and
    # halted the EA.
    with open(path, "w", newline="") as f:
        f.write(f"META,{stamp},{rebalance_id},{equity:.2f},{len(t)},{fingerprint}\n")
        f.write("symbol,kind,side,lots,weight,notional_usd,stop_atr_mult\n")
        for r in t.itertuples():
            f.write(f"{r.symbol},{r.kind},{r.side},{r.lots:.2f},{r.weight:.5f},"
                    f"{r.notional_usd:.0f},{r.stop_atr_mult:g}\n")
    print("wrote", path)
    mt5.shutdown()


if __name__ == "__main__":
    main()
