"""Nightly reconciliation of the live book: one ledger line a day.

Compares what the terminal actually did with what the engine asked for and
what the simulator's cost model says it should have cost. Four flags, the
four that gate the demo's exit criteria:

    FILL    a fill more than FILL_TOL_BPS beyond the model (half-spread plus
            slippage), judged against the mid of the minute it filled in
    FIN     overnight financing more than FIN_TOL_FRAC from the model in a month
    ROLL    a contract roll outside its window, or with mismatched legs
    DIVERG  held lots differing from target for more than one engine run

STRICTLY READ-ONLY WITH RESPECT TO THE TERMINAL. The EA is the only writer of
its files; a second writer can re-splice the price history underneath a
running engine. Everything this script reads is first COPIED into an archive
folder and read from there.

Two grades of evidence:

    live      the target file was archived before the rebalance's trades, so
              held lots are compared with the target as it stood
    replayed  only the current target file exists (the EA rewrites it every
              night at that night's equity and prices), so only differences
              no re-sizing can explain are flagged

Run it on the rebalance day itself, before the deals land, or the check can
only grade `replayed` and cannot catch a sizing error.

    python -m live.reconcile                # archive, then reconcile
    python -m live.reconcile --no-mt5       # CSV only
    python -m live.reconcile --no-archive
"""
import argparse
import datetime as dt
import os
import shutil

import numpy as np
import pandas as pd

from .specs import SPECS

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
MAGIC = 20260915
DEMO_START = "2026-09-15"
ROLL_DAYS = 5              # InpRollDays on the EA
SLIPPAGE_BPS = 1.0         # the simulator's assumption

FILL_TOL_BPS = 5.0         # beyond model half-spread + slippage
FIN_TOL_FRAC = 0.10        # per month, against the model
ROLL_WINDOW_SLACK_D = 2    # the EA's maintenance runs once a day, so a roll
                           # can land a day after the window opens (09-24)

# Outputs go to CTA_LIVE_DIR (default ./live_output), never into the terminal.
DATA = os.environ.get("CTA_LIVE_DIR", os.path.join(os.getcwd(), "live_output"))
ARCHIVE = os.path.join(DATA, "recon_archive")
LEDGER = os.path.join(DATA, "reconcile_ledger.csv")
REPORT = os.path.join(DATA, "reconcile_report.txt")

# files copied out of the terminal each run, before anything else touches them
ARCHIVE_FILES = ["cta_targets_engine.csv", "cta_targets.csv", "cta_deals.csv",
                 "cta_fills.csv", "cta_state.txt", "cta_port_prices.csv"]

BROKER_ROOTS = list(SPECS["broker"])               # SPX500, NDAQ100, ...
ROOT_TO_SYM = {row.broker: s for s, row in SPECS.iterrows()}

# Divergences that are known and deliberate. Reported every day with their
# reason, never counted as a defect, so the ledger does not cry wolf for
# months over something already decided. Removing a root from here is how
# the flag comes back.
# Held a Gilt entry while the broker was rejecting every Gilt order and the
# demo kept retrying it; empty since the Gilt was removed from the live spec.
ACCEPTED_DIVERGENCE: dict = {}


# ---------------------------------------------------------------------------
# terminal location and the archive step
# ---------------------------------------------------------------------------
def terminal_files_dir(use_mt5=True):
    """MQL5\\Files of the running terminal. Read-only use only."""
    if use_mt5:
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                ti = mt5.terminal_info()
                if ti is not None:
                    return os.path.join(ti.data_path, "MQL5", "Files")
        except Exception:
            pass
    base = os.path.join(os.environ["APPDATA"], "MetaQuotes", "Terminal")
    for d in os.listdir(base):
        p = os.path.join(base, d, "MQL5", "Files")
        if os.path.exists(os.path.join(p, "cta_targets_engine.csv")):
            return p
    raise SystemExit("cannot locate the terminal's MQL5\\Files")


def archive(files_dir, stamp=None):
    """Copy the terminal's state out before reading it. Never writes back."""
    stamp = stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    dest = os.path.join(ARCHIVE, stamp)
    os.makedirs(dest, exist_ok=True)
    got = []
    for f in ARCHIVE_FILES:
        src = os.path.join(files_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))
            got.append(f)
    return dest, got


def archived_targets_by_rebalance():
    """Every archived engine-target file, keyed by its META rebalance id.
    The first archive for a rebalance id is the one closest to the trade."""
    out = {}
    if not os.path.isdir(ARCHIVE):
        return out
    for stamp in sorted(os.listdir(ARCHIVE)):
        p = os.path.join(ARCHIVE, stamp, "cta_targets_engine.csv")
        if not os.path.exists(p):
            continue
        meta, tg = read_targets(p)
        if meta is None:
            continue
        out.setdefault(meta["rebalance_id"], (stamp, meta, tg))
    return out


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def read_targets(path):
    """META line then a header then one row per root. Returns (meta, frame)."""
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines or not lines[0].startswith("META"):
        return None, None
    m = lines[0].split(",")
    # m[1] is when the EA WROTE the file; m[6] is the date of the newest
    # PRICE behind the targets, appended by the EA from 2026-10-16. Files
    # archived before that recompile have no m[6], hence the default.
    meta = dict(written=m[1], rebalance_id=m[2], equity=float(m[3]),
                n=int(m[4]), engine=m[5] if len(m) > 5 else "",
                data_asof=m[6] if len(m) > 6 else "")
    tg = pd.DataFrame([ln.split(",") for ln in lines[2:]],
                      columns=lines[1].split(","))
    for c in ("side", "lots", "weight", "notional_usd", "stop_atr_mult"):
        tg[c] = pd.to_numeric(tg[c], errors="coerce")
    # the engine writes unsigned lots with the direction in `side`
    tg["target_lots"] = tg["side"] * tg["lots"].abs()
    return meta, tg.set_index("symbol")


DEAL_COLS = ["time", "symbol", "side", "entry", "lots", "price", "commission",
             "swap", "profit", "deal", "order", "position", "comment"]


def _empty_deals():
    """Typed empty frame: an untyped one makes every downstream `.dt` and
    `.str` call raise instead of simply finding nothing."""
    d = pd.DataFrame(columns=DEAL_COLS)
    d["time"] = pd.to_datetime(d["time"])
    d["comment"] = d["comment"].astype(str)
    for c in ("lots", "price", "commission", "swap", "profit"):
        d[c] = pd.to_numeric(d[c])
    return d


def read_deals_csv(path):
    """cta_deals.csv is headerless. Incomplete before the EA's logging was
    deployed, so MT5's account history is preferred where available."""
    if not os.path.exists(path):
        return _empty_deals()
    d = pd.read_csv(path, header=None, names=DEAL_COLS)
    d["time"] = pd.to_datetime(d["time"], format="%Y.%m.%d %H:%M:%S")
    d["comment"] = d["comment"].fillna("")
    return d


def read_deals_mt5(since):
    """The authoritative record: every deal under our magic."""
    import MetaTrader5 as mt5
    raw = mt5.history_deals_get(since, dt.datetime.now() + dt.timedelta(days=2))
    if not raw:
        return _empty_deals()
    d = pd.DataFrame([x._asdict() for x in raw])
    d = d[d["magic"] == MAGIC].copy()
    if d.empty:
        return _empty_deals()
    d["time"] = pd.to_datetime(d["time"], unit="s")
    d["side"] = np.where(d["type"] == 0, "BUY", np.where(d["type"] == 1, "SELL", "OTHER"))
    d["entry"] = d["entry"].map({0: "in", 1: "out", 2: "inout"}).fillna("other")
    d = d.rename(columns={"volume": "lots", "ticket": "deal", "position_id": "position"})
    return d[DEAL_COLS].sort_values("time").reset_index(drop=True)


def read_fills_csv(path):
    """cta_fills.csv: what was SENT, with the retcode. Rejections live here."""
    cols = ["time", "symbol", "side", "lots", "price", "deal", "retcode", "note"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    f = pd.read_csv(path, header=None, names=cols)
    f["time"] = pd.to_datetime(f["time"], format="%Y.%m.%d %H:%M:%S")
    f["note"] = f["note"].fillna("")
    return f


def root_of(symbol):
    """COPPERZ6 -> COPPER, AUDUSD.a -> AUDUSD.a. Longest prefix wins so
    USOIL and UKOIL cannot be confused."""
    cands = [r for r in BROKER_ROOTS if symbol == r or symbol.startswith(r)]
    return max(cands, key=len) if cands else None


# ---------------------------------------------------------------------------
# reference prices, straight from the terminal's bars for the CONTRACT traded
# ---------------------------------------------------------------------------
class Prices:
    """D1 and M1 closes per contract symbol. Nothing here touches the
    spliced research history: a fill is judged against the instrument that
    actually traded."""

    def __init__(self, use_mt5=True):
        self.ok = False
        self._d1, self._m1 = {}, {}
        self.server_offset = dt.timedelta(0)
        if not use_mt5:
            return
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.ok = mt5.initialize()
        except Exception:
            self.ok = False
        if self.ok:
            self.server_offset = self._measure_offset()

    def _measure_offset(self):
        """Deal times and bar times are SERVER time; copy_rates_range takes
        UTC. The broker's server runs UTC+3, but measure it rather than assume it:
        getting this wrong reads a fill against a bar from another hour and
        manufactures tens of bps of phantom slippage."""
        import time as _t
        best = None
        for s in ("SPX500Z6", "GOLDZ6", "AUDUSD.a", "EURUSD.a"):
            tick = self.mt5.symbol_info_tick(s)
            if tick is not None and tick.time:
                best = tick.time - _t.time()
                break
        if best is None:
            return dt.timedelta(0)
        return dt.timedelta(hours=round(best / 3600.0))

    def d1(self, symbol, start, end):
        key = (symbol, start.date(), end.date())
        if key in self._d1:
            return self._d1[key]
        out = pd.DataFrame()
        if self.ok:
            self.mt5.symbol_select(symbol, True)
            # wide window, matched on the bar's own date label: see m1_mid
            r = self.mt5.copy_rates_range(symbol, self.mt5.TIMEFRAME_D1,
                                          start - dt.timedelta(days=10),
                                          end + dt.timedelta(days=3))
            if r is not None and len(r):
                out = pd.DataFrame(r)
                out["date"] = pd.to_datetime(out["time"], unit="s").dt.normalize()
                out = out.set_index("date")
        self._d1[key] = out
        return out

    def m1_mid(self, symbol, when):
        """Mid of the minute bar containing `when`: the closest thing to the
        price on the screen at the moment of the fill.

        `copy_rates_range` interprets its naive datetimes in a frame that is
        neither UTC nor server time on this machine (measured: the returned
        bars sat 8 hours from the requested window), while the bar times it
        returns ARE correctly stamped on the server clock, the same clock
        the deal times use. So rather than correct the request, ask for a
        wide window and match on the bar's own label. No timezone arithmetic
        is then relied on anywhere.
        """
        if not self.ok:
            return np.nan
        day = pd.Timestamp(when).normalize()
        key = (symbol, day)
        if key not in self._m1:
            self.mt5.symbol_select(symbol, True)
            r = self.mt5.copy_rates_range(symbol, self.mt5.TIMEFRAME_M1,
                                          day.to_pydatetime() - dt.timedelta(days=2),
                                          day.to_pydatetime() + dt.timedelta(days=2))
            b = pd.DataFrame()
            if r is not None and len(r):
                b = pd.DataFrame(r)
                b["t"] = pd.to_datetime(b["time"], unit="s")
                b = b.set_index("t").sort_index()
            self._m1[key] = b
        bars = self._m1[key]
        if bars.empty:
            return np.nan
        gap = pd.Series((bars.index - pd.Timestamp(when)).values,
                        index=bars.index).abs()
        if gap.min() > pd.Timedelta(minutes=2):
            return np.nan          # no bar near the fill: not a reference
        row = bars.loc[gap.idxmin()]
        return float((row["open"] + row["close"]) / 2.0)

    def expiry(self, symbol):
        if not self.ok:
            return None
        info = self.mt5.symbol_info(symbol)
        if info is None or not info.expiration_time:
            return None
        return dt.datetime.fromtimestamp(info.expiration_time, dt.timezone.utc).replace(tzinfo=None)

    def fx_usd(self, ccy, when=None):
        """Multiplier taking one unit of `ccy` to USD."""
        if ccy == "USD":
            return 1.0
        if ccy == "USc":
            return 0.01
        if not self.ok:
            return np.nan
        pair, invert = {"GBP": ("GBPUSD.a", False), "JPY": ("USDJPY.a", True),
                        "AUD": ("AUDUSD.a", False)}.get(ccy, (None, False))
        if pair is None:
            return np.nan
        t = self.mt5.symbol_info_tick(pair)
        if t is None or not t.bid:
            return np.nan
        return (1.0 / t.bid) if invert else float(t.bid)

    def lot_value_usd(self, root, price):
        """USD notional of one lot of `root` at `price`, per specs.py."""
        sym = ROOT_TO_SYM.get(root)
        if sym is None or not np.isfinite(price):
            return np.nan
        row = SPECS.loc[sym]
        if row.cls == "fx":
            # an FX lot is `contract` units of the base currency
            base = {"aud": "AUD", "gbp": "GBP", "jpy": "USD"}[sym]
            return row.contract * (1.0 if base == "USD" else self.fx_usd(base))
        return row.contract * price * self.fx_usd(row.ccy)


# ---------------------------------------------------------------------------
# what was actually held, day by day, rebuilt from the deals
# ---------------------------------------------------------------------------
def held_lots_daily(deals, end=None):
    """Signed lots per CONTRACT symbol per day, cumulated from the deals.
    Returned on a calendar-day index so a divergence can be aged in days."""
    if deals.empty:
        return pd.DataFrame()
    d = deals.copy()
    d["signed"] = np.where(d["side"] == "BUY", d["lots"], -d["lots"])
    d["date"] = d["time"].dt.normalize()
    daily = d.pivot_table(index="date", columns="symbol", values="signed",
                          aggfunc="sum").fillna(0.0).cumsum()
    end = pd.Timestamp(end or dt.date.today()).normalize()
    idx = pd.date_range(daily.index.min(), max(daily.index.max(), end), freq="D")
    return daily.reindex(idx).ffill().fillna(0.0)


def by_root(held):
    """Contract-level lots collapsed to the root the engine targets."""
    if held.empty:
        return held
    out = {}
    for col in held.columns:
        r = root_of(col)
        if r is None:
            continue
        out[r] = out.get(r, 0.0) + held[col]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# check A: fill quality
# ---------------------------------------------------------------------------
def check_fills(deals, px):
    """Each deal against the mid of the minute it filled in. The model
    allows half the spread plus slippage; anything past that by more than
    FILL_TOL_BPS is flagged."""
    rows = []
    for _, d in deals.iterrows():
        root = root_of(d["symbol"])
        sym = ROOT_TO_SYM.get(root)
        if sym is None:
            continue
        mid = px.m1_mid(d["symbol"], d["time"].to_pydatetime())
        model_bps = SPECS.loc[sym, "spread_bps"] / 2.0 + SLIPPAGE_BPS
        if not np.isfinite(mid) or mid <= 0:
            rows.append(dict(time=d["time"], symbol=d["symbol"], root=root,
                             side=d["side"], lots=d["lots"], price=d["price"],
                             ref_mid=np.nan, slip_bps=np.nan, model_bps=model_bps,
                             excess_bps=np.nan, flag="", note="no M1 reference"))
            continue
        # positive = we paid away from mid, in the direction that costs us
        sign = 1.0 if d["side"] == "BUY" else -1.0
        slip_bps = sign * (d["price"] - mid) / mid * 1e4
        excess = slip_bps - model_bps
        rows.append(dict(time=d["time"], symbol=d["symbol"], root=root,
                         side=d["side"], lots=d["lots"], price=d["price"],
                         ref_mid=mid, slip_bps=slip_bps, model_bps=model_bps,
                         excess_bps=excess,
                         flag="FILL" if excess > FILL_TOL_BPS else "", note=""))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# check B: financing
# ---------------------------------------------------------------------------
def check_financing(deals, held, px, months=None):
    """Swap actually charged against the model, per month.

    Actual: DEAL_SWAP on closing deals plus the swap accrued on positions
    still open. Model: notional x rate x days / 365 from specs.py, which is
    zero on dated contracts. Only markets held as CFDs can breach this.
    """
    import MetaTrader5 as mt5
    actual = {}
    if not deals.empty:
        d = deals.copy()
        d["month"] = d["time"].dt.to_period("M").astype(str)
        for (m, s), g in d.groupby(["month", "symbol"]):
            actual[(m, root_of(s))] = actual.get((m, root_of(s)), 0.0) + float(g["swap"].sum())
    if px.ok:
        for p in (mt5.positions_get() or []):
            if p.magic != MAGIC:
                continue
            m = str(pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M"))
            k = (m, root_of(p.symbol))
            actual[k] = actual.get(k, 0.0) + float(p.swap)

    # model side
    from ctadata.rates import load_rates
    rates = load_rates()
    if getattr(rates.index, "tz", None) is not None:
        rates = rates.tz_localize(None)       # the held index is naive
    roots = by_root(held)
    rows = []
    if roots.empty:
        return pd.DataFrame(rows)
    for root in roots.columns:
        sym = ROOT_TO_SYM.get(root)
        if sym is None:
            continue
        spec = SPECS.loc[sym]
        lots = roots[root]
        model_m = {}
        for day, v in lots.items():
            if v == 0 or spec.kind == "dated":
                continue
            bars = px.d1(_contract_on(deals, root, day) or root, day, day)
            close = float(bars["close"].reindex([day]).ffill().iloc[0]) if len(bars) else np.nan
            lv = px.lot_value_usd(root, close)
            if not np.isfinite(lv):
                continue
            notional = v * lv
            # as P&L, to match DEAL_SWAP: negative = we paid
            r = _annual_fin_earned(sym, spec, rates, day, notional >= 0)
            key = str(day.to_period("M"))
            model_m[key] = model_m.get(key, 0.0) + abs(notional) * r / 365.0
        for m, model in model_m.items():
            act = actual.get((m, root), 0.0)
            denom = abs(model) if abs(model) > 1e-9 else np.nan
            dev = abs(act - model) / denom if np.isfinite(denom) else np.nan
            rows.append(dict(month=m, root=root, actual_swap=act, model_swap=model,
                             dev_frac=dev,
                             flag="FIN" if (np.isfinite(dev) and dev > FIN_TOL_FRAC) else ""))
    return pd.DataFrame(rows)


def _annual_fin_earned(sym, spec, rates, day, is_long):
    """The simulator's own sign convention: positive = the position EARNS
    it. DEAL_SWAP uses the same convention, so the two are comparable
    without a flip."""
    r = rates.reindex([pd.Timestamp(day)], method="ffill").ffill().bfill().iloc[0]
    if spec.cls == "fx":
        base, quote = {"aud": ("AUD", "USD"), "gbp": ("GBP", "USD"),
                       "jpy": ("JPY", "USD")}[sym]
        diff = float(r[base] - r[quote])
        earn = (diff - spec.fin_long / 100) if is_long else (-diff - spec.fin_short / 100)
    else:
        ref = float(r["JPY" if sym == "nikkei" else "USD"])
        earn = -(ref + spec.fin_long / 100) if is_long else (ref - spec.fin_short / 100)
    return earn


def _contract_on(deals, root, day):
    """Which contract of `root` was held on `day`, per the deals."""
    if deals.empty:
        return None
    d = deals[(deals["time"] <= day + pd.Timedelta(days=1))]
    d = d[[root_of(s) == root for s in d["symbol"]]]
    return d["symbol"].iloc[-1] if len(d) else None


# ---------------------------------------------------------------------------
# check C: rolls
# ---------------------------------------------------------------------------
def check_rolls(deals, px):
    """A roll is a same-day pair: into the new contract, out of the old.
    Window: the EA rolls when the held contract is within ROLL_DAYS of
    expiry, and maintenance runs once a day, so a roll may legitimately land
    up to ROLL_WINDOW_SLACK_D late. Size must match on both legs."""
    rows = []
    if deals.empty:
        return pd.DataFrame(rows)
    d = deals.copy()
    d["date"] = d["time"].dt.normalize()
    d["root"] = [root_of(s) for s in d["symbol"]]
    for (day, root), g in d.groupby(["date", "root"]):
        syms = set(g["symbol"])
        if root is None or len(syms) < 2:
            continue
        ins = g[g["comment"].str.contains("roll-in", na=False)]
        outs = g[(~g.index.isin(ins.index)) & (g["entry"] == "out")]
        if ins.empty:
            continue
        new_sym, old_sym = ins["symbol"].iloc[0], (outs["symbol"].iloc[0] if len(outs) else "")
        in_lots, out_lots = float(ins["lots"].sum()), float(outs["lots"].sum())
        exp_old = px.expiry(old_sym) if old_sym else None
        days_to_exp = (exp_old - day.to_pydatetime()).days if exp_old else np.nan
        early = np.isfinite(days_to_exp) and days_to_exp > ROLL_DAYS
        late = np.isfinite(days_to_exp) and days_to_exp < ROLL_DAYS - ROLL_WINDOW_SLACK_D
        size_ok = abs(in_lots - out_lots) < 1e-9
        notes = []
        if early:
            notes.append(f"rolled {days_to_exp}d before expiry, window is {ROLL_DAYS}d")
        if late:
            notes.append(f"rolled {days_to_exp}d before expiry, later than the window")
        if not size_ok:
            notes.append(f"in {in_lots:.2f} vs out {out_lots:.2f}")
        rows.append(dict(date=day, root=root, new=new_sym, old=old_sym,
                         in_lots=in_lots, out_lots=out_lots,
                         days_to_expiry=days_to_exp,
                         flag="ROLL" if (early or late or not size_ok) else "",
                         note="; ".join(notes)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# check D: held vs target
# ---------------------------------------------------------------------------
def check_divergence(held, targets, meta, grade):
    """Held lots per root against the target of the rebalance the EA
    executed, aged in days.

    The comparison must be against the target AS AT that rebalance. Between
    rebalances the EA holds a fixed lot count by design, while the engine
    rewrites `cta_targets_engine.csv` nightly at the day's equity and the
    day's price, so target lots move even though nothing is wrong. Comparing
    a held lot count to a nightly file therefore flags ordinary drift.

    grade `live`     the target was archived at the rebalance: compare lots
                     strictly, half a lot step of tolerance.
    grade `replayed` the target has been re-sized since: only differences
                     that no re-sizing can explain are flagged, namely a
                     sign mismatch and a flat book against a live target.
                     Everything else is reported as `drift_pct`, unflagged.
    """
    roots = by_root(held)
    if roots.empty or targets is None:
        return pd.DataFrame()
    rows = []
    for root in sorted(set(roots.columns) | set(targets.index)):
        tgt = float(targets["target_lots"].get(root, 0.0))
        series = roots[root] if root in roots.columns else pd.Series(0.0, index=roots.index)
        step = float(SPECS.loc[ROOT_TO_SYM[root], "vstep"]) if root in ROOT_TO_SYM else 0.01
        diverged = 0
        for day, v in series.items():
            if grade == "live":
                off = abs(v - tgt) > step / 2.0 + 1e-9
                why = "lots differ from the archived target"
            else:
                flat_vs_live = (abs(v) < step / 2.0) and (abs(tgt) >= step / 2.0)
                sign_wrong = (v * tgt < 0)
                off = flat_vs_live or sign_wrong
                why = ("flat against a live target" if flat_vs_live
                       else "wrong side" if sign_wrong else "")
            diverged = diverged + 1 if off else 0
            drift = (v - tgt) / abs(tgt) if abs(tgt) > 1e-9 else np.nan
            accepted = root in ACCEPTED_DIVERGENCE
            flag = "DIVERG" if (diverged > 1 and not accepted) else ""
            if diverged > 1 and accepted:
                flag = "ACCEPTED"
                why = ACCEPTED_DIVERGENCE[root]
            rows.append(dict(date=day, root=root, held_lots=v, target_lots=tgt,
                             diff_lots=v - tgt, drift_pct=100.0 * drift,
                             days_diverged=diverged,
                             rebalance_id=meta["rebalance_id"] if meta else "",
                             grade=grade, flag=flag,
                             note=why if diverged > 1 else ""))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# rejections, which are not one of the four flags but belong on the ledger
# ---------------------------------------------------------------------------
def summarise_rejections(fills):
    if fills.empty:
        return pd.DataFrame()
    bad = fills[~fills["retcode"].isin([10008, 10009, 10010, 0])].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["date"] = bad["time"].dt.normalize()
    g = bad.groupby(["date", "symbol", "retcode"]).size().rename("n").reset_index()
    g["flag"] = "REJECT"
    return g


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEMO_START)
    ap.add_argument("--no-mt5", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    a = ap.parse_args()

    use_mt5 = not a.no_mt5
    files_dir = terminal_files_dir(use_mt5)
    print(f"terminal files: {files_dir}  (read-only)")

    if not a.no_archive:
        dest, got = archive(files_dir)
        print(f"archived {len(got)} files to {dest}")
    else:
        dest = files_dir

    px = Prices(use_mt5)
    print(f"MT5 reference prices: {'available' if px.ok else 'UNAVAILABLE, fills unjudged'}")

    since = dt.datetime.fromisoformat(a.since)
    deals_csv = read_deals_csv(os.path.join(files_dir, "cta_deals.csv"))
    deals = read_deals_mt5(since) if px.ok else deals_csv
    source = "MT5 account history" if px.ok else "cta_deals.csv"
    if px.ok and len(deals_csv) < len(deals):
        print(f"note: cta_deals.csv has {len(deals_csv)} rows, the account history "
              f"has {len(deals)}. The EA's deal log started after the first "
              f"trades; using the account history.")
    deals = deals[deals["time"] >= since]
    fills = read_fills_csv(os.path.join(files_dir, "cta_fills.csv"))
    fills = fills[fills["time"] >= since]
    print(f"deals: {len(deals)} from {source}; fills sent: {len(fills)}")

    meta, targets = read_targets(os.path.join(files_dir, "cta_targets_engine.csv"))
    arch = archived_targets_by_rebalance()
    grade = "replayed"
    if meta and meta["rebalance_id"] in arch:
        stamp, ameta, atg = arch[meta["rebalance_id"]]
        # `live` only if the target was archived before the trades THAT
        # rebalance produced, not merely before the demo's first ever trade
        rb = deals[deals["comment"].str.contains("rebalance", na=False)]
        rb = rb[rb["time"].dt.to_period("M").astype(str) == meta["rebalance_id"]]
        first = rb["time"].min().strftime("%Y-%m-%d") if len(rb) else None
        if first is None or stamp <= first:
            meta, targets, grade = ameta, atg, "live"
    asof = (meta or {}).get("data_asof", "")
    note = f", prices to {asof}" if asof else ", prices to UNKNOWN (EA predates the asof field)"
    if asof:
        # the EA's own guard is InpMaxDataAgeD = 5; say so here too, because
        # this script is the thing that gets read every morning. A format we
        # do not recognise must not take the whole reconciliation down.
        try:
            age = (dt.date.today() - dt.datetime.strptime(asof, "%Y.%m.%d").date()).days
            note += f" ({age} days old)" + ("  <-- STALE" if age > 5 else "")
        except ValueError:
            note += " (unparseable date)"
    print(f"targets: rebalance {meta['rebalance_id'] if meta else '?'} "
          f"at equity {meta['equity'] if meta else 0:,.0f}, grade={grade}{note}")

    held = held_lots_daily(deals)

    f_fills = check_fills(deals, px)
    f_rolls = check_rolls(deals, px)
    f_div = check_divergence(held, targets, meta, grade)
    try:
        f_fin = check_financing(deals, held, px)
    except Exception as e:
        print(f"financing check skipped: {e}")
        f_fin = pd.DataFrame()
    f_rej = summarise_rejections(fills)

    # one ledger line a day
    days = pd.date_range(since, dt.date.today(), freq="D")
    led = pd.DataFrame(index=days)
    led.index.name = "date"
    led["grade"] = grade
    led["deals"] = deals.set_index(deals["time"].dt.normalize()).groupby(level=0).size().reindex(days).fillna(0).astype(int)
    for name, frame, key in [("fill", f_fills, "time"), ("roll", f_rolls, "date"),
                             ("diverg", f_div, "date"), ("reject", f_rej, "date")]:
        if frame.empty:
            led[f"{name}_flags"] = 0
            continue
        k = frame[key].dt.normalize() if np.issubdtype(frame[key].dtype, np.datetime64) else frame[key]
        n = frame.assign(_d=k)[lambda x: ~x["flag"].isin(["", "ACCEPTED"])].groupby("_d").size()
        led[f"{name}_flags"] = n.reindex(days).fillna(0).astype(int)
    led["fin_flags_month"] = int((f_fin["flag"] != "").sum()) if not f_fin.empty else 0
    led["clean"] = (led[[c for c in led.columns if c.endswith("_flags")]].sum(axis=1) == 0)

    os.makedirs(DATA, exist_ok=True)
    led.to_csv(LEDGER)

    # report
    out = []
    w = out.append
    w(f"CTA demo reconciliation, {dt.datetime.now():%Y-%m-%d %H:%M}")
    w(f"window {a.since} to {dt.date.today()}, deals from {source}, grade {grade}")
    w("")
    w("LEDGER")
    w(led.to_string())
    for title, frame in [("FILLS", f_fills), ("ROLLS", f_rolls),
                         ("FINANCING (per month)", f_fin),
                         ("HELD VS TARGET, flagged days only",
                          f_div[f_div["flag"] == "DIVERG"] if not f_div.empty else f_div),
                         ("HELD VS TARGET, accepted and not counted (latest day)",
                          f_div[(f_div["flag"] == "ACCEPTED")
                                & (f_div["date"] == f_div["date"].max())]
                          if not f_div.empty else f_div),
                         ("REJECTIONS", f_rej)]:
        w("")
        w(title)
        w(frame.to_string(index=False) if not frame.empty else "  none")
    if not f_div.empty:
        w("")
        w("HELD VS TARGET, latest day")
        last = f_div[f_div["date"] == f_div["date"].max()]
        w(last.to_string(index=False))
    text = "\n".join(out)
    with open(REPORT, "w") as fh:
        fh.write(text)
    print()
    print(text)
    print()
    print(f"ledger -> {LEDGER}")
    print(f"report -> {REPORT}")
    if px.ok:
        import MetaTrader5 as mt5
        mt5.shutdown()


if __name__ == "__main__":
    main()
