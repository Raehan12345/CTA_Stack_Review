# %% [markdown]
# # Chapter 9. From backtest to a live account
#
# **Question.** A backtest trades fractional positions at closing prices with
# a cost table. A live account trades whole lots, a day late, on CFDs that
# charge financing every night, through an engine written in a different
# language. How much of the backtest survives, and how do we know the live
# engine is the same strategy?
#
# **Covers** (original exec_sim_v1, the port harness, the reconciliation and
# marking tools):
# 1. the execution simulator, one effect at a time;
# 2. verifying the MQL5 port against the Python engine;
# 3. marking a live book from its deals;
# 4. the daily reconciliation.
#
# **Configuration:** the live book (A-12 plus BTC and ETH, five HRP buckets),
# chosen configuration, monthly, 13% target, sleeve only, Sharpe after the
# risk-free rate.

# %%
from _setup import *
from ctadata import load_live_universe
from ctadata.rates import load_rates
from live.specs import SPECS, spec_buckets
from live.simulator import simulate, held_weights, lot_value_usd
from live.marking import position_book_daily

LIVE = load_live_universe()
SYMS = list(SPECS.index)
RATES = load_rates()
KW = {**DEFAULT_PARAMS, **CHOSEN, "universe": spec_buckets()}
gross = build_trend_nav(LIVE, symbols=SYMS, **{**KW, "apply_costs": False})
net = build_trend_nav(LIVE, symbols=SYMS, **KW)
r_gross, r_net = returns(gross[0]), returns(net[0])
held, is_rebal = held_weights(gross, LIVE, "M")
RET = LIVE[SYMS].pct_change().fillna(0.0)
SPECS[["broker", "kind", "spread_bps", "vmin", "vstep"]]

# %% [markdown]
# ## 1. The simulator ladder
#
# `live/simulator.py` takes the research engine's HELD weights and re-trades
# them on the broker's instruments. Step 0 checks the reconstruction: held
# weights times market returns must give back the research gross return,
# otherwise every later step measures a bug.

# %%
recon = (RET * held).sum(axis=1)
print(f"reconstruction against research gross: max daily error {(recon - r_gross).abs().max():.1e}")

# %%
START = OOS
yrs = (LIVE.index[-1] - pd.Timestamp(START, tz=LIVE.index.tz)).days / 365.25


def step(**kw):
    out = simulate(LIVE, RET, held, is_rebal, RATES, start=START, **kw)
    navm = out["nav"].loc[START:].mean()
    s = stats(out["ret"].loc[START:], RF)
    s.update({"spread_pct_yr": 100 * out["cost_spread"].loc[START:].sum() / navm / yrs,
              "financing_pct_yr": 100 * out["cost_fin"].loc[START:].sum() / navm / yrs,
              "trades_yr": out["trades"] / yrs})
    return s

ladder = {
    "research gross": stats(r_gross.loc[START:], RF),
    "research net (cost table)": stats(r_net.loc[START:], RF),
    "0. sim, no costs, no lag, no rounding": step(capital=1e6, lag=0, slippage_bps=0.0, round_lots=False,
                                                  financing=False, spreads=False),
    "1. + spread and slippage": step(capital=1e6, lag=0, round_lots=False, financing=False),
    "2. + one-day lag": step(capital=1e6, lag=1, round_lots=False, financing=False),
    "3. + CFD financing": step(capital=1e6, lag=1, round_lots=False),
    "4. + whole lots, $1m": step(capital=1e6, lag=1),
    "4. + whole lots, $100k": step(capital=1e5, lag=1),
}
pd.DataFrame(ladder).T

# %% [markdown]
# Read it top to bottom; each line adds one effect.
#
# * **Step 0** should equal research gross: it is the same book re-traded
#   with every friction switched off.
# * **Spread and slippage** land near the research cost table, which is what
#   the table was built to approximate.
# * **The one-day lag** is the largest single step, about 0.12 of Sharpe.
#   The research engine sizes at one close and earns from the next; live,
#   the signal is computed after a close and can only trade at the next one,
#   so the first day of every rebalance is earned on the old position. Why
#   one day in twenty-one costs this much was not decomposed further; a
#   two-day lag is the check that it is a lag effect and not a bug.
# * **Financing** is the next largest, around 0.8% a year here: the CFDs
#   (the Nikkei, the three currencies, the two coins) pay the reference rate
#   plus a spread on their full notional every night. The dated contracts pay
#   none. It is the reason the equity leg of the stack must never sit in a
#   CFD.
# * **Whole lots** cost almost nothing even at $100k, because the broker's
#   minimum lots are small. (The simulator rounds to nearest; the EA floors.
#   That shortcut is commented in the code and makes these lines slightly
#   optimistic.)
#
# From research net 0.63 to a $100k account 0.44: that gap, mostly the lag,
# is the realistic haircut, before any selection premium (Chapter 4).
#
# **Never compute a lot value by hand.** `lot_value_usd` converts each
# instrument's quote currency. Multiplying the broker's price by its
# contract size without that conversion was once off by a factor of 158 on
# the yen-quoted instruments.

# %%
lot_value_usd(LIVE)[SYMS].iloc[-1].round(0).rename("USD per lot, last day").to_frame().T

# %% [markdown]
# ## 2. Is the live engine the same strategy?
#
# The live engine runs inside the trading terminal with no Python at run
# time, so `mql5/CTAEngine.mqh` is a second implementation of
# `build_trend_nav`: barbell signal, both volatility tiers, HRP with its own
# clustering, and the regime gate with the portable k-means (a xorshift
# random generator written identically in both languages, so the clustering
# is the same on the same data).
#
# The harness (`live/port_check.py`): export the prices and the Python
# engine's daily leverage, regime multiplier and held weights; run
# `mql5/CTAEngineTest.mq5` in the terminal on the same prices; compare every
# day and every column. Below, the files from the verification run.

# %%
import os
from live.port_check import compare
PORT = os.environ.get("CTA_PORT_DIR", "../port_check")
if os.path.exists(os.path.join(PORT, "cta_port_mql5.csv")):
    compare(PORT, tol=1e-6)
else:
    print("no port-check files here: run live.port_check export, then CTAEngineTest.mq5")

# %% [markdown]
# The regime multiplier matches on every day, and leverage and held weights
# agree to within 2e-8, which is floating-point summation order. A
# tolerance of exactly zero is not achievable between two languages that sum
# in different orders; 1e-6 is far below one lot of any market.
#
# **What this run does NOT cover.** It is the 13-market, four-bucket book
# (it still has the Gilt, dropped when the broker rejected every order for
# it, and no crypto). The live book is 14 markets in five buckets, so the
# verification must be redone for that shape before it can be called
# verified. That is an open item, and it is stated rather than assumed.
#
# ## 3. Marking a book from its deals
#
# The account's equity is rebuilt from the deal history alone, using
# average-cost accounting in deal order. A small synthetic history shows the
# four cases `position_book_daily` has to get right:

# %%
deals = pd.DataFrame({
    "time": pd.to_datetime(["2026-09-15 02:00", "2026-09-16 02:00", "2026-09-17 02:00",
                            "2026-09-18 02:00", "2026-09-21 02:00"]),
    "symbol": ["GOLD"] * 5,
    "side": ["BUY", "BUY", "SELL", "SELL", "BUY"],
    "lots": [1.0, 1.0, 1.0, 3.0, 2.0],
    "price": [2000.0, 2100.0, 2200.0, 2150.0, 2100.0]})
lots, entry = position_book_daily(deals, end="2026-09-21")
pd.DataFrame({"lots": lots["GOLD"], "avg_entry": entry["GOLD"]}).assign(
    case=["open", "add: entry blends", "reduce: entry unchanged",
          "flip through flat: entry resets", "", "", "close to flat: no entry"])

# %% [markdown]
# Floating P&L is then lots times (now minus entry) times the USD value of a
# point, with one subtlety: an FX lot's P&L lands in the QUOTE currency, so
# a USD/JPY position must be converted from yen. Marking FX on the base side
# was a live bug (bug-fix log, X13 and X14).
#
# ## 4. The daily reconciliation
#
# `live/reconcile.py` runs read-only against the terminal once a day and
# writes a ledger. Each check compares the account to what the model says
# should have happened:
#
# | check | compares | tolerance |
# |---|---|---|
# | fills | each fill against the one-minute mid at fill time | model half-spread plus slippage plus 5 bps |
# | financing | swap charged each month against the swap table | 10% of the model |
# | rolls | each dated contract rolled inside its window | the EA's roll days plus 2 |
# | divergence | held lots against the engine's targets after a rebalance | one minimum lot |
# | staleness | the price file's last date against the terminal's | the EA's maximum data age |
#
# Why it exists: it was built and then not run, and ten days of frozen
# prices and a silently dropped silver position were found by eye instead.
# A check that is not scheduled is not a check.
#
# ## What this chapter established
#
# * Most of the gap between the backtest and a live account is the one-day
#   execution lag, then financing; lot rounding is negligible at the
#   broker's lot sizes.
# * The live engine reproduces the research engine to floating-point
#   precision on the shape it was verified on; the live shape still needs the
#   same verification.
# * The live book is marked and reconciled from the broker's own records,
#   every day, by code rather than by eye.
