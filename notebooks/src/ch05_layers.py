# %% [markdown]
# # Chapter 5. The layers, and why most were rejected
#
# **Question.** The bare engine decides direction by trend and size by
# volatility. Can an extra layer improve how big each position is allowed
# to be?
#
# Every layer here changes SIZE, never direction. Each was tested against
# the same baseline, at matched realised volatility where size differs, per
# two-year window, and by the nested test (Chapter 3).
#
# **Covers:**
# rebalancing frequency; HRP; Economic Trend; the ATR trailing stop; the
# Efficiency Ratio gate; the VIX gate; the COT throttle; fast and slow as
# separate sleeves.
#
# **Configuration:** A-12, monthly, net of costs, 12% target, Sharpe after
# the risk-free rate, out of sample from 2008-09-04.

# %%
from _setup import *
from cta.walkforward import two_year_windows, nested
from ctadata.macro import load_econ, load_eer, load_econ_intl, load_cot
from ctadata.rates import load_vix
from cta.signals import _efficiency_ratio

UNI = buckets(A12)
NET_M = dict(MONTHLY, apply_costs=True)
WINDOWS = two_year_windows(df.index, OOS)


def run(**kw):
    return returns(build_trend_nav(df, symbols=A12, **{**NET_M, **kw})[0])


def verdict(name, r, base):
    """One row: full-sample Sharpe change, windows won, nested chained change."""
    won = sum(sharpe(r.loc[a:b], RF) > sharpe(base.loc[a:b], RF) for a, b in WINDOWS)
    _, chain, bchain = nested({"baseline": base, name: r}, WINDOWS, OOS, rf=RF, base="baseline")
    return dict(sharpe=sharpe(oos(r), RF), d_sharpe=sharpe(oos(r), RF) - sharpe(oos(base), RF),
                vol=vol(oos(r)), max_dd=max_drawdown(oos(r)),
                windows_won=f"{won}/{len(WINDOWS)}",
                nested_minus_base=sharpe(chain, RF) - sharpe(bchain, RF))

base = run()
describe({"A-12 baseline": base}, stacked=True)

# %% [markdown]
# ## 1. Weekly or monthly
#
# The signal is a sign of two lookback returns, so it changes little from
# week to week; weekly rebalancing mostly re-trades noise in the volatility
# estimates. Monthly captures the same signal for less cost.

# %%
weekly = returns(build_trend_nav(df, symbols=A12, apply_costs=True, **DEFAULT_PARAMS)[0])
gross_w = returns(build_trend_nav(df, symbols=A12, **DEFAULT_PARAMS)[0])
gross_m = returns(build_trend_nav(df, symbols=A12, **MONTHLY)[0])
rows = {}
for a, b in WINDOWS:
    rows[f"{a.year}-{b.year}"] = dict(
        weekly_net=sharpe(weekly.loc[a:b], RF), monthly_net=sharpe(base.loc[a:b], RF),
        weekly_cost_pts=100 * (cagr(gross_w.loc[a:b]) - cagr(weekly.loc[a:b])),
        monthly_cost_pts=100 * (cagr(gross_m.loc[a:b]) - cagr(base.loc[a:b])))
t = pd.DataFrame(rows).T
print(f"monthly cheaper in {(t.monthly_cost_pts < t.weekly_cost_pts).sum()} of {len(t)} windows")
t

# %% [markdown]
# **Signal-only rebalancing** was also tried: enter when a market's score
# changes and then never resize. Without periodic resizing, positions drift
# with price, realised volatility rises above the calendar book's, and the
# risk-adjusted return falls sharply: the calendar resize is part of the
# risk control, not just a cost.

# %%
signal_only = run(rebalance_mode="signal")
pd.DataFrame({"monthly calendar": stats(oos(base), RF), "signal only": stats(oos(signal_only), RF)}).T

# %% [markdown]
# ## 2. Hierarchical Risk Parity, within each asset class
#
# Inverse-volatility sizing stops one market dominating, but it cannot see
# that gold, silver and copper often move as one. HRP clusters each class by
# correlation and splits that class's risk down the tree, so correlated
# markets share an allocation. The scale is normalised so a class where HRP
# lands on equal weights is left unchanged. (Implementation: `cta/hrp.py`.)

# %%
hrp = run(hrp=True, universe=UNI)
pd.DataFrame({"HRP": verdict("HRP", hrp, base)}).T

# %% [markdown]
# HRP was kept. Two bugs were found on the way: it silently returned equal
# weights on every call (a scipy change broke `pypfopt`, and a broad `except`
# swallowed the error), and once fixed it was live only on rebalance days
# because its scale frame was pre-filled with 1.0. Both are pinned by tests.
#
# ## 3. Economic Trend
#
# AQR's "Economic Trend": five macro themes (growth, inflation, monetary
# policy, risk aversion, trade), each with a sign per asset class. When the
# fundamentals agree with the price trend, size up to 1.5x; when they
# disagree, down to 0.5x. Continuous and bounded, deliberately never a hard
# gate.

# %%
ECON = dict(economic_trend=True, universe=UNI, econ=load_econ(), eer=load_eer(),
            econ_intl=load_econ_intl())
econ = run(**ECON)
econ_hrp = run(hrp=True, **ECON)
pd.DataFrame({"Economic Trend": verdict("econ", econ, base),
              "HRP + Economic Trend": verdict("hrp+econ", econ_hrp, base)}).T

# %% [markdown]
# Alone it hurts. Combined with HRP it looks positive, but less so than HRP
# on its own: HRP supplies the within-class differentiation that Economic
# Trend structurally cannot.
#
# Why it fails on this book: macro data is country-level, so every market
# without its own country (all five commodities) falls back to US data and
# receives the SAME signal. The layer cannot tell gold from oil.

# %%
from cta.layers import economic_trend_score
sc = economic_trend_score(df, A12, UNI, ECON["econ"], ECON["eer"], ECON["econ_intl"])
comm = UNI["commodities"]
print("pairwise correlation of the Economic Trend score across the five commodities:")
sc[comm].corr().round(4)

# %% [markdown]
# ## 4. The ATR trailing stop
#
# Exit a position when it falls more than 3 ATRs from its best price since
# entry, checked daily. The first version scored a Sharpe of 2.5: the stop
# was checked against today's close and the position zeroed before today's
# return was booked, so the book never suffered the loss that triggered the
# stop. Fixed by booking today's return first (pinned by a test).

# %%
atr = run(trailing_stop=True, atr_window=14, atr_multiplier=3.0)
pd.DataFrame({"ATR 3x": verdict("atr", atr, base)}).T

# %% [markdown]
# Rejected at this width; Chapter 7 re-derives the width against the final
# configuration and finds that no width helps.
#
# ## 5. The Efficiency Ratio gate
#
# Flatten any market whose 40-day efficiency ratio (net move over total
# path) is below 0.3, keeping only "clean" trends.

# %%
er = run(er_gate=True, er_window=40, er_threshold=0.3)
print("share of days each market qualifies as trending:")
print((_efficiency_ratio(df[A12], 40) > 0.3).mean().round(3).to_string())
pd.DataFrame({"ER gate": verdict("er", er, base)}).T

# %% [markdown]
# Catastrophic, and for a structural reason. Each market qualifies only a
# few percent of days, so the book is mostly empty; Tier-2 leverage is sized
# off the near-zero volatility of an empty book; and when a few markets do
# qualify together they receive maximum leverage. This gate is also what
# exposed the Gilt roll bug in Chapter 2: it concentrated the book into the
# Gilt on the day of the unadjusted roll.
#
# ## 6. The VIX gate and the COT throttle

# %%
vix = run(vix=load_vix())
cot = run(cot=load_cot())
pd.DataFrame({"VIX gate": verdict("vix", vix, base), "COT throttle": verdict("cot", cot, base)}).T

# %% [markdown]
# * **VIX gate**: cut the portfolio target when the VIX spikes. Harmful, and
#   for an instructive reason: this sleeve's best periods ARE the volatility
#   spikes (2008, 2020, 2022). Cutting risk when fear rises cuts it exactly
#   where the sleeve earns. Its sizing already responds to volatility, market
#   by market, so the gate also duplicates work.
# * **COT throttle**: cut a position when speculators are crowded on the same
#   side (CFTC data, z-scored, lagged three days for publication). CFTC data
#   covers few of the markets, so the layer is mostly inactive.
#
# ## 7. Fast and slow as separate sleeves
#
# When the two legs disagree the blended score goes flat. The alternative:
# run each horizon as its own sleeve at half the risk budget, and add them.

# %%
from cta import build_fast_slow_nav
fs_nav, _, _ = build_fast_slow_nav(df, symbols=A12, **NET_M)
fs = returns(fs_nav)
pd.DataFrame({"fast + slow sleeves": verdict("fast/slow", fs, base)}).T

# %% [markdown]
# ## Summary

# %%
summary = pd.DataFrame({
    "HRP": verdict("x", hrp, base), "Economic Trend": verdict("x", econ, base),
    "HRP + Econ": verdict("x", econ_hrp, base), "ATR 3x": verdict("x", atr, base),
    "ER gate": verdict("x", er, base), "VIX gate": verdict("x", vix, base),
    "COT": verdict("x", cot, base), "fast/slow": verdict("x", fs, base),
    "signal-only": verdict("x", signal_only, base)}).T
summary.sort_values("d_sharpe", ascending=False)

# %% [markdown]
# Only HRP survives as a layer. The rest either do nothing measurable or do
# harm, and the pattern behind the harmful ones is the same: they cut risk
# at the moments a trend follower earns, or they concentrate the book.
# The next chapter takes up the one family of ideas that remained: deciding
# the overall size of the book by the market regime.
