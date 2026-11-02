# %% [markdown]
# # Chapter 7. The final configuration, and how it compares
#
# **Covers:**
# 1. the configuration and its headline;
# 2. where the return comes from;
# 3. the stop-loss, re-derived against this configuration;
# 4. four ideas tested after the configuration was fixed;
# 5. managed-futures benchmarks, compared fairly;
# 6. the return-stacked ETF this project set out to replicate.
#
# ```python
# build_trend_nav(df, symbols=A12, apply_costs=True,
#                 hrp=True, universe=buckets(A12), hrp_lag=1,
#                 rebalance_freq="M", base_target_vol=0.13,
#                 regime_gate=True, regime_mult_mode="learned",
#                 regime_features=("vol", "corr", "er", "disp", "agree"),
#                 regime_auto_k=(2, 10), regime_kmeans="portable")
# ```

# %%
from _setup import *
from cta.walkforward import two_year_windows, nested
from cta.engine import LAST_STOP_STATS
from ctadata.paths import data_dir
from ctadata.benchmarks import load_mbxax, load_nilssonhedge_cta, load_sg_trend_index, load_rsst

UNI = buckets(A12)
KW = {**DEFAULT_PARAMS, **CHOSEN, "universe": UNI}
WINDOWS = two_year_windows(df.index, OOS)
r = chosen()
describe({"S&P alone": SPX, "sleeve (chosen)": r, "S&P + sleeve": SPX + r})

# %% [markdown]
# The sleeve on its own is a modest return stream and was never meant to be
# held alone. Stacked on the S&P it raises the return of the combined book.
# Remember Chapter 4: this headline carries a selection premium of about
# 0.19 Sharpe, so a live expectation is nearer 0.35 than the figure above.
#
# ## 2. Where the return comes from
#
# Attribution is recorded inside the engine's loop (`collect_contrib=True`),
# from the same quantities that form each day's return, so it reconciles to
# the NAV exactly. Computing it from the returned weights would be wrong:
# those are TARGET weights, not the positions actually held.

# %%
nav, lev, w, contrib, cost = build_trend_nav(df, symbols=A12, collect_contrib=True, **KW)
err = (contrib.sum(axis=1) - cost.sum(axis=1) - returns(nav)).abs().max()
print(f"attribution reconciles to the NAV within {err:.1e}")
net = (contrib - cost).loc[OOS:]
by_mkt = net.sum().sort_values()
cls_of = {s: k for k, v in UNI.items() for s in v}
by_cls = by_mkt.groupby(cls_of).sum()
pd.DataFrame({"contribution, % of NAV (summed)": by_cls * 100}).round(2)

# %%
by_mkt.mul(100).round(2).rename("% of NAV").to_frame().T

# %% [markdown]
# Commodities, gold and copper above all, carry most of the return. Fixed
# income (one market) contributes least and sterling is a net loser over the
# period. Neither is a reason to cut: fixed income is a diversifier rather
# than a return source (in the original analysis, removing it deepened the
# drawdown in every configuration tested), and cutting the losers is exactly
# the selection-on-the-sample error of Chapter 4.

# %%
yearly = r.loc[OOS:].groupby(r.loc[OOS:].index.year).apply(lambda s: (1 + s).prod() - 1)
spx_y = SPX.loc[OOS:].groupby(SPX.loc[OOS:].index.year).apply(lambda s: (1 + s).prod() - 1)
pd.DataFrame({"sleeve": yearly, "S&P": spx_y}).mul(100).round(1).T

# %% [markdown]
# ## 3. The stop-loss, re-derived against this configuration
#
# The 3x ATR stop in Chapter 5 was tested against the plain sleeve. But a
# stop and the regime gate are substitutes, both cut risk, so the width has
# to be re-derived against the configuration actually run. Seven widths, the
# exit charged at the cost table, and two re-entry rules: wait for the next
# rebalance, or re-enter the day the score changes. Results at matched
# volatility, since a stop changes the book's size.

# %%
stops = {"no stop": r}
fires = {}
for m in (3, 5, 8, 12):
    stops[f"{m}x, wait"] = returns(build_trend_nav(df, symbols=A12, trailing_stop=True,
                                                   atr_multiplier=m, stop_cost=True, **KW)[0])
    fires[f"{m}x, wait"] = LAST_STOP_STATS.get("fires", 0)
stops["3x, re-enter on signal"] = returns(build_trend_nav(df, symbols=A12, trailing_stop=True,
                                                          atr_multiplier=3, stop_cost=True,
                                                          stop_reentry="signal", **KW)[0])
years = len(r.loc[OOS:]) / 252
t = describe(stops, match_to="no stop")
t["fires_per_year"] = pd.Series({k: v / years for k, v in fires.items()})
t

# %%
picks, chain, base_chain = nested(stops, WINDOWS, OOS, rf=RF, base="no stop")
print("width chosen per window:", picks.chose.tolist())
pd.DataFrame({"width picked per window": stats(chain, RF), "no stop": stats(base_chain, RF)}).T

# %% [markdown]
# No width meaningfully beats no stop. Tight stops lose clearly; wide ones
# (8x, 12x) fire a few times a year and land within noise of no stop, one a
# hair above and one below. Re-entering on a new signal is worse at the same
# width: a stop fires on a pullback while the trend is intact, the score has
# not changed, so the re-entry buys the reversal. Picking the width per
# window mostly picks the tightest, and loses.
#
# The book already controls risk three ways (each position sized to its own
# volatility, the book sized to a target, the regime gate). A per-position
# stop on top removes positions in intact trends. **No stop in the strategy.**
# The live account carries a 10x daily-ATR rail on the broker's server, which
# fires a few times a year and costs nothing measurable; it is there for what
# a backtest cannot see (a weekend gap, the terminal down).
#
# ## 4. Four ideas tested after the configuration was fixed
#
# Each reshapes the existing signal rather than adding information.

# %%
late = {
    "chosen (sign)": r,
    "sized signal, continuous": chosen(score_mode="continuous"),
    "sized signal, response": chosen(score_mode="response"),
    "rank regimes by drawdown": chosen(regime_rank_by="drawdown"),
    "no-trade band 20%, edge": chosen(rebalance_band=0.2, band_style="edge",
                                      band_relative=True, band_on_rebal_only=True),
    "hold half size on disagreement": chosen(score_mode="dominant"),
}
rows = {}
for k, v in late.items():
    if k == "chosen (sign)":
        continue
    won = sum(sharpe(v.loc[a:b], RF) > sharpe(r.loc[a:b], RF) for a, b in WINDOWS)
    _, ch, bch = nested({"chosen": r, k: v}, WINDOWS, OOS, rf=RF, base="chosen")
    rows[k] = dict(d_sharpe=sharpe(oos(v), RF) - sharpe(oos(r), RF),
                   windows_won=f"{won}/{len(WINDOWS)}",
                   nested_minus_chosen=sharpe(ch, RF) - sharpe(bch, RF))
pd.DataFrame(rows).T

# %% [markdown]
# All inside the band every rejected layer sits in. Holding a half position
# when the two legs disagree is the clearest case: disagreement usually marks
# a turning point, and going flat there is the useful part of the signal. The
# space of "same signal, different shape" is flat on this book.
#
# ## 5. Against managed-futures benchmarks
#
# Two fairness rules, both learned the hard way:
# * **sleeve against sleeve.** A stacked book runs about 200% exposure; a
#   managed-futures fund does not. The first benchmark comparison set the
#   stacked book against unstacked funds and was withdrawn.
# * **at the benchmark's own volatility.** Managers do not publish their
#   targets, so the sleeve is scaled to each benchmark's realised volatility
#   over the benchmark's own window.

# %%
def vs(bench_ret, name):
    b = bench_ret.dropna()
    b.index = pd.to_datetime(b.index).tz_localize(None) if getattr(b.index, "tz", None) else pd.to_datetime(b.index)
    s = r.copy(); s.index = s.index.tz_localize(None)
    idx = b.index.intersection(s.index)
    b, s = b.loc[idx], s.loc[idx]
    rf = RF.copy(); rf.index = rf.index.tz_localize(None)
    sm = matched(s, vol(b))
    return {f"{name} ({idx.min():%Y-%m} on)": stats(b, rf),
            f"sleeve at {name}'s vol": stats(sm, rf)}

etf = pd.read_parquet(data_dir() / "cta_etfs_yahoo.parquet")
rows = {}
rows.update(vs(load_mbxax().pct_change(), "MBXAX"))
rows.update(vs(load_nilssonhedge_cta(), "Nilsson Hedge CTA"))
rows.update(vs(load_sg_trend_index().pct_change(), "SG Trend Indicator"))
for t in ("DBMF", "KMLM"):
    rows.update(vs(etf[(t, "adjclose")].pct_change(), t))
pd.DataFrame(rows).T

# %% [markdown]
# At each benchmark's own volatility the sleeve earns more than three of the
# five (MBXAX, the SG Trend Indicator, KMLM) and less than two (DBMF and the
# Nilsson Hedge composite), and its worst loss is shallower than all five.
# Theirs are net of fees on real fills; this is a backtest at closing prices. Neither
# direction of that comparison is the pitch. The pitch is the stack.
#
# ## 6. Against the return-stacked ETF
#
# RSST holds 100% US equities plus 100% trend-following managed futures:
# the same structure as the stacked book, so here the stacked comparison IS
# fair. Its series is price only (distributions not reinvested), which
# understates it by perhaps a point a year; ours charges no management fee.
# A futures-based book also earns nothing on its idle collateral, so a second
# line adds the T-bill rate.

# %%
rsst = load_rsst().pct_change().dropna()
idx = rsst.index.intersection(r.index)
ours = (SPX + r).loc[idx]
funded = ours + RF.loc[idx]
cmp_ = {"RSST (price only, fee inside)": rsst.loc[idx], "ours, stacked": ours,
        "ours, stacked + T-bills on collateral": funded}
print(f"window {idx.min().date()} to {idx.max().date()}, daily correlation "
      f"{rsst.loc[idx].corr(ours):.2f}")
pd.DataFrame({k: stats(v, RF) for k, v in cmp_.items()}).T

# %%
fig, ax = plt.subplots()
for k, v in cmp_.items():
    ax.plot((1 + v).cumprod(), label=k)
ax.set_title("RSST against the stacked book over RSST's life"); ax.legend(); plt.show()

# %% [markdown]
# The two track each other closely until April 2025, when RSST's trend book
# was long equities into the tariff sell-off. Most of the gap is that one
# episode, which is how the gate is supposed to work (insurance that pays
# occasionally), and one episode is not a track record.
#
# **How the equity leg is held matters as much as any layer.** With the
# sleeve identical, holding the S&P as a CFD (paying the broker's financing
# on the full notional every night), as a future, or as the ETF in cash
# differs by several points a year, purely from financing compounding. The
# live demo holds the dated S&P future, which is what the backtest models;
# for real money the equity leg belongs in cash securities, never a CFD.
