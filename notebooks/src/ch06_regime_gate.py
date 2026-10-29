# %% [markdown]
# # Chapter 6. The regime gate
#
# **Question.** Can the overall size of the book be set by recognising the
# market's state, running smaller in regimes where the sleeve has
# historically lost?
#
# **Covers** (original notebooks v2/05, v2/06, v2/07):
# 1. how the gate works: features, k-means on an expanding window, learned
#    multipliers;
# 2. the number of regimes, and the elbow rule ("auto-k");
# 3. WK-means, the clever alternative that lost, and why;
# 4. **the comparison error**: judging two books at different volatility;
# 5. the nested test over gate variants, and the decision.
#
# **Configuration:** A-12, monthly, net, HRP on, Sharpe after the risk-free
# rate, out of sample from 2008-09-04.

# %%
from _setup import *
from cta.walkforward import two_year_windows, nested
from cta.regime import _regime_features, _REGIME_CACHE

UNI = buckets(A12)
BASE = dict(MONTHLY, apply_costs=True, hrp=True, universe=UNI)
LEARN = dict(regime_gate=True, regime_mult_mode="learned")
TREND5 = ("vol", "corr", "er", "disp", "agree")
WINDOWS = two_year_windows(df.index, OOS)


def run(target=0.12, **kw):
    out = build_trend_nav(df, symbols=A12, **{**BASE, "base_target_vol": target, **kw})
    return returns(out[0])

# %% [markdown]
# ## 1. How the gate works
#
# Each day is described by five statistics of the whole book, all lagged a
# day: volatility, average correlation between markets, the efficiency ratio
# (trend quality), cross-sectional dispersion, and **agreement**, the share
# of markets whose fast and slow legs point the same way.
#
# k-means groups days into regimes on an EXPANDING window, refitting about
# once a year; each new day is classified by the latest fit, never by a fit
# that saw it. Each regime gets a multiplier on the portfolio volatility
# target, from 1.0 down to 0.5.
#
# **Learned, not assumed.** A long-only intuition says violent regimes are
# the dangerous ones. For a trend follower that is backwards: by the time a
# crash is recognisable the book is usually already short, so the "crisis"
# cluster tends to PAY. The gate therefore ranks regimes by what the sleeve
# actually earned over the following 21 days, counting only windows that had
# closed before the decision date, and gives the best regime 1.0.

# %%
feats = _regime_features(df, A12)[list(TREND5)]
feats.loc[OOS:].describe().T[["mean", "std", "min", "max"]]

# %%
gate = run(**LEARN, regime_features=TREND5, regime_auto_k=(2, 10))
mult = list(_REGIME_CACHE.values())[-1]
fig, ax = plt.subplots()
ax.plot(mult.loc[OOS:], lw=0.8)
ax.set_title("Regime multiplier on the portfolio volatility target (auto-k, learned)")
plt.show()
print("share of days at each multiplier:")
print(mult.loc[OOS:].round(2).value_counts(normalize=True).sort_index().round(3).to_string())

# %% [markdown]
# ## 2. How many regimes? The elbow rule
#
# The first version fixed k = 3, inherited from a proposal and never tested.
# The elbow rule chooses k at every refit from the shape of the clustering
# alone: fit k = 2..10, draw a line from the first point of the inertia curve
# to the last, take the k furthest below it. No return, Sharpe or drawdown
# enters the choice, so it cannot be tuned to performance.

# %%
rows = {"no gate": run()}
for k in (2, 3, 5, 7):
    rows[f"k={k}"] = run(**LEARN, regime_features=TREND5, regime_clusters=k)
rows["auto-k"] = gate
describe(rows, stacked=True)

# %% [markdown]
# The same pattern at every k: the gate cuts the sleeve's worst loss, and
# at a common 12% target its Sharpe is lower at every k (section 4 explains
# why a common target is the wrong comparison). A
# single fixed k that happens to look best (k=7 did, in the original run) is
# one value out of seven tried: a cherry pick. The best fixed k measured
# window by window moved between several values, so no fixed choice was ever
# available in advance; auto-k removes the guess.
#
# ## 3. WK-means: better labels, worse trading
#
# Wasserstein k-means (Horvath, Issa and Muguruza, 2021) clusters the whole
# return DISTRIBUTION of each window instead of hand-picked features.
# Nothing is hand-picked; volatility, skew and tails all enter. It produced
# the most credible regime labels of any method tried.

# %%
wk = run(**LEARN, regime_method="wasserstein")
describe({"no gate": rows["no gate"], "feature k-means, auto-k": gate,
          "WK-means": wk}, stacked=True)

# %% [markdown]
# Against no gate it loses, and it is the only variant that deepens the
# STACKED book's worst loss: it cuts risk at the moments the sleeve should be
# cushioning the equity position. (On A-12 it happens to score above the
# feature gate on Sharpe; on A-13, where the decision was made, it was worse
# on every measure.) The reason it cuts at the wrong moments is structural.
# WK-means represents a window as a sorted list of returns. Sorting is exactly what makes the distance cheap,
# and sorting **destroys time order**:

# %%
rng = np.random.default_rng(1)
up = np.r_[np.full(31, 0.01), np.full(32, -0.01)]          # a clean rise, then a clean fall
shuffled = rng.permutation(up)                               # the same days, as noise
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(np.cumprod(1 + up), label="trend then reversal")
ax[0].plot(np.cumprod(1 + shuffled), label="same returns, shuffled"); ax[0].legend()
ax[0].set_title("Two paths a trend follower treats very differently")
ax[1].plot(np.sort(up), "o", ms=3); ax[1].set_title("What WK-means sees: one identical sorted list")
plt.show()
print("WK-means distance between the two windows:", float(np.abs(np.sort(up) - np.sort(shuffled)).mean()))

# %% [markdown]
# The feature that does most of the work in the feature-based gate,
# agreement between the fast and slow legs, is a path statistic, and so is
# the efficiency ratio. A distribution-based method cannot see either.
#
# ## 4. The comparison error
#
# At a common 12% target, the gate looked like it cost return. That
# conclusion stood for a day. The gate runs the book SMALLER, and scaling a
# strategy up or down does not change its return per unit of risk. Two books
# at different volatility cannot be compared directly.

# %%
pair = {"no gate, 12%": rows["no gate"], "gate, 12%": gate}
t = describe(pair)
print(t[["vol", "cagr", "max_dd"]])
scaled = matched(gate, vol(oos(rows["no gate"])))
describe({"no gate, 12%": rows["no gate"], "gate, scaled to the same realised vol": scaled})

# %% [markdown]
# Scaling to the same realised risk removes most of the apparent cost:
# comparing at different volatility had made the gate look far worse than it
# is. To compare properly across the whole defensible range, run both at
# targets from 12% to 18% and plot return against drawdown.

# %%
front = []
for tv in (0.12, 0.13, 0.14, 0.16, 0.18):
    for label, kw in (("no gate", {}), ("gate", dict(LEARN, regime_features=TREND5, regime_auto_k=(2, 10)))):
        r = run(target=tv, **kw)
        front.append(dict(gate=label, target=tv, vol=vol(oos(r)), cagr=cagr(oos(r)),
                          max_dd=max_drawdown(oos(r)),
                          stacked_cagr=cagr(oos(r + SPX)), stacked_dd=max_drawdown(oos(r + SPX))))
front = pd.DataFrame(front)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for g, sub in front.groupby("gate"):
    ax[0].plot(-sub.max_dd, sub.cagr, "o-", label=g)
    ax[1].plot(-sub.stacked_dd, sub.stacked_cagr, "o-", label=g)
ax[0].set_title("Sleeve: return against worst loss"); ax[1].set_title("Stacked on the S&P")
for a in ax:
    a.set_xlabel("worst loss"); a.set_ylabel("CAGR"); a.legend()
plt.show()
front.set_index(["gate", "target"])

# %% [markdown]
# On A-12, today, the two frontiers cross: the gate gives a shallower sleeve
# drawdown, but at the same realised risk it earns less, and the stacked
# book barely benefits.
#
# **That is not what the decision was based on.** The gate was chosen on
# 11 September on A-13, the same markets plus the Gilt. Re-running the same
# matched-volatility test on both universes shows what changed:
#
# %%
A13 = sorted(A12 + ["gilt"])
rows13 = []
for name, syms in (("A-13 (with the Gilt)", A13), ("A-12 (without)", A12)):
    kw = dict(MONTHLY, apply_costs=True, hrp=True, universe=buckets(syms))
    ng = returns(build_trend_nav(df, symbols=syms, **kw)[0])
    g = returns(build_trend_nav(df, symbols=syms, **{**kw, **GATE, "base_target_vol": 0.13, "hrp_lag": 1})[0])
    gm = matched(oos(g), vol(oos(ng)))
    rows13.append(dict(universe=name, no_gate_sharpe=sharpe(oos(ng), RF), gate_sharpe=sharpe(oos(g), RF),
                       cagr_gain_at_matched_vol=cagr(gm) - cagr(oos(ng)),
                       dd_gain_at_matched_vol=max_drawdown(gm) - max_drawdown(oos(ng))))
pd.DataFrame(rows13).set_index("universe")

# %% [markdown]
# On A-13 the gate wins at matched risk on both return and drawdown, which
# reproduces the original decision. On A-12 it does not. Removing one bond
# market, the only other fixed-income market besides the US 10-year, changed
# the verdict, and the gate was never re-tested when the Gilt was dropped on
# 24 September. Two things follow:
#
# * The gate stays in the live configuration, because the project's rule is
#   not to tune it on past performance, and because with crypto in the book
#   it helps again (Chapter 8).
# * It is a clear open item: the gate's justification rests on a universe
#   the live book no longer trades.
#
# Either way it is insurance, not an engine: where it helps, the benefit
# comes from a few crisis windows.

# ## 5. Can the gate be tuned? The nested test
#
# Ten plausible gate variants (feature sets, fixed or auto k, forward
# windows, WK-means). Choose among them on training data only, judge on the
# next window, chain.

# %%
ORIG, RICH = ("vol", "corr", "er"), ("vol", "corr", "er", "dist_ma", "dd", "vol_ratio",
                                    "chg", "disp", "agree", "rsi")
CANDIDATES = {
    "orig3 k=3": dict(**LEARN, regime_features=ORIG),
    "orig3 auto-k": dict(**LEARN, regime_features=ORIG, regime_auto_k=(2, 10)),
    "trend5 k=3": dict(**LEARN, regime_features=TREND5),
    "trend5 auto-k": dict(**LEARN, regime_features=TREND5, regime_auto_k=(2, 10)),
    "rich10 auto-k": dict(**LEARN, regime_features=RICH, regime_auto_k=(2, 10)),
    "min2 auto-k": dict(**LEARN, regime_features=("vol", "agree"), regime_auto_k=(2, 10)),
    "path3 auto-k": dict(**LEARN, regime_features=("er", "agree", "chg"), regime_auto_k=(2, 10)),
    "trend5 auto-k fwd10": dict(**LEARN, regime_features=TREND5, regime_auto_k=(2, 10), regime_fwd_window=10),
    "trend5 auto-k fwd42": dict(**LEARN, regime_features=TREND5, regime_auto_k=(2, 10), regime_fwd_window=42),
    "wasserstein k=3": dict(**LEARN, regime_method="wasserstein"),
}
cand = {k: run(**v) for k, v in CANDIDATES.items()}
picks, chain, _ = nested(cand, WINDOWS, OOS, rf=RF)
nogate_chain = pd.concat([rows["no gate"].loc[a:b] for a, b in WINDOWS])
nogate_chain = nogate_chain[~nogate_chain.index.duplicated()]
print("chosen per window:", picks["chose"].tolist())
pd.DataFrame({"gate chosen per window": stats(chain, RF), "no gate": stats(nogate_chain, RF)}).T

# %% [markdown]
# Choosing among gate variants on past performance does worse than running
# no gate at all (originally 0.345 against 0.410). The variant that looks
# best with hindsight cannot be picked in advance, and trying costs more than
# the gate is worth.
#
# **Decision (11 September, on A-13):** one variant, fixed a priori for its
# drawdown behaviour (five features, auto-k, learned ranking), never tuned. The live engine
# also uses a portable k-means (k-means++ on a small xorshift generator that
# MQL5 can reproduce exactly) and a one-day HRP lag; together they move the
# Sharpe by 0.0002.

# %%
describe({"no gate, 12%": rows["no gate"], "chosen: gate, 13%, portable": chosen()}, stacked=True)
