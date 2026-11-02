# %% [markdown]
# # Chapter 3. How a result is validated
#
# **Question.** Which numbers can be trusted, and how do you tell?
#
# **Covers:**
# 1. walk-forward: re-fitting a parameter per fold, against leaving it fixed;
# 2. the risk-free rate in the Sharpe ratio;
# 3. the out-of-sample convention, and why 2008 is never out of sample;
# 4. replaying 2008 inside the out-of-sample window;
# 5. the nested test that every later decision had to pass;
# 6. how much the start date alone moves the answer.

# %%
from _setup import *
from cta.walkforward import rolling_folds, two_year_windows, nested
from ctadata import LEAN_7, SLEEVE_A_15, symbols_of

LEAN = symbols_of(LEAN_7)

# %% [markdown]
# ## 1. Re-fitting the lookbacks per fold
#
# The first walk-forward: five-year training windows, one-year test windows.
# In each fold, pick the lookback pair with the best training Sharpe from a
# small grid, trade it on the test year, and chain the test years. Compare
# with keeping the literature's 40/252 fixed throughout.
#
# The vectorised engine is used here: with no optional layers it is the same
# computation as the main loop, about 100 times faster.

# %%
GRID = [(20, 252), (40, 252), (60, 378)]
folds = rolling_folds(df, train_size=1260, test_size=252, step=252, history_buffer=400)
rows, oos_reopt = [], []
for f in folds:
    def tr_sharpe(pair):
        nav = build_trend_nav_fast(f.train, symbols=LEAN, **{**DEFAULT_PARAMS,
                                   "short_lookback": pair[0], "long_lookback": pair[1]})[0]
        return sharpe(returns(nav))
    best = max(GRID, key=tr_sharpe)
    nav = build_trend_nav_fast(f.test, symbols=LEAN, **{**DEFAULT_PARAMS,
                               "short_lookback": best[0], "long_lookback": best[1]})[0]
    seg = returns(nav).loc[f.test_range[0]:f.test_range[1]]
    oos_reopt.append(seg)
    rows.append(dict(test=f"{f.test_range[0].date()}", picked=best, oos_sharpe=sharpe(seg)))
fold_table = pd.DataFrame(rows).set_index("test")
fold_table

# %%
reopt = pd.concat(oos_reopt)
reopt = reopt[~reopt.index.duplicated()]
frozen = returns(build_trend_nav_fast(df, symbols=LEAN, **DEFAULT_PARAMS)[0]).reindex(reopt.index)
print(f"window {reopt.index.min().date()} to {reopt.index.max().date()}, gross, 0% rate")
pd.DataFrame({"re-optimised per fold": stats(reopt), "frozen 40/252": stats(frozen)}).T

# %% [markdown]
# Re-fitting per fold loses to leaving the parameters alone. The grid picks a
# different pair in many folds and the fold Sharpes swing widely: the
# signature of fitting each training window's noise. **The lookbacks are
# never re-optimised again**, and the general lesson, that a small search on
# one sample fits noise, comes back in every chapter.
#
# ## 2. The risk-free rate
#
# The common convention is Sharpe = mean / std of raw returns, a 0% rate.
# From here on every headline subtracts the actual US short rate. It lowers
# every Sharpe by a roughly constant amount and changes no ranking.

# %%
r_a = returns(build_trend_nav(df, symbols=symbols_of(SLEEVE_A_15), **DEFAULT_PARAMS)[0])
s, st = oos(r_a), oos(r_a + SPX)
pd.DataFrame({"sleeve": [sharpe(s), sharpe(s, RF)], "stacked": [sharpe(st), sharpe(st, RF)]},
             index=["0% rate", "after the US short rate"])

# %% [markdown]
# ## 3. The out-of-sample convention
#
# The data starts in September 2006. The slow leg needs 252 days of history,
# and a training window needs another year, so results are reported from
# **2008-09-04**. Two consequences worth stating plainly:
#
# * The 2008 crisis, the strategy's best episode, is only partly inside the
#   window, and the strategy's design was motivated by knowing 2008 happened.
# * Early notebooks used a single holdout from 2011-07-20 and called it "the
#   walk-forward window". It is one split, not a rolling walk-forward, and it
#   excludes 2008 entirely. Some conclusions changed when the window changed;
#   for instance HRP looked unhelpful from 2011 and helpful from 2008.

# %%
from ctadata import buckets as _b
A15 = symbols_of(SLEEVE_A_15)
base = returns(build_trend_nav(df, symbols=A15, **DEFAULT_PARAMS)[0])
hrp = returns(build_trend_nav(df, symbols=A15, hrp=True, universe=SLEEVE_A_15, **DEFAULT_PARAMS)[0])
rows = {}
for label, start in (("from 2007-08 (1-year buffer)", df.index[252]),
                     ("from 2008-08 (2-year buffer)", df.index[504]),
                     ("from 2011-07 (5-year buffer)", df.index[1260])):
    rows[label] = dict(baseline=sharpe(base.loc[start:], RF), hrp=sharpe(hrp.loc[start:], RF))
t = pd.DataFrame(rows).T
t["hrp_minus_baseline"] = t["hrp"] - t["baseline"]
t

# %% [markdown]
# The same layer, the same data, opposite verdicts depending only on where
# scoring starts. HRP exists to reduce concentrated correlated risk, and
# correlations spike in a crisis, so its benefit shows up when a crisis is in
# the scored window. The conclusion carried forward is the narrower one: HRP
# helps when correlations spike.
#
# ## 4. 2008, replayed
#
# Because 2008 is mostly outside the out-of-sample window, the window's
# drawdowns are a floor, not a ceiling. To measure how far below the floor a
# 2008-scale event could go, each market's real returns from Lehman to the
# March 2009 low are spliced into the out-of-sample window at four dates, and
# the strategy is re-run on the altered history.

# %%
CRISIS = df.loc["2008-09-02":"2009-03-09", A12].pct_change().iloc[1:]

def splice_crisis(prices, at):
    idx = prices.index
    pos = idx.get_indexer([pd.Timestamp(at, tz="UTC")], method="nearest")[0]
    real = prices[A12].pct_change()
    out = prices.copy()
    factor = np.ones((len(idx) - pos, len(A12)))
    for i in range(1, len(idx) - pos):
        r = CRISIS.iloc[i - 1].values if i <= len(CRISIS) else real.iloc[pos + i].values
        factor[i] = factor[i - 1] * (1 + np.nan_to_num(r, nan=0.0))
    out.loc[idx[pos:], A12] = prices[A12].iloc[pos].values * factor
    return out

kw = {**DEFAULT_PARAMS, "rebalance_freq": "M"}
real = returns(build_trend_nav_fast(df, symbols=A12, **kw)[0])
rows = {"real history": dict(sleeve_dd=max_drawdown(oos(real)),
                             stacked_dd=max_drawdown(oos(real + SPX)))}
for at in ("2013-01-02", "2016-06-01", "2019-06-03", "2023-01-03"):
    alt = splice_crisis(df, at)
    r = returns(build_trend_nav_fast(alt, symbols=A12, **kw)[0])
    spx_alt = alt["spx"].pct_change().fillna(0.0)
    rows[f"2008 spliced at {at}"] = dict(sleeve_dd=max_drawdown(oos(r)),
                                         stacked_dd=max_drawdown(oos(r + spx_alt)))
pd.DataFrame(rows).T

# %% [markdown]
# A 2008-scale shock inside the window deepens the stacked drawdown well past
# anything the real out-of-sample window shows, and how much depends on
# when it lands, because the trend signal's state going into the shock
# matters: at one of the four dates it does not deepen the drawdown at all,
# at another it nearly doubles it. One historical scenario is not a worst
# case; it shows the reported drawdowns are not one either.
#
# ## 5. The nested test
#
# A full-sample comparison of two variants answers "which was better over
# this history?", which is not the question. The question is "could you have
# chosen the better one in advance?". `cta.walkforward.nested` answers it:
#
# 1. split the out-of-sample period into two-year windows;
# 2. before each window, score every variant on everything before it;
# 3. take the winner, run it on the window, never revisit the choice;
# 4. chain the windows into one track record that could have been followed.
#
# Illustrated on the lookback pair again, monthly, gross, on A-12.

# %%
variants = {f"{a}/{b}": returns(build_trend_nav_fast(df, symbols=A12, **{**kw,
            "short_lookback": a, "long_lookback": b})[0]) for a, b in GRID}
windows = two_year_windows(df.index, OOS)
picks, chain, base_chain = nested(variants, windows, OOS, rf=RF, base="40/252")
picks

# %%
pd.DataFrame({"picked per window": stats(chain, RF), "40/252 throughout": stats(base_chain, RF),
              **{f"{k} throughout": stats(pd.concat([v.loc[a:b] for a, b in windows]).pipe(
                  lambda s: s[~s.index.duplicated()]), RF) for k, v in variants.items()}}).T

# %% [markdown]
# Choosing per window ends roughly level with the baseline, and well below
# the best fixed variant, which is only knowable with hindsight. The variant
# that looks best over the whole sample cannot be picked in advance.
#
# This is the standard every later layer had to meet: beat the baseline when
# chosen this way, not merely over the full sample. Measured across the
# project, choosing in advance has cost roughly the full-sample advantage
# almost every time, and offering a wider menu of variants made the chained
# result worse twice.
#
# ## 6. How much the start date alone moves the answer
#
# The chosen configuration's Sharpe, measured from different start dates.

# %%
r_chosen = chosen()
starts = ["2008-09-04", "2009-09-04", "2010-09-04", "2011-09-04", "2012-09-04", "2013-09-04"]
pd.Series({s: sharpe(r_chosen.loc[s:], RF) for s in starts}, name="sharpe from this date").to_frame()

# %% [markdown]
# The spread across start dates is larger than almost any improvement any
# layer in this project produced. That is why full-sample results are
# treated as hypotheses, and why extending the history further back (to
# evaluate 2008 genuinely out of sample) was the top open item when research
# closed.
