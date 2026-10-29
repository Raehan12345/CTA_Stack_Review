# %% [markdown]
# # Chapter 8. Extensions: crypto, a wider universe, macro, and a class-level trend
#
# **Question.** Once the configuration was fixed, the remaining ways to
# improve it were new markets and new information. What worked?
#
# **Covers** (original notebooks v2/10, v2/11 and the two sandbox notebooks):
# 1. BTC and ETH as a fifth asset class (adopted, and live);
# 2. a 56-market universe, and the diagnosis of why it loses;
# 3. a macro filter for the wide universe's bad window (rejected);
# 4. a class-level P&L trend override (survived its nested test, not deployed).
#
# **Configuration:** the chosen configuration (HRP with a one-day lag,
# monthly, learned regime gate, 13% target), net of costs, crypto CFD
# financing charged wherever crypto is held, Sharpe after the risk-free rate.

# %%
from _setup import *
from cta.walkforward import two_year_windows
from cta.overlays import class_pnl_override
from ctadata import frame, load_crypto, tradeable_pool, CRYPTO
from live.simulator import held_weights
from live.costs import crypto_carry

PX = frame(include_crypto=True)


def book(symbols, px=PX, charge_crypto=True, contrib=False, **over):
    syms = sorted(set(symbols))
    out = build_trend_nav(px, symbols=syms, **{**DEFAULT_PARAMS, **CHOSEN,
                          "universe": buckets(syms), "collect_contrib": contrib, **over})
    r = returns(out[0])
    cy = [c for c in CRYPTO if c in syms]
    if cy and charge_crypto:
        held, _ = held_weights(out, px, "M")
        r = r - crypto_carry(held[cy], px.index)
    return (r, out) if contrib else r

# %% [markdown]
# ## 1. Crypto
#
# The broker trades BTC and ETH as rolling CFDs, so the underlying is spot,
# not the CME future. They form their own asset class, a fifth HRP bucket.
# A CFD is not a future: its financing is NOT in the price (15% a year paid on
# a long, 7.5% earned on a short), so it is charged explicitly on top of the
# usual costs. Measured from ETH's first date so both books see the same
# window.

# %%
FROM = "2018-02-08"
a12 = book(A12)
a12c = book(A12 + CRYPTO)
a12c_free = book(A12 + CRYPTO, charge_crypto=False)
cmp_ = {"A-12": a12, "A-12 + BTC + ETH, financing charged": a12c,
        "A-12 + BTC + ETH, no financing": a12c_free,
        "A-12 + BTC only": book(A12 + ["btc"]), "A-12 + ETH only": book(A12 + ["eth"])}
t = pd.DataFrame({k: stats(v.loc[FROM:], RF) for k, v in cmp_.items()}).T
t["stacked_cagr"] = [cagr((v + SPX).loc[FROM:]) for v in cmp_.values()]
t

# %%
folds = [("2018-02-08", "2020-02-08"), ("2020-02-08", "2022-02-08"),
         ("2022-02-08", "2024-02-08"), ("2024-02-08", "2026-09-28")]
pd.DataFrame({f"{a[:7]} to {b[:7]}": dict(a12=sharpe(a12.loc[a:b], RF),
              with_crypto=sharpe(a12c.loc[a:b], RF)) for a, b in folds}).T.assign(
    diff=lambda x: x.with_crypto - x.a12)

# %% [markdown]
# Every fold improves, both coins are needed, and financing costs little,
# because crypto ends up a small share of gross notional and is flat or
# short much of the time.
#
# **The correction that matters.** A first draft claimed the edge was
# two-sided. Decomposing the crypto P&L by the side of the position:

# %%
_, out = book(A12 + CRYPTO, contrib=True)
contrib, cost = out[3], out[4]
held, _ = held_weights(out, PX, "M")
rows = {}
for a, b in folds:
    cpnl = (contrib[CRYPTO] - cost[CRYPTO]).loc[a:b]
    side = np.sign(held[CRYPTO].shift(1)).loc[a:b]
    rows[f"{a[:7]} to {b[:7]}"] = dict(
        crypto_pnl_pct=100 * cpnl.sum().sum(),
        from_long_pct=100 * cpnl.where(side > 0, 0).sum().sum(),
        from_short_pct=100 * cpnl.where(side < 0, 0).sum().sum(),
        btc_days_long_pct=100 * (side["btc"] > 0).mean())
pd.DataFrame(rows).T

# %% [markdown]
# The contribution comes from the long side; the short leg has not worked in
# this sample. The sleeve is long only part of the time, so this is TIMED
# exposure to an asset that rose roughly tenfold over the window, not a
# direction-neutral edge, and it has not been tested through a long crypto
# bear market. Crypto was adopted in research and deployed to the live book
# on 25 September as a fifth bucket.
#
# **And the regime gate.** On A-12 the gate does not earn its place
# (Chapter 6). With crypto in the book, compare the gate against HRP alone:

# %%
pd.DataFrame({
    "A-12, HRP only": stats(book(A12, regime_gate=False).loc[FROM:], RF),
    "A-12, HRP + gate": stats(a12.loc[FROM:], RF),
    "A-12 + crypto, HRP only": stats(book(A12 + CRYPTO, regime_gate=False).loc[FROM:], RF),
    "A-12 + crypto, HRP + gate": stats(a12c.loc[FROM:], RF)}).T

# %% [markdown]
# The gate runs the book at lower volatility, so the CAGR column cannot be
# compared; Sharpe and drawdown can. On A-12 alone over this window the gate
# costs Sharpe. With crypto it leaves Sharpe unchanged and cuts the worst
# loss by about a third. The live book carries the gate on that basis, and
# the A-12 result is an open item, not a settled one.
#
# %% [markdown]
# ## 2. A wider universe
#
# Grinold's law says more independent bets should help. A 56-market pool
# (every tradeable future with enough history) against A-12, same
# configuration, same window:

# %%
POOL = tradeable_pool(PX)
pool = book(POOL)
print(f"{len(POOL)} markets")
describe({"A-12": a12, "56-market pool": pool})

# %%
WIN = [("2008-09-04", "2012-12-31"), ("2013-01-01", "2017-12-31"),
       ("2018-01-01", "2021-12-31"), ("2022-01-01", "2026-09-28")]
pd.DataFrame({f"{a[:4]}-{b[:4]}": dict(a12=sharpe(a12.loc[a:b], RF), pool=sharpe(pool.loc[a:b], RF))
              for a, b in WIN}).T.assign(gap=lambda x: x.pool - x.a12)

# %% [markdown]
# The gap is essentially one window, 2018-2021. What failed in it?

# %%
_, pout = book(POOL, contrib=True)
pc = (pout[3] - pout[4]).loc["2018-01-01":"2021-12-31"]
cls = {s: c for c, v in buckets(POOL).items() for s in v}
by = pc.sum().groupby(cls).agg(["sum", "size"])
by["sum"] *= 100
by.columns = ["contribution_pct", "n_markets"]
by

# %%
def mean_corr(names):
    c = PX[names].pct_change().loc[OOS:].corr().values
    return c[np.triu_indices(len(names), 1)].mean()

pd.DataFrame({k: dict(n=len(v), mean_pairwise_corr=mean_corr(v))
              for k, v in buckets(POOL).items() if len(v) > 1}).T

# %% [markdown]
# **Fake breadth, not breadth.** The pool holds sixteen equity indices whose
# returns are highly correlated: close to one bet held sixteen ways. When
# trend following on equity indices failed through 2018-2021, all of them
# failed together. And HRP divides each class's risk among however many
# markets sit in it, so the pool's gold or copper carries a fraction of the
# risk it has in A-12: the pool diluted its winners without diluting its
# losers. Deleting the near-duplicate equity indices closes about half the gap:

# %%
EQ_KEEP = ["spx", "nasdaq", "nikkei"]
trimmed = [s for s in POOL if cls[s] != "equities" or s in EQ_KEEP]
describe({"A-12": a12, "56 pool": pool,
          f"pool, equities trimmed to 3 ({len(trimmed)} markets)": book(trimmed)})

# %% [markdown]
# Trimming to exactly A-12's three equity indices imports A-12's own choice,
# so this is an upper bound and a hypothesis, not a result. The structural
# idea it points to (a cap on correlation within a class, fixed in advance
# rather than chosen on performance) was the next open item.
#
# ## 3. A macro filter for the bad window: rejected
#
# Could the Economic Trend layer (Chapter 5) have cut risk into the
# 2018-2021 equity collapse?

# %%
from ctadata.macro import load_econ, load_eer, load_econ_intl
ECON = dict(economic_trend=True, econ=load_econ(), eer=load_eer(), econ_intl=load_econ_intl())
pool_e = book(POOL, **ECON)
a12_e = book(A12, **ECON)
pd.DataFrame({k: dict(full=sharpe(oos(v), RF), w2018_2021=sharpe(v.loc["2018":"2021"], RF))
              for k, v in {"56 pool": pool, "56 pool + macro": pool_e,
                           "A-12": a12, "A-12 + macro": a12_e}.items()}).T

# %% [markdown]
# It makes the bad window worse. The likely mechanism: the 2020 crash
# arrived with no macro warning (GDP and CPI are quarterly and monthly and
# published in arrears), and the recovery ran while macro data was at its
# worst, so a filter that cuts size on bad fundamentals was most defensive
# through the strongest trend period of the decade. Most of the pool is also
# judged on US data, as no per-country macro exists on disk for Europe.
#
# ## 4. A class-level P&L trend: the override
#
# Run the book's own 40/252 barbell on each asset class's realised P&L, and
# stand a class down while both legs point down (`cta/overlays.py`). The
# prediction, written down before building it: it should help the wide pool,
# where sixteen equity indices fail as one, and hurt a concentrated book,
# which has little within-class redundancy to protect.

# %%
FOLDS = [("2010-09-04", "2012-09-04"), ("2012-09-04", "2014-09-04"), ("2014-09-04", "2016-09-04"),
         ("2016-09-04", "2018-09-04"), ("2018-09-04", "2020-09-04"), ("2020-09-04", "2022-09-04"),
         ("2022-09-04", "2024-09-04"), ("2024-09-04", "2026-09-28")]


def override_nested(symbols):
    syms = sorted(set(symbols))
    _, out = book(syms, contrib=True, charge_crypto=False)
    held, _ = held_weights(out, PX, "M")
    sl = slice(OOS, None)
    base, over, down = class_pnl_override(held.loc[sl], out[3].loc[sl], out[4].loc[sl], buckets(syms))
    chains = {"base": [], "override": [], "nested": []}
    chose = []
    for a, b in FOLDS:
        tr_b, tr_o = base.loc[:a].iloc[:-1], over.loc[:a].iloc[:-1]
        use = sharpe(matched(tr_o, vol(tr_b)), RF) > sharpe(tr_b, RF)   # training data only
        chose.append("override" if use else "base")
        seg_b, seg_o = base.loc[a:b], matched(over.loc[a:b], vol(base.loc[a:b]))
        chains["base"].append(seg_b); chains["override"].append(seg_o)
        chains["nested"].append(seg_o if use else seg_b)
    res = {k: sharpe(pd.concat(v), RF) for k, v in chains.items()}
    res["times_override_chosen"] = f"{chose.count('override')}/{len(chose)}"
    return res, down

rows, downs = {}, {}
for name, syms in (("56 pool", POOL), ("A-12 + crypto", A12 + CRYPTO), ("A-12", A12)):
    rows[name], downs[name] = override_nested(syms)
pd.DataFrame(rows).T

# %%
(pd.DataFrame(downs) * 100).round(1).rename_axis("share of days stood down, %")

# %% [markdown]
# The prediction held: the override helps the wide pool and hurts the
# concentrated books. For the pool the nested decision chose it in every
# window. For the concentrated books it did not reliably switch it off (it
# chose it in 5 of 8 and 4 of 8 windows) and paid for that, so the case
# against deploying it on the live book rests on the prediction plus the
# full-sample result, not on the nested rule alone. It does
# not close the gap to A-12, it stands classes down a large share of the
# time (a slow allocation overlay, not a crash stop), and a production
# version would need a real shadow book. Not deployed.
#
# ## What this chapter established
#
# * New information can help (crypto); more of the same markets does not.
# * The wide universe's failure has a specific cause, correlated equity
#   indices, and a specific structural remedy to test next.
# * The class override is the one sandbox idea that survived the nested test,
#   and only where it was predicted to.
