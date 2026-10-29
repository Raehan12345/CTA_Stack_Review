# %% [markdown]
# # Chapter 2. The data: loading, cleaning, and restitching futures across rolls
#
# **Question.** Is the price history the engine trades on actually the market
# it claims to be?
#
# Five defects were found in the raw data over the life of the project. Each
# was found because a number looked wrong, and each has a fix in `ctadata`
# with a test in `tests/`:
#
# | | Defect | Found by |
# |---|---|---|
# | 1 | holiday dates injected by an outer join | a baseline that drifted |
# | 2 | an unadjusted Gilt roll, 2011-12-29 | a -108% day under a regime gate |
# | 3 | January spike-and-revert artefacts in a benchmark index | a scan for >7% days |
# | 4 | the last row of every pull was an intraday tick | a refresh that disagreed with the old file |
# | 5 | **roll jumps left in every generic** | the S&P's post-expiry Mondays |
#
# The fifth is the big one, and most of this chapter is about it.

# %%
from _setup import *
from ctadata.paths import data_dir
from ctadata.workbooks import load_workbook
from ctadata.prices import PULLS, load_seconds, frame, load_crypto
from ctadata.universe import TICKER_INFO
from ctadata.cleaning import GILT_ROLL_FIX_DATE, GILT_ROLL_FIX_RATIO, fix_spike_and_revert
from ctadata import restitch

# %% [markdown]
# ## 1. One calendar, left-joined
#
# Every Bloomberg pull has the same layout (tickers on row 1, a formula on
# row 2, dates and closes below). The first pull's calendar is the master,
# and every later pull is LEFT joined onto it.
#
# The original code used an outer join. The second pull's markets keep
# different national holidays, so the outer join added dates on which every
# original market was blank. The engine fills a missing return with zero, so
# each one became a fabricated flat trading day, and some rebalance dates
# moved.

# %%
d = data_dir()
p1, p2 = load_workbook(d / PULLS[0]), load_workbook(d / PULLS[1])
outer = p1.join(p2, how="outer")
injected = outer.index.difference(p1.index)
print(f"pull 1: {len(p1)} dates   pull 2: {len(p2)} dates")
print(f"dates an outer join adds to the original markets: {len(injected)}")
print("examples:", [x.date() for x in injected[:6]])
print("on those dates the original markets are all blank:",
      bool(outer.loc[injected, p1.columns].isna().all().all()))

# %% [markdown]
# ## 2. The Gilt's unadjusted roll
#
# On 2011-12-29 the Gilt generic fell 12.2% and stayed there. A 10-year gilt
# future does not do that: the generic jumped from the expiring contract onto
# the next one without adjusting the price. It sat unnoticed in every
# backtest, diluted across the book, until a regime gate that concentrated
# the book into the Gilt that week produced a -108% day.
#
# The second-month generic, pulled later, confirms the diagnosis: the same
# day the front fell 12.2% while the second ROSE, and the front closed almost
# exactly at the second's price. The fix rebases the series from that date by
# the jump ratio, which corrects that one day's return and leaves every other
# day's return unchanged.

# %%
raw_g1 = p1["G 1 Comdty"].astype(float)
g2 = load_workbook(d / "G2_GC2_EC2_AD2_DataPull.xlsx")
g2 = g2[[c for c in g2.columns if str(c).startswith("G 2")][0]].astype(float)
day = GILT_ROLL_FIX_DATE
win = slice(day - pd.Timedelta(days=6), day + pd.Timedelta(days=4))
pd.DataFrame({"front G1 (raw)": raw_g1.loc[win], "second G2 (raw)": g2.reindex(raw_g1.index).loc[win],
              "front, patched": df["gilt"].loc[win]})

# %%
print(f"that day: front {raw_g1.pct_change().loc[day]:+.2%}, "
      f"second {g2.pct_change().reindex([day]).iloc[0]:+.2%}")
print(f"patch ratio {GILT_ROLL_FIX_RATIO:.6f}; largest |daily return| in the patched series "
      f"{df['gilt'].pct_change().abs().max():.2%}")

# %% [markdown]
# A consequence worth knowing: after the patch the column is an adjusted
# index, not a price. Comparing its LEVEL with the raw second month gives a
# spurious basis of about 11%, which was once written up as "the second-month
# ticker is broken". It was not. Level comparisons use raw series.
#
# ## 3. Spike-and-revert artefacts in a benchmark
#
# The SG Trend Indicator shows dozens of days with moves over 7%, nearly all
# in early January, each reversing within a day or two. A diversified trend
# index does not do that every January. The cleaner rebases from each
# flagged day, the Gilt fix generalised.

# %%
from ctadata.benchmarks import SG_TREND_FILE
sg_raw = pd.read_excel(d / SG_TREND_FILE, header=None).iloc[2:, [0, 1]]
sg_raw.columns = ["date", "v"]
sg_raw = sg_raw.dropna().set_index(pd.to_datetime(sg_raw.dropna()["date"], utc=True))["v"].astype(float)
big = sg_raw.pct_change().abs() > 0.07
print(f"raw: {int(big.sum())} days over 7%, of which {int(big[big].index.month.isin([1]).sum())} in January")
print(f"cleaned: {int((fix_spike_and_revert(sg_raw).pct_change().abs() > 0.07).sum())} days over 7%")

# %% [markdown]
# ## 4. The last row of every pull was an intraday tick
#
# Each pull was run during the trading day, so its final row was a live
# price, not a settlement. When the data was refreshed, about 40 values
# changed, some by 5%. `ctadata.cleaning.merge_pull` now treats each
# column's last observation as provisional (the new settled value wins
# there) and refuses to merge if any EARLIER shared day disagrees, because
# that means the file is not the same series. Tested in
# `tests/test_cleaning.py`.
#
# ## 5. Roll jumps in every generic
#
# A futures contract expires, so a 20-year history is a chain of contracts.
# Bloomberg's generic tickers splice that chain **without adjustment**: on a
# roll day the series steps from the expiring contract to the next one, and
# the difference between them appears as a return nobody earned. In contango
# it is a spurious gain; in backwardation, a spurious loss.
#
# The tell was the S&P's Monday after each quarterly expiry: on average
# -0.84% in 2010-2021 and +1.61% in 2023-2026, exactly the sign flip of the
# S&P calendar spread between the zero-rate years and the higher-rate years.
#
# ### 5a. A roll calendar from the contract chain
#
# The full chain: every contract of every market, expired ones included,
# with its last tradeable date and delivery month. On each date the front is
# the nearest contract that has not expired; the day it changes is a roll.

# %%
contracts = restitch.load_contracts()
print(f"{len(contracts):,} contracts across {contracts['root'].nunique()} roots")
px = frame()
gaps, rolls = restitch.roll_calendar(px.index, contracts, restitch.root_to_name())
print(f"calendar built for {gaps.shape[1]} markets")

# %% [markdown]
# ### 5b. Validate the calendar against the prices before using it
#
# If a market's derived roll days are right, the unadjusted generic should
# jump on them: the mean absolute move on roll days should be clearly larger
# than on other days. The threshold (1.2x, on at least 10 rolls) was fixed
# before looking at which markets pass.
#
# Markets whose chains list contract months that never actually trade fail
# this test. Gold lists a contract every month but its generic trades only
# five; the chain-derived calendar flags hundreds of ordinary days as rolls
# and the ratio collapses toward 1. Failing markets are left UNADJUSTED: an
# unadjusted series is a known, measurable error, while adjusting on a wrong
# calendar would inject new false moves.

# %%
mask, report = restitch.validated_rolls(px, gaps, rolls)
report["ratio"] = report["ratio"].astype(float)
print(f"{int(report['passed'].sum())} markets pass, {int((~report['passed'].astype(bool)).sum())} fail")
print("excluded by hand:", sorted(restitch.EXCLUDE))
report.sort_values("ratio")

# %% [markdown]
# ### 5c. Stitch returns contract by contract
#
# ```
# ordinary day   r_t = F1_t / F1_{t-1} - 1      same contract on both days
# roll day       r_t = F1_t / F2_{t-1} - 1      F2 yesterday IS F1 today
# ```
#
# The second line is exact: on the day the front generic rolls, the contract
# it rolls into is the one the second generic was tracking the day before.
# No back-adjustment of price levels is needed at all.

# %%
sec = load_seconds()
dist = restitch.distortion(px, sec, mask, start=OOS)
dist = dist.join(pd.Series({c: ctadata_cls for c in dist.index
                            for ctadata_cls in [__import__("ctadata").asset_class(c)]}, name="class"))
print(f"mean absolute distortion: {dist['fake_pa'].abs().mean():.2%} a year, "
      f"across {len(dist)} adjusted markets")
dist.sort_values("fake_pa")

# %%
st = restitch.stitched_returns(px, sec, mask)
check = []
for c in mask.columns[mask.any()]:
    if c not in st.columns:
        continue
    m = mask[c]
    raw_r, adj_r = px[c].pct_change().abs(), st[c].abs()
    check.append(dict(market=c, raw_ratio=raw_r[m].mean() / raw_r[~m].mean(),
                      stitched_ratio=adj_r[m].mean() / adj_r[~m].mean()))
check = pd.DataFrame(check).set_index("market")
print("roll-day move relative to ordinary days, median across markets:")
print(check.median().round(2).to_string())

# %% [markdown]
# Before stitching, roll days move far more than ordinary days. After, they
# look like ordinary days: the jump was the basis, and it is gone.
#
# ### 5d. What it does to the live book
#
# The live book is A-12 plus BTC and ETH. Six of its twelve futures have
# validated calendars and are stitched; the metals and the currency futures
# fail validation and stay raw; crypto is spot and has no rolls. Same
# configuration, same window, crypto financing charged in both.

# %%
from live.simulator import held_weights
from live.costs import crypto_carry
from cta.config import CHOSEN

LIVE = sorted(A12 + ["btc", "eth"])
live_px = px.join(load_crypto(), how="left")[LIVE]
adj_ix = restitch.stitched_index(px, sec, mask)
live_adj = live_px.copy()
for c in LIVE:
    if c in mask.columns and mask[c].any():
        live_adj[c] = adj_ix[c].where(live_px[c].notna())

def sleeve(prices):
    out = build_trend_nav(prices, symbols=LIVE, universe=buckets(LIVE),
                          **{**DEFAULT_PARAMS, **CHOSEN})
    held, _ = held_weights(out, prices, "M")
    return returns(out[0]) - crypto_carry(held[["btc", "eth"]], prices.index)

print("stitched in the live book:", [c for c in LIVE if c in mask.columns and mask[c].any()])
describe({"raw generics": sleeve(live_px), "roll-stitched": sleeve(live_adj)}, stacked=True)

# %% [markdown]
# The Sharpe barely moves, which is what made it safe to keep publishing
# figures on the raw generics: the six stitched markets are the liquid index
# and energy contracts, whose roll distortion is small next to their trend
# moves, and it partly cancels across long and short positions. The sleeve's
# own worst loss improves by about 2.5 points.
#
# The stacked book's worst loss is deeper on stitched data, though. This was
# not investigated further; the likely mechanism is that the stitched equity
# and bond series differ from the raw ones exactly in the stacked book's
# worst stretch, which is dominated by the S&P leg. A reminder that one
# summary statistic can move the opposite way from another; both are
# reported.
#
# The engine's default data path stays on the raw generics (plus the Gilt
# patch), with this comparison as the evidence that the choice barely
# changes this book's risk-adjusted return. It would NOT be harmless for a
# carry strategy, which is exactly where the stitched returns were used.
#
# ## 6. The live splice: the same problem, solved by re-anchoring
#
# The research history ends when the Bloomberg files were last copied. The
# live engine extends it to yesterday with the broker's own daily closes,
# which track the same market with a different basis and sometimes a
# different quote (the broker quotes USDJPY; the research series is USD per
# JPY). `ctadata.restitch.ratio_splice` takes the ratio of the two prices on
# the last date both have and scales the newer broker closes by it.
#
# Run every day with the result written back, the anchor is always
# yesterday. So when the broker's dated contract rolls, the new contract is
# level-matched to yesterday and the switch never shows up as a jump. That
# is the live answer to the same roll problem.
#
# A demonstration on real data: treat the S&P second-month generic as "the
# broker's series" (a different contract, a different basis) and splice it
# onto the front-month history.

# %%
cut = pd.Timestamp("2026-06-01", tz="UTC")
hist = df["spx"].where(df.index < cut)
extended, ratio, anchor = restitch.ratio_splice(hist, sec["spx"])
new = extended.loc[extended.index >= cut]
print(f"anchor {anchor.date()}, ratio {ratio:.5f}")
print("returns after the splice equal the second month's own returns:",
      bool(np.allclose(new.pct_change().dropna(), sec["spx"].loc[new.index].pct_change().dropna())))
print("level jump at the splice:", f"{extended.loc[new.index[0]] / extended.loc[anchor] - 1:+.3%}",
      f"(the second month's own move that day: {sec['spx'].pct_change().loc[new.index[0]]:+.3%})")

# %% [markdown]
# ## What this chapter established
#
# * Every published number runs on a frame with one fixed calendar and the
#   Gilt roll patched.
# * The generics carry roll jumps worth several points a year in some
#   markets. A validated roll calendar and contract-by-contract stitching
#   remove them; the comparison above shows the live book is barely affected,
#   which is why the engine's default stays on the raw generics.
# * The live engine solves the same problem by daily ratio re-anchoring.
