"""The K-means regime gate, and the Wasserstein variants that lost to it.

Each day is described by a few statistics of the whole book. K-means groups
similar days into regimes on an expanding window, the current day is
classified with the latest fit, and the portfolio volatility target is
scaled by that regime's multiplier (1.0 down to 0.5).

Three findings shaped this module:

  * LEARNED, not assumed, multipliers. A long-only intuition says violent
    regimes are bad. For a trend follower the "crisis" cluster has the BEST
    forward returns; the costly regime is the choppy middle. Clusters are
    therefore ranked by the sleeve's own measured forward return.
  * AUTO-K. The number of regimes is re-derived at every refit by the elbow
    rule. It never picks the inherited k=3; it picks 5 or 6.
  * WK-MEANS LOST. Clustering sorted return distributions gives the best
    regime labels and the worst trading result, because sorting discards
    the order of returns and order is what a trend follower trades.
"""
import numpy as np
import pandas as pd

from .signals import _efficiency_ratio


# ---------------------------------------------------------------------------
# LAYER 6 -- K-means regime gate.
#
# Two references. Oliveira et al. cluster macro
# states on an expanding window and backtest one month ahead, choosing k by the
# elbow heuristic. Virolainen uses Wasserstein k-means for VALUE AT RISK, not
# return timing -- worth remembering, because volatility-shaped regimes are
# genuinely useful for risk estimation and that is not evidence they time a
# trend book. Consistent with what we see: both WK gates correlate +0.31..0.35
# with our own leverage, i.e. they largely rediscover the vol targeting.
#
# Features come from the instrument book itself, so no new vendor dependency.
# They are computed equal-weighted rather than on the strategy's own weights:
# the regime should describe the market, not the book we happen to hold, or the
# gate feeds back on its own past sizing.
#
# Labels are re-sorted at every refit so cluster 0 is always the calmest.
# K-means label ordering is arbitrary and without this the gate whipsaws on
# pure relabelling. Multipliers hit portfolio target vol, the same slot as the
# VIX layer, not per instrument.
#
# k was fixed at 3 originally. regime_auto_k re-derives it per refit by the
# elbow rule instead; on this data it picks 5-6.
_REGIME_MULT_DEFAULT = {0: 1.0, 1: 0.75, 2: 0.5}   # calm-trending / choppy / crisis

# Memoization for the regime multiplier -- see build_trend_nav's _regime_key.
# Process-lifetime only; a fresh kernel starts empty.
_REGIME_CACHE: dict = {}


def _learned_ladder(labels: pd.Series, fwd: pd.Series, cutoff, n_clusters: int,
                    ladder: list) -> dict:
    """Assign risk multipliers to clusters by MEASURED strategy performance
    instead of by assumption.

    The fixed schedule (calm 1.0 / choppy 0.75 / crisis 0.5) encodes a
    long-only intuition: the more violent the market, the less risk you want.
    For a trend follower that is often exactly backwards -- by the time a
    crash is identified the barbell has usually already flipped short, so the
    crash PAYS. Measured forward returns confirm it: the "crisis" cluster is
    consistently the sleeve's most profitable state, and the genuinely costly
    state is the choppy middle, which whipsaws the book long AND short.

    So rather than ranking clusters by volatility and assuming the violent one
    is worst, this ranks them by what the sleeve actually earned in each, and
    hands the largest multiplier to the best-performing cluster. Same ladder of
    values, so the overall risk budget is comparable -- only the assignment
    changes.

    `cutoff` excludes any date whose forward window has not fully closed by the
    decision date, so no future information reaches the ranking.
    Returns {cluster_rank: multiplier}; falls back to the neutral assignment
    when there is not enough labelled history to rank on.
    """
    hist = pd.concat([labels.rename("k"), fwd.rename("f")], axis=1).dropna()
    hist = hist[hist.index <= cutoff]
    default = {i: (ladder[i] if i < len(ladder) else ladder[-1]) for i in range(n_clusters)}
    if len(hist) < 60:
        return default
    # Under auto-k the number of clusters changes between refits, so labels
    # carried forward from an earlier fit can exceed the current k. Rank only
    # the label space the current model can actually emit.
    hist = hist[hist["k"] < n_clusters]
    perf = hist.groupby("k")["f"].mean()
    if len(perf) < n_clusters:
        return default
    order = perf.sort_values(ascending=False).index      # best cluster first
    rungs = (ladder if len(ladder) == len(order)
             else list(np.linspace(max(ladder), min(ladder), len(order))))
    return {int(c): rungs[j] for j, c in enumerate(order)}


# --- WK-means (Wasserstein k-means), Horvath, Issa & Muguruza (2021),
# "Clustering Market Regimes using the Wasserstein Distance", arXiv:2110.11848.
#
# The feature-based gate below (_regime_features) hand-picks three summary
# statistics and clusters those. WK-means instead clusters the RETURN
# DISTRIBUTIONS themselves: each time window becomes an empirical measure, and
# distance is the p-Wasserstein distance between measures. Nothing is discarded
# by a feature choice -- vol, skew and tail shape all enter automatically.
#
# The paper's two results that make this cheap (its Section 2, eq. 21 and
# Prop. 2.6): for empirical measures with N atoms each,
#   * W_p(mu,nu)^p = (1/N) * sum_i |alpha_i - beta_i|^p   over SORTED atoms
#     -- i.e. the Lp distance between the two windows' sorted return vectors,
#     because sorting IS the optimal transport map in one dimension;
#   * the p=1 Wasserstein barycentre of a set of such measures is the
#     coordinate-wise MEDIAN of their sorted atom vectors (for p=2, the mean).
# So the whole algorithm is Lloyd's algorithm run on sorted return vectors.
#
# Paper's own hyperparameters were (h1, h2) = (35, 28) on HOURLY SPY -- windows
# of ~1 week with adjacent windows one day apart, and k=2. The daily analogue
# used here keeps the shape rather than the literal numbers: a window of
# regime_window trading days, re-evaluated at every weekly rebalance.
def _wasserstein_kmeans(atoms: np.ndarray, k: int, p: int = 1, seed: int = 0,
                         n_init: int = 5, max_iter: int = 50, tol: float = 1e-12):
    """Lloyd's algorithm under the p-Wasserstein distance on 1-D empirical
    measures. `atoms` is (M, N): M windows, each row the N returns of that
    window ALREADY SORTED ascending. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    M = atoms.shape[0]
    best = None
    for _ in range(max_init := n_init):
        cent = atoms[rng.choice(M, k, replace=False)].copy()
        labels = np.zeros(M, dtype=int)
        for _ in range(max_iter):
            # (M, k) distances; sorting already applied, so this is just Lp.
            d = np.stack([(np.abs(atoms - c) ** p).mean(axis=1) for c in cent], axis=1)
            new = d.argmin(axis=1)
            new_cent = cent.copy()
            for j in range(k):
                members = atoms[new == j]
                if len(members) == 0:
                    continue
                # Wasserstein barycentre: coordinate-wise median (p=1) or mean (p=2).
                new_cent[j] = np.median(members, axis=0) if p == 1 else members.mean(axis=0)
            shift = float((np.abs(new_cent - cent) ** p).mean())
            cent, labels = new_cent, new
            if shift < tol:
                break
        loss = float(np.stack([(np.abs(atoms - cent[j]) ** p).mean(axis=1) for j in range(k)],
                              axis=1).min(axis=1).sum())
        if best is None or loss < best[0]:
            best = (loss, labels.copy(), cent.copy())
    return best[1], best[2]


def _elbow_k(inertias: list, k_values: list) -> int:
    """Pick the number of clusters at the 'elbow' of the inertia curve, by the
    standard maximum-distance-to-chord rule: draw the straight line from the
    first point to the last, and take the k whose inertia sits furthest below
    it. Same criterion the `kneed` package implements for a convex decreasing
    curve, written out here so the engine does not gain a dependency for six
    lines of geometry.

    Used to re-derive k INSIDE each refit rather than fixing it at 3 globally.
    That matters: a fixed k is a hyperparameter chosen with hindsight over the
    whole sample, which is exactly the criticism this project levelled at its
    own universe selection."""
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertias, dtype=float)
    if len(x) < 3 or not np.isfinite(y).all() or np.ptp(y) == 0:
        return int(x[0])
    # Normalise both axes so the chord distance is scale-free.
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y.min()) / (y.max() - y.min())
    # Distance below the chord from (0, yn[0]) to (1, yn[-1]).
    chord = yn[0] + xn * (yn[-1] - yn[0])
    return int(x[int(np.argmax(chord - yn))])


def build_macro_features(prices: pd.DataFrame, econ: pd.DataFrame, eer: pd.DataFrame,
                          rates: pd.DataFrame = None, cot: pd.DataFrame = None,
                          z_window: int = 756) -> pd.DataFrame:
    """Regime features from data OUTSIDE the instrument book.

    Every feature the price-based gate uses is derived from the same 13 prices
    the engine already trades, so it can only ever re-describe what the engine
    can already see. These come from credit, the curve, policy and positioning
    instead.

    Publication lags follow the conventions already used elsewhere in this
    module: market data one day, CFTC three days (Tuesday as-of, Friday
    release), CPI 30 days, GDP 45 days. Everything is then shifted one more
    day, the same generic discipline as the barbell score.

    Returns whatever it can build from what it is given -- missing inputs drop
    their features rather than failing, so a caller without COT still gets the
    credit and curve columns.
    """
    idx = prices.index
    out = {}

    def z(s, w=z_window):
        return (s - s.rolling(w, min_periods=w // 3).mean()) / s.rolling(w, min_periods=w // 3).std()

    if econ is not None and "hy_oas" in econ.columns:
        oas = econ["hy_oas"].reindex(idx).ffill().shift(1)
        out["credit"] = z(oas)                      # how stressed is credit
        out["credit_chg"] = z(oas.diff(63))         # and which way is it moving

    # Curve slope proxy. These are bond FUTURES prices, so price up = yield
    # down; the 10y underperforming the 2y means long yields rose more, i.e.
    # a steepening. No new data needed, but it is information the book-level
    # features (vol, correlation, trend quality) never look at.
    if {"us2y", "us10y"}.issubset(prices.columns):
        slope = (prices["us10y"].pct_change(252) - prices["us2y"].pct_change(252))
        out["slope"] = z(-slope.shift(1))

    if rates is not None and "USD" in rates.columns:
        short = rates["USD"].reindex(idx).ffill().shift(1)
        out["policy_chg"] = z(short.diff(252))
        if econ is not None and "cpi_yoy" in econ.columns:
            # dropna FIRST: these arrive on a daily index but only carry a
            # value on release days, so shifting the index and forward-filling
            # would match a prior row whose value is NaN.
            cpi = econ["cpi_yoy"].dropna().copy()
            cpi.index = cpi.index + pd.Timedelta(days=30)     # publication lag
            cpi = cpi.reindex(idx, method="ffill").shift(1)
            out["real_rate"] = z(short * 100 - cpi)

    if eer is not None:
        nom = [c for c in eer.columns if c.endswith("_nominal")]
        if nom:
            # The three trade-weighted rates are AUD/GBP/JPY, so their average
            # move is roughly the inverse of a broad dollar factor.
            basket = eer[nom].reindex(idx).ffill().pct_change(63).mean(axis=1)
            out["usd"] = z(-basket.shift(1))

    if econ is not None and "gdp_qoq" in econ.columns:
        g = econ["gdp_qoq"].dropna().copy()
        g.index = g.index + pd.Timedelta(days=45)             # publication lag
        out["growth"] = z(g.reindex(idx, method="ffill").shift(1))

    if econ is not None and "cpi_yoy" in econ.columns:
        c = econ["cpi_yoy"].dropna().copy()
        c.index = c.index + pd.Timedelta(days=30)
        c = c.reindex(idx, method="ffill")
        out["infl_chg"] = z(c.diff(252).shift(1))

    if cot is not None and len(cot.columns):
        # How crowded speculators are overall, regardless of direction.
        cot = cot.dropna(how="all")
        cz = cot.apply(lambda s: (s - s.rolling(156, min_periods=52).mean())
                       / s.rolling(156, min_periods=52).std())
        crowd = cz.abs().mean(axis=1)
        crowd.index = crowd.index + pd.Timedelta(days=3)      # CFTC release lag
        out["crowding"] = crowd.reindex(idx, method="ffill").shift(1)

    return pd.DataFrame(out, index=idx)


def _regime_features(prices: pd.DataFrame, symbols: list[str], vol_window: int = 60,
                      er_window: int = 40, corr_window: int = 60,
                      macro: pd.DataFrame = None) -> pd.DataFrame:
    """Regime features, one row per date, each lagged one day so day t's
    features use only information available before day t.

    The first three (vol / corr / er) are the original hand-picked set. The
    remainder are the portfolio-level analogues of the single-stock feature
    family in the clustering-for-trading literature (distance from moving
    average, drawdown depth, RSI, short-window price change): the same ideas,
    but computed on the equal-weighted BOOK rather than on one instrument,
    because this gate scales a whole portfolio's risk rather than switching a
    single position on and off.

    Two of them have no single-asset equivalent and exist because this is a
    multi-instrument trend book: `disp` (how differently the instruments are
    behaving from one another) and `agree` (what fraction of the book has its
    fast and slow trend legs pointing the same way). Trend following lives or
    dies on those two, so if any feature beyond the original three earns its
    place it is most likely one of these."""
    rets = prices[symbols].pct_change()
    eq = rets.mean(axis=1)
    vol = eq.rolling(vol_window).std() * np.sqrt(252)
    # Mean pairwise correlation, cheaply: for an equal-weighted book,
    # var(mean) = (1/n)*mean_var*(1 + (n-1)*rho_bar), so rho_bar can be
    # recovered from the ratio of portfolio variance to mean instrument
    # variance without forming a full correlation matrix every day.
    n = len(symbols)
    mean_var = rets.rolling(corr_window).var().mean(axis=1)
    port_var = eq.rolling(corr_window).var()
    rho = ((port_var * n / mean_var) - 1.0) / (n - 1) if n > 1 else pd.Series(0.0, index=prices.index)
    er = _efficiency_ratio(prices[symbols], er_window).mean(axis=1)

    # --- book-level analogues of the single-stock feature family ---
    book = (1.0 + eq.fillna(0.0)).cumprod()          # equal-weighted book index
    dist_ma = book / book.rolling(100).mean() - 1.0  # distance from 100d average
    dd = book / book.rolling(252, min_periods=60).max() - 1.0   # drawdown depth
    vol_ratio = (eq.rolling(20).std() / eq.rolling(252).std()).replace([np.inf, -np.inf], np.nan)
    chg = book.pct_change(10)                        # 10-day book move
    # Cross-sectional dispersion: how differently the instruments are moving.
    disp = rets.std(axis=1).rolling(corr_window).mean()
    # Trend agreement: share of instruments whose 40d and 252d momentum agree.
    fast = np.sign(prices[symbols] / prices[symbols].shift(40) - 1.0)
    slow = np.sign(prices[symbols] / prices[symbols].shift(252) - 1.0)
    agree = (fast == slow).mean(axis=1)
    # RSI of the book, Wilder's 14, reusing the shape of calculate-RSI.
    d = book.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))

    feats = pd.DataFrame({"vol": vol, "corr": rho.clip(-1, 1), "er": er,
                          "dist_ma": dist_ma, "dd": dd, "vol_ratio": vol_ratio,
                          "chg": chg, "disp": disp, "agree": agree,
                          "rsi": rsi / 100.0,
                          # book-level skewness over the vol window (2026-09-21, after the
                          # CITIC skewness pages); tested as a sixth gate feature
                          "skew": eq.rolling(vol_window).skew()}).shift(1)
    if macro is not None:
        # build_macro_features applies its own publication lags and shift, so
        # these are joined AFTER the shift above rather than being shifted twice.
        feats = feats.join(macro.reindex(feats.index), how="left")
    return feats


def _wasserstein_regime_mult(prices: pd.DataFrame, symbols: list[str], is_rebal: np.ndarray,
                              n_clusters: int = 3, min_train: int = 504, refit_every: int = 13,
                              hysteresis: int = 2, regime_mult: dict = None, seed: int = 0,
                              regime_window: int = 63, p: int = 1,
                              sleeve_ret: pd.Series = None, fwd_window: int = 21) -> pd.Series:
    """WK-means version of _regime_target_mult: identical expanding-window,
    no-lookahead discipline and identical output contract (a portfolio-level
    target-vol multiplier), but the clustering is done on the return
    DISTRIBUTION of each window rather than on three hand-picked features.

    Windows are the trailing `regime_window` daily returns of the
    equal-weighted book, taken at each rebalance date (so adjacent windows
    overlap heavily, matching the paper's overlapping-partition design).
    Clusters are ranked by centroid dispersion so 0 is always the calmest."""
    eq = prices[symbols].pct_change().mean(axis=1)
    rebal_dates = prices.index[is_rebal]
    # One sorted-atom vector per rebalance date, from the trailing window
    # ending the day BEFORE that date (no same-day information).
    win = {}
    for dt in rebal_dates:
        hist = eq.loc[:dt].iloc[:-1].dropna()
        if len(hist) >= regime_window:
            win[dt] = np.sort(hist.iloc[-regime_window:].values)
    return _expanding_regime_labels(win, prices.index, n_clusters, min_train // regime_window,
                                    refit_every, hysteresis, regime_mult, seed, p,
                                    sleeve_ret=sleeve_ret, fwd_window=fwd_window)


def _sliced_wasserstein_embedding(rets: np.ndarray, directions: np.ndarray,
                                   n_quantiles: int = 16) -> np.ndarray:
    """Embed a (T x d) window of returns so that the ordinary Lp distance
    between two embeddings equals (a constant times) the SLICED p-Wasserstein
    distance between the two windows viewed as measures on R^d.

    Method is the one the paper prescribes for d > 1 (its Section 1.2, citing
    Rabin et al. and Kolouri et al.): project the d-dimensional cloud onto many
    directions on the unit sphere, take the cheap closed-form 1-D Wasserstein
    distance along each projection, and average over projections. Because the
    1-D distance is just the Lp distance between sorted projections, stacking
    the sorted projections into one vector turns sliced-Wasserstein k-means
    into ordinary Lp k-means on that vector -- so the same clusterer,
    barycentre (coordinate-wise median for p=1) and ranking logic all carry
    over unchanged from the univariate case.

    The quantile function of each projection is evaluated on a fixed grid of
    n_quantiles points rather than at all T atoms. That is an approximation
    (exact as n_quantiles -> T) taken purely for speed; it keeps the embedding
    small enough to refit many times over a 20-year expanding window.

    This is why the multivariate version exists: the projections are sensitive to CORRELATION, which averaging the book into one
    series destroys: a book whose instruments all move together is a cigar-
    shaped cloud whose spread depends strongly on projection direction, while
    an uncorrelated book is a spherical cloud with similar spread in every
    direction. The univariate WK-means cannot distinguish those two cases."""
    proj = rets @ directions.T                       # (T, M)
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return np.quantile(proj, qs, axis=0).ravel()     # (M * n_quantiles,)


def _expanding_regime_labels(win: dict, index: pd.DatetimeIndex, n_clusters: int,
                              min_train_windows: int, refit_every: int, hysteresis: int,
                              regime_mult: dict, seed: int, p: int,
                              sleeve_ret: pd.Series = None, fwd_window: int = 21) -> pd.Series:
    """Shared expanding-window classify-and-smooth loop for every WK-means
    variant. Kept in one place so the univariate and multivariate versions
    cannot silently diverge in their lookahead discipline or label handling.

    `win` maps rebalance date -> that window's embedding vector. At each date
    the clusterer is fitted on STRICTLY EARLIER windows only, then used to
    classify the current one."""
    regime_mult = regime_mult or _REGIME_MULT_DEFAULT
    dates = list(win)
    if len(dates) < 10:
        return pd.Series(1.0, index=index)
    atoms_all = np.vstack([win[d] for d in dates])

    raw = pd.Series(np.nan, index=index)
    cent = cent_rank = None
    since_refit = 0
    learn = sleeve_ret is not None
    if learn:
        fwd = sleeve_ret[::-1].rolling(fwd_window).sum()[::-1].shift(-1)
        ladder = sorted([regime_mult[i] for i in sorted(regime_mult)], reverse=True)
        maps = {}
    for i, dt in enumerate(dates):
        if i < max(min_train_windows, n_clusters + 1):
            continue
        if cent is None or since_refit >= refit_every:
            _, cent = _wasserstein_kmeans(atoms_all[:i], n_clusters, p=p, seed=seed)
            # Rank clusters by centroid dispersion so 0 is always the calmest,
            # making the label meaning stable across refits.
            cent_rank = {int(orig): int(r) for r, orig in enumerate(np.argsort(cent.std(axis=1)))}
            since_refit = 0
        since_refit += 1
        d = [(np.abs(atoms_all[i] - c) ** p).mean() for c in cent]
        raw.loc[dt] = cent_rank[int(np.argmin(d))]
        if learn:
            cutoff = dt - pd.Timedelta(days=int(fwd_window * 1.45))
            maps[dt] = _learned_ladder(raw.loc[:dt], fwd, cutoff, n_clusters, ladder)

    raw = raw.ffill()
    if hysteresis > 1:
        stable, cur, run, prev = [], np.nan, 0, np.nan
        for v in raw.values:
            run = 1 if v != prev else run + 1
            prev = v
            if not np.isnan(v) and (np.isnan(cur) or run >= hysteresis):
                cur = v
            stable.append(cur)
        raw = pd.Series(stable, index=raw.index)

    if not learn:
        return raw.map(regime_mult).astype(float).fillna(1.0)
    mp = pd.Series(maps).reindex(index).ffill()
    out = pd.Series(
        [m.get(int(k), 1.0) if isinstance(m, dict) and not pd.isna(k) else 1.0
         for k, m in zip(raw.values, mp.values)], index=index)
    return out.astype(float).fillna(1.0)


def _wasserstein_mv_regime_mult(prices: pd.DataFrame, symbols: list[str], is_rebal: np.ndarray,
                                 n_clusters: int = 3, min_train: int = 504, refit_every: int = 13,
                                 hysteresis: int = 2, regime_mult: dict = None, seed: int = 0,
                                 regime_window: int = 63, p: int = 1, n_projections: int = 48,
                                 n_quantiles: int = 16, standardize: bool = False,
                                 sleeve_ret: pd.Series = None, fwd_window: int = 21) -> pd.Series:
    """Multivariate (sliced) WK-means regime gate -- the d > 1 extension the
    paper signposts in its Section 1.2 but does not itself run.

    Each window is the trailing `regime_window` days of the instruments'
    returns kept SEPARATE (a T x d cloud) rather than averaged into one series,
    so the clustering can see cross-sectional correlation structure.

    standardize: when True, each instrument is divided by its own trailing
    volatility before projection, so no single high-vol instrument (natural gas,
    say) dominates every projection. This preserves correlation structure while
    discarding some level-of-volatility information -- worth testing both ways,
    since it is not obvious a priori which matters more here."""
    rets = prices[symbols].pct_change()
    if standardize:
        # Trailing vol only -- shifted, so no same-day information leaks in.
        rets = rets / rets.rolling(252, min_periods=60).std().shift(1)
    rets = rets.replace([np.inf, -np.inf], np.nan)

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, len(symbols)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    win = {}
    for dt in prices.index[is_rebal]:
        hist = rets.loc[:dt].iloc[:-1]
        if len(hist) < regime_window:
            continue
        block = hist.iloc[-regime_window:]
        if block.isna().all(axis=None):
            continue
        win[dt] = _sliced_wasserstein_embedding(block.fillna(0.0).values, directions, n_quantiles)
    return _expanding_regime_labels(win, prices.index, n_clusters, min_train // regime_window,
                                    refit_every, hysteresis, regime_mult, seed, p,
                                    sleeve_ret=sleeve_ret, fwd_window=fwd_window)



def _forward_stat(sleeve_ret: pd.Series, fwd_window: int, rank_by: str = "return") -> pd.Series:
    """Per-day statistic of the sleeve's return over the NEXT fwd_window days
    (starting tomorrow), used by _learned_ladder to rank clusters. NaN where
    the window is truncated or contains a NaN, matching the rolling-sum
    convention of the original.

      "return"    sum of the next fwd_window returns (the original)
      "drawdown"  worst cumulative loss inside the window (<= 0): the deepest
                  point the sleeve fell to from the labelled day
      "downside"  sum of the negative daily returns in the window (<= 0)

    All three rank "best first" under a descending sort, so _learned_ladder
    is unchanged. Added 2026-09-16 for the rank-by-drawdown test; the gate's
    reliable effect is on drawdown, so ranking on forward return is arguably
    the wrong objective."""
    if rank_by == "return":
        return sleeve_ret[::-1].rolling(fwd_window).sum()[::-1].shift(-1)
    r = sleeve_ret.values.astype(float)
    n = len(r)
    out = np.full(n, np.nan)
    if n > fwd_window:
        # rows t = 0 .. n-fwd_window-1 have a full window r[t+1 .. t+fwd_window]
        win = np.lib.stride_tricks.sliding_window_view(r[1:], fwd_window)      # (n-fwd_window, fwd_window)
        ok = ~np.isnan(win).any(axis=1)
        if rank_by == "drawdown":
            stat = np.minimum(np.cumsum(win, axis=1).min(axis=1), 0.0)
        elif rank_by == "downside":
            stat = np.where(win < 0, win, 0.0).sum(axis=1)
        else:
            raise ValueError(f"regime_rank_by must be 'return', 'drawdown' or 'downside', got {rank_by!r}")
        out[:n - fwd_window] = np.where(ok, stat, np.nan)
    return pd.Series(out, index=sleeve_ret.index)

def _regime_target_mult(prices: pd.DataFrame, symbols: list[str], is_rebal: np.ndarray,
                         n_clusters: int = 3, min_train: int = 504, refit_every: int = 13,
                         hysteresis: int = 2, regime_mult: dict = None, seed: int = 0,
                         vol_window: int = 60, er_window: int = 40,
                         features: tuple = ("vol", "corr", "er"),
                         sleeve_ret: pd.Series = None, fwd_window: int = 21,
                         auto_k: tuple = None, macro: pd.DataFrame = None,
                         kmeans_impl: str = "sklearn", rank_by: str = "return") -> pd.Series:
    """Portfolio-level target-vol multiplier from an expanding-window K-means
    regime classifier. Returns 1.0 (no effect) everywhere until min_train
    observations of valid features exist.

    refit_every: refit the clusterer every Nth rebalance rather than every
    one -- the expanding window barely changes week to week, and refitting
    ~quarterly (13 weekly rebalances) cuts the cost by an order of magnitude
    with no material difference to the labels. Between refits the most recent
    fitted model is used to CLASSIFY new days, which is still strictly
    out-of-sample.

    hysteresis: require this many consecutive days classified into a new
    regime before actually switching to it. Guards against single-day regime
    churn, which would otherwise create turnover with no informational
    content."""
    if kmeans_impl == "sklearn":
        from sklearn.cluster import KMeans
    elif kmeans_impl == "portable":
        # Same algorithm on a generator that MQL5 can reproduce; the
        # reference for the in-terminal port. See cta/kmeans_portable.py.
        from .kmeans_portable import PortableKMeans as KMeans
    else:
        raise ValueError(f"kmeans_impl must be 'sklearn' or 'portable', got {kmeans_impl!r}")

    regime_mult = regime_mult or _REGIME_MULT_DEFAULT
    feats = _regime_features(prices, symbols, vol_window, er_window, macro=macro)
    # `features` selects which of vol / corr / er the clusterer may see, so the
    # contribution of each can be ablated.
    missing = [f for f in features if f not in feats.columns]
    if missing:
        raise ValueError(f"unknown regime feature(s) {missing}; available: {list(feats.columns)}")
    feats = feats[list(features)]
    valid = feats.dropna()
    out = pd.Series(1.0, index=prices.index)
    if len(valid) <= min_train:
        return out

    rebal_dates = prices.index[is_rebal]
    model = None
    mu = sd = None
    since_refit = 0
    raw = pd.Series(np.nan, index=prices.index)
    # Learned mode: forward return of the sleeve at constant risk, used to rank
    # clusters by what the strategy ACTUALLY earned in each (see _learned_ladder).
    learn = sleeve_ret is not None
    if auto_k and not learn:
        # With a variable k there is no fixed cluster-id -> multiplier table to
        # map onto, and ranking by volatility would assume the violent regime is
        # the bad one, which for this sleeve is backwards. Auto-k is therefore
        # only meaningful alongside the learned ladder.
        raise ValueError("auto_k requires regime_mult_mode='learned'")
    if learn:
        fwd = _forward_stat(sleeve_ret, fwd_window, rank_by)
        base_ladder = sorted([regime_mult[i] for i in sorted(regime_mult)], reverse=True)
        ladder = base_ladder
        maps = {}
    k_here = n_clusters

    for i, dt in enumerate(rebal_dates):
        hist = valid.loc[:dt]
        if len(hist) <= min_train:
            continue
        # Drop the current row from the training set: fit on strictly prior
        # information, then classify today with the resulting model.
        train = hist.iloc[:-1]
        if model is None or since_refit >= refit_every:
            mu, sd = train.mean(), train.std().replace(0, 1.0)
            Z = ((train - mu) / sd).values
            k_here = n_clusters
            if auto_k:
                # Re-derive k from THIS fold's training data only, by the elbow
                # rule. n_clusters then acts purely as a fallback.
                ks = [k for k in range(auto_k[0], auto_k[1] + 1) if k < len(Z)]
                if len(ks) >= 3:
                    inert = [KMeans(n_clusters=k, n_init=10, random_state=seed)
                             .fit(Z).inertia_ for k in ks]
                    k_here = _elbow_k(inert, ks)
            # Fit on the raw ndarray (not the DataFrame) so the later
            # .predict() on an ndarray doesn't trip sklearn's feature-name check.
            model = KMeans(n_clusters=k_here, n_init=10,
                           random_state=seed).fit(Z)
            if learn:
                # Stretch the same top-to-bottom risk range over however many
                # clusters this refit chose, so the overall risk budget stays
                # comparable as k varies.
                ladder = list(np.linspace(max(base_ladder), min(base_ladder), k_here))
            # Sort cluster ids so 0 is always the calmest -- by the vol centroid
            # when vol is one of the selected features, otherwise by overall
            # centroid magnitude (the ablations may exclude vol entirely).
            if "vol" in features:
                key = model.cluster_centers_[:, list(features).index("vol")]
            else:
                key = np.linalg.norm(model.cluster_centers_, axis=1)
            order = np.argsort(key)
            model._rank = {int(orig): int(rank) for rank, orig in enumerate(order)}
            since_refit = 0
        since_refit += 1
        lbl = model.predict(((hist.iloc[[-1]] - mu) / sd).values)[0]
        raw.loc[dt] = model._rank[int(lbl)]
        if learn:
            # Rank clusters on labels assigned so far, using only forward
            # windows that have fully closed before today.
            cutoff = dt - pd.Timedelta(days=int(fwd_window * 1.45))
            maps[dt] = _learned_ladder(raw.loc[:dt], fwd, cutoff, k_here, ladder)

    raw = raw.ffill()
    if hysteresis > 1:
        stable, cur, run, prev = [], np.nan, 0, np.nan
        for v in raw.values:
            if v != prev:
                run, prev = 1, v
            else:
                run += 1
            if not np.isnan(v) and (np.isnan(cur) or run >= hysteresis):
                cur = v
            stable.append(cur)
        raw = pd.Series(stable, index=raw.index)

    if not learn:
        return raw.map(regime_mult).astype(float).fillna(1.0)
    # Apply whichever learned map was in force on each date.
    mp = pd.Series(maps).reindex(prices.index).ffill()
    out = pd.Series(
        [m.get(int(k), 1.0) if isinstance(m, dict) and not pd.isna(k) else 1.0
         for k, m in zip(raw.values, mp.values)], index=prices.index)
    return out.astype(float).fillna(1.0)
