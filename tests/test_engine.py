"""Engine regression tests on synthetic prices. Each one pins a bug that was
found and fixed, so it cannot come back quietly (see docs/BUGFIX_LOG.md)."""
import numpy as np
import pandas as pd
import pytest

from cta import build_trend_nav, build_trend_nav_fast, DEFAULT_PARAMS
from cta.engine import analyze_trailing_stop_events
from cta.hrp import _hrp_scale
from cta.regime import _REGIME_CACHE

SYMS = list("abcdef")


def test_warmup_guard_stops_the_book_opening_at_max_leverage(prices):
    """E4: the 252-day warm-up produced exact-zero returns, a zero vol
    estimate, and leverage pinned at the cap."""
    _, lev_bug, _ = build_trend_nav(prices, symbols=SYMS, warmup_guard=False, **DEFAULT_PARAMS)
    _, lev_fix, _ = build_trend_nav(prices, symbols=SYMS, **DEFAULT_PARAMS)
    cap = DEFAULT_PARAMS["max_leverage"]
    assert (lev_bug >= cap - 1e-12).any(), "the old behaviour hits the cap"
    first_live = lev_fix[lev_fix > 0].index[0]
    assert lev_fix.loc[first_live] < cap, "the guarded book must not open at the cap"


def test_warmup_guard_only_blanks_the_leading_warmup(prices):
    """E5: a later day with every market gapped must NOT be blanked; the test
    is 'has the book ever been live', not 'is it live today'."""
    p = prices.copy()
    late = p.index[1000]
    p.loc[late] = np.nan                       # a full-universe data gap mid-sample
    _, lev, _ = build_trend_nav(p, symbols=SYMS, **DEFAULT_PARAMS)
    window = lev.loc[late:].iloc[1:70]
    assert (window > 0).all(), "one gapped day must not zero leverage for a vol window"


def test_stop_cannot_dodge_the_loss_that_triggers_it(prices):
    """E3: today's return is booked on the weights held BEFORE the stop check.
    So the stopped and unstopped books are identical up to and including the
    first fire day; they can only differ from the day after."""
    kw = dict(DEFAULT_PARAMS, rebalance_freq="M")
    nav0, _, _ = build_trend_nav(prices, symbols=SYMS, **kw)
    nav1, _, _ = build_trend_nav(prices, symbols=SYMS, trailing_stop=True,
                                 atr_window=14, atr_multiplier=2.0, **kw)
    ev = analyze_trailing_stop_events(prices, symbols=SYMS, atr_window=14, atr_multiplier=2.0, **kw)
    assert len(ev), "the synthetic data should fire at least one stop"
    first_fire = ev["stop_date"].min()
    r0, r1 = nav0.pct_change(), nav1.pct_change()
    differs = r0.index[(r0 - r1).abs() > 1e-15]
    assert differs.min() > first_fire


def test_fast_engine_matches_the_loop(prices):
    """The vectorised engine is the loop with every optional layer off. On the
    research data it is bit-identical (difference exactly 0.0); on arbitrary
    data the summation order can differ in the last bit."""
    for freq in ("W", "M"):
        kw = dict(DEFAULT_PARAMS, rebalance_freq=freq)
        a = build_trend_nav(prices, symbols=SYMS, **kw)[0]
        b = build_trend_nav_fast(prices, symbols=SYMS, **kw)[0]
        assert float((a - b).abs().max()) < 1e-12


def test_attribution_reconciles_to_the_nav(prices, universe):
    """E7: attribution is recorded inside the loop, so it sums to the NAV."""
    nav, lev, w, contrib, cost = build_trend_nav(
        prices, symbols=SYMS, apply_costs=False, hrp=True, universe=universe,
        collect_contrib=True, **{**DEFAULT_PARAMS, "rebalance_freq": "M"})
    err = (contrib.sum(axis=1) - cost.sum(axis=1) - nav.pct_change().fillna(0.0)).abs().max()
    assert err < 1e-12


def test_hrp_scale_is_carried_between_rebalances(prices, universe):
    """E2: pre-filling with 1.0 left ffill nothing to fill, so HRP was live
    one day a month."""
    rets = prices.pct_change().fillna(0.0)
    period = prices.index.tz_localize(None).to_period("M").values
    is_rebal = np.r_[False, period[1:] != period[:-1]]
    scale = _hrp_scale(rets, universe, is_rebal, hrp_window=252)
    reb = prices.index[is_rebal]
    d0, d1 = reb[20], reb[21]
    between = scale.loc[d0:d1].iloc[:-1]
    assert (between.nunique() == 1).all(), "scale must be constant between rebalances"
    assert (scale.loc[d0] != 1.0).any(), "HRP must actually tilt, not stay at equal weight"
    # each class's scale averages 1: HRP redistributes within a class, never adds risk
    for cls, syms in universe.items():
        assert scale.loc[d0, syms].mean() == pytest.approx(1.0)


def test_regime_cache_key_tracks_the_sleeve_return(prices, universe):
    """E6: two configurations differing only in HRP must not share a cached
    regime multiplier, because the learned gate ranks clusters by the
    sleeve's own return, which HRP changes."""
    kw = dict(DEFAULT_PARAMS, rebalance_freq="M", regime_gate=True,
              regime_mult_mode="learned", regime_min_train=300)
    n0 = len(_REGIME_CACHE)
    build_trend_nav(prices, symbols=SYMS, **kw)
    build_trend_nav(prices, symbols=SYMS, hrp=True, universe=universe, **kw)
    assert len(_REGIME_CACHE) == n0 + 2


def test_target_weights_are_not_held_weights(prices):
    """The engine returns TARGET weights, which change daily, while the book
    only trades on rebalance days."""
    _, _, w = build_trend_nav(prices, symbols=SYMS, **{**DEFAULT_PARAMS, "rebalance_freq": "M"})
    daily_changes = (w.diff().abs().sum(axis=1) > 0).mean()
    assert daily_changes > 0.5, "targets move most days; turnover from them is meaningless"
