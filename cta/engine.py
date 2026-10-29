"""The backtest engine.

build_trend_nav is the strategy. Per day, in order:

  1. score       barbell trend signal per market, lagged a day (signals.py)
  2. Tier 1      weight = score * instrument target vol / own trailing vol,
                 optionally scaled by the COT, Economic Trend and HRP layers
  3. Tier 2      leverage = portfolio target / trailing vol of the unlevered
                 book, optionally scaled by the regime gate
  4. rebalance   on the first trading day of each period the target weights
                 and leverage are adopted; between rebalances they are held
  5. P&L         today's return is booked on the weights held BEFORE any
                 stop is checked, then costs are charged on turnover

It returns TARGET weights, not held positions. Anything computed from them
(turnover especially) is meaningless; per-instrument attribution is recorded
inside the loop instead (collect_contrib=True) and reconciles to the NAV to
1e-16.
"""
import hashlib
import os

import numpy as np
import pandas as pd

from .config import TRANSACTION_COST_BPS
from .signals import _tsmom_score, _efficiency_ratio, _atr_proxy
from .layers import _vix_target_mult, _cot_throttle_mult, economic_trend_score, _econ_trend_mult
from .hrp import _hrp_scale
from .regime import (_REGIME_CACHE, _REGIME_MULT_DEFAULT, _regime_target_mult,
                     _wasserstein_regime_mult, _wasserstein_mv_regime_mult)

# CTA_DEBUG=1 prints a per-run trace: which layers are live, when the book
# first trades, leverage and terminal NAV. Off by default so sweeps stay quiet.
DEBUG = bool(os.environ.get("CTA_DEBUG"))


def _log(msg):
    if DEBUG:
        print(f"[engine] {msg}")


# Stop-fire and re-entry counts from the most recent run with
# trailing_stop=True (calendar mode). Diagnostics only.
LAST_STOP_STATS: dict = {}


def build_trend_nav(prices: pd.DataFrame, symbols: list[str], short_lookback: int = 40,
                     long_lookback: int = 252, vol_window: int = 60, inst_target_vol: float = 0.20,
                     max_inst_weight: float = 3.0, base_target_vol: float = 0.12, max_leverage: float = 4.0,
                     rebalance_freq: str = "W", vix: pd.Series = None, stress_window: int = 252,
                     stress_scale: float = 1.0, cot: pd.DataFrame = None, cot_window: int = 156,
                     cot_threshold: float = 1.5, cot_shrink: float = 0.5,
                     cot_release_lag_days: int = 3, apply_costs: bool = False,
                     cost_multiplier: float = 1.0, er_gate: bool = False, er_window: int = 40,
                     er_threshold: float = 0.3, universe: dict = None, economic_trend: bool = False,
                     econ: pd.DataFrame = None, eer: pd.DataFrame = None, econ_intl: pd.DataFrame = None,
                     econ_strength: float = 0.5, econ_floor: float = 0.5, econ_ceiling: float = 1.5,
                     econ_z_window: int = 756, hrp: bool = False, hrp_window: int = 252,
                     hrp_lag: int = 0,
                     trailing_stop: bool = False, atr_window: int = 14,
                     atr_multiplier: float = 3.0, stop_reentry: str = "rebalance",
                     stop_cost: bool = False,
                     regime_gate: bool = False, regime_clusters: int = 3,
                     regime_min_train: int = 504, regime_refit_every: int = 13,
                     regime_hysteresis: int = 2, regime_mult: dict = None,
                     regime_method: str = "features", regime_window: int = 63,
                     regime_p: int = 1, regime_projections: int = 48,
                     regime_quantiles: int = 16, regime_standardize: bool = False,
                     regime_features: tuple = ("vol", "corr", "er"),
                     regime_mult_mode: str = "fixed", regime_fwd_window: int = 21,
                     regime_auto_k: tuple = None, regime_macro: pd.DataFrame = None,
                     regime_kmeans: str = "sklearn", regime_rank_by: str = "return",
                     rebalance_band: float = None, rebalance_mode: str = "calendar",
                     band_style: str = "target", band_relative: bool = False,
                     band_on_rebal_only: bool = False,
                     warmup_guard: bool = True,
                     score_mode: str = "sign", score_clip: float = 2.0, skew_tilt: float = 0.0,
                     collect_contrib: bool = False) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Returns (nav, leverage, weights_history), plus per-instrument
    contributions and costs when collect_contrib=True.

    Note that weights_history is the TARGET weight, not the weight actually
    held. Turnover computed from it is meaningless -- it comes out identical
    across configurations. Use collect_contrib, or gross-minus-net Sharpe as a
    cost proxy.

    Optional layers, all off by default:

    vix        shrinks the Tier-2 portfolio target when VIX is elevated.
               Same slot as regime_gate -- running both double-counts.
    cot        throttles Tier-1 inst_target_vol when speculators are crowded
               on the side we are already on.
    er_gate    forces a score to flat when that instrument's Efficiency Ratio
               is below er_threshold.
    economic_trend  scales inst_target_vol by macro agreement. Needs
               `universe`, `econ`, `eer`.
    hrp        within-bucket Hierarchical Risk Parity tilt on
               inst_target_vol. Needs `universe`.
    trailing_stop   ratcheting ATR stop, checked daily rather than only at
               rebalances. stop_reentry="rebalance" (default) leaves a
               stopped position flat until the next rebalance re-reads the
               score; "signal" re-enters it the day its score changes to a
               non-zero value, sized at the leverage in force. stop_cost
               charges the cost table on the stop exit (default False: the
               exit is free, which flatters tight stops). Both need
               calendar rebalancing.
    regime_gate     K-means regime classifier on the Tier-2 target.

    `universe` (asset_class -> symbols) is required by economic_trend and hrp,
    which both need to know which bucket an instrument sits in.

    apply_costs charges TRANSACTION_COST_BPS on turnover in EFFECTIVE
    (leverage-scaled) weight. cost_multiplier scales the whole table for
    stress tests without redefining it.

    warmup_guard (default True) holds the book flat until the portfolio vol
    estimate is built from days that carried a real signal.

    Without it: _tsmom_score needs 252d, so the first year of
    port_ret_unlevered is exact zeros, its rolling std is ~0, and Tier-2
    leverage (target / ~0) pins to max_leverage and stays there. The book then
    opens at the 4x cap sized off artificial flatness. On the 30-instrument
    sample that produced single-day NAV moves of +10% and +23% against a 12%
    ANNUAL target, and 1.93x of free NAV before the strategy had really
    started. Anything measured from 2011 onward is unaffected.

    Set warmup_guard=False only to reproduce a number produced before the
    fix. It is a bug switch, not a modelling choice."""
    if DEBUG:
        on = [n for n, flag in (("vix", vix is not None), ("cot", cot is not None),
                                ("er_gate", er_gate), ("econ", economic_trend),
                                ("hrp", hrp), ("atr", trailing_stop),
                                ("regime", regime_gate), ("costs", apply_costs)) if flag]
        _log(f"{len(symbols)} symbols, {rebalance_freq} rebal, "
             f"target {base_target_vol:.0%}, layers: {', '.join(on) or 'none'}")

    returns = prices[symbols].pct_change().fillna(0.0)
    score = _tsmom_score(prices[symbols], short_lookback, long_lookback,
                         mode=score_mode, vol_window=vol_window, clip=score_clip)

    if er_gate:
        er = _efficiency_ratio(prices[symbols], er_window)
        score = score.where(er > er_threshold, 0.0)

    n = len(prices)
    period = prices.index.tz_localize(None).to_period(rebalance_freq).values
    is_rebal = np.empty(n, dtype=bool)
    is_rebal[1:] = period[1:] != period[:-1]
    is_rebal[0] = False

    effective_inst_target_vol = inst_target_vol
    if cot is not None:
        cot_mult = _cot_throttle_mult(cot, score, prices.index, symbols, cot_window, cot_threshold,
                                       cot_shrink, cot_release_lag_days)
        effective_inst_target_vol = effective_inst_target_vol * cot_mult
    if economic_trend:
        if universe is None or econ is None or eer is None:
            raise ValueError("economic_trend=True requires universe, econ, and eer")
        econ_score = economic_trend_score(prices, symbols, universe, econ, eer, econ_intl, econ_z_window)
        econ_mult = _econ_trend_mult(econ_score, score, econ_strength, econ_floor, econ_ceiling)
        effective_inst_target_vol = effective_inst_target_vol * econ_mult
    if hrp:
        if universe is None:
            raise ValueError("hrp=True requires universe")
        universe_key = tuple(sorted((cls, tuple(syms)) for cls, syms in universe.items()))
        hrp_cache_key = (id(prices), tuple(symbols), rebalance_freq, hrp_window, universe_key, hrp_lag)
        hrp_scale = _hrp_scale(returns, universe, is_rebal, hrp_window, cache_key=hrp_cache_key, lag=hrp_lag)
        effective_inst_target_vol = effective_inst_target_vol * hrp_scale[symbols]

    inst_vol = returns.rolling(vol_window).std().shift(1) * np.sqrt(252)
    inst_weight = (score * effective_inst_target_vol / inst_vol).replace([np.inf, -np.inf], np.nan)
    inst_weight = inst_weight.clip(lower=-max_inst_weight, upper=max_inst_weight).fillna(0.0)
    if skew_tilt:
        # Each market's rolling skew (average of 130/195/260-day windows, as in
        # the CITIC recipe), z-scored against its own expanding history, lagged
        # a day. Tilt = 1 - skew_tilt * z, clipped to [0.5, 1.5]: a market whose
        # skew is unusually negative for it runs larger, unusually positive
        # runs smaller. Time-series, so it needs no breadth. Research only.
        sk = sum(returns.rolling(w).skew() for w in (130, 195, 260)) / 3
        z = (sk - sk.expanding(min_periods=252).mean()) / sk.expanding(min_periods=252).std()
        tilt = (1.0 - skew_tilt * z.shift(1)).clip(0.5, 1.5).fillna(1.0)
        inst_weight = inst_weight * tilt

    # Computed here, BEFORE the regime gate, because a learned gate scores each
    # regime against the sleeve's own unlevered return stream. Unlevered is the
    # right reference: it is the strategy at constant risk, so it does not
    # depend on the leverage the gate is about to set (which would be circular).
    port_ret_unlevered = (returns[symbols] * inst_weight).sum(axis=1)
    if warmup_guard:
        # Blank out the pre-signal run of exact zeros so the vol estimate is
        # NaN (rolling std needs a full window of real observations) rather
        # than ~0.0 -- see warmup_guard in the docstring. leverage_at_rebal's
        # own .fillna(0.0) below then holds the book flat until the estimate
        # is built from vol_window days that all carried a live signal.
        #
        # .cummax() matters: the test is "has the book EVER been live", not
        # "is it live today". Scores go NaN again on any later day where every
        # instrument in this particular universe has a data gap, and blanking
        # those would poison a whole rolling window each time -- forcing
        # leverage to zero for vol_window days at a stretch in the middle of
        # the live sample. Only the leading warm-up run may be blanked.
        port_ret_unlevered = port_ret_unlevered.where(score.notna().any(axis=1).cummax())

    if vix is not None:
        adaptive_target_vol = base_target_vol * _vix_target_mult(vix, prices.index, stress_window, stress_scale)
    else:
        adaptive_target_vol = pd.Series(base_target_vol, index=prices.index)

    if regime_gate:
        # Same slot as the VIX layer -- both scale the Tier-2 portfolio target.
        # Composable with it (they multiply) but note the two are answering the
        # same question from different data, so stacking them double-counts.
        # The regime multiplier depends only on prices/symbols and the regime
        # parameters -- never on the other layers -- so it is identical across
        # every config in a sweep that shares them. Memoized process-lifetime,
        # same pattern and same id(prices) caveat as _HRP_CACHE above.
        _regime_key = (id(prices), tuple(symbols), regime_method, rebalance_freq,
                       regime_clusters, regime_min_train, regime_refit_every,
                       regime_hysteresis, regime_window, regime_p, regime_projections,
                       regime_quantiles, regime_standardize, vol_window, er_window,
                       tuple(regime_features), regime_mult_mode, regime_fwd_window,
                       regime_auto_k, regime_kmeans, regime_rank_by,
                       None if regime_macro is None else
                       (tuple(regime_macro.columns), int(regime_macro.notna().sum().sum())),
                       tuple(sorted((regime_mult or _REGIME_MULT_DEFAULT).items())))
        if regime_mult_mode not in ("fixed", "learned"):
            raise ValueError("regime_mult_mode must be 'fixed' or 'learned', "
                             f"got {regime_mult_mode!r}")
        # In learned mode the gate ranks its clusters by the sleeve's own
        # measured forward return in each, using training data only.
        _sr = port_ret_unlevered if regime_mult_mode == "learned" else None
        # That return stream depends on hrp, universe, economic_trend, cot,
        # er_gate, the lookbacks, inst_target_vol and more -- none of which are
        # in the key above. Listing them would rot the moment a new layer is
        # added, so fingerprint the series itself: two configurations share a
        # cached multiplier only if the book it was derived from is identical.
        # Without this, running plain+regime before hrp+regime silently served
        # the second the first one's gate (measured: 0.4714 -> 0.4830).
        _sr_key = (None if _sr is None else
                   hashlib.sha1(np.ascontiguousarray(_sr.values, dtype="float64")).hexdigest())
        _regime_key = _regime_key + (_sr_key,)
        if _regime_key in _REGIME_CACHE:
            _rm = _REGIME_CACHE[_regime_key]
        elif regime_method == "wasserstein":
            _rm = _wasserstein_regime_mult(
                prices, symbols, is_rebal, regime_clusters, regime_min_train,
                regime_refit_every, regime_hysteresis, regime_mult,
                regime_window=regime_window, p=regime_p,
                sleeve_ret=_sr, fwd_window=regime_fwd_window)
        elif regime_method == "wasserstein_mv":
            _rm = _wasserstein_mv_regime_mult(
                prices, symbols, is_rebal, regime_clusters, regime_min_train,
                regime_refit_every, regime_hysteresis, regime_mult,
                regime_window=regime_window, p=regime_p,
                n_projections=regime_projections, n_quantiles=regime_quantiles,
                standardize=regime_standardize,
                sleeve_ret=_sr, fwd_window=regime_fwd_window)
        elif regime_method == "features":
            _rm = _regime_target_mult(
                prices, symbols, is_rebal, regime_clusters, regime_min_train,
                regime_refit_every, regime_hysteresis, regime_mult,
                vol_window=vol_window, er_window=er_window,
                features=regime_features, auto_k=regime_auto_k, macro=regime_macro,
                sleeve_ret=_sr, fwd_window=regime_fwd_window, kmeans_impl=regime_kmeans,
                rank_by=regime_rank_by)
        else:
            raise ValueError("regime_method must be 'features', 'wasserstein' or "
                             f"'wasserstein_mv', got {regime_method!r}")
        _REGIME_CACHE[_regime_key] = _rm
        adaptive_target_vol = adaptive_target_vol * _rm

    port_vol_lagged = port_ret_unlevered.rolling(vol_window).std().shift(1) * np.sqrt(252)
    leverage_at_rebal = (adaptive_target_vol / port_vol_lagged).clip(upper=max_leverage).fillna(0.0)

    if apply_costs:
        missing = [s for s in symbols if s not in TRANSACTION_COST_BPS]
        if missing:
            raise ValueError(f"apply_costs=True but no TRANSACTION_COST_BPS entry for {missing}")
        cost_bps = pd.Series({s: TRANSACTION_COST_BPS[s] * cost_multiplier for s in symbols})

    if stop_reentry not in ("rebalance", "signal"):
        raise ValueError(f"stop_reentry must be 'rebalance' or 'signal', got {stop_reentry!r}")
    if (stop_reentry == "signal" or stop_cost) and (rebalance_mode != "calendar"
                                                    or rebalance_band is not None):
        raise NotImplementedError("stop_reentry='signal' and stop_cost are implemented "
                                  "for calendar rebalancing only")
    if trailing_stop:
        atr = _atr_proxy(prices[symbols], atr_window)
        best_price = pd.Series(np.nan, index=symbols)
        position_side = pd.Series(0.0, index=symbols)
        # Names knocked out by the stop since the last rebalance, and the
        # exit cost decided at today's close, charged tomorrow.
        stopped_out = pd.Series(False, index=symbols)
        carry_cost = pd.Series(0.0, index=symbols)
        n_fires = n_reentries = 0

    nav = 1.0
    navs = np.empty(n)
    leverages = np.empty(n)
    navs[0] = nav
    leverages[0] = 0.0
    # Return attribution. Recorded inside the loop from the SAME quantities that
    # form port_ret, so the per-instrument contributions reconcile to the NAV
    # exactly rather than being reconstructed afterwards from target weights
    # (which are not the weights actually held -- see weights_history below).
    if collect_contrib:
        if rebalance_mode != "calendar" or rebalance_band is not None:
            raise NotImplementedError(
                "collect_contrib is implemented for calendar rebalancing only; "
                "signal and band modes track effective weights per instrument "
                "and would need their own attribution.")
        contrib = np.zeros((n, len(symbols)))
        contrib_cost = np.zeros((n, len(symbols)))
    current_weights = pd.Series(0.0, index=symbols)
    current_leverage = 0.0
    # No-trade-band mode: positions are held until the target drifts far enough
    # away to be worth the commission, instead of being reset on a calendar.
    # Tracks EFFECTIVE weight (leverage x instrument weight) directly, because
    # once trading is per-instrument there is no single portfolio leverage --
    # each position carries the leverage in force when it was last traded.
    band_mode = rebalance_band is not None
    if band_mode:
        current_eff = pd.Series(0.0, index=symbols)
        traded_any = False

    # Signal-only mode: start flat, open a position when an instrument's barbell
    # score changes, then HOLD IT UNCHANGED until the score changes again. There
    # is no periodic resizing at all -- positions are left to drift with price,
    # exactly like buying a contract and not touching it. A strengthening
    # conviction (score 0.5 -> 1.0) is itself a score change, so it tops the
    # position up; a weakening or flipping signal trims or reverses it.
    #
    # Weights are drifted each day by w * (1+r_i) / (1+r_portfolio), which is
    # what genuinely holding a fixed position does to your portfolio weights.
    # Holding the WEIGHT fixed instead would silently mean trading every day to
    # maintain it, which is the thing this mode exists to avoid.
    signal_mode = rebalance_mode == "signal"
    if rebalance_mode not in ("calendar", "signal"):
        raise ValueError(f"rebalance_mode must be 'calendar' or 'signal', got {rebalance_mode!r}")
    if signal_mode:
        current_eff = pd.Series(0.0, index=symbols)
        last_score = pd.Series(np.nan, index=symbols)

    for i in range(1, n):
        cost = 0.0
        if signal_mode:
            lev_i = float(leverage_at_rebal.iloc[i])
            sc = score.iloc[i]
            changed = sc.notna() & (sc != last_score)
            if lev_i > 0 and changed.any():
                target = lev_i * inst_weight.iloc[i]
                if apply_costs:
                    d = (target - current_eff).abs().where(changed, 0.0)
                    cost = float((d * cost_bps).sum())
                current_eff = current_eff.where(~changed, target)
                last_score = last_score.where(~changed, sc)
                if trailing_stop:
                    today_price = prices[symbols].iloc[i]
                    new_side = np.sign(current_eff)
                    reset = new_side != position_side
                    best_price = best_price.where(~reset, today_price)
                    position_side = new_side
            r_i = returns[symbols].iloc[i]
            port_ret = float((r_i * current_eff).sum()) - cost
            nav *= (1 + port_ret)
            navs[i] = nav
            leverages[i] = float(current_eff.abs().sum())
            # let the held positions drift -- this is what "no rebalancing" means
            if port_ret > -0.99:
                current_eff = current_eff * (1 + r_i) / (1 + port_ret)
            if trailing_stop:
                today_price = prices[symbols].iloc[i]
                is_long = position_side > 0
                is_short = position_side < 0
                best_price = pd.Series(
                    np.where(is_long, np.maximum(best_price, today_price),
                             np.where(is_short, np.minimum(best_price, today_price), best_price)),
                    index=symbols)
                stop_level = pd.Series(
                    np.where(is_long, best_price - atr.iloc[i] * atr_multiplier,
                             np.where(is_short, best_price + atr.iloc[i] * atr_multiplier, np.nan)),
                    index=symbols)
                stopped = ((is_long & (today_price <= stop_level)) |
                           (is_short & (today_price >= stop_level)))
                if stopped.any():
                    current_eff = current_eff.where(~stopped, 0.0)
                    position_side = position_side.where(~stopped, 0.0)
            continue

        if band_mode:
            lev_i = float(leverage_at_rebal.iloc[i])
            # band_on_rebal_only (2026-09-16): apply the band only on calendar
            # rebalance days, as a filter on small resizes. Evaluated daily,
            # the sign score changes ~730 times a year across 13 markets
            # against ~74 sampled monthly, and a band cannot filter a sign
            # flip, so a daily band is daily rebalancing on a noisy signal.
            if band_on_rebal_only and not is_rebal[i]:
                lev_i = 0.0
            if lev_i > 0:
                target_eff = lev_i * inst_weight.iloc[i]
                gap = target_eff - current_eff
                dev = gap.abs()
                # Trade only the names whose target has moved further than the
                # band. A signal flip or a strengthening conviction both move
                # the target enough to trigger; a small drift in trailing
                # volatility does not.
                # band_relative: the band is a fraction of |target| rather than
                # an absolute effective weight, so a small position is not
                # re-traded on noise that would be invisible on a large one.
                width = rebalance_band * target_eff.abs() if band_relative else rebalance_band
                trade = dev > width
                if not traded_any:
                    trade = target_eff.abs() > 0        # establish the book once
                    traded_any = bool(trade.any())
                if trade.any():
                    # band_style="edge" (2026-09-16): trade back to the nearer
                    # band edge, not all the way to target. The original
                    # "target" style traded the full gap on every trigger and
                    # churned more than weekly rebalancing; that result was
                    # withdrawn. Trading to the edge is the standard no-trade
                    # region rule and is what the withdrawn test meant to do.
                    if band_style == "edge":
                        new_eff = target_eff - np.sign(gap) * width
                        new_eff = new_eff.where(trade, current_eff)
                        # the first trade establishes the book at target
                        if traded_any and not (current_eff != 0).any():
                            new_eff = target_eff.where(trade, current_eff)
                    elif band_style == "target":
                        new_eff = target_eff
                    else:
                        raise ValueError(f"band_style must be 'target' or 'edge', got {band_style!r}")
                    if apply_costs:
                        cost = float(((new_eff - current_eff).abs().where(trade, 0.0) * cost_bps).sum())
                    current_eff = current_eff.where(~trade, new_eff)
                    if trailing_stop:
                        today_price = prices[symbols].iloc[i]
                        new_side = np.sign(current_eff)
                        reset = new_side != position_side
                        best_price = best_price.where(~reset, today_price)
                        position_side = new_side
            port_ret = float((returns[symbols].iloc[i] * current_eff).sum()) - cost
            nav *= (1 + port_ret)
            navs[i] = nav
            leverages[i] = float(current_eff.abs().sum())
            if trailing_stop:
                today_price = prices[symbols].iloc[i]
                is_long = position_side > 0
                is_short = position_side < 0
                best_price = pd.Series(
                    np.where(is_long, np.maximum(best_price, today_price),
                             np.where(is_short, np.minimum(best_price, today_price), best_price)),
                    index=symbols)
                stop_level = pd.Series(
                    np.where(is_long, best_price - atr.iloc[i] * atr_multiplier,
                             np.where(is_short, best_price + atr.iloc[i] * atr_multiplier, np.nan)),
                    index=symbols)
                stopped = ((is_long & (today_price <= stop_level)) |
                           (is_short & (today_price >= stop_level)))
                if stopped.any():
                    current_eff = current_eff.where(~stopped, 0.0)
                    position_side = position_side.where(~stopped, 0.0)
            continue

        cost_vec = None
        if trailing_stop and carry_cost.any():
            cost_vec = carry_cost.copy()
            carry_cost = pd.Series(0.0, index=symbols)
        if is_rebal[i] and not pd.isna(leverage_at_rebal.iloc[i]):
            new_weights = inst_weight.iloc[i]
            new_leverage = float(leverage_at_rebal.iloc[i])
            if apply_costs:
                # Turnover on *effective* (leverage-scaled) weight -- what
                # actually gets bought/sold, not the raw instrument weight.
                turnover = (new_leverage * new_weights - current_leverage * current_weights).abs()
                cost_vec = turnover * cost_bps if cost_vec is None else cost_vec + turnover * cost_bps
            current_weights = new_weights
            current_leverage = new_leverage
            if trailing_stop:
                today_price = prices[symbols].iloc[i]
                new_side = np.sign(current_weights)
                reset = new_side != position_side
                best_price = best_price.where(~reset, today_price)
                position_side = new_side
                stopped_out[:] = False
        elif trailing_stop and stop_reentry == "signal" and stopped_out.any():
            # A stopped name comes back the day its score changes to
            # something non-zero: a new signal, not a bounce that would only
            # re-trigger the same stop. Same convention as a rebalance: traded
            # at today's close, earns today's return, pays the cost table.
            sc, sc_prev = score.iloc[i], score.iloc[i - 1]
            reenter = stopped_out & sc.notna() & (sc != sc_prev) & (sc != 0)
            if reenter.any():
                w_new = inst_weight.iloc[i].where(reenter, current_weights)
                if apply_costs:
                    re_cost = (current_leverage * w_new.where(reenter, 0.0)).abs() * cost_bps
                    cost_vec = re_cost if cost_vec is None else cost_vec + re_cost
                current_weights = w_new
                today_price = prices[symbols].iloc[i]
                best_price = best_price.where(~reenter, today_price)
                position_side = position_side.where(~reenter, np.sign(w_new))
                stopped_out = stopped_out & ~reenter
                n_reentries += int(reenter.sum())
        if cost_vec is not None:
            cost = float(cost_vec.sum())

        # Realize today's return using the weights in force *before* any
        # stop check below -- a position held into today's close earns
        # today's full move, whether or not that move is what ends up
        # triggering the stop. Checking the stop against today's price and
        # then zeroing today's weight (the original, buggy order) let the
        # backtest dodge the very loss that triggered the stop -- lookahead,
        # not genuine improvement, and it's what produced an implausible
        # ~2.5 Sharpe on the first attempt. The stop below only ever affects
        # tomorrow onward.
        gross_i = current_leverage * (returns[symbols].iloc[i] * current_weights)
        port_ret = float(gross_i.sum()) - cost
        nav *= (1 + port_ret)
        navs[i] = nav
        leverages[i] = current_leverage
        if collect_contrib:
            # Per-instrument contribution to TODAY's portfolio return, net of
            # that instrument's own share of the trading cost. By construction
            # these sum to port_ret exactly, so the attribution reconciles to
            # the NAV rather than approximating it.
            contrib[i] = gross_i.values
            if cost_vec is not None:
                contrib_cost[i] = cost_vec.reindex(symbols).values

        if trailing_stop:
            today_price = prices[symbols].iloc[i]
            is_long = position_side > 0
            is_short = position_side < 0
            # Ratchet the high/low-water-mark -- only ever moves in the
            # position's favor, never loosens.
            best_price = pd.Series(
                np.where(is_long, np.maximum(best_price, today_price),
                         np.where(is_short, np.minimum(best_price, today_price), best_price)),
                index=symbols)
            stop_level = pd.Series(
                np.where(is_long, best_price - atr.iloc[i] * atr_multiplier,
                         np.where(is_short, best_price + atr.iloc[i] * atr_multiplier, np.nan)),
                index=symbols)
            stopped = ((is_long & (today_price <= stop_level)) | (is_short & (today_price >= stop_level)))
            if stopped.any():
                if stop_cost and apply_costs:
                    carry_cost = carry_cost + (current_leverage * current_weights.where(stopped, 0.0)).abs() * cost_bps
                current_weights = current_weights.where(~stopped, 0.0)
                position_side = position_side.where(~stopped, 0.0)
                stopped_out = stopped_out | stopped
                n_fires += int(stopped.sum())

    if trailing_stop and not (signal_mode or band_mode):
        LAST_STOP_STATS.clear()
        LAST_STOP_STATS.update(fires=n_fires, reentries=n_reentries)
    nav_out = pd.Series(navs, index=prices.index, name="nav")
    lev_out = pd.Series(leverages, index=prices.index, name="leverage")

    if DEBUG:
        live = lev_out[lev_out > 0]
        first = live.index[0].date() if len(live) else None
        capped = float((lev_out >= max_leverage - 1e-12).mean())
        _log(f"first live {first}, leverage median {live.median():.2f} "
             f"max {lev_out.max():.2f}, {capped:.1%} of days at the {max_leverage}x cap")
        _log(f"terminal NAV {nav_out.iloc[-1]:.4f}x")
        if capped > 0.05:
            _log(f"WARNING: {capped:.0%} of days pinned at the cap -- check the "
                 f"vol estimate, this is what warmup_guard exists to prevent")

    if collect_contrib:
        return (nav_out, lev_out, inst_weight,
                pd.DataFrame(contrib, index=prices.index, columns=symbols),
                pd.DataFrame(contrib_cost, index=prices.index, columns=symbols))
    return nav_out, lev_out, inst_weight


def build_trend_nav_fast(prices: pd.DataFrame, symbols: list[str], short_lookback: int = 40,
                          long_lookback: int = 252, vol_window: int = 60, inst_target_vol: float = 0.20,
                          max_inst_weight: float = 3.0, base_target_vol: float = 0.12,
                          max_leverage: float = 4.0, rebalance_freq: str = "W",
                          warmup_guard: bool = True) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Vectorized equivalent of build_trend_nav() for the no-optional-layers
    case only (i.e. every parameter build_trend_nav exposes beyond what
    this signature also takes -- cot, economic_trend, hrp, trailing_stop,
    apply_costs, vix, er_gate -- must be off/None). Deliberately excludes
    those from its own signature rather than accepting and ignoring them,
    so a caller can't silently ask for a layer this function doesn't
    implement.

    Exists purely for throughput: build_trend_nav's day-by-day Python loop
    profiles as ~95%+ pandas label-indexing overhead, not real computation
    -- with every optional layer off, that loop reduces to "hold the
    weights set at the last rebalance until the next one," which is a
    forward-fill and needs no loop at all. Verified bit-identical
    (max|nav difference| == 0.0) against build_trend_nav across 15 random
    instrument combinations of every size 5-30 (tests/test_engine.py keeps
    checking it). ~178x faster in that verification (23ms vs.
    4.1s per call, n=15). Built for Monte Carlo / exhaustive-search work
    over many combinations, where build_trend_nav's per-call cost is the
    binding constraint; not a replacement for build_trend_nav
    anywhere an optional layer is needed."""
    returns = prices[symbols].pct_change().fillna(0.0)
    score = _tsmom_score(prices[symbols], short_lookback, long_lookback)

    n = len(prices)
    period = prices.index.tz_localize(None).to_period(rebalance_freq).values
    is_rebal = np.empty(n, dtype=bool)
    is_rebal[1:] = period[1:] != period[:-1]
    is_rebal[0] = False

    inst_vol = returns.rolling(vol_window).std().shift(1) * np.sqrt(252)
    inst_weight = (score * inst_target_vol / inst_vol).replace([np.inf, -np.inf], np.nan)
    inst_weight = inst_weight.clip(lower=-max_inst_weight, upper=max_inst_weight).fillna(0.0)

    adaptive_target_vol = pd.Series(base_target_vol, index=prices.index)
    port_ret_unlevered = (returns[symbols] * inst_weight).sum(axis=1)
    if warmup_guard:
        # Mirrors build_trend_nav's own warmup_guard -- see its docstring,
        # including why this is .cummax() and not the instantaneous test.
        port_ret_unlevered = port_ret_unlevered.where(score.notna().any(axis=1).cummax())
    port_vol_lagged = port_ret_unlevered.rolling(vol_window).std().shift(1) * np.sqrt(252)
    leverage_at_rebal = (adaptive_target_vol / port_vol_lagged).clip(upper=max_leverage).fillna(0.0)

    rebal_mask = pd.Series(is_rebal, index=prices.index)
    held_weights = inst_weight.where(rebal_mask).ffill().fillna(0.0)
    held_leverage = leverage_at_rebal.where(rebal_mask).ffill().fillna(0.0)

    port_ret = held_leverage.values * (returns[symbols].values * held_weights.values).sum(axis=1)
    nav_vals = np.empty(n)
    nav_vals[0] = 1.0
    nav_vals[1:] = np.cumprod(1.0 + port_ret[1:])

    return (pd.Series(nav_vals, index=prices.index, name="nav"),
            pd.Series(held_leverage.values, index=prices.index, name="leverage"),
            inst_weight)


def analyze_trailing_stop_events(prices: pd.DataFrame, symbols: list[str],
                                  short_lookback: int = 40, long_lookback: int = 252, vol_window: int = 60,
                                  inst_target_vol: float = 0.20, max_inst_weight: float = 3.0,
                                  base_target_vol: float = 0.12, max_leverage: float = 4.0,
                                  rebalance_freq: str = "W", atr_window: int = 14,
                                  atr_multiplier: float = 3.0) -> pd.DataFrame:
    """Diagnostic companion to build_trend_nav's trailing_stop -- NOT the
    NAV builder itself, a separate pass that mirrors its exact stop logic
    (same lookahead-safe today's-return-before-today's-stop-check
    ordering) but records every trigger instead of just acting on it.

    For each stop event, computes the counterfactual: what would this
    instrument's return have been from the stop date through the *next
    scheduled rebalance* had the position simply been held at its
    pre-stop weight/leverage instead of going flat? That's the fair
    comparison -- in a no-stop world the weight wouldn't have changed
    until that same next rebalance anyway, so "held to the next
    rebalance" is what actually would have happened, not an arbitrary
    horizon.

    Returns one row per stop event: symbol, stop_date, next_rebal_date,
    side (+1 long / -1 short), the weight/leverage that was in force,
    and counterfactual_return_% -- positive means the stop cost real
    profit (a whipsaw that would have kept running), negative means the
    stop avoided a real loss (the position kept moving against its
    original direction). Aggregate counterfactual_return_% directly to
    answer "how much did the stop cost/save on average" -- MUST be kept
    in sync with build_trend_nav's own trailing_stop block if that logic
    ever changes."""
    returns = prices[symbols].pct_change().fillna(0.0)
    score = _tsmom_score(prices[symbols], short_lookback, long_lookback)
    inst_vol = returns.rolling(vol_window).std().shift(1) * np.sqrt(252)
    inst_weight = (score * inst_target_vol / inst_vol).replace([np.inf, -np.inf], np.nan)
    inst_weight = inst_weight.clip(lower=-max_inst_weight, upper=max_inst_weight).fillna(0.0)

    n = len(prices)
    period = prices.index.tz_localize(None).to_period(rebalance_freq).values
    is_rebal = np.empty(n, dtype=bool)
    is_rebal[1:] = period[1:] != period[:-1]
    is_rebal[0] = False
    rebal_idx = np.flatnonzero(is_rebal)

    port_ret_unlevered = (returns[symbols] * inst_weight).sum(axis=1)
    port_vol_lagged = port_ret_unlevered.rolling(vol_window).std().shift(1) * np.sqrt(252)
    leverage_at_rebal = (base_target_vol / port_vol_lagged).clip(upper=max_leverage).fillna(0.0)

    atr = _atr_proxy(prices[symbols], atr_window)
    best_price = pd.Series(np.nan, index=symbols)
    position_side = pd.Series(0.0, index=symbols)
    current_weights = pd.Series(0.0, index=symbols)
    current_leverage = 0.0

    events = []
    for i in range(1, n):
        if is_rebal[i] and not pd.isna(leverage_at_rebal.iloc[i]):
            current_weights = inst_weight.iloc[i]
            current_leverage = float(leverage_at_rebal.iloc[i])
            today_price = prices[symbols].iloc[i]
            new_side = np.sign(current_weights)
            reset = new_side != position_side
            best_price = best_price.where(~reset, today_price)
            position_side = new_side

        today_price = prices[symbols].iloc[i]
        is_long = position_side > 0
        is_short = position_side < 0
        best_price = pd.Series(
            np.where(is_long, np.maximum(best_price, today_price),
                     np.where(is_short, np.minimum(best_price, today_price), best_price)),
            index=symbols)
        stop_level = pd.Series(
            np.where(is_long, best_price - atr.iloc[i] * atr_multiplier,
                     np.where(is_short, best_price + atr.iloc[i] * atr_multiplier, np.nan)),
            index=symbols)
        stopped = ((is_long & (today_price <= stop_level)) | (is_short & (today_price >= stop_level)))

        if stopped.any():
            next_rebal_pos = np.searchsorted(rebal_idx, i, side="right")
            next_rebal_i = rebal_idx[next_rebal_pos] if next_rebal_pos < len(rebal_idx) else n - 1
            next_rebal_date = prices.index[next_rebal_i]
            stop_date = prices.index[i]
            for sym in symbols:
                if not stopped[sym]:
                    continue
                price_now = prices[sym].iloc[i]
                price_next_rebal = prices[sym].iloc[next_rebal_i]
                counterfactual_ret = (current_leverage * current_weights[sym]
                                       * (price_next_rebal / price_now - 1))
                events.append({
                    "symbol": sym, "stop_date": stop_date, "next_rebal_date": next_rebal_date,
                    "side": int(position_side[sym]), "weight": current_weights[sym],
                    "leverage": current_leverage, "counterfactual_return_%": counterfactual_ret * 100,
                })
            current_weights = current_weights.where(~stopped, 0.0)
            position_side = position_side.where(~stopped, 0.0)

    return pd.DataFrame(events)


def build_fast_slow_nav(prices: pd.DataFrame, symbols: list[str], short_lookback: int = 40,
                         long_lookback: int = 252, vol_window: int = 60, inst_target_vol: float = 0.20,
                         max_inst_weight: float = 3.0, base_target_vol: float = 0.12, max_leverage: float = 4.0,
                         rebalance_freq: str = "W", **kwargs) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Alternative to the blended barbell score: runs the fast
    (short_lookback) and slow (long_lookback) horizons as two fully
    independent sub-sleeves -- each gets its own Tier-1/Tier-2 vol
    targeting and its own leverage decision -- combined by *summing
    returns* at the portfolio level (the same additive-stacking
    mechanic this whole project already uses for SPX + CTA), rather
    than blending the two signals into one score before sizing.

    The structural difference from the baseline barbell: when the two
    horizons disagree, the blended score cancels toward zero and the
    instrument goes flat. Here, disagreement instead means one sub-sleeve
    holds a long position and the other holds an independently-sized
    short on the *same* instrument at the *same* time -- they only net
    to exactly zero if both sub-sleeves happen to size the position
    identically, not automatically. "Sometimes a fast and slow view
    genuinely warrant different position sizes, not just an averaged
    direction" is the whole premise being tested.

    Reuses build_trend_nav() itself for each sub-sleeve (short_lookback
    == long_lookback collapses the barbell formula to a pure
    single-horizon score -- no new signal code needed) rather than
    reimplementing the sizing/NAV machinery. Each sub-sleeve targets
    HALF of base_target_vol by default -- a natural risk-parity split
    between the two speeds, not a fitted choice; the two streams' actual
    correlation determines whether the combined book ends up above or
    below the original single-sleeve target, which is itself part of
    what this test is meant to reveal, not something to force back to
    exactly base_target_vol.

    Returns (nav_combined, ret_fast, ret_slow) -- the two standalone
    return series are returned alongside the combined NAV specifically
    so their correlation and individual behavior can be inspected, not
    just the blended result."""
    half_vol = base_target_vol / 2
    nav_fast, _, _ = build_trend_nav(prices, symbols=symbols, short_lookback=short_lookback,
                                      long_lookback=short_lookback, vol_window=vol_window,
                                      inst_target_vol=inst_target_vol, max_inst_weight=max_inst_weight,
                                      base_target_vol=half_vol, max_leverage=max_leverage,
                                      rebalance_freq=rebalance_freq, **kwargs)
    nav_slow, _, _ = build_trend_nav(prices, symbols=symbols, short_lookback=long_lookback,
                                      long_lookback=long_lookback, vol_window=vol_window,
                                      inst_target_vol=inst_target_vol, max_inst_weight=max_inst_weight,
                                      base_target_vol=half_vol, max_leverage=max_leverage,
                                      rebalance_freq=rebalance_freq, **kwargs)
    ret_fast = nav_fast.pct_change().fillna(0.0)
    ret_slow = nav_slow.pct_change().fillna(0.0)
    nav_combined = (1 + ret_fast + ret_slow).cumprod()
    return nav_combined, ret_fast, ret_slow
