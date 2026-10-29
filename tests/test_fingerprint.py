"""Behavioural fingerprint of the engine on the research data.

Run before and after any change that is meant to be cosmetic. Each case is a
configuration that was published at some point; its NAV is hashed and must
not move. The hashes in fingerprints.json were produced by the original
engine before it was reorganised into this package, and this package
reproduces all of them bit for bit.

Skipped when CTADATA_DIR is not set (the data is licensed and not included).
"""
import hashlib
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("CTADATA_DIR"),
                                reason="needs the licensed research data (CTADATA_DIR)")

HERE = os.path.dirname(os.path.abspath(__file__))


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes()).hexdigest()[:16]


def _cases():
    from cta import build_trend_nav, build_trend_nav_fast, DEFAULT_PARAMS
    from ctadata import load_cta_data, A12, UNTRADEABLE, buckets
    from ctadata.macro import load_econ, load_eer, load_econ_intl
    df = load_cta_data()
    pool = sorted(s for s in df.columns if s not in UNTRADEABLE)
    uni = buckets(A12)
    econ = dict(economic_trend=True, econ=load_econ(), eer=load_eer(), econ_intl=load_econ_intl())
    m = {**DEFAULT_PARAMS, "rebalance_freq": "M"}
    gate = dict(regime_gate=True, regime_mult_mode="learned",
                regime_features=("vol", "corr", "er", "disp", "agree"), regime_auto_k=(2, 10))
    return df, {
        "fast_pool_weekly": lambda: build_trend_nav_fast(df, symbols=pool, **DEFAULT_PARAMS),
        "a12_monthly_net": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, **m),
        "a12_hrp": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, hrp=True, universe=uni, **m),
        "a12_atr": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, trailing_stop=True,
                                           atr_window=14, atr_multiplier=3.0, **m),
        "a12_econ_hrp": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, hrp=True,
                                                universe=uni, **econ, **m),
        "a12_regime_sklearn": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, **gate, **m),
        "pool_weekly_net": lambda: build_trend_nav(df, symbols=pool, apply_costs=True, **DEFAULT_PARAMS),
        "CHOSEN": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, hrp=True, universe=uni,
                                          hrp_lag=1, regime_kmeans="portable", **gate,
                                          **{**m, "base_target_vol": 0.13}),
        "wk_means": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, regime_gate=True,
                                            regime_method="wasserstein", **m),
        "band_edge": lambda: build_trend_nav(df, symbols=A12, apply_costs=True, rebalance_band=0.2,
                                             band_style="edge", band_relative=True,
                                             band_on_rebal_only=True, **m),
        "signal_mode": lambda: build_trend_nav(df, symbols=A12, apply_costs=True,
                                               rebalance_mode="signal", **m),
        "continuous": lambda: build_trend_nav(df, symbols=A12, apply_costs=True,
                                              score_mode="continuous", **m),
    }


def compute() -> dict:
    df, cases = _cases()
    out = {"data": _h(df.values.ravel())}
    for name, fn in cases.items():
        r = fn()
        out[name] = {"nav": _h(r[0].values), "lev": _h(r[1].values)}
    return out


def test_fingerprints_unchanged():
    expected = json.load(open(os.path.join(HERE, "fingerprints.json")))
    got = compute()
    moved = [k for k in expected if got.get(k) != expected[k]]
    assert not moved, f"behaviour changed in: {moved}"


if __name__ == "__main__":
    json.dump(compute(), open(os.path.join(HERE, "fingerprints.json"), "w"), indent=1)
    print("wrote fingerprints.json")
