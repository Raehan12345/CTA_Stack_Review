import numpy as np
import pandas as pd
import pytest

from ctadata.cleaning import (fix_spike_and_revert, patch_known_data_issues, merge_pull,
                              GILT_ROLL_FIX_DATE, GILT_ROLL_FIX_RATIO)


def test_gilt_patch_changes_exactly_one_return():
    idx = pd.bdate_range("2011-12-20", "2012-01-10", tz="UTC")
    px = pd.Series(133.29, index=idx)
    px[idx >= GILT_ROLL_FIX_DATE] = 117.06               # the unadjusted roll
    df = pd.DataFrame({"gilt": px, "spx": np.linspace(1200, 1260, len(idx))})
    raw = df.pct_change()
    fixed = patch_known_data_issues(df.copy()).pct_change()
    changed = (raw - fixed).abs() > 1e-12
    assert changed["gilt"].sum() == 1, "only the roll day's return may change"
    assert fixed.loc[GILT_ROLL_FIX_DATE, "gilt"] == pytest.approx(0.0, abs=1e-12)
    assert not changed["spx"].any(), "no other column is touched"


def test_spike_and_revert_removes_each_spike_and_keeps_other_returns():
    idx = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
    s = pd.Series([100, 101, 102, 130, 103, 104, 105, 106, 107, 108.0], index=idx)
    out = fix_spike_and_revert(s, threshold=0.07)
    r = out.pct_change().dropna()
    assert r.abs().max() < 0.07
    # an untouched ordinary day keeps its exact return
    assert out.pct_change().iloc[2] == pytest.approx(s.pct_change().iloc[2])


def test_merge_pull_prefers_the_settled_last_row():
    idx = pd.bdate_range("2026-09-21", periods=5)
    old = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.5]}, index=idx)        # last row an intraday tick
    new = pd.DataFrame({"x": [5.0, 6.0]}, index=[idx[-1], idx[-1] + pd.offsets.BDay()])
    out = merge_pull(old, new)
    assert out.loc[idx[-1], "x"] == 5.0, "the settled close wins on the provisional day"
    assert len(out) == 6


def test_merge_pull_refuses_a_disagreement_on_a_settled_day():
    idx = pd.bdate_range("2026-09-21", periods=5)
    old = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    new = pd.DataFrame({"x": [9.0, 5.0]}, index=[idx[-2], idx[-1]])        # disagrees on day -2
    with pytest.raises(ValueError):
        merge_pull(old, new)


def test_merge_pull_refuses_unknown_columns():
    idx = pd.bdate_range("2026-09-21", periods=2)
    with pytest.raises(ValueError):
        merge_pull(pd.DataFrame({"x": [1.0, 2.0]}, index=idx),
                   pd.DataFrame({"y": [1.0, 2.0]}, index=idx))
