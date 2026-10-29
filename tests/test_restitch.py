"""Restitching across contract rolls, on synthetic contracts with a known roll
jump. No licensed data needed."""
import numpy as np
import pandas as pd
import pytest

from ctadata.restitch import (stitched_returns, stitched_index, validated_rolls,
                              roll_ratio, ratio_splice, roll_calendar)


def two_contract_world(n=300, roll_at=150, basis=0.05, seed=1):
    """A front and a second generic for one market. The underlying moves the
    same for both contracts; the second trades at a constant `basis` premium
    (contango). On day `roll_at` the front generic jumps onto the next
    contract, which until yesterday was the second generic."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    spot = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    near = pd.Series(spot, index=idx)
    far = near * (1 + basis)
    nxt = far * (1 + basis)                       # the contract after `far`
    front = near.copy()
    front.iloc[roll_at:] = far.iloc[roll_at:]     # the generic rolls: near -> far
    second = far.copy()
    second.iloc[roll_at:] = nxt.iloc[roll_at:]
    rolls = pd.Series(False, index=idx)
    rolls.iloc[roll_at] = True
    return (front.to_frame("x"), second.to_frame("x"), rolls.to_frame("x"),
            near.pct_change().to_frame("x"), roll_at)


def test_raw_generic_books_the_basis_as_a_fake_return():
    front, second, rolls, true_r, k = two_contract_world()
    raw = front.pct_change()
    fake = raw["x"].iloc[k] - true_r["x"].iloc[k]
    assert fake == pytest.approx(0.05, rel=0.05), "the roll should show the basis as a jump"


def test_stitched_return_on_the_roll_day_is_the_true_return():
    front, second, rolls, true_r, k = two_contract_world()
    st = stitched_returns(front, second, rolls)
    # F1_t / F2_{t-1}: the contract rolled into is the one the second generic held yesterday
    assert st["x"].iloc[k] == pytest.approx(true_r["x"].iloc[k], abs=1e-12)
    # ordinary days are untouched
    others = [i for i in range(1, len(st)) if i != k]
    assert np.allclose(st["x"].iloc[others], true_r["x"].iloc[others])


def test_stitched_index_has_no_level_jump():
    front, second, rolls, true_r, k = two_contract_world()
    ix = stitched_index(front, second, rolls)["x"]
    jump = abs(ix.iloc[k] / ix.iloc[k - 1] - 1)
    assert jump < 0.05, "the stitched index must not jump by the basis"


def test_validation_passes_a_true_calendar_and_rejects_a_polluted_one():
    """A correct calendar puts its roll days on the jumps. A calendar with
    extra inactive months (gold lists every month, trades five) flags many
    ordinary days and its ratio collapses."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=1200, tz="UTC")
    r = rng.normal(0, 0.01, len(idx))
    true_rolls = np.zeros(len(idx), bool)
    true_rolls[60::63] = True                       # quarterly
    r[true_rolls] += 0.04                           # unadjusted roll jumps
    px = pd.Series(100 * np.cumprod(1 + r), index=idx)
    good = pd.Series(true_rolls, index=idx)
    polluted = pd.Series(np.zeros(len(idx), bool), index=idx)
    polluted.iloc[::21] = True                      # "monthly" listings
    assert roll_ratio(px, good) > 2.0
    assert roll_ratio(px, polluted) < 1.2

    frame = px.to_frame("m")
    gaps = pd.DataFrame({"m": 3.0}, index=idx)
    mask, report = validated_rolls(frame, gaps, good.to_frame("m"))
    assert bool(report.loc["m", "passed"]) and mask["m"].sum() == good.sum()
    mask, report = validated_rolls(frame, gaps, polluted.to_frame("m"))
    assert not bool(report.loc["m", "passed"]) and mask["m"].sum() == 0


def test_roll_calendar_from_a_chain():
    idx = pd.bdate_range("2024-01-01", "2024-12-31", tz="UTC")
    contracts = pd.DataFrame({
        "root": ["ES1 Index"] * 4,
        "contract": ["ESH4", "ESM4", "ESU4", "ESZ4"],
        "ltd": pd.to_datetime(["2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20"]),
        "cmon": [2024 * 12 + 3, 2024 * 12 + 6, 2024 * 12 + 9, 2024 * 12 + 12],
    })
    gaps, rolls = roll_calendar(idx, contracts, {"ES1 Index": "spx"})
    assert set(gaps["spx"].dropna().unique()) == {3.0}        # quarterly cycle
    roll_days = rolls.index[rolls["spx"]]
    # the trading day after each expiry, including December's
    assert [d.month for d in roll_days] == [3, 6, 9, 12]


def test_ratio_splice_anchors_and_scales():
    idx = pd.bdate_range("2025-01-01", periods=10, tz="UTC")
    hist = pd.Series([100.0] * 6 + [np.nan] * 4, index=idx)
    live = pd.Series(np.linspace(50, 59, 10), index=idx)     # same market, different basis
    out, ratio, anchor = ratio_splice(hist, live)
    assert anchor == idx[5]
    assert ratio == pytest.approx(100.0 / live.iloc[5])
    # the extension continues from the history's level with the live series' returns
    assert out.iloc[6] == pytest.approx(live.iloc[6] * ratio)
    assert (out.loc[:idx[5]] == 100.0).all(), "nothing before the splice date changes"


def test_ratio_splice_refuses_without_overlap():
    a = pd.Series([1.0, 2.0], index=pd.bdate_range("2025-01-01", periods=2, tz="UTC"))
    b = pd.Series([3.0, 4.0], index=pd.bdate_range("2025-02-03", periods=2, tz="UTC"))
    with pytest.raises(ValueError):
        ratio_splice(a, b)


def test_ratio_splice_inverts_the_quote():
    idx = pd.bdate_range("2025-01-01", periods=4, tz="UTC")
    hist = pd.Series([0.0066, 0.0067, np.nan, np.nan], index=idx)   # USD per JPY
    live = pd.Series([151.5, 149.3, 150.0, 148.0], index=idx)        # USDJPY
    out, ratio, _ = ratio_splice(hist, live, invert=True)
    # a fall in USDJPY is a rise in USD-per-JPY
    assert out.iloc[3] > out.iloc[2]
