"""Live P&L marking (bugs X13 and X14). No MT5, no data, no network.

1. The entry price was one average over the WHOLE period, so a day's mark
   used entries made after it, and a long and a later short were blended.
2. FX was marked by the change in lot notional, which is identically zero
   because an FX lot's USD notional does not depend on the price.
"""
import numpy as np
import pandas as pd
import pytest

from live import reconcile as rd
from live.marking import position_book_daily, mark_usd


class FakePrices:
    """Just what mark_usd touches, with fixed rates so the sums are checkable."""
    USDJPY = 152.0

    def fx_usd(self, ccy):
        return {"USD": 1.0, "USc": 0.01, "JPY": 1.0 / self.USDJPY,
                "AUD": 0.6928, "GBP": 1.3194}[ccy]

    def lot_value_usd(self, root, price):
        return rd.Prices.lot_value_usd(self, root, price)


def deals(rows):
    d = pd.DataFrame(rows, columns=["time", "symbol", "side", "lots", "price"])
    d["time"] = pd.to_datetime(d["time"])
    return d


GBP = deals([("2026-09-15 09:00", "GBPUSD.a", "BUY", 0.38, 1.34721),
             ("2026-10-01 02:10", "GBPUSD.a", "SELL", 0.76, 1.31936)])


def test_entry_price_does_not_see_the_future():
    lots, entry = position_book_daily(GBP, end="2026-10-13")
    sep20 = pd.Timestamp("2026-09-20")
    assert lots.at[sep20, "GBPUSD.a"] == pytest.approx(0.38)
    assert entry.at[sep20, "GBPUSD.a"] == pytest.approx(1.34721)
    blended = (0.38 * 1.34721 + 0.76 * 1.31936) / 1.14
    assert abs(entry.at[sep20, "GBPUSD.a"] - blended) > 1e-4, "still blending the two legs"


def test_a_flip_resets_the_entry():
    lots, entry = position_book_daily(GBP, end="2026-10-13")
    oct5 = pd.Timestamp("2026-10-05")
    assert lots.at[oct5, "GBPUSD.a"] == pytest.approx(-0.38)
    assert entry.at[oct5, "GBPUSD.a"] == pytest.approx(1.31936)


def test_adding_blends_and_reducing_does_not():
    d = deals([("2026-09-15", "GOLDZ6", "BUY", 1.0, 100.0),
               ("2026-09-16", "GOLDZ6", "BUY", 1.0, 110.0),
               ("2026-09-17", "GOLDZ6", "SELL", 1.0, 200.0)])
    l, e = position_book_daily(d, end="2026-09-18")
    assert e.at[pd.Timestamp("2026-09-16"), "GOLDZ6"] == pytest.approx(105.0)
    assert e.at[pd.Timestamp("2026-09-17"), "GOLDZ6"] == pytest.approx(105.0)
    assert l.at[pd.Timestamp("2026-09-17"), "GOLDZ6"] == pytest.approx(1.0)


def test_closing_to_flat_leaves_no_entry():
    d = deals([("2026-09-15", "AUDUSD.a", "BUY", 0.5, 0.71223),
               ("2026-10-01", "AUDUSD.a", "SELL", 0.5, 0.69280)])
    l, e = position_book_daily(d, end="2026-10-05")
    assert abs(l.at[pd.Timestamp("2026-10-02"), "AUDUSD.a"]) < 1e-9
    assert np.isnan(e.at[pd.Timestamp("2026-10-02"), "AUDUSD.a"])


def test_fx_positions_are_actually_marked():
    px = FakePrices()
    # short 0.38 GBPUSD from 1.31936 to 1.31000: 0.38 x 100,000 x 0.00936
    assert mark_usd(px, "GBPUSD.a", -0.38, 1.31936, 1.31000) == pytest.approx(355.68, abs=0.5)
    assert mark_usd(px, "AUDUSD.a", 0.5, 0.71223, 0.69280) == pytest.approx(
        0.5 * 100000 * (0.69280 - 0.71223), abs=0.5)
    # USDJPY: P&L lands in yen and converts back at the yen rate
    assert mark_usd(px, "USDJPY.a", 0.3, 150.0, 152.0) == pytest.approx(
        0.3 * 100000 * 2.0 / 152.0, abs=0.5)


def test_the_old_fx_formula_was_identically_zero():
    px = FakePrices()
    old = -0.38 * (px.lot_value_usd("GBPUSD.a", 1.31000) - px.lot_value_usd("GBPUSD.a", 1.31936))
    assert abs(old) < 1e-9


def test_non_fx_goes_through_lot_value():
    px = FakePrices()
    # copper is quoted in US cents: 250 x price x 0.01
    assert mark_usd(px, "COPPER", 2.0, 400.0, 410.0) == pytest.approx(2.0 * 250 * 10.0 * 0.01)
    assert mark_usd(px, "COPPER", 2.0, np.nan, 410.0) == 0.0
    assert mark_usd(px, "COPPER", 0.0, 400.0, 410.0) == 0.0
