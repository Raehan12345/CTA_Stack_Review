import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def random_walk_prices(n_days=1600, symbols=("a", "b", "c", "d", "e", "f"), seed=7,
                       drift=0.0002, vol=0.012) -> pd.DataFrame:
    """Synthetic daily closes on a UTC business-day calendar, with mild
    trends so the engine actually takes positions."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days, tz="UTC")
    r = rng.normal(drift, vol, size=(n_days, len(symbols)))
    # slow regime-like drift changes so the 40/252 barbell has trends to find
    r += np.sin(np.arange(n_days)[:, None] / 120.0 + np.arange(len(symbols))) * 0.0008
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=list(symbols))


@pytest.fixture
def prices():
    return random_walk_prices()


@pytest.fixture
def universe():
    return {"equities": ["a", "b"], "commodities": ["c", "d"], "currencies": ["e", "f"]}
