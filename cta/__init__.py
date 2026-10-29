"""Time-series momentum sleeve: the managed-futures half of a return-stacked
book (100% equities plus a trend-following overlay on the same capital)."""
from .config import DEFAULT_PARAMS, CHOSEN, GATE, TRANSACTION_COST_BPS, chosen_kwargs
from .engine import build_trend_nav, build_trend_nav_fast, build_fast_slow_nav

__all__ = ["build_trend_nav", "build_trend_nav_fast", "build_fast_slow_nav",
           "DEFAULT_PARAMS", "CHOSEN", "GATE", "TRANSACTION_COST_BPS", "chosen_kwargs"]
