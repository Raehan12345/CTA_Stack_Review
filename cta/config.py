"""Parameters, the cost table, and the chosen configuration."""
import pandas as pd

# The engine's defaults are the ORIGINAL research settings (weekly, 12%), so
# that every early chapter can be reproduced. The configuration actually run
# is CHOSEN below.
DEFAULT_PARAMS = dict(
    short_lookback=40,       # about 2 months: the barbell's fast leg
    long_lookback=252,       # about 12 months: the slow leg
    vol_window=60,           # both instrument and portfolio volatility estimates
    inst_target_vol=0.20,    # Tier 1: each instrument's own volatility target
    max_inst_weight=3.0,     # cap on one instrument's weight; binds on ~11% of days
    base_target_vol=0.12,    # Tier 2: the whole book's volatility target
    max_leverage=4.0,        # safety rail on Tier 2; never binds in the chosen book
    rebalance_freq="W",
)

# The research configuration, settled on 14 September 2026 and live since
# the 15th. Two settings exist only so the MQL5 port can reproduce it exactly:
# `regime_kmeans="portable"` (k-means on a generator any language can
# implement) and `hrp_lag=1` (HRP's window ends the day before it is used,
# which also removes a one-day peek). Together they move Sharpe from 0.4804
# to 0.4806.
#
# DO NOT TUNE the regime parameters. Choosing among ten gate variants on
# training data and judging on the next window scored 0.345 against 0.410
# for running no gate at all.
CHOSEN = dict(
    apply_costs=True,
    hrp=True, hrp_lag=1,
    rebalance_freq="M",
    base_target_vol=0.13,
    regime_gate=True,
    regime_mult_mode="learned",
    regime_features=("vol", "corr", "er", "disp", "agree"),
    regime_auto_k=(2, 10),
    regime_kmeans="portable",
)
GATE = {k: CHOSEN[k] for k in ("regime_gate", "regime_mult_mode", "regime_features",
                               "regime_auto_k", "regime_kmeans")}


def chosen_kwargs(universe: dict) -> dict:
    """DEFAULT_PARAMS overlaid with CHOSEN, plus the HRP bucket map."""
    return {**DEFAULT_PARAMS, **CHOSEN, "universe": universe}


# Expense-ratio drag from comparable products (2026-09): 0.35%, 0.75%,
# 0.85% and 0.99%. Charged to the sleeve only, as a constant daily drag.
FEE_BPS = {"low": 35, "base": 75, "high": 99}


def apply_fee_drag(returns: pd.Series, scenario: str = "base") -> pd.Series:
    """Annual fee spread evenly across trading days. Not a per-trade cost:
    see TRANSACTION_COST_BPS for that."""
    return returns - (FEE_BPS[scenario] / 10_000) / 252


# One-way cost per unit of notional traded: half-spread, commission and
# slippage together. CTA replication research calls a volume-weighted 2 bp
# round turn across 24 contracts "realistic, if anything conservative";
# practitioner estimates are 2-3 bp one way, less for index futures and
# bonds, more for thin contracts. The table is tiered on that. The original
# seven markets were sourced; later additions extend the same tiers by
# judgment.
#
# No financing charge: a future's cost of carry is already in its price, so
# charging it again would double-count. Crypto is the exception (a CFD, not a
# future); its financing is charged separately (see live.costs).
TRANSACTION_COST_BPS = {
    # tier 1: the most liquid
    "spx": 0.00020, "us10y": 0.00020, "eur": 0.00020, "us2y": 0.00020,
    "nasdaq": 0.00020, "jpy": 0.00020, "gbp": 0.00020, "canada10y": 0.00020,
    "us5y": 0.00020, "btc": 0.00020,
    # tier 2
    "eurobund": 0.00025, "gold": 0.00035, "wti": 0.00040, "aud": 0.00035,
    "eurostoxx": 0.00025, "brent": 0.00030, "jgb": 0.00030, "gilt": 0.00025,
    "copper": 0.00035, "nikkei": 0.00030, "hangseng": 0.00030, "dax": 0.00030,
    "cad": 0.00025, "us_ultra": 0.00025, "australia10y": 0.00030, "eurobobl": 0.00025,
    # tier 3: thinner or more volatile
    "silver": 0.00040, "natgas": 0.00045, "coffee": 0.00045, "oj": 0.00045,
    "corn": 0.00040, "eth": 0.00045,     # crypto spreads read live: 1.8 and 6.5 bp
    # the wider universe
    "wheat": 0.00040, "soybean": 0.00040, "soymeal": 0.00040, "soyoil": 0.00040,
    "heatoil": 0.00045, "gasoline": 0.00045, "gasoil": 0.00045,
    "sugar": 0.00045, "cocoa": 0.00045, "cotton": 0.00045,
    "cattle": 0.00045, "hogs": 0.00050, "feeder": 0.00050,
    "platinum": 0.00045, "palladium": 0.00055,
    "dow": 0.00020, "russell": 0.00025, "ftse": 0.00025, "cac": 0.00025,
    "smi": 0.00030, "ftsemib": 0.00030, "ibex": 0.00030, "asx": 0.00030,
    "tsx": 0.00030, "aex": 0.00030,
    "chf": 0.00020, "nzd": 0.00025, "dxy": 0.00025, "mxn": 0.00045,
    "schatz": 0.00020, "euribor": 0.00020, "france10y": 0.00025, "btp": 0.00030,
    "vix": 0.00060,
}
