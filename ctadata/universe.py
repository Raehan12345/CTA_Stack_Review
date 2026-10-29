"""Every market the project ever loaded, its asset class, and which ones the
broker can actually trade.

Single source of truth. Universes are derived from these tables, never typed
out again elsewhere: an untradeable-instrument list that was once copy-pasted
into each notebook went stale in one copy, and a sleeve was reported as
tradeable when its largest position (the US 2-year) was not. A hand-kept
class map later went stale the same way when crypto was added.
"""

# ---------------------------------------------------------------------------
# Bloomberg generic front months -> short names. Four pulls, widening the
# universe each time: the original nine, twelve more for breadth, four more
# fixed-income markets, then agriculturals and two equity indices.
# ---------------------------------------------------------------------------
TICKER_INFO = {
    "TY1 Comdty": "us10y",
    "TU1 Comdty": "us2y",
    "RX1 Comdty": "eurobund",
    "G 1 Comdty": "gilt",
    "ES1 Index": "spx",
    "GC1 Comdty": "gold",
    "CL1 Comdty": "wti",
    "EC1 Curncy": "eur",
    "AD1 Curncy": "aud",
    "NQ1 Index": "nasdaq",
    "VG1 Index": "eurostoxx",
    "HG1 Comdty": "copper",
    "NG1 Comdty": "natgas",
    "JY1 Curncy": "jpy",
    "BP1 Curncy": "gbp",
    "JB1 Comdty": "jgb",
    "CO1 Comdty": "brent",
    "SI1 Comdty": "silver",
    "NK1 Index": "nikkei",
    "CD1 Curncy": "cad",
    # Tagged Curncy in the export and named for the US Ultra Bond, but the
    # values are the German Buxl. Kept under its original name so every
    # historical result stays reproducible; it is untradeable either way.
    "WN1 Curncy": "us_ultra",
    "FV1 Comdty": "us5y",
    "OE1 Comdty": "eurobobl",
    "XM1 Comdty": "australia10y",
    "CN1 Comdty": "canada10y",
    "C 1 Comdty": "corn",
    "KC1 Comdty": "coffee",
    "JO1 Comdty": "oj",
    "GX1 Index": "dax",
    "HI1 Index": "hangseng",
}

# ---------------------------------------------------------------------------
# Research markets the broker does not offer. Verified against all of the
# broker's live symbols; its whole government-bond shelf is twelve dated
# futures and the only US Treasury among them is the 10-year note.
# ---------------------------------------------------------------------------
UNTRADEABLE = frozenset({
    "jgb",
    "us2y",          # missing from an earlier copy of this list: see module note
    "us5y",
    "us_ultra",
    "australia10y",  # the broker's AUS200 is the equity index, not the bond
    "eurobobl",      # Schatz and Bund are listed, Bobl is not
    # Listed, and passes the terminal's own order check, but every order was
    # rejected server-side (1,816 attempts over 13 days). Dropped from
    # research on 2026-09-24, which turned A-13 into A-12.
    "gilt",
})

# The universes in the order the project built them. Kept because each
# chapter re-runs its own stage of the history on the universe of the time.
LEAN_7 = {
    "fixed_income": ["us10y", "eurobund"],   # Fed against ECB is a different bet;
    "equities": ["spx"],                      # two points on one curve is not
    "commodities": ["gold", "wti"],
    "currencies": ["eur", "aud"],
}
EXPANDED_19 = {
    "fixed_income": ["us10y", "eurobund", "us2y", "gilt", "jgb"],
    "equities": ["spx", "nasdaq", "eurostoxx", "nikkei"],
    "commodities": ["gold", "wti", "copper", "natgas", "brent", "silver"],
    "currencies": ["eur", "aud", "jpy", "gbp"],
}
# Sleeve A: the 19 less the weakest leave-one-out contributor in each class
# (eurostoxx, eurobund, natgas, eur).
SLEEVE_A_15 = {
    "fixed_income": ["us10y", "us2y", "gilt", "jgb"],
    "equities": ["spx", "nasdaq", "nikkei"],
    "commodities": ["gold", "wti", "copper", "brent", "silver"],
    "currencies": ["aud", "jpy", "gbp"],
}


def symbols_of(universe: dict) -> list:
    return [s for group in universe.values() for s in group]


# A-12: Sleeve A less whatever the broker cannot trade. Derived, not typed,
# so dropping or restoring an instrument propagates everywhere.
A12 = sorted(set(symbols_of(SLEEVE_A_15)) - UNTRADEABLE)
CRYPTO = ["btc", "eth"]
LIVE_14 = sorted(A12 + CRYPTO)

# ---------------------------------------------------------------------------
# Asset classes. HRP and the Economic Trend layer both need to know which
# bucket a market sits in. An unknown market raises rather than falling into
# a default bucket: a bond in the wrong bucket silently changes every HRP
# weight in the book.
# ---------------------------------------------------------------------------
CLASS = {
    # equities
    "spx": "equities", "nasdaq": "equities", "nikkei": "equities",
    "eurostoxx": "equities", "dax": "equities", "hangseng": "equities",
    "dow": "equities", "russell": "equities", "ftse": "equities",
    "cac": "equities", "smi": "equities", "ftsemib": "equities",
    "ibex": "equities", "asx": "equities", "tsx": "equities", "aex": "equities",
    "vix": "equities",
    # fixed income
    "us10y": "fixed_income", "us2y": "fixed_income", "us5y": "fixed_income",
    "us_ultra": "fixed_income", "gilt": "fixed_income", "jgb": "fixed_income",
    "eurobund": "fixed_income", "eurobobl": "fixed_income",
    "australia10y": "fixed_income", "canada10y": "fixed_income",
    "btp": "fixed_income", "schatz": "fixed_income", "france10y": "fixed_income",
    "euribor": "fixed_income",
    # currencies
    "aud": "currencies", "gbp": "currencies", "jpy": "currencies",
    "eur": "currencies", "cad": "currencies", "chf": "currencies",
    "nzd": "currencies", "mxn": "currencies", "dxy": "currencies",
    # commodities
    "gold": "commodities", "silver": "commodities", "copper": "commodities",
    "platinum": "commodities", "palladium": "commodities",
    "wti": "commodities", "brent": "commodities", "natgas": "commodities",
    "heatoil": "commodities", "gasoline": "commodities", "gasoil": "commodities",
    "corn": "commodities", "wheat": "commodities", "soybean": "commodities",
    "soymeal": "commodities", "soyoil": "commodities", "sugar": "commodities",
    "cocoa": "commodities", "cotton": "commodities", "coffee": "commodities",
    "oj": "commodities", "cattle": "commodities", "hogs": "commodities",
    "feeder": "commodities",
    # crypto is its own class: a fifth HRP bucket, not a commodity
    "btc": "crypto", "eth": "crypto",
}


def asset_class(sym: str) -> str:
    try:
        return CLASS[sym]
    except KeyError:
        raise KeyError(f"no asset class for {sym!r}; add it to CLASS rather "
                       f"than letting it fall into a default bucket") from None


def buckets(symbols) -> dict:
    """{asset class: [symbols]} for HRP, sorted within each class."""
    out = {}
    for s in symbols:
        out.setdefault(asset_class(s), []).append(s)
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The wider universe (Tier A and Tier B), front and second months.
# ---------------------------------------------------------------------------
WIDE_FRONT = {
    "HO1 Comdty": "heatoil", "XB1 Comdty": "gasoline", "QS1 Comdty": "gasoil",
    "W 1 Comdty": "wheat", "S 1 Comdty": "soybean", "SM1 Comdty": "soymeal",
    "BO1 Comdty": "soyoil", "SB1 Comdty": "sugar", "CC1 Comdty": "cocoa",
    "CT1 Comdty": "cotton", "LC1 Comdty": "cattle", "LH1 Comdty": "hogs",
    "FC1 Comdty": "feeder",
    "PL1 Comdty": "platinum", "PA1 Comdty": "palladium",
    "Z 1 Index": "ftse", "CF1 Index": "cac",
    "SM1 Index": "smi",           # NOT soybean meal: the yellow key matters
    "ST1 Index": "ftsemib", "IB1 Index": "ibex", "XP1 Index": "asx",
    "PT1 Index": "tsx", "EO1 Index": "aex",
    "RTY1 Index": "russell",      # from 2017-07-10 only
    "SF1 Curncy": "chf", "NV1 Curncy": "nzd", "PE1 Curncy": "mxn",
    "DM1 Index": "dow", "DX1 Curncy": "dxy", "UX1 Index": "vix",
    "IK1 Comdty": "btp", "DU1 Comdty": "schatz", "ER1 Comdty": "euribor",
    "OAT1 Comdty": "france10y",   # from 2012-04-16, the contract's full life
}
# Pulled, deliberately not mapped.
UNUSABLE = {
    "QAT1 Comdty": "not the French OAT; replaced by OAT1 Comdty",
    "CRA1 Comdty": "524 rows ending 2009-03-31; the contract is discontinued",
}

WIDE_SECOND = {
    "HO2 Comdty": "heatoil", "XB2 Comdty": "gasoline", "QS2 Comdty": "gasoil",
    "W 2 Comdty": "wheat", "S 2 Comdty": "soybean", "SM2 Comdty": "soymeal",
    "BO2 Comdty": "soyoil", "SB2 Comdty": "sugar", "CC2 Comdty": "cocoa",
    "CT2 Comdty": "cotton", "LC2 Comdty": "cattle", "LH2 Comdty": "hogs",
    "FC2 Comdty": "feeder", "PL2 Comdty": "platinum", "PA2 Comdty": "palladium",
    "NG2 Comdty": "natgas", "C 2 Comdty": "corn", "KC2 Comdty": "coffee",
    "JO2 Comdty": "oj",
    "TY2 Comdty": "us10y", "ES2 Index": "spx", "NQ2 Index": "nasdaq",
    "NK2 Index": "nikkei", "SI2 Comdty": "silver", "HG2 Comdty": "copper",
    "CL2 Comdty": "wti", "CO2 Comdty": "brent", "BP2 Curncy": "gbp",
    "JY2 Curncy": "jpy", "GC2 Comdty": "gold", "AD2 Curncy": "aud",
    "EC2 Curncy": "eur",
}
# Second months pulled later, keyed by the same short name as the front.
# SM2 Index is the SMI; SM2 Comdty is soybean meal and is in WIDE_SECOND.
WIDE_SECOND_EXTRA = {
    "EO2 Index": "aex", "XP2 Index": "asx", "XM2 Comdty": "australia10y",
    "IK2 Comdty": "btp", "CF2 Index": "cac", "CD2 Curncy": "cad",
    "CN2 Comdty": "canada10y", "SF2 Curncy": "chf", "GX2 Index": "dax",
    "DM2 Index": "dow", "DX2 Curncy": "dxy", "ER2 Comdty": "euribor",
    "OE2 Comdty": "eurobobl", "RX2 Comdty": "eurobund", "VG2 Index": "eurostoxx",
    "OAT2 Comdty": "france10y", "Z 2 Index": "ftse", "ST2 Index": "ftsemib",
    "HI2 Index": "hangseng", "IB2 Index": "ibex", "JB2 Comdty": "jgb",
    "PE2 Curncy": "mxn", "NV2 Curncy": "nzd", "RTY2 Index": "russell",
    "DU2 Comdty": "schatz", "SM2 Index": "smi", "PT2 Index": "tsx",
    "TU2 Comdty": "us2y", "FV2 Comdty": "us5y", "WN2 Curncy": "us_ultra",
    "UX2 Index": "vix",
}

CRYPTO_FRONT = {"XBTUSD Curncy": "btc", "XETUSD Curncy": "eth"}
# BTC before 2015 is not the same instrument: 127-134% annualised vol and 27
# days above a 25% move, from $0.08 with no retail venue. ETH's Bloomberg
# series starts 2018-02-08 regardless of the asset's launch.
CRYPTO_START = {"btc": "2015-01-01", "eth": None}
# Broker financing on the crypto CFDs, annual % of notional: long pays 15%,
# short earns 7.5%. Not in a spot price, so charged explicitly.
CRYPTO_SWAP = {"btc": (-0.150, 0.075), "eth": (-0.150, 0.075)}

COMMODITIES_WITH_CURVE = sorted([
    "gold", "silver", "copper", "wti", "brent",
    "natgas", "corn", "coffee", "oj", "platinum", "palladium",
    "heatoil", "gasoline", "gasoil", "wheat", "soybean", "soymeal", "soyoil",
    "sugar", "cocoa", "cotton", "cattle", "hogs", "feeder",
])
