# Bug-fix log

Every defect found in this project, how it was found, what caused it, how it
was fixed and how the fix was verified. Ordered by area, then date.

Most of these were found the same way: a number that looked wrong was
treated as a bug to find, not a result to report. A Sharpe of 2.5, a +23%
day on a 12% annual target, a -108% day, a result byte-identical to the
baseline: each one was a defect.

| # | Area | Bug | Found by | Impact |
|---|---|---|---|---|
| D1 | Data | Holiday dates injected by an outer join | Baseline drift check | Fake flat days, shifted rebalances |
| D2 | Data | Unadjusted Gilt roll on 2011-12-29 | -108% day under the ER gate | One false -12.2% return |
| D3 | Data | Benchmark index spike-and-revert artefacts | Scan for >7% days | False January spikes |
| D4 | Data | Roll jumps left in every generic | Post-expiry Monday statistics | 3.49%/yr mean distortion |
| D5 | Data | Contract chains list inactive months | Roll-calendar validation | Wrong roll dates for 15 markets |
| D6 | Data | Last row of each pull was an intraday tick | Refresh merge disagreed with old file | ~40 wrong closes |
| D7 | Data | Price quote units (cents, cents per 100 yen) | $787m JPY contract | Analysis error, caught before use |
| D8 | Data | Mislabelled and empty tickers | Row counts | Wrong market under a name |
| D9 | Data | Patched column compared with a raw series | Retracted finding | A false "broken second month" claim |
| D10 | Data | Rolling windows NaN across holiday gaps | Near-empty book | Silent zero positions |
| E1 | Engine | HRP silently failed on every call | Result identical to baseline | Layer was a no-op |
| E2 | Engine | HRP scale not carried between rebalances | Inspection after E1 | Layer active one day a month |
| E3 | Engine | Trailing-stop look-ahead | Sharpe 2.5 | Stop dodged its own loss |
| E4 | Engine | Start-up leverage pinned at 4x | +10% and +23% days | Every full-sample figure inflated |
| E5 | Engine | First warm-up fix blanked later days | Leverage zero for 60 days in 2012 | Fix-of-the-fix |
| E6 | Engine | Regime cache key incomplete | Order-dependent results | Gate benefit inflated |
| E7 | Engine | Turnover computed from target weights | Identical across configurations | Two wrong conclusions |
| E8 | Engine | No-trade band traded to target | Churned more than weekly | Result withdrawn |
| E9 | Engine | HRP optimiser failure swallowed | Code review | Stale risk split, silently |
| E10 | Engine | Untradeable list copy-pasted per notebook | Audit | Untradeable sleeve called tradeable |
| M1 | Method | Stacked book compared to unstacked funds | Review of the benchmark chart | Unfair comparison |
| M2 | Method | Books compared at different volatility | Matched-vol re-test | Wrong verdict on the gate for a day |
| M3 | Method | Circular selection test | Nested walk-forward | Selection looked like skill |
| M4 | Method | In-sample rank quoted as evidence | Review | "3rd of 3,876" was in-sample |
| M5 | Method | Over-broad claims later corrected | Re-tests | HRP, crypto, wide universe |
| M6 | Method | Rounding and convention slips | Re-derivation | 18.8 vs 18.7, CAGR convention |
| M7 | Method | Regime gate not re-tested after the universe changed | Re-run for this repository | Gate's advantage does not hold on A-12 |
| M8 | Method | A selection rule re-derived did not reproduce | Re-run for this repository | 0.231 against 0.261 for the pool |
| X1 | Execution | FX lot value converted twice | Dry run | AUD 0.80 lots instead of 0.57 |
| X2 | Execution | Forming bar included in the splice | Code reading | Partial bar as a close |
| X3 | Execution | One rejected symbol blocked the book | Daily re-trades in the demo | 12 markets re-traded daily |
| X4 | Execution | Nine EA defects | External code review | See entry |
| X5 | Execution | New dated contracts had no history | Roll planning | A market would flatten at its first roll |
| X6 | Execution | Ten days on frozen prices | Reconciliation by eye | Engine "succeeded" on stale data |
| X7 | Execution | Doubled carriage returns | EA halted on header check | 25-minute outage |
| X8 | Execution | Engine fired inside the oil session break | First unattended rebalance | Late fills on oil |
| X9 | Execution | Lot value from MT5's profit currency | Ad-hoc analysis, then code review | 158x and 100x errors |
| X10 | Execution | `0 * NaN = NaN` for unlisted markets | Simulator NAV went NaN | Partial-history markets |
| X11 | Execution | Hand-kept universe list went stale | KeyError on import | File mode dead for 3 weeks |
| X12 | Execution | No price date in the targets file | Audit | Age check read the clock |
| X13 | Execution | Live P&L marked with future entry prices | Audit of the weekly report | Look-ahead in the live curve |
| X14 | Execution | FX positions never marked | Same audit | Largest risk line missing |
| X15 | Execution | Null total-return series accepted | Audit | Bad benchmark data let through |
| X16 | Execution | Simulator rounds lots, EA floors | Lot audit | Known, deliberately not fixed |

---

## Data

### D1. Holiday dates injected by an outer join

- **Symptom.** After merging a second Bloomberg pull, the original 7-market
  baseline Sharpe moved.
- **Cause.** The new tickers traded on different national holiday calendars.
  An outer join added about 33 dates to the original markets, and
  `fillna(0.0)` turned each into a fake flat-return day, shifting some
  rebalance dates.
- **Fix.** Left-join every later pull onto the first pull's calendar.
- **Verified.** An assertion that the 7-market baseline reproduces exactly.

### D2. Unadjusted Gilt roll on 2011-12-29

- **Symptom.** A -108% single day under the Efficiency Ratio gate.
- **Cause.** The Gilt generic fell 12.2% in one day and stayed there. A UK
  10-year bond future does not do that; the generic had jumped from the
  expiring contract onto the next one without adjustment. The gate had
  concentrated the whole book into the Gilt that week, removing the dilution
  that had hidden it in every earlier run.
- **Fix.** Rebase the series from that date by the jump ratio
  (133.29 / 117.06), which corrects that one day's return and leaves every
  other day's return unchanged.
- **Verified.** A scan of every market for daily moves above 8%: every other
  hit is a real dated event. Later confirmed independently: on that day the
  front generic fell 12.2% while the second-month generic rose 0.4%, and the
  two closed within 0.06 of each other.

### D3. Benchmark index spike-and-revert artefacts

- **Symptom.** About 39 days of over 7% moves in the SG Trend Indicator, all
  in early January, each fully reversed within two days.
- **Fix.** A general detector for a spike followed by a reversal, applied in
  the benchmark loader.

### D4. Roll jumps left in every generic

- **Symptom.** The S&P's Monday after quarterly expiry averaged -0.84% in
  2010 to 2021 and +1.61% in 2023 to 2026: exactly the sign flip of the
  calendar spread.
- **Cause.** Bloomberg generics splice contracts without adjustment, so each
  roll books the basis as a return nobody earned.
- **Fix.** Stitched returns from the contract chain: on a roll day the return
  is `F1_t / F2_{t-1} - 1`, because the contract the front rolls into is the
  one the second generic tracked the day before. Exact, with no back-
  adjustment.
- **Verified.** On adjusted series, roll-day moves fall back to ordinary-day
  size. Mean distortion removed: 3.49% a year.

### D5. Contract chains list inactive months

- **Symptom.** The chain-derived calendar showed 216 gold rolls.
- **Cause.** Gold lists a contract every month but the generic trades only
  February, April, June, August and December.
- **Fix.** A validation rule fixed before seeing results: trust a market's
  chain only if its derived roll days move at least 1.2x more than ordinary
  days. 49 pass; the 15 that fail (metals, FX futures, three equity indices)
  are left unadjusted. One false pass (Euribor) excluded by hand.

### D6. The last row of each pull was an intraday tick

- **Symptom.** A refresh disagreed with the existing file on its last date.
- **Cause.** Each pull was run during the trading day, so its final row was a
  live price, not a settlement. About 40 values were wrong, some by 5%.
- **Fix.** The merge treats each file's last row as provisional, prefers the
  settled value, and refuses to merge any file that disagrees on an earlier
  day.

### D7. Price quote units

- **Symptom.** A $787 million JPY futures contract.
- **Cause.** AUD, GBP and copper futures are quoted in US cents; JPY in cents
  per 100 yen.
- **Fix.** A per-market price divisor. The same lesson reappears in X9.

### D8. Mislabelled and empty tickers

- The column labelled US Ultra Bond held the German Buxl.
- `QAT1` was not the French OAT and started in 2021; `BAX1` was empty after
  its benchmark rate was discontinued.
- **Fix.** Relabel; re-pull or drop.

### D9. A patched column compared with a raw series

- **Symptom.** The Gilt appeared to trade at a +10.8% basis to its second
  month, so the second-month ticker was declared broken.
- **Cause.** The loader's Gilt column carries the D2 patch, so after 2011 it
  is an adjusted index, not a price. Raw front against raw second is a normal
  0.55% basis.
- **Fix.** The finding was retracted. Rule: level comparisons use raw series.

### D10. Rolling windows NaN across holiday gaps

- **Symptom.** A sized-signal test and a skewness test each produced a nearly
  empty book; a 32-market book scored identically to a 24-market one.
- **Cause.** `pct_change().rolling(n)` is NaN whenever the window contains a
  holiday gap, which for long windows is most of them.
- **Fix.** Rolling statistics run on `pct_change().fillna(0)`, as the engine's
  own volatility estimate always did.

## Engine

### E1. HRP silently failed on every call

- **Symptom.** The first HRP result was byte-identical to the baseline.
- **Cause.** `pypfopt` 1.6.0 validates its linkage method against a private
  scipy attribute that newer scipy no longer has. Every call raised, a broad
  `except` caught it, and the scale fell back to 1.0.
- **Fix.** A compatibility shim restoring the attribute.
- **Later (E9).** The silent fallback itself was removed.

### E2. HRP scale not carried between rebalances

- **Cause.** The scale frame was pre-filled with 1.0, so `ffill()` had no
  gaps to fill and every non-rebalance day kept the placeholder.
- **Fix.** Pre-fill with NaN.

### E3. Trailing-stop look-ahead

- **Symptom.** Sharpe 2.5 full sample, above 2.0 walk-forward.
- **Cause.** The stop was checked against today's close and the position
  zeroed before today's return was booked, so the book never suffered the
  move that triggered the stop.
- **Fix.** Book today's return on the weights held before the check; the stop
  affects tomorrow onward.

### E4. Start-up leverage pinned at 4x

- **Symptom.** Single days of +10% and +23% against a 12% annual target.
- **Cause.** The 252-day signal needs a year of history, so the first year of
  portfolio returns is exact zeros. Their 60-day volatility is about zero,
  and Tier-2 leverage `target / vol` pinned to the 4x cap for 197 days. The
  book opened at maximum size on a fake risk estimate.
- **Fix.** `warmup_guard`: blank the pre-signal run so the vol estimate is
  NaN, which holds leverage at zero until 60 real days exist.
- **Verified.** Post-2011 returns identical to 1e-16, so every walk-forward
  figure survived. On the 24-market pool terminal NAV fell from 3.09x to
  1.87x.

### E5. The first warm-up fix blanked later days

- **Cause.** The first version tested "is any market live today". A later
  day with every market gapped also failed, blanking a whole rolling window
  and forcing leverage to zero for 60 days in 2012.
- **Fix.** "Has the book ever been live": `.cummax()`.
- **Verified.** 60 random universes of every size: out-of-sample window
  unchanged, worst difference 2e-15.

### E6. Regime cache key incomplete

- **Symptom.** The same configuration scored 0.4714 or 0.4830 depending on
  what had run before it in the same session.
- **Cause.** The learned gate ranks clusters by the sleeve's own return,
  which depends on HRP, the universe, and every other layer. The cache key
  listed only the regime parameters, so a run with HRP reused the
  multiplier computed without it.
- **Fix.** Add a SHA-1 hash of the sleeve return series to the key, so it
  cannot go stale as layers are added.

### E7. Turnover computed from target weights

- **Cause.** The engine returns target weights, not held positions. Turnover
  from them came out identical across configurations and produced two wrong
  conclusions.
- **Fix.** `collect_contrib=True` records per-market contribution and cost
  inside the loop; it reconciles to NAV to 1.7e-16.

### E8. No-trade band traded to target

- **Cause.** The band traded all the way to target on every trigger, so it
  churned more than weekly rebalancing.
- **Fix.** Result withdrawn; rebuilt to trade to the band edge
  (`band_style="edge"`), then tested properly (a wash).

### E9. HRP optimiser failure swallowed

- **Cause.** On a failed optimisation the code kept the previous rebalance's
  weights and printed nothing.
- **Fix.** A printed warning naming the class and date.

### E10. Untradeable list copy-pasted per notebook

- **Symptom.** A sleeve was reported as tradeable when its largest position
  (the US 2-year) was not.
- **Fix.** One list, `NO_MT5_INSTRUMENT`, imported everywhere.

## Method

### M1. Stacked book compared with unstacked funds
The sleeves were compared *stacked* (200% exposure) against managed-futures
funds that are not. Corrected to sleeve-only; on the fair basis the result
went from "beats both" to "beats one".

### M2. Books compared at different volatility
The regime gate looked like it cost 0.6 points of return at a common 12%
target, but it ran the book at 12.2% realised vol against 13.8%. At matched
vol it added return and cut drawdown. The rule since: match realised
volatility before comparing.

### M3. Circular selection test
A selected sleeve beat all 15,000 random portfolios, but it was chosen and
scored on the same window. The nested walk-forward replaced it: aggressive
selection 0.25 against 0.37 for no selection.

### M4. In-sample rank quoted as evidence
Sleeve A's "3rd of 3,876" was its full-sample rank; out of sample it is 5th.

### M5. Over-broad claims later corrected
- "HRP is a walk-forward improvement": only when 2008 is in the scored window.
- "Crypto's edge is two-sided": all of it is long-side.
- "More markets is arithmetically incapable of working": the real cause was
  16 highly correlated equity indices.
- A claimed 2,300x speed-up: the real figure was about 170 to 350x.

### M6. Rounding and convention slips
- Stacked CAGR quoted as 18.8% from a four-decimal print of 0.1875; it is
  18.7%.
- CAGR computed as `252 / len` changed to calendar years.

### M7. The regime gate was never re-tested after the universe changed
- **Symptom.** Re-running every chapter for this repository, the gate's
  matched-volatility advantage did not reproduce on A-12.
- **Cause.** The gate was chosen on A-13. When the Gilt was dropped the
  headline was recomputed, but the gate-against-no-gate comparison was not.
- **Measured.** At matched volatility from 2008-09-04, A-13: +0.21 points of
  CAGR and 3.3 points shallower drawdown. A-12: -0.85 points of CAGR and 1.2
  points shallower drawdown. With crypto in the book (the live shape) the
  gate leaves Sharpe unchanged and cuts the worst loss by about a third.
- **Status.** Open. A universe change invalidates every comparison made on
  the old universe, not only the headline.

### M8. A selection rule re-derived did not reproduce
- **Symptom.** The "mild" universe-selection rule had been reported as
  beating holding the whole pool (0.42 against 0.38). Re-derived here on the
  current engine and data, it scores 0.231 against 0.261 for the pool.
- **Cause.** Not isolated. The engine, the roll-adjusted data and the
  tradeable list all changed between the two runs.
- **Consequence.** The conclusion of Chapter 4 strengthens rather than
  weakens: no selection rule tested beats holding the pool out of sample, and
  the selection premium in the headline is about 0.19.

## Execution and live

### X1. FX lot value converted twice
In the first dry run AUD sized at 0.80 lots instead of 0.57: the lot was
converted by the base currency on top of the pair's price.

### X2. Forming bar included in the splice
`CopyRates` returns today's incomplete bar; the splice drops it.

### X3. One rejected symbol blocked the book
The broker rejected every Gilt order, so the rebalance never completed and
the EA re-traded the other 12 markets daily. Fixed with per-root completion
tracking: completed roots are recorded and only the failed one is retried.
Tested for real on 1 October, when Brent could not trade for two hours and
the other 13 completed immediately.

### X4. Nine EA defects from an external code review
1. `PositionClose` returns true on a server rejection; retcodes now checked
   and a close is verified flat.
2. Rolls closed the old contract before opening the new one; now open first.
3. Cached targets had no age limit; now 4 days.
4. A market that could not be extended went NaN and would have been
   flattened; now the whole run fails and previous targets stand.
5. Zero-divides in two regime features; guarded to reproduce pandas.
6. File-mode targets barely validated; now range-checked.
7. A failed ATR read could send an order with no stop; now refused.
8. A failed stop modify still marked the day done; now reported.
9. A residual of exactly half a lot step retried forever; tolerances made
   inclusive.

### X5. New dated contracts had no history
A contract carries bars only from shortly before it lists, so the splice
would never overlap and the market would be flattened at its first roll. The
EA now appends its spliced days to the history file daily, so the splice
anchor is always yesterday.

### X6. Ten days on frozen prices
The rebalance could not complete (X3), and the staleness guard measured how
long since the engine ran, not how old the prices were; the history extender
returned success when it found nothing new. Fix: a price-age guard
(`InpMaxDataAgeD`) checked against the data date.

### X7. Doubled carriage returns
`to_csv` wrote CRLF and `Path.write_text` translated the LF again, giving
CR CR LF. MQL5 kept the stray CR, so the header check compared `"eth\r"`
with `"eth"` and halted. Rule: files for MQL5 are written with `newline=""`.

### X8. Engine fired inside the oil session break
22:00 UTC sits inside the ICE Brent and CME WTI daily breaks, so oil fills
landed hours late with no reference price. Moved to 02:00 UTC.

### X9. Lot value from the wrong currency
In analysis, `price x contract_size` without converting the quote currency
made the two JPY-quoted markets look 158x too large. In the signal service,
MT5's profit currency reported USD for copper, which is quoted in cents
(100x). Fix: one SPECS-driven `lot_value_usd()`, used everywhere.

### X10. `0 * NaN = NaN` for unlisted markets
A flat position in a market before its listing date poisoned the
simulator's whole NAV. Fix: an explicit validity mask.

### X11. Hand-kept universe list went stale
When crypto entered the instrument spec, a hand-kept class map did not, and
the file-mode signal path raised `KeyError` on import for three weeks. Fix:
buckets derived from the spec table.

### X12. No price date in the targets file
The file carried its write time only, so the archived age check was reading
the clock. A price-date field was appended (appended, so old files still
parse).

### X13 and X14. Live P&L marking
The weekly live report marked open positions against one entry price
averaged over the whole period, putting later entries into earlier days. And
it never marked FX at all: an FX lot is a fixed amount of the base currency,
so its USD notional does not depend on price, and marking by change in
notional gives zero. Fix: average-cost accounting walked in deal order, and
FX P&L as `contract x (now - entry)` in the quote currency, converted.
Verified against the broker's own P&L to within a dollar; reproduced as
assertions in `tests/test_live_marking.py`.

### X15. Null total-return series accepted
The ETF fetch tested whether the adjusted-close list was non-empty, not
whether it held values. Now checks the contents.

### X16. Simulator rounds lots, EA floors (known, not fixed)
The simulator rounds lots to nearest; the EA floors, so a target under one
minimum lot is not traded. Backtests are therefore slightly optimistic.
Left as is deliberately: fixing it restates published figures, and a data
backfill will force a full restatement anyway.
