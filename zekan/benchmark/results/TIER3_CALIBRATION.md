# Tier 3 Phase B — HistGradientBoosting recalibration evidence

This is evidence for a decision, not the decision itself. The default
estimator has not been changed; nothing here flips it. Phase C (the actual
default flip, if any) happens only after this is reviewed.

## What was added (Part 1, committed separately)

`"histgb"` added to `_ESTIMATOR_ALLOWLIST` -> `HistGradientBoostingClassifier`,
with exactly `random_state=42, early_stopping=False`. `early_stopping=False`
is a deliberate deviation from sklearn's own `'auto'` default: `'auto'`
silently enables an internal validation split once a dataset crosses 10,000
rows, meaning the model's fit procedure would change shape purely because
row count crossed a threshold -- unacceptable for a trust gate that has to
mean the same thing at every scale. `"gbm"` (the old
`GradientBoostingClassifier`) is unchanged, still present, still the same
slow implementation it always was.

## What was run (Part 2)

`nsl_boundary_sweep --estimator histgb --jobs 12` (fixed_v1 null stopping,
NOT sequential -- one variable changed at a time from the F2b/Tier 2
baselines). Same 25-cell fixture as F2b (5 alpha levels x 5 injector seeds,
`n_entities=2000`, `n_permutations=100`). histgb here is the exact factory
recipe from Part 1 (`zekan.severity.estimators._build_factory("histgb")`),
not a separate sweep-tuned variant -- this measures exactly what a real
`--estimator histgb` audit would produce.

**Wall clock: 1275s (~21.3 min), versus the rf baseline's ~37 min** -- roughly
1.7x faster, consistent with the scaling advantage measured in Phase A.

Raw evidence: `f2b_calibration_histgb.csv` (this directory). Baseline for
comparison: `f2b_calibration_spawn_v2.csv`.

## THE HARD QUESTION: does NSL >= 1.0 remain a valid materiality gate?

**Yes.** All 25 cells produce the **identical verdict and detection_channel**
as the rf baseline -- zero mismatches, zero flips:

| alpha | rf verdict/channel | histgb verdict/channel | match |
|---|---|---|---|
| 0.00 (5 seeds) | pass / (none) | pass / (none) | yes, all 5 |
| 0.60 (5 seeds) | fail / both | fail / both | yes, all 5 |
| 1.10 (5 seeds) | fail / both | fail / both | yes, all 5 |
| 1.60 (5 seeds) | fail / both | fail / both | yes, all 5 |
| 2.50 (5 seeds) | fail / both | fail / both | yes, all 5 |

NSL ranges (within-entity):

| | rf | histgb |
|---|---|---|
| clean cells (alpha=0.00) | [-2.65, -0.17] | [-1.64, -0.38] |
| leaked cells (alpha>=0.60) | [+3.84, +9.21] | [+5.87, +12.53] |

Across-entity channel:

| | rf | histgb |
|---|---|---|
| clean cells | [-2.54, -0.16] | [-1.75, -0.61] |
| leaked cells | [+11.50, +28.21] | [+16.09, +40.42] |

**The [1.0, 2.0) band remains empty under histgb** -- confirmed directly
(no cell, clean or leaked, on either channel, lands in that range). If
anything, histgb shows a **wider** separation gap than rf on the leaked
side (NSL up to 12.5 within / 40.4 across, versus rf's 9.2 / 28.2) --
not a narrower or more ambiguous one.

`null_iqr` (within-entity) sat in **[0.00132, 0.00288]** across the 25
cells, versus the rf-derived scaling law's point prediction of
`0.176/sqrt(20000) ~= 0.00124` at this n. Same order of magnitude, slightly
above the law's point estimate but consistent with the same
size-of-dataset-driven narrowing rf showed -- this law was fit to rf, and
histgb's null_iqr sitting close to (not wildly off from) that prediction
is itself informative, not something this phase claims to re-derive
precisely for histgb.

**Conclusion: no calibration failure. The NSL>=1.0 gate transfers cleanly
to histgb on this fixture, with margin to spare.**

## Floor re-anchoring evidence (Part 3)

Methodology reconstructed from `label_proxy_sweep.py`'s own documentation
(its `_clf()` comment and print string confirm the original rf temporal
ceiling was measured at `n_entities=200`, default `SplitPolicy`,
`RandomForestClassifier(n_estimators=20, random_state=0)`, sweeping
`inject_graded_future_leak` across alpha in
`[0.00, 0.60, 0.90, 1.10, 1.60, 2.50, 3.00]`, median fl over 5 injector
seeds per alpha). **Before trusting this for histgb, the same methodology
was re-run under rf first, to validate the reconstruction**: it reproduced
the documented reference table almost exactly (0.0956 vs the documented
0.096 ceiling at alpha=1.60; 0.0280 vs 0.028 clean baseline) -- high
confidence the reconstruction is faithful, not a guess.

| alpha | rf fl_med | histgb fl_med |
|---|---|---|
| 0.00 | +0.0280 | +0.0064 |
| 0.60 | +0.0728 | +0.0579 |
| 0.90 | +0.0777 | +0.0723 |
| 1.10 | +0.0874 | +0.0753 |
| 1.60 | +0.0956 (rf ceiling) | +0.0822 |
| 2.50 | +0.0935 | +0.0837 (histgb ceiling) |
| 3.00 | +0.0899 | +0.0818 |

**histgb's temporal saturation ceiling is 0.0837 (at alpha=2.50), lower
than rf's 0.0956 (at alpha=1.60).** `warn_floor=0.10` sits above the
histgb ceiling with **more** margin than it does above rf's (0.0163 vs
0.0044) -- floor anchoring is not broken; if anything it is slightly
safer under histgb on this fixture. **No stop condition triggered.**

## What remains unvalidated (stated plainly, not glossed over)

- **This is one benchmark DGP** (`make_clean_dataset`'s AR(1) rho=0.80
  synthetic panel, at n=2000/200 entities respectively) and **one leak
  shape** (`inject_graded_future_leak`'s row-level graded signal). Real-data
  behavior (Test B's Diabetes-130 panel) has not been re-run under histgb
  as part of this phase -- that would be a natural next check before any
  default flip, not something this phase substitutes for.
- **The rf-derived `null_iqr ~= 0.176/sqrt(n)` scaling law was not
  independently re-derived for histgb** -- only spot-checked for
  plausibility at one n. A dedicated histgb-specific scaling sweep (mirroring
  how the rf law was originally earned) has not been run.
- **The across-entity boundary's own "validated-safe but not fully earned"
  caveat from F2b still applies identically here** -- this phase did not
  change or re-examine that status; `inject_graded_future_leak` still isn't
  the entity-constant-aggregate shape the across-entity null specifically
  targets.
- **Only `fixed_v1` null stopping was tested here** (one variable at a time,
  per the task's own instruction) -- histgb has not been measured in
  combination with Tier 2's `sequential_v1` stopping. That combination is
  untested.
- **Wall-clock numbers are single runs**, not repeated/averaged -- real but
  not statistically hardened measurements.
- This document answers "does the existing rf-calibrated gate transfer to
  histgb," not "should the default be flipped." That is a separate decision
  (Phase C) resting on considerations beyond calibration transfer alone
  (e.g. broader real-data validation, operational familiarity, library
  dependency surface).

## Provenance

Raw 25-cell evidence: `f2b_calibration_histgb.csv` (this directory).
Baseline for comparison: `f2b_calibration_spawn_v2.csv` (this directory).
Regenerate via:

```
python -m zekan.benchmark.nsl_boundary_sweep --estimator histgb --jobs N --output PATH
```

Temporal-ceiling re-measurement was a scratch-only script
(`scratch/tier3_temporal_ceiling.py`, not committed) reconstructing
`label_proxy_sweep.py`'s documented n=200 methodology; re-running it
requires that script or an equivalent rebuild from the methodology
described above.
