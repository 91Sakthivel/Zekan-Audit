# Upgrade 1 step 1d — undeclared-feature screen calibration

This is evidence for calibrating the undeclared-feature screen pre-registered
in `UPGRADE1_PREREGISTRATION.md`. No probe code exists yet (that's step 1e) —
this measures the honest-feature score distribution and the leak anchor, and
reports what the pre-registration's own falsification conditions say about
the design as specified. Same discipline as `TIER3_CALIBRATION.md`: this is
evidence for a decision, written down honestly whichever way it goes, not a
run massaged until the named anchors land where hoped.

**Headline: the SUSPECTED tier as pre-registered (BH-FDR over per-feature
univariate-AUC-vs-0.5 p-values) fails its own pre-committed falsification
condition.** `number_inpatient` is flagged at every tested FDR level, at both
n. This is not a threshold-tuning problem — reasoning and evidence below.
The NEAR_CERTAIN tier (a separate, absolute criterion) is unaffected and
calibrates cleanly.

## AUC → p methodology (resolved before scoring; item A)

The pre-registration locked the *signal* (univariate AUC on temporal
expanding folds, histgb) but left the AUC→p conversion open. Resolved here as
the **analytic route**, confirmed implementable and used as follows:

**Per fold**, treat the fold's AUC as a Mann-Whitney U statistic
(`U = auc * n_pos * n_neg`) and test it against the null AUC=0.5 using the
closed-form, no-ties normal-approximation null variance of U:

```
Var(U)   = n_pos * n_neg * (n_pos + n_neg + 1) / 12
Var(auc) = Var(U) / (n_pos * n_neg)^2 = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)
z_fold   = (auc - 0.5) / sqrt(Var(auc))
```

`n_pos`/`n_neg` come straight from each fold's own `FoldMeta.test_rows` /
`test_base_rate` (already populated by `temporal_expanding_folds`) — no raw
predictions are needed, so this adds zero extra model fits, exactly as the
task required. `_univariate_auc` itself only returns the scalar mean AUC;
this calibration script called `evaluate_folds` directly to get
`EvaluationResult.fold_evals` (per-fold `FoldEval.meta`), which is a strict
superset of what `_univariate_auc` uses — nothing about how
`_univariate_auc` aggregates folds (arithmetic mean of per-fold AUC) made the
per-fold analytic route unsound; the two are answering different questions
(point estimate vs. significance) and don't need to share a weighting scheme.

**Combining folds into one feature-level p-value**: a size-weighted
Stouffer's Z, `z_combined = sum(w_i * z_i) / sqrt(sum(w_i^2))` with
`w_i = sqrt(n_test_i)`, converted to a one-sided p-value (H1: AUC > 0.5, the
leaky direction) via the standard normal survival function. Weighting by
`sqrt(n)` gives larger, more-informative folds more say in the combined
statistic — a standard weighted-Stouffer convention. In practice this barely
mattered here: `temporal_expanding_folds` buckets periods into
equal-count blocks, so every fold's test set is close to the same size
(e.g. 16,961 rows in every one of the 5 valid B-2 100k folds) — the weights
are nearly uniform. This is recorded as a locked methodology decision, not
pinned by the pre-registration.

**This route is sound and was implementable exactly as scoped.** It is not
what failed below — see the FDR finding.

## Frames and n

| frame | n_rows | source |
|---|---|---|
| B2_10k | 10,008 | `scratch/testB2_strat.csv` (pre-existing stratified sample) |
| B2_100k | 101,766 | `testB2_sensitivity.csv`, regenerated via `prepare_test_b.py` (not committed to the repo; lives alongside the raw Diabetes-130 file) |
| B3_100k | 101,766 | `testB3_honest_unknown.csv`, same regeneration |
| B3_10k | 10,008 | same row subset (by `encounter_id`) as `testB2_strat.csv`, applied to the B-3 frame — built here for n-stability on the anchor, not a pre-existing file |
| F2b_honest / F2b_graded | 20,000 | `make_clean_dataset(n_entities=2000, seed=0)` + `inject_graded_future_leak`, same fixture as `nsl_boundary_sweep.py` |

**Housekeeping correction (not an edit to the pre-registration — recorded
here, as the task specified):** the pre-registration states "47 legitimate
Diabetes-130 features ... 48 features minus the 1 declared-forbidden
`planted_leak`." Recomputing directly from `engine.py`'s own
`_feature_cols`/`safe_features` logic against the real B-2 frame gives
**48** non-forbidden features, not 47 (52 columns − 3 role columns
`{patient_nbr, period_ordinal, readmitted_lt30}` − 1 forbidden
`planted_leak` = 48). All 48 were scored; none were arbitrarily dropped to
match the pre-registration's count. This looks like a one-off arithmetic
slip in the pre-registration, not a sign of a different exclusion the code
actually applies — verified there is no other implicit exclusion (e.g.
`encounter_id`, which pre-flight flags as ID-like, is still a real scored
feature; it scored `auc=0.5000, p=0.5` at both n, i.e. it carries no
temporal-fold signal despite being ID-like).

Split policy: B-2/B-3 used the engine's own default
(`n_splits=5, min_test_rows_per_fold=100, min_positive=20, min_negative=20`);
F2b used its own established policy
(`n_splits=5, min_test_rows=50, min_pos=10, min_neg=10`, matching
`nsl_boundary_sweep.py`). All runs produced **5/5 valid folds, 0 skipped**,
at every n and every frame — no partial-coverage caveat applies to this
calibration.

## n-stability finding (item B)

**AUC point estimates are stable across n; p-values are not — by design,
and that instability is the root of the falsification below.**

| feature | AUC @10k | AUC @100k | Δ AUC | p @10k | p @100k |
|---|---|---|---|---|---|
| number_inpatient | 0.6018 | 0.6079 | +0.006 | 5.25e-25 | 2.55e-257 |
| discharge_disposition_id | 0.5712 | 0.5805 | +0.009 | 2.06e-13 | 7.29e-145 |
| diag_1 | 0.5362 | 0.5644 | +0.028 | 1.31e-4 | 3.33e-93 |

The AUC values move by hundredths — genuine sampling noise at a much smaller
scale than 10x more data would suggest, consistent with AUC being a stable
population quantity. The **p-values move by hundreds of orders of
magnitude** for the *same* underlying effect. This isn't a bug in the
combination method — it's `Var(auc) ∝ 1/n` doing exactly what the formula
says: any real, nonzero deviation from AUC=0.5, no matter how small,
becomes arbitrarily significant as n grows. A screen that thresholds on
p-value alone is therefore not just "a little sensitive to n" — its
detection boundary at a fixed q level moves in AUC-space as n changes, and
at real-world dataset sizes it moves toward including modest, honestly
predictive features. This is exactly what happens below.

## Full distributions (item 4)

Raw per-feature/per-cell evidence: `f2b_calibration_undeclared_screen.csv`
(this directory, 129 rows — 48×2 B-2 honest scores, 2 B-3 anchor scores,
6 F2b honest feature scores, 25 F2b graded-leak scores).

### B-2 honest distribution, n=10,008 (48 features, sorted by AUC)

| feature | univariate_auc | p_value |
|---|---|---|
| number_inpatient | 0.6018 | 5.25e-25 |
| discharge_disposition_id | 0.5712 | 2.06e-13 |
| diag_1 | 0.5362 | 0.000131 |
| number_diagnoses | 0.5270 | 0.00322 |
| diag_2 | 0.5246 | 0.00643 |
| time_in_hospital | 0.5236 | 0.00902 |
| diabetesMed | 0.5232 | 0.00968 |
| number_emergency | 0.5232 | 0.00985 |
| age | 0.5212 | 0.0164 |
| metformin | 0.5210 | 0.0166 |
| num_lab_procedures | 0.5165 | 0.0513 |
| payer_code | 0.5164 | 0.0491 |
| diag_3 | 0.5145 | 0.0706 |
| number_outpatient | 0.5145 | 0.072 |
| insulin | 0.5129 | 0.0951 |
| medical_specialty | 0.5117 | 0.123 |
| A1Cresult | 0.5096 | 0.171 |
| glyburide | 0.5081 | 0.204 |
| glipizide | 0.5045 | 0.321 |
| num_procedures | 0.5040 | 0.347 |
| admission_type_id | 0.5034 | 0.367 |
| num_medications | 0.5020 | 0.432 |
| glimepiride | 0.5013 | 0.45 |
| admission_source_id | 0.5012 | 0.465 |
| pioglitazone | 0.5005 | 0.484 |
| race | 0.5003 | 0.484 |
| weight | 0.5003 | 0.498 |
| nateglinide | 0.5001 | 0.498 |
| metformin-pioglitazone | 0.5000 | 0.5 |
| encounter_id | 0.5000 | 0.5 |
| tolbutamide | 0.5000 | 0.5 |
| acetohexamide | 0.5000 | 0.5 |
| glimepiride-pioglitazone | 0.5000 | 0.5 |
| metformin-rosiglitazone | 0.5000 | 0.5 |
| chlorpropamide | 0.5000 | 0.5 |
| miglitol | 0.5000 | 0.5 |
| examide | 0.5000 | 0.5 |
| citoglipton | 0.5000 | 0.5 |
| tolazamide | 0.5000 | 0.5 |
| troglitazone | 0.5000 | 0.5 |
| glipizide-metformin | 0.5000 | 0.5 |
| max_glu_serum | 0.4999 | 0.511 |
| acarbose | 0.4996 | 0.516 |
| repaglinide | 0.4991 | 0.537 |
| rosiglitazone | 0.4989 | 0.546 |
| glyburide-metformin | 0.4984 | 0.562 |
| change | 0.4881 | 0.886 |
| gender | 0.4867 | 0.913 |

### B-2 honest distribution, n=101,766 (48 features, sorted by AUC)

| feature | univariate_auc | p_value |
|---|---|---|
| number_inpatient | 0.6079 | 2.55e-257 |
| discharge_disposition_id | 0.5805 | 7.29e-145 |
| diag_1 | 0.5644 | 3.33e-93 |
| diag_3 | 0.5617 | 4.52e-86 |
| diag_2 | 0.5537 | 2.13e-65 |
| time_in_hospital | 0.5434 | 1.25e-43 |
| number_diagnoses | 0.5420 | 4.2e-41 |
| insulin | 0.5377 | 2.3e-33 |
| number_emergency | 0.5362 | 1.34e-30 |
| num_medications | 0.5333 | 1.57e-26 |
| medical_specialty | 0.5231 | 6e-14 |
| number_outpatient | 0.5229 | 1.89e-13 |
| age | 0.5220 | 1.53e-12 |
| diabetesMed | 0.5180 | 4.61e-09 |
| admission_source_id | 0.5177 | 1.02e-08 |
| payer_code | 0.5173 | 2.2e-08 |
| change | 0.5154 | 5.16e-07 |
| metformin | 0.5154 | 5.33e-07 |
| num_procedures | 0.5135 | 7.52e-06 |
| num_lab_procedures | 0.5113 | 0.000189 |
| admission_type_id | 0.5105 | 0.000463 |
| A1Cresult | 0.5098 | 0.000918 |
| glyburide | 0.5030 | 0.172 |
| race | 0.5028 | 0.181 |
| glimepiride | 0.5027 | 0.198 |
| rosiglitazone | 0.5025 | 0.214 |
| max_glu_serum | 0.5020 | 0.261 |
| repaglinide | 0.5017 | 0.294 |
| gender | 0.5017 | 0.3 |
| pioglitazone | 0.5005 | 0.443 |
| chlorpropamide | 0.5001 | 0.489 |
| acarbose | 0.5001 | 0.493 |
| tolazamide | 0.5001 | 0.493 |
| tolbutamide | 0.5000 | 0.498 |
| miglitol | 0.5000 | 0.499 |
| encounter_id | 0.5000 | 0.5 |
| metformin-pioglitazone | 0.5000 | 0.5 |
| citoglipton | 0.5000 | 0.5 |
| acetohexamide | 0.5000 | 0.5 |
| examide | 0.5000 | 0.5 |
| metformin-rosiglitazone | 0.5000 | 0.5 |
| glimepiride-pioglitazone | 0.5000 | 0.5 |
| troglitazone | 0.5000 | 0.5 |
| glipizide-metformin | 0.5000 | 0.5 |
| glyburide-metformin | 0.4997 | 0.532 |
| nateglinide | 0.4997 | 0.536 |
| weight | 0.4996 | 0.558 |
| glipizide | 0.4980 | 0.728 |

### F2b synthetic honest features, n=20,000 (`make_clean_dataset`, seed=0)

| feature | univariate_auc | p_value | design signal |
|---|---|---|---|
| feature_0 | 0.7630 | ~0 (underflow) | strong entity-level signal (coeff 0.7) |
| feature_1 | 0.7609 | ~0 (underflow) | strong entity-level signal (coeff 0.7) |
| feature_2 | 0.5808 | 1.67e-61 | moderate signal (coeff 0.3) |
| feature_3 | 0.5012 | 0.345 | pure noise |
| feature_4 | 0.5040 | 0.192 | pure noise |
| feature_5 | 0.4970 | 0.747 | pure noise |
| graded_future_leak, alpha=0.0 (5 seeds) | 0.497–0.508 | 0.13–0.70 | honest by construction (`is_leak=False`) |

**This matters beyond graded coverage**: `feature_0`/`feature_1` at AUC≈0.76
are honest by construction, yet score well above anything in the real B-2
frame (max 0.6079). This is exactly why the pre-registration wanted the
synthetic fixtures included — a real dataset alone doesn't show how high a
*legitimate* feature's univariate AUC can go. It also means the true
honest-tail ceiling used for margin/threshold reasoning below is **0.7630**
(F2b), not 0.6079 (B-2) alone.

### F2b graded leak (`graded_future_leak`, alpha>0, 5 seeds each)

| alpha | SNR | AUC range (5 seeds) | p range |
|---|---|---|---|
| 0.60 | 1.2 | 0.601–0.615 | 6.9e-88 – 2.2e-116 |
| 1.10 | 2.2 | 0.632–0.642 | 1.6e-148 – 2.2e-172 |
| 1.60 | 3.2 | 0.641–0.649 | 5.1e-168 – 1.2e-189 |
| 2.50 | 5.0 | 0.647–0.655 | 4.4e-181 – 8.3e-206 |

Notable, unprompted-but-relevant observation: alpha=0.60's AUC range
(0.601–0.615) sits almost exactly on top of `number_inpatient`'s AUC
(0.602–0.608) — a real, modest, genuine partial leak and a real, modest,
genuinely-honest strong predictor are **numerically indistinguishable** at
the raw-AUC level. This doesn't itself trigger a falsification condition
(those are specifically about the B-2/B-3 real-data anchors), but it's a
second, independent confirmation of the same structural issue as the
`number_inpatient` finding: **AUC magnitude alone, in the moderate range,
cannot separate "honest but predictive" from "leaky but imperfect."** This
is already acknowledged in the pre-registration's own two-tier design
(SUSPECTED is explicitly a judgment call) — the FDR-level problem below is
about SUSPECTED failing even on features it was never supposed to need
judgment for.

### Leak anchor (`readmitted`, raw, B-3 frame)

| n | mean AUC | per-fold AUC (all 5 folds) | p |
|---|---|---|---|
| 10,008 | 1.0000 | [1.0, 1.0, 1.0, 1.0, 1.0] | ~0 (underflow) |
| 101,766 | 1.0000 | [1.0, 1.0, 1.0, 1.0, 1.0] | ~0 (underflow) |

Perfect separation, every fold, both n — no ambiguity in the anchor itself.

## Measured margin (item 5)

| | honest-tail max | leak anchor (min per-fold, either n) | gap |
|---|---|---|---|
| B-2 alone | 0.6079 (number_inpatient, 100k) | 1.0000 | 0.392 |
| B-2 + F2b honest | 0.7630 (feature_0, F2b) | 1.0000 | 0.237 |

**No overlap at either n, by a wide margin** — the honest distribution
(including the synthetic fixture's higher-signal features) tops out well
below 0.80; the leak sits at exactly 1.0 in every fold, both n. The
honest/leak-overlap falsification condition (below) does not trigger.

## Falsification conditions (item 6/7 gate)

1. **`number_inpatient` must not be flagged at the derived SUSPECTED
   threshold → TRIGGERED.** See below — reported as a design failure, not
   tuned around.
2. **Honest distribution and leak must not overlap → does not trigger.**
   Margin is 0.237–0.392 with zero data points in between (see above).
3. **Raw `readmitted` must clear NEAR_CERTAIN → does not trigger (clears
   trivially).** AUC=1.0 in every evaluated fold, both n (see above).

## Item 6: FDR level for SUSPECTED — design failure, not a threshold

Benjamini-Hochberg was run directly on the 48 p-values at both n, at
q ∈ {0.001, 0.01, 0.05, 0.10, 0.20}:

| q | flagged @10k (of 48) | number_inpatient flagged? | flagged @100k (of 48) | number_inpatient flagged? |
|---|---|---|---|---|
| 0.001 | 2 | **yes** | 20 | **yes** |
| 0.01 | 3 | **yes** | 22 | **yes** |
| 0.05 | 4 | **yes** | 22 | **yes** |
| 0.10 | 10 | **yes** | 22 | **yes** |
| 0.20 | 10 | **yes** | 22 | **yes** |

`number_inpatient` is flagged at **every** q tested, at **both** n.

**This is structural, not a tuning problem.** `number_inpatient` is the
rank-1 (smallest) p-value among the 48 honest features at both n. The BH
procedure's own definition — reject all hypotheses at rank ≤ k, where k is
the *largest* rank satisfying `p_(k) ≤ (k/m)·q` — means rank 1 is included
in the flagged set **whenever the flagged set is non-empty at all**, for any
q. At n=100k, the flagged set is non-empty for q as small as
~1.2×10⁻²⁵⁵ (`48 × p_(1)`) — a value with no meaning as an actual FDR
control level, and one that would also exclude every other honest feature's
legitimate signal, defeating the screen's purpose. There is no usable q that
protects `number_inpatient` while still functioning as a screen.

**Why this happens (root cause, tying back to the n-stability finding):**
`number_inpatient` is not a false positive under the null being tested
(`AUC = 0.5`, i.e. "no relationship to the target at all") — it genuinely,
legitimately has real predictive signal (it is Test B's own previously
identified strongest honest predictor). With enough data, the analytic test
detects that real relationship with overwhelming certainty
(p=2.55×10⁻²⁵⁷ at n=100k) — correctly, in the narrow statistical sense. The
problem is that **"detectably non-null" and "leak-like" are not the same
question**, and BH-FDR, applied to a per-feature significance test against
`AUC=0.5`, answers the former, not the latter. The F2b fixture reproduces
the identical pattern independently (`feature_0`/`feature_1`, honest by
construction, p≈0 at n=20,000) — this is not a Diabetes-130-specific
artifact.

**Per the pre-registered falsification protocol: this is reported as a
design failure of the SUSPECTED tier as currently specified, not adjusted
until it passes.** No FDR level is recommended here. Redesigning the
SUSPECTED-tier statistic (e.g. testing against a reference/null built from
the dataset's own honest-feature distribution rather than the fixed
`AUC=0.5` null, or requiring a permutation-based null the way the engine's
own `TEMPORAL_LEAKAGE` gate already does) is out of scope for this
calibration step and is not proposed as a fix here — this document's job was
to measure and report, which it does: **the SUSPECTED tier cannot proceed
to implementation (step 1e) on the method as pre-registered.**

## Item 7: NEAR_CERTAIN absolute criterion — calibrates cleanly

Unaffected by the SUSPECTED-tier finding: this is a separate, absolute,
per-fold criterion, not an FDR level.

- Honest-tail per-fold ceiling: `number_inpatient`'s worst (highest) single
  fold was 0.6195 (100k); F2b's `feature_0`/`feature_1` (mean 0.76) are the
  overall honest ceiling. No honest feature, real or synthetic, in any
  single fold, at any n, approached even 0.65.
- Leak anchor: **exactly 1.0 in every one of 5 folds, at both n** — no
  per-fold variation to be tie-robust against in this evidence, but the
  criterion is written as "every evaluated fold ≥ X" specifically so a
  near-but-imperfect duplicate (e.g. B-2's own `planted_leak`, which has 5%
  label noise) would still clear a threshold placed comfortably under 1.0.

**Derived: `X = 0.99`, every evaluated fold.** Positioned deep inside the
empty gap between the honest ceiling (~0.76) and the leak (1.0 exactly) —
the exact value is not pinned tightly by this data (nothing sits close
enough to 0.99 to adjudicate the precise cut, same honest caveat
`F2b_CALIBRATION.md` recorded for the NSL=1.0 boundary), but it sits deep
inside a wide, cleanly empty gap on both sides: comfortably above every
honest score ever observed (real or synthetic, ~0.30 of margin), and
comfortably below the leak's actual value with room for realistic
imperfection (a near-duplicate at 0.99–0.999 would still clear it; the
raw B-3 anchor clears it with the maximum possible margin at 1.0 exactly,
in every fold, both n).

## Clean bill of health carried over from step 1c

Registry names the pre-registration cited as precedent
(`CORRELATED_LEAK_PAIR`, `ENTITY_CONTAMINATION`/`ENTITY_CONTAMINATION_RISK`,
`TEMPORAL_LEAKAGE`) were verified against the real `_REGISTRY` in step 1c —
all three exist exactly as cited, no correction needed. Recorded here per
the task's own instruction that any 1c correction belongs in this document;
there was none.

## What remains unvalidated (stated plainly, not glossed over)

- **The SUSPECTED tier has no working design.** This calibration measured
  the pre-registered method and found it fails its own falsification gate.
  No alternative statistic was designed, evaluated, or even attempted here
  — that is explicitly out of scope for a measure-and-report calibration
  step. Step 1e cannot implement a SUSPECTED probe against the method as
  currently pre-registered.
- **Only histgb was measured**, per the pre-registration's own scope
  (estimator-coupled, matching Tier 3 Phase C's default). Not re-measured
  under rf or gbm.
- **Only one real dataset (Diabetes-130) and one synthetic DGP
  (`make_clean_dataset`'s AR(1) panel) were used.** The F2b honest features
  materially raised the observed honest ceiling versus B-2 alone (0.76 vs
  0.61) — a third dataset shape could plausibly raise it further. The
  NEAR_CERTAIN margin (0.237 at minimum) has room to absorb this, but this
  wasn't stress-tested beyond these two sources.
- **The screenability gate (minimum-information floor for
  mostly-missing columns) was not applied here.** All 48 B-2 features were
  scored regardless of missingness (e.g. `weight`, 97% missing per Test B's
  own setup, scored auc≈0.50 at both n — not a freak result here, but this
  calibration didn't stress-test the gate itself, only noted the column
  didn't misbehave without it).
- **The wide-data pre-rank (fast correlation-style pre-selection for
  datasets too large to fully temporal-score) was not exercised.** 48
  features was small enough to score every column directly; this
  calibration says nothing about pre-rank behavior at larger column counts.
- **`known_strong_features` contract suppression was not exercised** — no
  contract carrying that field exists yet. Whether it would have "solved"
  the `number_inpatient` falsification by suppression (rather than the
  screen never flagging it in the first place) was deliberately not
  investigated: per the pre-registration's own falsification protocol, the
  test is whether the *unsuppressed* screen would flag it, not whether an
  operator-provided allowlist can patch the result afterward.
- **Combination-method sensitivity was not swept.** The sqrt(n)-weighted
  Stouffer combination was used throughout; an unweighted Stouffer or a
  Fisher's-method combination were not run side-by-side to confirm the
  `number_inpatient` finding is combination-method-invariant. Given
  `number_inpatient`'s p-value margin over the BH threshold is enormous
  (multiple hundreds of orders of magnitude at n=100k), this is not
  expected to change the finding, but it was not empirically checked.

## Provenance

Raw evidence: `f2b_calibration_undeclared_screen.csv` (this directory, 129
rows: 48 B-2 honest features × 2 n, 2 B-3 anchor points, 6 F2b honest
features, 25 F2b graded-leak cells).

Regenerate via the scratch-only script (not committed, same convention as
`scratch/tier3_temporal_ceiling.py`):

```
.venv/Scripts/python.exe scratch/upgrade1_calibration.py
```

Requires `testB2_sensitivity.csv` and `testB3_honest_unknown.csv` (the full
101,766-row Test B frames) alongside the raw `diabetic_data.csv` — these are
regenerated by `zekan/benchmark/prepare_test_b.py` and were not copied into
the repo (only the 10k stratified sample, `scratch/testB2_strat.csv`, is
committed there). `scratch/testB2_strat.csv`'s own generation method (beyond
"417 rows per period_ordinal bucket, preserving temporal structure") was not
re-derived here — it was reused as-is, and the B-3 10k frame was built by
matching its exact `encounter_id` set for a directly comparable pair.

## Upgrade 1 step 1g — post-implementation validation

Step 1e implemented the probe (`zekan/detectors/undeclared_feature_probe.py`)
against this document's own findings: `NEAR_CERTAIN_UNDECLARED_LEAK` as
pre-registered (absolute, `AUC >= 0.99` every fold), and the SUSPECTED tier
replaced by an annotate-nothing ranked panel, per the FDR design failure
recorded above. Step 1f surfaced both in `text_view.py`/`html_view.py`. This
section closes the loop: does the shipped behavior match what the
pre-registration and this calibration predicted, run end-to-end on the real
101,766-row Test B frames, on the default path (no `--stability`, default
`n_jobs`)? Logs: `scratch/1g_B1.log`, `1g_B2.log`, `1g_B3.log`.

### Pre-registered validation conditions vs. what ran

| condition (`UPGRADE1_PREREGISTRATION.md`, "Validation conditions") | met? |
|---|---|
| B-1: verdict unchanged (TRUSTED); `number_inpatient` not annotated at either tier | **Yes.** TRUSTED, unchanged. No `NEAR_CERTAIN`. `number_inpatient` appears only in the panel (rank 1, AUC 0.6079), which carries no threshold and asserts no issue — not an annotation. |
| B-3: verdict unchanged (PASS); `NEAR_CERTAIN_UNDECLARED_LEAK` present, naming `readmitted`, rendered prominently, AUC shown in text | **Yes.** PASS, unchanged. `NEAR_CERTAIN` block renders immediately after the verdict headline, names `readmitted`, shows `AUC 1.0000` in the rendered text (not JSON-only). |
| B-2: `planted_leak` outside screen scope (declared forbidden), not duplicate-flagged; no honest feature newly flagged | **Yes.** `planted_leak` absent from the panel (it's excluded from candidates as declared-forbidden, per `probe_undeclared_feature_screen`'s `forbidden` set) and from `NEAR_CERTAIN` — its leakage is still caught, correctly, by the existing B/C attribution path (`+0.3089` AUC), which is a separate mechanism from this screen. No panel entry crossed the `NEAR_CERTAIN` floor. |
| Temporal-vs-random wiring test passes | **Yes**, but not exercised by these three runs — covered separately by `tests/test_undeclared_feature_probe.py::test_temporal_vs_random_invariant_screen_reports_temporal_score`, part of the "Full suite" pass below. |
| Resilience: a probe exception surfaces as a registered `PROBE_FAILED` record, not a crash; soft time budget respected | **Partially verified.** `PROBE_FAILED` isolation is generic infrastructure in `audit._run_structural_probes` (`tests/test_structural_probe_wiring.py`), and the undeclared-feature probe is registered through the same `_ProbeSpec` mechanism (`test_registered_in_probe_registry_with_correct_capability_flags`) — so an exception in this probe specifically is caught by construction, not by a probe-specific test. No failure was injected into this probe directly. The soft time budget (`deadline`) is accepted for calling-convention compatibility but not consulted by the probe (documented in its own docstring) — not a violation of the pre-registration's letter (which called it a "soft, cooperative" budget), but not actively enforced either. |

### Run summaries (full 101,766 rows, default path)

- **B-1** (`scratch/1g_B1.log`): verdict **TRUSTED**, unchanged. No `NEAR_CERTAIN`. `number_inpatient` ranked #1 in the panel, AUC 0.6079, reported as a candidate with no claim attached. Screened 48 of 48 non-forbidden features, 0 not screenable.
- **B-2** (`scratch/1g_B2.log`): verdict **FAILED**, unchanged. B−C fixable-leakage inflation +0.3095 AUC; `planted_leak` #1 in attribution at +0.3089 AUC. `planted_leak` correctly **absent** from the screen (declared forbidden, out of scope by construction). Screened 48 of 48. No honest feature newly flagged.
- **B-3** (`scratch/1g_B3.log`): verdict **PASS**, unchanged. `NEAR_CERTAIN` block names `readmitted`, AUC 1.0000, rendered immediately after the verdict. Panel shows `readmitted` at 1.0000 then a cliff to `number_inpatient` at 0.6079 — the same ~0.39 gap this document's own margin measurement (item 5, above) predicted. Screened 49 of 49 (49, not 48, because raw `readmitted` is itself an undeclared, unscreened-out candidate in this frame).
- **Full suite**: 837 passed.

### Measured cost correction

The pre-registration's own calibration plan did not fix a cost estimate; a
separate, earlier estimate (~5–15s per audit for the screen) was derived
from the 10k-row case and never re-measured at the full 101,766-row scale
before this task. That estimate was **low, and scoped to 10k** — the real
cost at full scale is materially higher.

Measured wall-clock (log file mtimes, start-to-finish):

| run | wall-clock | vs. pre-screen baseline |
|---|---|---|
| B-1 (TRUSTED) | ~54s | — |
| B-2 (FAILED) | **~1458s (24m18s)** | baseline (`TEST_B_RESULTS.md`) was 557s pre-screen — **~2.6x** |
| B-3 (PASS) | ~55s | — |

B-2 is the only one of the three that pays **both** the full null/ablation
attribution path (needed to measure and rank `planted_leak`'s damage) **and**
the undeclared-feature screen — B-1 and B-3 are fast because a TRUSTED/PASS
verdict does comparatively little attribution work. B-2's true screen-only
marginal cost (1458s − 557s ≈ 900s, roughly 15 minutes) is the honest number
for a full-attribution audit at this row count, not the pre-registration's
~5–15s estimate.

**Contributing factor**: 48 non-forbidden features sits just under
`WIDE_DATA_CAP = 50` (`undeclared_feature_probe.py:104`), so every feature on
this frame received a full temporal-fold model fit rather than the cheap
`_pre_rank` correlation pass — the wide-data cap path never triggered on
this data (also noted separately below, under "what remains unvalidated").
This is recorded as a finding, not a fix: the cap value itself is not
changed by this task.

**Follow-up flagged, not actioned here**: revisit `WIDE_DATA_CAP` (currently
50) — whether the pre-rank threshold should be materially lower than
"barely above Diabetes-130's own feature count" now that full scoring at
48 features has been measured to cost ~15 minutes of marginal wall-clock on
a 100k-row frame.

### What remains unvalidated (post-1g, in addition to step 1d's own list above)

- **The screen was not exercised under `--stability`.** All three runs used
  the default path. Whether the panel/`NEAR_CERTAIN` data survives
  `_apply_seed_stability`'s re-verdicting unchanged (or interacts with it at
  all) was not checked here.
- **The wide-data cap path never triggered on this data.** Both B-1/B-2 (48
  candidates) and B-3 (49) sit under `WIDE_DATA_CAP = 50`, so `_pre_rank`
  never ran in any of these three validation runs — only the
  step-1d calibration's own reasoning (not a real end-to-end run) speaks to
  its behavior.
- **Name-similarity corroboration contributes nothing here.** As documented
  in `undeclared_feature_probe.py`'s own module docstring, `_name_score`
  matches a fixed set of temporal-keyword shapes (`final_*`, `days_to_*`,
  `_after_`, etc.), not feature-vs-target name similarity — confirmed again
  in this run: `readmitted` (target `readmitted_lt30`) scores
  `name_pattern_score=0.0` despite being an almost-literal name match to the
  target. The corroboration field is present in every panel/`NEAR_CERTAIN`
  record but added no signal in the one case where a human would most
  expect it to.
- **`known_strong_features` remains unimplemented.** No `PredictionContract`
  field exists yet (only the unused `suppressed_by_known_strong_features`
  bool on the dormant `SuspectedUndeclaredLeakDetail` struct) — there is
  nothing for it to suppress while the SUSPECTED tier itself stays deferred,
  so this was correctly out of scope for 1g, not a gap introduced by it.
- **Resilience was verified generically, not with a failure injected into
  this specific probe** (see validation-conditions table above) — the
  isolation mechanism is shared infrastructure already tested elsewhere, but
  no test drives an exception through
  `probe_undeclared_feature_screen` itself to confirm it degrades the same
  way.
