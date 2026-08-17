# Fold-Level All-NaN Column Crash — Pre-Registration

**Status:** DEFECT RECORDED, MEASURED, FIX NOT YET DECIDED.
**Date:** 2026-08-17.

This document records a defect and the measurements that pin down its cause,
before any fix is chosen or any Python file is touched. No code is modified
by this document.

---

## 1. The crash

Running an audit on the Freddie Mac 2018Q1 frame table (91,323 rows, 64
feature columns, 22 declared categorical) terminated with exit code 1 after
19.17 seconds, inside the first `evaluate_folds` call at `engine.py:342`,
before any probe, verdict, or JSON output existed.

Exception: `ValueError: window shape cannot be larger than input array shape`,
raised in `sklearn/ensemble/_hist_gradient_boosting/binning.py`
`_find_binning_thresholds` via `sliding_window_view(distinct_values, 2)`,
inside a joblib worker. No column name appears anywhere in the traceback.

## 2. Root cause, established by measurement

Installed scikit-learn is `1.9.0`. Its `_find_binning_thresholds` contains an
explicit guard: `if len(distinct_values) == 1`, return an empty array. The
crash is in the NEXT branch: when `distinct_values` has length 0, the
condition `len(distinct_values) <= max_bins` is true and
`sliding_window_view` on a zero-length array raises.

Four probes against the exact estimator config Zekan uses
(`HistGradientBoostingClassifier(random_state=42, early_stopping=False)`),
each paired with a varying column so the fit is otherwise valid:

| probe | input | outcome |
|---|---|---|
| (a) constant, no NaN | `[1.0,1.0,1.0,1.0]` | PASS |
| (b) constant with NaN | `[1.0,1.0,nan,1.0,nan,1.0]` | PASS |
| (c) entirely NaN | `[nan,nan,nan,nan]` | `ValueError` (the crash) |
| (d) two distinct values, some NaN | `[1.0,2.0,nan,1.0,2.0,nan]` | PASS |

Record plainly: only an entirely-NaN column crashes. Constant columns are
handled correctly by this sklearn version.

## 3. Two hypotheses falsified before the correct one

Record both, with why each was wrong. They were reasonable and measurement
killed them; preserving that is the point of this document.

- **Hypothesis 1 (mine): whole-dataframe all-NaN columns.** FALSIFIED — the
  frame table has ZERO entirely-NaN columns.
- **Hypothesis 2 (external expert review): constant columns crash the
  binner**, and the fix is to keep vs. drop them, with the
  sparse-constant-plus-NaN case carrying missingness signal. FALSIFIED as a
  diagnosis by probes (a), (b) and (d) above — constant columns do not crash
  sklearn 1.9.0 at all. Record that the review's separate point about
  distinct-value semantics (`s.dropna().nunique()` vs
  `s.nunique(dropna=False)`) remains valid on its own terms and is noted
  below (section 8).

## 4. Actual cause — fold-level, not dataframe-level

The crashing column is entirely NaN within a FOLD, though not across the
whole dataframe. Folds built by
`temporal_expanding_folds(time_col='PERIOD', entity_col='LOAN IDENTIFIER',
target_col='target_delinquency', n_splits=5, min_test_rows=100, min_pos=20,
min_neg=20)` — all `SplitPolicy` defaults.

Fold table:

| fold | train | train period | test | test period |
|---|---|---|---|---|
| 0 | 25,371 | 201801-201904 | 26,941 | 201905-202009 |
| 1 | 52,312 | 201801-202009 | 16,474 | 202010-202202 |
| 2 | 68,786 | 201801-202202 | 10,922 | 202203-202306 |
| 3 | 79,708 | 201801-202306 | 8,817 | 202307-202410 |
| 4 | 88,525 | 201801-202410 | 2,798 | 202411-202603 |

Entirely-NaN in a TRAIN slice: `BANKRUPTCY CRAMDOWN COSTS` (folds 0 and 1),
`CUMULATIVE MODIFICATION COSTS` (fold 0), `CURRENT PERIOD MODIFICATION COSTS`
(fold 0). Entirely-NaN in a TEST slice: `BANKRUPTCY CRAMDOWN COSTS` (fold 0).

`BANKRUPTCY CRAMDOWN COSTS` non-NaN count per fold train slice: 0, 0, 3, 4, 8;
per test slice: 0, 3, 1, 4, 163.

Record the mechanism: the Freddie Mac guide states this field is populated
only in the period of zero balance and not at all for loans zero-balanced on
or before September 2025. It is therefore structurally absent early and
appears late, and expanding-window folds place the sparse tail last.

## 5. Why this is a fail-safe violation

Zekan's spine is "cannot verify, exit non-zero." Here the pre-flight gate
passes, then the fit dies inside a joblib worker with an sklearn traceback
that names no column, no verdict, and no exit contract.

Record the structural point: NO whole-dataframe pre-flight check can detect
this, because the condition is fold-dependent. A check inspecting `df` before
splitting will always pass while the fit still fails. Any fix must therefore
live where folds exist, not in `contract_checks.py` alongside the other
checks.

## 6. Generality

Record that this is not specific to Freddie Mac. Any temporal audit where a
field is introduced partway through the observation window hits it — schema
changes, new regulatory reporting fields, instrumentation added mid-history.
This is ordinary in real temporal data.

## 7. Open question, not decided here

Record without choosing: a column that is absent early and populated later is
itself a temporal-availability signal, which is arguably something a leakage
auditor should FLAG rather than route around. Whether the correct response is
to skip the fold, fail the audit, or treat this as a detectable condition in
its own right is NOT decided in this document.

Record the candidate responses without selecting one:

- (a) mark the fold skipped with a reason, reusing the existing
  `folds_skipped`/`skip_reasons` machinery;
- (b) fail the audit with a message naming the column and fold;
- (c) exclude the column from that fold's fit only — noting that this would
  make B and C see different feature sets across folds, changing what pooled
  `fl` means.

## 8. Separate, smaller items recorded here

- Zekan cannot read pipe-delimited files: `cli.py` uses plain `pd.read_csv`
  with no separator option, supporting only `.csv` and `.parquet`. This
  forced an external conversion of the frame table. A usability gap,
  recorded, not fixed here.
- The pre-flight constant-column NOTE uses distinct-value counting that
  includes NaN, so it detected 4 of 7 constant columns and missed the three
  with one real value plus NaN. Independent of this crash, since constant
  columns do not crash sklearn 1.9.0, but the semantics are wrong either way.

## 9. Integrity clause

Findings contradicting anything recorded here are recorded as dated addenda.
No threshold or gate is tuned to produce a desired outcome.
