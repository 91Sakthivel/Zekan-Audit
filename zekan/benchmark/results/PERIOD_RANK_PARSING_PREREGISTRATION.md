# PERIOD_RANK_PARSING_PREREGISTRATION

Dated 2026-08-18.

**Status: DEFECT RECORDED AND MEASURED. FIX SPECIFIED, NOT YET IMPLEMENTED.**

## 1. The defect

An audit on the Freddie Mac 2018Q1 frame table failed after 40.39s with
`RuntimeError: No interior folds produced OOF predictions for AUC_C.`, raised
at `null_baseline.py:798`, after all 10 pre-flight checks passed and `READY`
was reported.

Root cause: `engine.py` (~line 359) and `null_baseline.py` (~line 508) each
build `period_rank` as

```python
{pd.Timestamp(p).strftime("%Y-%m-%d"): i for i, p in enumerate(sorted_periods)}
```

where `p` is a RAW value from `df[contract.prediction_time]`. For an integer
YYYYMM period column, `pd.Timestamp(201801)` interprets the integer as
**nanoseconds since epoch** and returns `1970-01-01`. Every one of the 99
distinct periods therefore maps to the same key, the dict comprehension keeps
only the last write, and `n_periods` collapses from 99 to 1.

## 2. Measured consequence

With `n_periods = 1` and `leak_lookahead = 1`, `cutoff = 1 - 1 - 1 = -1`.
Every fold's `test_time_max` rank is 98, which exceeds -1, so `is_terminal`
is true for all 5 folds. `interior_fold_idxs` is empty, `_pool_oof_predictions`
returns `None`, and the null estimation raises.

Measured fold table (target_delinquency, Freddie Mac 2018Q1 frame table,
`n_splits=5`, `min_test_rows=100`, `min_pos=20`, `min_neg=20`,
`leak_lookahead=1`):

| fold_idx | train_n | test_n | skipped | terminal | train_pos/neg | test_pos/neg |
|---|---|---|---|---|---|---|
| 0 | 25371 | 26941 | False | True | 2139/23232 | 4682/22259 |
| 1 | 52312 | 16474 | False | True | 6821/45491 | 3787/12687 |
| 2 | 68786 | 10922 | False | True | 10608/58178 | 2577/8345 |
| 3 | 79708 | 8817 | False | True | 13185/66523 | 1240/7577 |
| 4 | 88525 | 2798 | False | True | 14425/74100 | 269/2529 |

5 folds, **0 skipped, 5 terminal, 0 interior**.

All folds clear `min_test_rows=100`, `min_pos=20`, and `min_neg=20` for both
`target_delinquency` and `target_creditevent` — this is **not** a
fold-viability problem. The train/test row counts and positive/negative
counts above are all comfortably above the configured floors.

`target_creditevent` produces an identical empty interior set: 0 skipped,
5 terminal, 0 interior, measured directly by rebuilding the same folds
against that target column. Fold boundaries and terminal classification
depend only on the period column, not on the target. The defect therefore
blocks Frames C, P, and D alike — the whole study, not one frame.

## 3. The inconsistency, stated precisely

`splitters.py`'s `temporal_expanding_folds` parses the whole column correctly
with `pd.to_datetime(df[time_col])` and derives `FoldMeta.test_time_max` from
that parsed series. `engine.py` and `null_baseline.py` instead parse each
**raw** value individually with `pd.Timestamp(p)`. Same column, two parsers,
different results.

The detail that makes this fail silently: the same lines sort correctly,
using `key=lambda x: pd.to_datetime(x)` on the raw values before enumeration
— so period **order** is right, while the period **keys** built immediately
after (via `pd.Timestamp(p).strftime(...)`, not `pd.to_datetime`) are wrong.
Nothing about the sort step looks broken, which is why this was not visible
by inspection.

## 4. The more serious case — silent wrong answers

Recorded plainly: the crash in section 1 only occurs because **all** folds
collapsed to terminal. A **partial** collapse — where some but not all raw
period values happen to parse to distinct `pd.Timestamp(p)` keys while others
collide — would produce a wrong interior set, a wrong pooled null AUC, and
**no error at all**. This is the more dangerous manifestation, and the reason
this defect is being recorded rather than quietly patched.

Recorded as an **open question requiring separate verification, not asserted
here**: whether any previously recorded artifact computed on an integer-period
column is affected by either the crashing or the silent form of this defect.
Test B's period column parses unambiguously under `pd.Timestamp` (it is not
an integer YYYYMM encoding), so Test B results are **expected** to be
unaffected — but this must be verified, not assumed.

## 5. Fix specified

Parse the period column **once**, consistently, using the same
`pd.to_datetime` call `splitters.py` already uses, and derive `period_rank`
from those parsed values — not from raw column values via `pd.Timestamp`.

`engine.py` and `null_baseline.py` must not each re-derive this
independently; a single shared helper is preferable so the two cannot drift
apart again.

What must **not** change: fold construction, the `leak_lookahead` policy, the
terminal/interior definition itself, and any threshold. This is a parsing
correction, not a policy change.

## 6. Falsification and guard, pre-committed

- After the fix, the Freddie Mac audit must produce a **non-empty** interior
  fold set. If it does not, the diagnosis in section 1 is wrong, and that is
  recorded as a finding.
- Test B must reproduce byte-identical `fixable_leakage`, verdict,
  `naive_auc`, `deployable_auc`, and `nsl` against
  `scratch/testB2_10k_histgb.json`. Any movement means Test B **was**
  affected by this defect, which would be a significant finding about prior
  recorded results and must be recorded, not absorbed.
- The full suite must pass.
- A test must pin correct `period_rank` construction for an **integer YYYYMM
  period column** specifically, since that is the case no existing test
  covers.

## 7. Fail-safe note

Recorded: raising `RuntimeError` from inside null estimation gives the user
no verdict, no exit contract, and no guidance — the same failure shape
recorded in `FOLD_LEVEL_ALLNAN_PREREGISTRATION.md` section 5, now in a third
location. Improving that error path is **not in scope here** and is listed
as outstanding work.

## 8. Integrity clause

Findings contradicting anything here are recorded as dated addenda. No
threshold or gate is tuned to produce a desired outcome.
