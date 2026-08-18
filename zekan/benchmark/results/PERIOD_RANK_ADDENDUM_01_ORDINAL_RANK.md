# Addendum 01 — Ordinal rank, not parsed dates

Addendum 01 to `PERIOD_RANK_PARSING_PREREGISTRATION.md` (4c4e3de). Dated
2026-08-18.

**Status: SECTION 3 AND SECTION 5 OF THE PRE-REGISTRATION ARE WRONG. FIX
RESPECIFIED, NOT YET IMPLEMENTED.**

## 1. The pre-registration's diagnosis was wrong

Section 3 claimed:

> `splitters.py`'s `temporal_expanding_folds` parses the whole column
> correctly with `pd.to_datetime(df[time_col])`

This is **false** for an integer YYYYMM column: `pd.to_datetime` on a bare
integer Series, with no `unit` or `format` supplied, also reads the integers
as nanoseconds since epoch. The reference implementation section 5 told the
fix to mirror was itself producing wrong dates.

Recorded honestly: a fix implemented exactly to section 5's specification —
"parse the period column once, consistently, using the same `pd.to_datetime`
call `splitters.py` already uses" — reproduces the defect. This was found by
implementing it and verifying against real data (`build_period_rank` still
returned `n_periods=1` on the Freddie Mac frame table), not by inspection.

## 2. What actually happens — measured

All three parse modes preserve 99 **distinct** values on the real `PERIOD`
column:

```
no arguments:   1970-01-01 00:00:00.000201802, ...   99 distinct
format="%Y%m":  2018-02-01, 2018-03-01, ...           99 distinct
unit="D":       2522-07-08, ...                       99 distinct
```

Only `format="%Y%m"` is calendar-correct.

The precise mechanism: parsing never collides. The collapse happens at
`.strftime("%Y-%m-%d")`, because no-argument parsing places all 99 values
within a sub-day span near the epoch, so day-formatting flattens them to one
string. The defect is therefore not "two parsers disagree" — it is one wrong
parse whose damage is invisible until day-formatting.

## 3. Consumer scope — measured

`train_time_min`, `train_time_max`, and `test_time_min` have **zero**
consumers anywhere in `zekan/`. `test_time_max` has exactly two: `engine.py`'s
terminal determination and `null_baseline.py`'s `_build_interior_fold_set`,
both doing the same `period_rank` lookup. Everything else is second-order
through `is_terminal` (`engine.py`'s `interior_leakages`, `verdict.py`,
`phase5_calibration.py`).

## 4. Test B was never affected — the section 4 open question is CLOSED

Test B's period column is `period_ordinal`, dtype `object` (string), values
like `'2000-01-01'`. `pd.to_datetime` parses it to 24 distinct calendar
dates. It is a string date column, not an integer encoding. Prior recorded
Test B results stand.

Also recorded: **no** existing test or fixture used an integer period column
before this session's uncommitted work, which is why this survived.

## 5. Respecified fix — rank on order, not on parsed dates

`period_rank` is consumed only as `ttm_rank > cutoff`, where
`cutoff = n_periods - 1 - leak_lookahead`. That is a pure **ordinal**
comparison: the logic needs to know which period is later, never what date
it is.

Zekan already has a correct total order — the sort at those same lines uses
`pd.to_datetime` as the sort key, which preserves distinctness and ordering
in every parse mode measured in section 2, including the wrong ones. Wrong
dates, right sequence.

**Fix**: rank periods by position in the sorted distinct list, keyed on the
**raw** column values. No date formatting anywhere in the rank path.

Preferred over the two alternatives considered:

- **(a) `format="%Y%m"`** — rejected: specific to one encoding. Integer
  YYYYMMDD, epoch seconds, or a period counter each need something
  different, and inferring which is the same class of error as inferring
  categorical columns, which this project deliberately refused.
- **(b) fail pre-flight on an ambiguous period column** — rejected: integer
  YYYYMM is ordinary panel encoding in finance and healthcare, not
  ambiguous data. Rejecting it would be brittle, not fail-safe.

## 6. `test_time_max` is a separate, real defect

Ranking on raw order does **not** fix `test_time_max` itself, which is a
**display** field and genuinely reports `1970-01-01` for integer period
columns. Zekan showing a user a false fold end date is its own defect.

It has only two consumers (section 3), both of which become raw-keyed under
this fix, so changing it is cheap. Two sub-options recorded, without
choosing between them:

- report the raw column value (always truthful, not pretty), or
- report the parsed timestamp without day-formatting.

## 7. Falsification, pre-committed

- After this fix, the Freddie Mac audit must produce a **non-empty** interior
  fold set. If it does not, this diagnosis is wrong too, and that is
  recorded as a further finding.
- Test B must still reproduce byte-identical `fixable_leakage`, verdict,
  `naive_auc`, `deployable_auc`, and `nsl`. Section 4 establishes Test B was
  never affected, so **any** movement means the fix broke the working case.
- The full suite must pass.
- The integer-YYYYMM test added in the uncommitted working tree must pass.

## 8. Integrity clause

This addendum records a pre-registered fix specification being falsified by
implementing it. The original document is not edited. No threshold or gate
is tuned to produce a desired outcome.
