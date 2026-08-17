# Categorical Feature Support — Pre-Registration

**Status:** PRE-REGISTERED, NOT YET IMPLEMENTED.
**Date:** 2026-08-17.

This document records the design decisions for native categorical-feature
support in Zekan, before any implementation. It is written against the
current state of the code as re-read on this date; every claim below cites
the specific file and lines it rests on. If any cited behavior is wrong, that
is a defect in this pre-registration and must be corrected here, not papered
over downstream.

---

## 1. Motivation and precedent

Zekan currently requires every candidate feature column to pass numeric
coercion before severity can be computed, and Zekan itself performs no
encoding of any kind.

`zekan/contract/contract_checks.py:204-242`, `_check_feature_columns_numeric`,
computes `feature_cols` as every dataframe column except `entity_id`,
`prediction_time`, `available_features_until`, and `target` (line 220-221 —
`forbidden_after_prediction` columns are deliberately **not** excluded, per
the function's own docstring: "engine.py still includes them in model A/B's
'all_features' set, so they must be numeric too"). For each candidate column
it runs `pd.to_numeric(df[col], errors="coerce")` and fails the whole check
the moment any column has one or more values that fail to coerce (lines
223-228). The function's docstring states why there is no WARN tier: the
downstream cast in `metrics.py` (see below) raises for the whole feature
block on a single bad value, "so ANY coercion failure here must FAIL the
gate, not just a widespread one" (lines 210-212). This check is one of ten
run by `validate_contract` (`contract_checks.py:270-288`) and its FAIL status
makes `ValidationResult.passed` False (`contract_checks.py:39-42`), which is
the harder of the two gates `cli.py` checks before running an audit.

The actual cast this check exists to protect happens in
`zekan/severity/metrics.py:54-84`, `_feature_matrix`:

```python
return df[feature_cols].to_numpy(dtype=np.float32)
```

wrapped in a try/except that the function's own docstring calls "defense in
depth" — `_check_feature_columns_numeric` is "the intended gate for this,"
this cast-level guard exists only for callers that reach `evaluate_folds`
directly, bypassing contract validation (`metrics.py:66-71`). `evaluate_folds`
(`metrics.py:87-160`) calls `_feature_matrix` when no pre-built matrix is
supplied and hands the resulting float32 array straight to the estimator.
There is no encoding step anywhere between the dataframe and this cast.
Zekan therefore cannot audit ordinary mixed-type tabular data without
external preprocessing.

**The precedent, recorded honestly:** Test B (Diabetes-130) hit this exact
failure first. `zekan/benchmark/prepare_test_b.py:87-97` (comment, quoted
verbatim):

> `# Addendum 3 (TEST_B_PREREGISTRATION_ADDENDUM_3.md): the first real B-1 run`
> `# crashed on evaluate_folds's df[feature_cols].to_numpy(dtype=float) cast,`
> `# because Diabetes-130 is mostly raw categorical text (race, gender, age`
> `# bucket, diag_1/2/3, ~20 drug-dosage columns). Addendum 3 pre-registered the`
> `# fix: ordinal/label encoding (sorted-unique -> 0..k-1, same semantics as`
> `# sklearn's OrdinalEncoder), not one-hot (diag_1/2/3 alone would explode into`
> `# 700+ columns each) and not target encoding (would inject target information`
> `# into the features, manufacturing leakage). The mapping uses ONLY each`
> `# column's own values -- no target, no randomness -- and every value present,`
> `# including the '?' sentinel, gets an ordinary code: nothing is cleaned or`
> `# imputed away.`

The fix was external: `_build_ordinal_mappings` / `_apply_ordinal_mappings`
(`prepare_test_b.py:189-218`) — sorted-unique → `0..k-1`, target-free,
computed once and applied outside Zekan, before the data ever reaches a
contract. The resulting mapping was written to
`zekan/benchmark/results/TEST_B_ENCODING_MAP.json` (path constant at
`prepare_test_b.py:106`) for full reproducibility.

**Consequence:** the calibrated Test B baseline — the honest-feature ceiling
and NEAR_CERTAIN/near-bijection thresholds this project's own calibration
documents rest on — was itself measured on externally ordinal-encoded data,
not raw categorical text. Native categorical support therefore **matches**
the conditions the existing calibration was measured under; it does not
depart from them.

**The trigger for this pre-registration:** the Freddie Mac 2018Q1 case-control
frame table (`frames_2018Q1_cc_for_audit.csv`) failed
`feature_columns_numeric` with 22 of 63 candidate columns non-numeric,
reported by the exploratory probe-search audit run on 2026-08-17 (background
task `b0nouyoei`).

---

## 2. Defect recorded separately — probes blocked by an unrelated precondition

This is recorded as a defect independent of categorical support: it exists
today regardless of whether categorical columns are ever declared.

`zekan/detectors/near_bijection_probe.py:76-130`, `_score_feature`, operates
on raw column values, not a numeric cast:

```python
x_cat = x.astype("object").where(x.notna(), _NAN_SENTINEL)
table = (
    pd.DataFrame({"x": x_cat, "y": y})
    .groupby("x", dropna=False)["y"]
    .agg(n1="sum", count="count")
)
```

(lines 91-96). It builds a value→label joint-frequency table via `groupby`
and computes Theil's U from binary entropy over that table
(`_binary_entropy`, lines 64-73). No estimator, no folds, no numeric cast.
`probe_near_bijection` (lines 133-210) calls `candidate_features(contract, df)`
then scores every candidate directly. This is also reflected in the probe's
own registration: `zekan/severity/audit.py:92`, `_ProbeSpec(fn=probe_near_bijection,
needs_folds=False)` — no `needs_model`, `needs_matrix`, or `needs_budget`
flags are set, unlike the undeclared-feature screen immediately above it
(lines 81-88), confirming that Upgrade H's own precondition is nothing more
than a loaded dataframe and a contract.

`zekan/detectors/undeclared_feature_probe.py:225-241`, `_score_one_feature`,
by contrast, calls `evaluate_folds(df, [feature], target_col, folds,
model_factory, X_all=X_sub, y_all=y_all)` (line 238) — the same
`evaluate_folds` covered in Section 1, which goes through
`metrics._feature_matrix`'s `to_numpy(dtype=np.float32)` cast. Upgrade 1
genuinely requires numeric input: it fits a real estimator per candidate
feature on temporal folds.

`zekan/cli.py:112-129` gates both. `result = validate_contract(cfg.contract, df)`
(line 103) is followed by:

```python
if result.passed and result.can_compute_severity:
    ...
elif result.passed:
    ...
    return None, None
else:
    failed = [c.name for c in result.checks if c.status == CheckStatus.FAIL]
    typer.echo(f"CONTRACT FAILED: {', '.join(failed)}, severity will not be computed.", ...)
    raise typer.Exit(1)
```

(lines 112-129). `run_audit` — the function that eventually calls
`_run_structural_probes` and therefore both `probe_near_bijection` and
`probe_undeclared_feature_screen` (`zekan/severity/audit.py:307-313`) — is
imported and invoked only later in the same function (`cli.py:155-168`),
strictly after this gate. When `feature_columns_numeric` FAILs,
`result.passed` is False, the `else` branch fires, and `typer.Exit(1)` is
raised at line 129 **before `run_audit` is ever called**. Upgrade H never
runs, even though its own precondition (a dataframe and a contract, nothing
more) was already satisfied.

As a secondary, currently-unreached detail for completeness: even if
`cli.py`'s gate were bypassed, `run_severity_analysis` carries its own
internal gate (`zekan/severity/engine.py:224-239`) — `if not
val.can_compute_severity: return SeverityResult(status="unavailable", ...)`
— which would also short-circuit before `_run_structural_probes` is reached
from inside `run_audit`. Both gates test the same undifferentiated
`passed`/`can_compute_severity` booleans; neither distinguishes "the AUC
engine needs this" from "Upgrade H doesn't." The `cli.py` gate is the one
that actually fires in the observed Freddie Mac failure, since it sits
upstream of `run_audit` entirely.

**Recorded defect:** a numeric-coercion failure on any candidate feature
column currently blocks a deterministic, dtype-agnostic structural probe
that never needed numeric input, costing the user a full round trip to learn
something Zekan already had the information to tell them.

---

## 3. Decisions

**(a) Declared, not inferred.** The contract gains `categorical_features:
list[str]` (default `[]`) on `PredictionContract`
(`zekan/contract/prediction_contract.py:30-46`). Zekan never infers a
column's semantics — a nominal column and a column of numeral-looking
strings are different things, and only the user knows which is which.
Columns not listed in `categorical_features` that still fail numeric
coercion continue to FAIL `feature_columns_numeric` exactly as today.
Fail-safe is preserved: declaring nothing changes nothing.

**(b) The failing check becomes actionable.** When `_check_feature_columns_numeric`
finds non-numeric columns not covered by a `categorical_features`
declaration, its FAIL message emits a copy-pasteable
`categorical_features:` YAML fragment listing exactly the offending column
names (the same names `bad_cols` already collects at
`contract_checks.py:223-235`), so the user is not left guessing which
columns to declare. The tool does the labour of identifying them; the human
keeps the authority of deciding whether they are truly nominal.

**(c) Probes run when their own preconditions are met.** On contract failure
the exit code stays non-zero, severity is not computed, and TRUSTED (or any
PASS-shaped verdict) is never emitted — none of that changes. But Upgrade H
runs regardless, because Section 2 establishes that its precondition (a
loaded dataframe and a contract) is independent of numeric coercion. The
governing asymmetry: a positive finding (Upgrade H flagging a near-bijection)
is reportable on its own terms; the absence of a finding from a probe that
never ran is not clearance and must not be presented as if it were.
Withholding a discovered near-bijection because an unrelated check failed
costs the user a round trip to learn something Zekan already knew at the
moment the pre-flight checks ran.

**(d) Encoding is ordinal**, sorted-unique to `0..k-1`, deterministic,
target-free — matching `prepare_test_b.py`'s existing scheme
(`_build_ordinal_mappings`, lines 189-204: "Uses only each column's own
values -- no target, no randomness -- so the mapping is deterministic and
reproducible"). One-hot is rejected for the same reason Addendum 3 rejected
it there (cardinality explosion on high-cardinality nominal columns); target
encoding is rejected for the same reason (it would inject target information
into the features, manufacturing leakage).

**(e) The encoding map is emitted in provenance.** Without it, a JSON result
that used categorical encoding is not independently reproducible — the same
class of artifact as the existing data/contract hashes and estimator
identity that `zekan/reports/provenance.py`'s `build_provenance` already
threads through every audit (`cli.py:200-210`). The mapping belongs
alongside those, not as a side file the user has to separately remember
exists.

**(f) The contract schema gains `extra="forbid"`.** Confirmed by inspection:
`zekan/contract/prediction_contract.py` sets no `model_config` /
`ConfigDict` / `extra=` anywhere (grepped across the `contract/` module,
zero matches), so `PredictionContract` runs on plain Pydantic v2 default
behavior — `extra="ignore"` — meaning an unrecognized top-level key is
silently dropped, not rejected. `zekan/config/schema.py`'s `ZekanConfig` and
its sub-models are in the same state. Adding a new, semantically load-bearing
field (`categorical_features`) to a schema that currently swallows typos
silently would create exactly the silent-failure class this project exists
to catch: a user who misspells `categorical_features` as
`categorical_feature` or `categorial_features` would get no error, no
warning, and a contract that behaves as if the field were never declared.
`extra="forbid"` closes that gap contract-wide, not just for this one field.

**(g) Not done: passing `categorical_features` through to
`HistGradientBoostingClassifier`'s native `categorical_features` parameter.**
`zekan/severity/estimators.py:34-62`, `_build_factory`, currently constructs
`HistGradientBoostingClassifier(random_state=42, early_stopping=False)`
(lines 50-60) with no `categorical_features` argument, despite sklearn
supporting it. This is deliberately not changed here: passing it would let
the estimator itself treat declared columns as unordered categories (better
than an arbitrary ordinal split at every internal node), but it would change
what the estimator does with the same input data — changing what
`fixable_leakage` measures relative to every existing calibration run,
including Test B, which was measured against plain ordinal-encoded input
with no native categorical handling. That would break comparability with the
calibrated baseline this whole project rests on. If native
`categorical_features` support to the estimator is wanted later, it needs
its own separate calibration sweep, not a silent piggyback on this change.

---

## 4. Calibration claim — Upgrade H

**Claim:** sorted-unique ordinal encoding is a bijection between a column's
raw distinct values and `0..k-1`. Theil's U (`near_bijection_probe.py:64-130`)
is computed entirely from a value→label joint-frequency table built by
`groupby("x")["y"].agg(n1="sum", count="count")` — i.e., from the *partition*
of rows induced by each distinct value of `x`, not from the numeric identity
of the codes themselves. A bijective relabelling of the values (raw string
→ ordinal integer) preserves that partition exactly: every row that shared a
raw value still shares its ordinal code, and no two rows that had different
raw values are ever merged into the same code. Rare-value pooling
(`SUPPORT_FLOOR`, lines 41-51) operates on the same partition either way,
since pooling groups by `count`, a property of the partition, not of value
identity. Therefore Theil's U for a given column is **invariant** under
sorted-unique ordinal encoding, and the `CRITERION = 0.99` threshold
(`near_bijection_probe.py:58`) requires no recalibration for encoded
columns.

**Falsification condition:** if a measured Theil's U value differs between a
raw-string run and a sorted-unique-ordinally-encoded run of the same column
against the same target, this argument is wrong, and that discrepancy must
be recorded as a finding — not silently reconciled by adjusting the
encoding, the threshold, or the claim after the fact.

---

## 5. Stated limitation — Upgrade 1

Recorded honestly, not claimed away: Upgrade 1's univariate AUC
(`undeclared_feature_probe.py:225-241`, via `evaluate_folds` /
`_feature_matrix`) is rank-based. Sorted-unique ordinal encoding imposes an
arbitrary total order on what may be an unordered nominal column (e.g.
`SERVICER NAME`) — codes 0, 1, 2, ... carry no meaning beyond
alphabetical-sort position. A tree-based estimator (the default `histgb`,
`estimators.py:31`) can recover most of a nominal partition's information
through repeated binary splits at different thresholds, but this is not
guaranteed to exactly reconstruct the original partition the way Theil's U's
groupby does in Section 4 — a single ordinal column can only be split a
bounded number of times per tree, and some partitions of a high-cardinality
column are not efficiently reachable through threshold splits on an
arbitrary integer ordering. Upgrade 1 results (`NEAR_CERTAIN_AUC_FLOOR =
0.99`, `undeclared_feature_probe.py:81`) on ordinally-encoded nominal
features are therefore **not** claimed invariant the way Upgrade H's are.
This is a stated limitation, not a solved problem, and is not addressed by
this pre-registration.

---

## 6. Regression guard — pre-committed

Before this change is considered complete:

1. The existing test suite (887 tests at last count, per the task
   instruction — not independently re-verified in this document, since
   verifying it would require running the suite, which this pre-registration
   explicitly does not do) must pass in full.
2. A Test B frame must be re-run end-to-end through the categorical-aware
   path, with the same declared columns Addendum 3's external encoding
   already covers, to confirm the **numeric-only path produces unchanged
   results** — i.e., a column that was already numeric before this change,
   or a column that was already externally encoded by `prepare_test_b.py`
   before reaching the contract, must produce byte-identical
   `fixable_leakage`, verdict, and Upgrade 1/Upgrade H output to the
   pre-change baseline.

If Test B's numbers move at all under this comparison, the change has
altered the calibrated baseline and must be reverted or explained — never
accepted silently. This is the same discipline every prior addendum in this
project has applied to its own predictions: a contradicting result is a
finding, not something to quietly tune away.

---

## 7. Integrity clause

Findings that contradict anything recorded here — including the Section 4
invariance claim, the Section 5 limitation, or the Section 6 regression
guard — are recorded as dated addenda to this document. The original text
above is never edited to make a later finding fit. No threshold in this
document (`CRITERION`, `NEAR_CERTAIN_AUC_FLOOR`, or any new categorical-path
constant introduced during implementation) is tuned to produce a desired
outcome.
