# Addendum 02 — Regression guard run and passed

Addendum 02 to `FOLD_INERT_FEATURES_PREREGISTRATION.md` (0fc7487), following
addendum 01 (4390f7a). Implemented across `6ae4d40` (machinery) and `8d30a84`
(reporting). Dated 2026-08-18.

**Status: REGRESSION GUARD RUN AND PASSED.**

## 1. Guard requirement

Pre-registration section 11, "Regression guard — pre-committed":

> - The full suite must pass.
> - Test B must reproduce byte-identical `fixable_leakage`, verdict,
>   `naive_auc`, `deployable_auc`, and `nsl` against
>   `scratch/testB2_10k_histgb.json`. No Test B column is ever
>   training-empty, so the adapter must never fire there. Any movement means
>   the change is not inert on the calibrated baseline and must be reverted
>   or explained, never accepted.
> - The monotonicity invariant (section 7) must have a test.
> - ZK-EST-04 must have a test.

Addendum 01 section 8, "Effect on the regression guard":

> Section 11's guard is UNCHANGED except that "ZK-EST-04 must have a test" is
> replaced by: a test asserting that the inert-column set is correctly
> identified and reported, since the probability-identity claim is no longer
> testable. The Test B byte-identical requirement and the monotonicity test
> both stand.

## 2. Result — byte-identical

Baseline: `scratch/testB2_10k_histgb.json`, `estimator_identity` = `histgb`.
Contract: `zekan/benchmark/test_b_contracts/testB2_sensitivity.yml`.
Data: `scratch/testB2_strat.csv` (10,008 rows).

Re-run under the same contract and data, all fields match exactly:

| field | baseline | re-run | match |
|---|---|---|---|
| `fixable_leakage` | `0.40084697565613614` | `0.40084697565613614` | exact |
| `verdict` | `FAIL` | `FAIL` | exact |
| `naive_auc` | `0.9663757037727894` | `0.9663757037727894` | exact |
| `deployable_auc` | `0.58944909861267` | `0.58944909861267` | exact |
| `nsl` | `5.451647315703834` | `5.451647315703834` | exact |
| `estimator_identity` | `histgb` | `histgb` | exact |

`data_sha256` is identical between baseline and re-run
(`d9fda80f...480f1e72`), confirming the same input data was used in both
runs.

## 3. The bypass is confirmed, not assumed

The value match in section 2 is not, by itself, sufficient evidence that the
adapter is inert here — a byte-identical result on Test B alone does not
distinguish "the adapter never fired" from "the adapter fired but happened
to change nothing." This is the load-bearing check: if the adapter had fired
on Test B, a byte-identical result would have been luck, not a demonstrated
bypass.

Measured directly:

- `fold_inert_columns` and `fold_feature_coverage` are both present as JSON
  keys in the re-run output, with value `null`.
- `FEATURE COVERAGE` appears zero times in the human-readable output.
- `PARTIAL` likewise does not appear anywhere in the human-readable output.

Keys-present-with-null follows the same additive-field convention already
established for `categorical_encoding`: it lets `zekan diff` distinguish "no
inerting occurred on this run" (key present, `null`) from "this artifact
predates the fold-inert feature" (key absent).

## 4. Expected difference

`contract_sha256` differs between baseline and re-run:

- baseline: `4625cce4e6a7ebb629f7caf5a3920b76bb6c56f4e2a307bb8efa083fd2505d02`
- re-run: `3257331847a3edec56ff1405f7f1c6a073ce1afa0dac2873d65982b46f3300f3`

This is expected, not a failure. The contract schema gained
`categorical_features` in the earlier categorical-support work, which
changes `contract.model_dump_json()` independent of behavior. This is
already recorded in `CATEGORICAL_SUPPORT_ADDENDUM_01_REGRESSION_GUARD.md`.

## 5. Full suite

899 passed, up from the 895-test baseline recorded before this work, with
four new tests covering: the coverage block's presence when inerting
occurred, the coverage block's absence when it did not, the "active from
fold N" per-column line, and the monotonicity assertion (section 7).

## 6. Still outstanding

Recorded here as open items, not resolved by this guard:

- The adapter is scoped to temporal folds only. The random grouped-CV
  baseline (eval A) is not covered by it; grouped CV has no nesting
  guarantee, so the monotonicity invariant (section 7) is not meaningful
  there — but a fold-inert column could in principle arise on other data
  under grouped CV, unreported.
- Addendum 01 section 7's standing limitation stands unresolved: whether
  `max_features` and `random_state` make column removal perturb the fitted
  model beyond dropping the removed columns' contribution remains
  unresolvable on this stack.
- The opt-in `--from-period` second scope (pre-registration section 10) is
  designed but unbuilt.
- The exact first period per inert column is not reported; it would require
  a per-row scan the machinery never performs, and pre-registration section
  8 already scoped this out as not cheaply available.

## 7. Integrity clause

This addendum records a pre-registered guard's outcome as measured. No
threshold or gate is tuned to produce a desired outcome. Findings
contradicting anything here would be recorded as a further dated addendum,
not silently absorbed.
