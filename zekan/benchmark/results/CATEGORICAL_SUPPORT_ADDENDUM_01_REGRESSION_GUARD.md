# Addendum 01 — Regression Guard Result

Addendum 01 to `CATEGORICAL_SUPPORT_PREREGISTRATION.md` (`a2875e0`), implemented
across `1d1aa05`, `c8a1627`, `c6c13aa`.

**Date:** 2026-08-17.
**Status:** REGRESSION GUARD RUN AND PASSED.

---

## 1. Guard requirement

Quoted from `CATEGORICAL_SUPPORT_PREREGISTRATION.md` section 6:

> Before this change is considered complete:
>
> 1. The existing test suite (887 tests at last count, per the task
>    instruction...) must pass in full.
> 2. A Test B frame must be re-run end-to-end through the categorical-aware
>    path, with the same declared columns Addendum 3's external encoding
>    already covers, to confirm the **numeric-only path produces unchanged
>    results** — i.e., a column that was already numeric before this change,
>    or a column that was already externally encoded by `prepare_test_b.py`
>    before reaching the contract, must produce byte-identical
>    `fixable_leakage`, verdict, and Upgrade 1/Upgrade H output to the
>    pre-change baseline.
>
> If Test B's numbers move at all under this comparison, the change has
> altered the calibrated baseline and must be reverted or explained — never
> accepted silently.

---

## 2. First attempt was invalid — recorded, not hidden

The guard was first run against B-1 (`testB1_specificity.yml` +
`scratch/testB1_run2.json` as baseline). This comparison was **invalid**,
not a failure of the change:

- That baseline carries `estimator_identity: "default"` — a
  pre-Tier-3-Phase-C artifact, recorded from when the codebase's default
  estimator was `rf` and before `cli.py` was fixed to resolve
  `estimator_identity` to a concrete name instead of leaving it as the
  literal word `"default"`. The current run correctly recorded
  `estimator_identity: "histgb"`. Two different estimators, not a
  before/after comparison of the same one.
- `naive_auc` and `deployable_auc` differed accordingly: baseline
  `0.651759978716761` and `0.6310995739151736`; measured
  `0.6809911860203827` and `0.6549738844620387`.
- The baseline predates Upgrade 1 and Upgrade H entirely — it carries
  neither a `structural_annotations` key nor an `undeclared_feature_panel`
  key at all, consistent with `TEST_B_RESULTS.md`'s own "What remains
  unvalidated" section stating Upgrade 1 was unbuilt when that document was
  written. No screen comparison was possible against it.
- B-1 is the clean frame (`forbidden_after_prediction: []`), so its
  `fixable_leakage` of `0.0` matching is weak evidence on its own: with
  `safe_features == all_features` whenever nothing is declared forbidden, B
  and C are the identical feature set on the identical folds by
  construction — `fixable_leakage = B − C` is structurally pinned to `0.0`
  regardless of which estimator ran, or whether the categorical-encoding
  change did anything at all.

This attempt is reported here rather than discarded, because a guard result
that turned out not to be usable is itself part of the record — silently
dropping it and only reporting the second, valid attempt would misrepresent
how the check was actually carried out.

---

## 3. Valid guard — B-2 at 10k, same estimator

Baseline: `scratch/testB2_10k_histgb.json`, `estimator_identity: "histgb"` —
a same-estimator baseline, valid for comparison. Data: `scratch/testB2_strat.csv`
(10,008 rows, stratified), contract: `testB2_sensitivity.yml`.

Side by side, all exact matches:

| field | baseline | measured |
|---|---|---|
| `fixable_leakage` | `0.40084697565613614` | `0.40084697565613614` |
| `verdict` | `FAIL` | `FAIL` |
| `naive_auc` | `0.9663757037727894` | `0.9663757037727894` |
| `deployable_auc` | `0.58944909861267` | `0.58944909861267` |
| `temporal_all_auc` | `0.9628105119380574` | `0.9628105119380574` |
| `nsl` | `5.451647315703834` | `5.451647315703834` |
| `data_sha256` | `d9fda80f...480f1e72` | `d9fda80f...480f1e72` (identical, confirming same input data) |

The numeric-only path is confirmed unchanged.

Full suite result: **895 passed**, up from 887, with 8 new tests (4 covering
categorical-mapping sentinel-collision/unseen-value/determinism behavior in
`tests/test_contract.py`, 4 covering the Part B gate behavior — Upgrade H
running and Upgrade 1 not running on a failed contract, non-zero exit, no
PASS-shaped verdict — in `tests/test_cli.py`).

---

## 4. Expected difference — contract hash

`contract_sha256` differs: baseline `4625cce4e6a7ebb629f7caf5a3920b76bb6c56f4e2a307bb8efa083fd2505d02`,
measured `3257331847a3edec56ff1405f7f1c6a073ce1afa0dac2873d65982b46f3300f3`.
This is because `PredictionContract` gained the additive `categorical_features`
field in step 1, which changes what `hash_contract` hashes via
`contract.model_dump_json()`.

**Consequence, recorded plainly:** every contract hash in the project changed
with this release, so `zekan diff` against any pre-change artifact will
report a contract mismatch. This is correct behaviour — the contract schema
genuinely did change — but it is a real thing users comparing a pre-change
JSON result against a post-change one will encounter, not a silent
no-op upgrade.

---

## 5. Still unverified

- The pre-registration's section 4 Theil's U invariance claim (sorted-unique
  ordinal encoding is a bijection, so Theil's U is invariant under it) is
  supported by a unit test covering sentinel collision
  (`test_categorical_mapping_nan_sentinel_collision_gets_distinct_codes`),
  but has **not** been verified end-to-end by comparing a raw run against an
  encoded run of the same real column through Upgrade H itself. Recorded as
  outstanding.
- The pre-registration's section 5 Upgrade 1 limitation — that univariate
  AUC on ordinally-encoded nominal features is not claimed invariant, since
  the encoding imposes an arbitrary order a tree-based estimator may not
  fully recover — remains a stated limitation, unmeasured. No calibration
  run has been performed to characterize how much (if any) Upgrade 1 signal
  is lost on a genuinely high-cardinality nominal column under this
  encoding.
