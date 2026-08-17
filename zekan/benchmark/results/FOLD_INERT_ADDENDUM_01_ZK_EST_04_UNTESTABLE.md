# Addendum 01 — ZK-EST-04 As Formulated Is Untestable

Addendum 01 to `FOLD_INERT_FEATURES_PREREGISTRATION.md` (`0fc7487`).

**Date:** 2026-08-17.
**Status:** ZK-EST-04 AS FORMULATED IS UNTESTABLE. INVARIANT REFORMULATED.
NOT YET IMPLEMENTED.

---

## 1. What section 5 required

Quoted from the pre-registration: "adding or removing a training-all-missing
feature must not alter fitted probabilities for otherwise identical Zekan
estimator inputs," proven on a representative matrix, with the falsification
condition that "if ZK-EST-04 cannot be demonstrated on the pinned stack,
this design FAILS and Zekan must fail pre-flight rather than inert
anything."

## 2. What was measured

Real data: the Freddie Mac 2018Q1 frame table, 64 feature columns after the
same categorical encoding the audit applies, same temporal folds
(`SplitPolicy` defaults).

Record the correction made mid-measurement: `engine.py`'s `_feature_cols`
excludes only `entity_id`, `prediction_time`, `available_features_until`,
and `target` — NOT `forbidden_after_prediction` columns, since those remain
in model B and are removed only for model C. `target_creditevent` is
therefore correctly among the 64. A first draft that excluded it (giving 63)
was corrected before running.

Arms and results:

- **Arm A** — fold 2 train, 64 real columns plus one artificially appended
  all-NaN column: EXCEPTION on fit, `ValueError: window shape cannot be
  larger than input array shape`.
- **Arm C** — fold 0 train, all 64 real columns including the 3 genuinely
  all-NaN-in-train columns: EXCEPTION on fit, same error. Confirms the
  defect recorded in `0baafc1` reproduces.
- **Arm D** — fold 0 train, 61 columns with the 3 inert columns removed:
  FIT SUCCEEDED, AUC `0.902769608630054`.
- **Arm E** — fold 0 train, 60 columns, the same 3 inert removed plus one
  arbitrary ACTIVE column (`CURRENT ACTUAL UPB`) also removed: FIT
  SUCCEEDED, AUC `0.9027686490903688`.
- **Arm D vs Arm E** `predict_proba` on fold 0's test slice: max absolute
  difference `0.07501896132549113`, mean absolute difference
  `2.2295035071277258e-05`, bitwise identical FALSE.

## 3. Finding — the invariant is untestable as formulated

Record plainly: both "with the all-NaN column" arms crash. There is no
runnable fit that includes a training-all-missing column on this stack, so
there is nothing to compare a removal against. ZK-EST-04 as written compares
two states only one of which can exist.

Record that this is a defect in the invariant's **formulation**, not a
failure of the fold-inerting design and not a passing result. It is neither
demonstrated nor refuted — it is unaskable.

## 4. The control worked — the measurement was sensitive

Record that Arm E establishes the comparison could have detected a real
change: removing one genuinely active column moved predicted probabilities
by up to `0.075` and shifted AUC in the seventh decimal. The measurement was
sensitive enough to detect column-removal effects; it simply had no valid
"with" arm for the all-NaN case.

## 5. Reformulated invariant

Record the reformulation and its consequence.

The empirically meaningful question is: does a matrix from which a
zero-information column has been removed fit identically to an otherwise
identical matrix that never contained it? This is trivially true — it is
the same matrix — and therefore carries no empirical content.

Consequence: removal is the ONLY available path to a fit. The question is
not whether removal is inert relative to an alternative, because no
alternative exists. ZK-EST-04 is therefore **downgraded from a gating
invariant to a documentation obligation**: Zekan must report exactly which
columns were inerted in which folds, because the user cannot infer it and
there is no counterfactual run to compare against.

Record explicitly that this **weakens** a pre-registered gate, disclosed as
such.

## 6. What this strengthens instead

Because the invariant cannot gate the design, the reporting requirements in
section 8 of the pre-registration carry more weight, not less: per-fold
active and inert counts, and per inert column the first period with usable
training support and the fold from which it becomes active. Record that
these move from "good practice" to "the only mechanism by which a user
learns this happened."

## 7. Unresolved concern, recorded not dismissed

The external review's underlying worry — that `max_features` defaulting to
`1.0` and `random_state` governing binning subsampling could make column
removal perturb the fitted model — is NOT resolved by this measurement. It
cannot be resolved on this stack, because the comparison it requires is
unrunnable. Record it as a standing limitation of the design rather than an
answered question.

## 8. Effect on the regression guard

Section 11's guard is UNCHANGED except that "ZK-EST-04 must have a test" is
replaced by: a test asserting that the inert-column set is correctly
identified and reported, since the probability-identity claim is no longer
testable. The Test B byte-identical requirement and the monotonicity test
both stand.

## 9. Integrity clause

This addendum records a pre-registered condition firing in an unanticipated
way. The original section 5 text is not edited. No threshold or gate is
tuned to produce a desired outcome.
