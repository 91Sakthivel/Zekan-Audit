# Fold-Local Inert Features — Pre-Registration

**Status:** PRE-REGISTERED, NOT YET IMPLEMENTED.
**Date:** 2026-08-17.

This is the design document following
`FOLD_LEVEL_ALLNAN_PREREGISTRATION.md` (`0baafc1`), which recorded the defect
and the measurements that pinned its cause, and left the fix open. This
document does not supersede that one — it builds the design on top of it and
references it throughout. No Python file is modified by this document, and
nothing here has been run.

---

## 1. Problem, in one paragraph

Summarised from `0baafc1`: a declared feature that is entirely NaN within a
fold's train slice crashes `histgb`'s binner
(`sliding_window_view` on a zero-length `distinct_values` array). Measured
there: only zero-support slices fail — a column with 3 non-NaN values of
68,786 rows fits fine. Constant columns (one real value, with or without
NaN) do not crash sklearn 1.9.0 at all; that was a falsified hypothesis, not
the cause.

## 2. Estimand — the reframe this design rests on

Record the decision: **fold-local inerting**, not fold skipping, not global
column exclusion, not window truncation.

State the estimand plainly. Zekan does not ask "what would leakage look like
if today's schema had existed throughout history." It asks "how much leakage
would this declared system have exhibited through time, given the
information actually available at each point." For a temporal auditor this
is the more defensible question — it audits the system as it actually
existed, fold by fold, rather than either pretending absent fields were
always present (fold skipping loses periods; global exclusion loses the
feature everywhere) or truncating the observation window to only the part
where every declared feature happens to be populated (window truncation
throws away real, usable early history for every OTHER feature just because
one feature arrived late).

## 3. The corrected pooling rule

Record that an earlier, blunter rule — never pool B-C across folds with
differing feature sets — was too strong. Expanding-window CV already fits a
different model per fold as training data accumulates; identical effective
information across folds is unachievable on temporal data by construction.

The correct invariant is **identical feature-availability policy** across
folds: for every fold, B may use every declared feature with usable training
information, and C follows the same rule after removing forbidden features.
Pooling remains valid under that policy — what must be identical across
folds is not the feature *set*, but the *rule* by which each fold's feature
set is derived.

## 4. The boundary — non-negotiable

**Zero usable training observations -> may be treated as fold-inert.**
**One or more -> a real feature; the estimator must handle it or Zekan
fails.**

Record why: with zero observations, neither values nor missingness vary
within that fold's train slice, so there is nothing learnable — the column
carries no information for that fit, by construction, not by estimation.
With one or more, missingness itself varies and `histgb` can split on it;
that is real, usable signal, however faint.

Record explicitly: "sparse", "few observations", "one distinct value", and
"99.9% missing" are **NOT** equivalent to zero and must never be inerted.

Record that an earlier claim in this project's own discussion — that three
columns with ~171 non-NaN values "were never going to carry signal" — was an
unmeasured assumption and is **retracted** here. Whether a column with a
handful of observations carries signal is an empirical question the audit
itself is built to answer, not a premise to bake into the fix.

## 5. ZK-EST-04 — gating invariant

State it: **adding or removing a training-all-missing feature must not alter
fitted probabilities for otherwise identical Zekan estimator inputs.**

Requirement: predicted probabilities identical, or within a tight
deterministic tolerance — not merely equal AUC. AUC is rank-invariant and
can hide a real change in the fitted function; probability identity is the
actual claim being made when a column is inerted.

Record why this must be proven, not assumed: `max_features` defaults to
`1.0` and `random_state` also governs binning subsampling inside
`HistGradientBoostingClassifier`, so removing a physically all-NaN column
from the matrix passed to `.fit()` is not guaranteed a priori to be inert on
the fitted model — it changes the feature index space the estimator's
internal randomness operates over, even though the column itself carries no
information.

Record the proof regime: proven once per `(estimator_identity, sklearn
version)`, on a matrix representative of real use — not a two-column
synthetic case like the probes in `0baafc1` — the first time inerting fires
in a real run; result cached and recorded in provenance; re-proven when the
stack changes (estimator identity or sklearn version).

**FALSIFICATION CONDITION**: if ZK-EST-04 cannot be demonstrated on the
pinned stack, this design **FAILS** and Zekan must fail pre-flight rather
than inert anything. Record that this is a real possible outcome, not a
formality — the design is not entitled to assume the invariant holds.

## 6. Efficiency requirements — part of the design, not an optimisation pass

- Inert sets computed **once** at fold construction: one `notna().any()`
  pass per column per fold train slice, stored as per-fold index lists. B
  and C slice a precomputed array; availability is never recomputed per fit.
- When **no** fold has any inert column — the common case, including Test B
  — the adapter is bypassed entirely and the existing code path runs
  untouched, with zero overhead. ZK-EST-04 need not hold when never
  exercised.
- The permutation null (`estimate_fixable_leakage_null`, default 100 draws)
  reuses the same precomputed per-fold inert sets. Record explicitly:
  recomputing availability per draw would multiply cost by the draw count
  (100x by default), for a quantity that cannot change across permutation
  draws — inertness is a property of the training slice's temporal
  structure, not of the permuted target.

## 7. Monotonicity invariant

Record: expanding-window training sets are nested (`0baafc1` fold table:
fold 1's train set is fold 0's train set plus fold 0's test set, and so on),
so once a column becomes active it can never become inert again within the
same audit. The per-fold inert set is therefore computable in a single
forward pass, and the coverage table is always a staircase — inert, then
active, never active-inert-active.

Record this as a testable invariant: if violated, the folds are not
expanding-window and something upstream is wrong. Zekan should assert it.

Record the legibility benefit: output reads "active from fold 2 onward"
rather than an arbitrary per-fold list — the staircase property is what
makes a single "first active fold" number a complete and correct summary.

## 8. Reporting — coverage is per column, not only per fold

Record that the audit must surface, in human output and JSON:

- per fold: active count / inert count;
- per inert column: the first period at which it has usable training
  support, and the fold from which it becomes active.

Record the unresolved sub-question: whether a column inert in early folds
and active later constitutes full or partial severity coverage for that
column. The position taken here is **partial**, and the coverage statement
must say so per column. Record that this is a judgement, not a measurement —
the column WAS quantified once it activated, but the audit's severity
estimate for it is built from fewer folds than a column active throughout,
and the reported coverage must not imply otherwise.

## 9. Verdict rule

Record: a CLEAN verdict covers only the intersection of temporal population
and feature universe for which severity was actually quantified. Structural
inspection (Upgrade H, Upgrade 1) does **not** extend severity coverage to
unmodelled features — a structural probe can say a feature looks suspicious
but cannot say how much predictive optimism it contributes. Severity
coverage and structural coverage are reported separately and must not be
conflated in the verdict text.

## 10. Optional second scope — opt-in, not default

Record: a later-window audit (`--from-period`) is a **separate, opt-in
mode** answering a different question — modern-schema leakage severity — not
a completion of the historical audit. Not run by default.

Record the reasoning: running it automatically doubles fits, nulls, and
verdicts on every affected dataset, against Zekan's efficiency and
simplicity constraints, and "cannot verify, don't claim clearance" is a rule
about epistemic claims rather than a requirement to execute every
conceivable analysis.

Record that this is only defensible **because** fold-local inerting keeps
all 64 declared features in the audit; if the design had instead excluded
columns globally, an opt-in second scope would **not** be sufficient and the
audit would have to return UNVERIFIED with a non-zero exit — a globally
excluded feature's severity would never be quantified by either scope, so
silence about it would be a false claim of clearance, not a deferred
convenience.

## 11. Regression guard — pre-committed

- The full suite must pass.
- Test B must reproduce byte-identical `fixable_leakage`, verdict,
  `naive_auc`, `deployable_auc`, and `nsl` against
  `scratch/testB2_10k_histgb.json`. No Test B column is ever training-empty,
  so the adapter must never fire there. Any movement means the change is not
  inert on the calibrated baseline and must be reverted or explained, never
  accepted.
- The monotonicity invariant (section 7) must have a test.
- ZK-EST-04 must have a test.

## 12. Integrity clause

Findings contradicting anything here are recorded as dated addenda. No
threshold or gate is tuned to produce a desired outcome.
