# Bootstrap CI Pre-Registration (CI-A)

## 1. Date and status

Dated 2026-08-16.

**Status: PRE-REGISTERED, NOT YET IMPLEMENTED.**

This document records a defect and a design decision *before* any fix is
written, per this project's record-before-fix discipline (see the addendum
convention used elsewhere in `zekan/benchmark/results/`). No Python file is
touched by this document. No audit was run to produce it.

## 2. Defect found (record before fix)

All claims below were re-verified directly against the current code and
tracked docs immediately before writing this file (line numbers confirmed
via grep, not assumed):

- The verdict/policy ladder gates on the **pooled** out-of-fold `fl`.
  `zekan/severity/engine.py:388`: `fixable_leakage = auc_b_pool - auc_c_pool`.
  `zekan/severity/verdict.py:756`: `fl = result.fixable_leakage` — this is
  the value passed into `_policy_verdict()` and compared against
  `warn_floor`/`fail_floor` to produce PASS/NOTE/WARN/FAIL/UNCONFIRMED_HIGH_DAMAGE.

- The shipped `fold_ci` block (`zekan/severity/verdict.py`, class `FoldCI`
  at line 206, built by `_compute_fold_ci` at line 351) computes a 95%
  t-interval **centered on the fold mean** of interior per-fold leakages —
  not on pooled. The class docstring says this explicitly: "CI is centered
  on the fold-level mean, not pooled OOF."

- Therefore the gated `fl` (pooled) is **not guaranteed by construction** to
  lie within its own published `[ci_low, ci_high]` — pooled and the CI's
  center (fold mean) are two different statistics over overlapping but
  non-identical inputs.

- The divergence is recorded as `pooled_vs_fold_gap = pooled - fl_mean` and
  documented in the field's docstring as "reflects estimator differences
  (normal)," but nothing enforces or surfaces this gap outside the raw JSON
  field itself — there is no check, warning, or gate on its magnitude.

- **Measured instance** (`scratch/testB2_10k_histgb.json`, B-2 at 10,008
  rows, re-read and confirmed directly from the file):
  - pooled: `0.40084697565613614`
  - fold_mean: `0.3810372079645289`
  - gap: `0.01980976769160725` (~0.0198)
  - ci_low: `0.3266536733854439` (~0.3267)
  - ci_high: `0.4354207425436139` (~0.4354)

  Pooled (0.4008) falls inside `[0.3267, 0.4354]` here, but incidentally —
  not by construction, since the interval was never built to bracket the
  pooled statistic in the first place.

- **Degrees of freedom**: `k` = count of interior (non-terminal) folds.
  Confirmed from the same JSON: `folds_evaluated = 5`, and back-solving
  `k` from `fl_fold_se = fl_fold_std / sqrt(k)`
  (`0.03417718909431018 / 0.01708859454715509 ≈ 2.0` ⟹ `k ≈ 4`) gives
  `k = 4` — one terminal fold excluded from the 5 evaluated, consistent
  with `df = k - 1 = 3` and `t_crit = scipy.stats.t.ppf(0.975, df=3) ≈ 3.18`.
  `zekan/severity/splitters.py`'s default `n_splits=5` makes this the
  typical case, not an edge case.

  `temporal_expanding_folds` (`zekan/severity/splitters.py`) produces
  **nested, expanding-window** training sets — each fold's training set is
  a superset of the previous fold's. The t-interval's i.i.d.-sample
  assumption is not met by construction under this nesting; the fold
  leakage values it's computed over are correlated, not independent draws.

  The existing coverage test (`tests/test_verdict.py`,
  `TestCIcoverage.test_t_ci_covers_zero_on_null_folds`, lines 302-321)
  draws its 4 fold leakages via `rng.normal(0.0, 0.020, size=4).tolist()` —
  **simulated, independent** draws. It does not exercise, and cannot
  detect, the nesting correlation that the real expanding-window folds
  actually carry.

- **Surface scope**: `ci_low`/`ci_high`/`ci_ratio`/`confidence_tier` are
  serialized into JSON (`zekan/reports/json_export.py:98`,
  `"fold_ci": raw["fold_ci"]`, the entire block verbatim) but are **never
  rendered in human-readable text output** — confirmed by grep across
  `zekan/reports/text_view.py` and `zekan/reports/messages.py`:
  `text_view.py` surfaces only `folds_skipped`, `folds_evaluated`,
  `skip_reasons`, `stability_seeds_checked`, and `seed_instability_note`
  from the `fold_ci` block; none of the CI-numeric fields appear in any
  rendered string. `messages.py` has zero references to `fold_ci` at all.

## 3. Decision: canonical estimator

**Pooled is canonical `fl`.**

Rationale:
- It is what the gate already acts on (`verdict.py:756`) — changing the
  canonical estimator to fold-mean would be a silent behavior change to
  every existing verdict, not a documentation fix.
- It uses every out-of-fold prediction exactly once (pooled OOF AUC over
  all interior folds' predictions, computed as one AUC rather than an
  average of per-fold AUCs).
- It does not inherit the small-`k` fragility of a 4-fold nested-window
  t-interval (`df ≤ 3` in the typical case).

Fold mean is retained as a labeled **diagnostic**, not an estimate of the
gated quantity.

## 4. Design to be implemented (CI-A)

- **Paired row bootstrap** over the pooled out-of-fold arrays (`y_pool`,
  `pooled_proba_b`, `pooled_proba_c` — see `zekan/severity/engine.py:372-388`),
  resampled **together** so B and C see identical rows on each draw.
- **Models held fixed. No refitting.** This is an evaluation-set precision
  interval only — it quantifies sampling uncertainty in the pooled AUC
  difference given the existing out-of-fold predictions, not model-fitting
  variance.
- Resampling must be:
  - **fold-stratified** — predictions come from a different model per
    temporal fold, so an unstratified bootstrap could draw a sample that
    misrepresents each fold's contribution to the pool;
  - **entity-clustered where an entity column exists** — rows within an
    entity are correlated (the same entity contributes multiple rows/
    periods), so resampling individual rows independently understates
    true sampling variance.
  - A naive i.i.d. row bootstrap would be **anti-conservative** on both
    counts.
- **Percentile interval** (not t-interval, not BCa at this stage).
- Draw count to be fixed at implementation time and **justified by measured
  evidence**, not chosen for convenience.
- **Explicitly out of scope**: any interval requiring refits across seeds.
  That is CI-B and belongs to the `--stability` work item
  (`zekan/cli.py`'s `--stability`/`--seeds` flags on `zekan audit`), which
  already reruns the permutation null across seeds — a distinct mechanism
  from this evaluation-set bootstrap.

## 5. Falsification condition — measured first, before any surfacing

Tracked evidence, re-verified directly against
`zekan/benchmark/results/TEST_B_RESULTS.md` (lines 81-118) before writing
this section:

> | n | fl | naive | deployable | NSL (within) | estimator |
> |---|---|---|---|---|---|
> | 10,008 (stratified) | 0.4008 | 0.9664 | 0.5894 | 5.45 | histgb |
> | 40,008 (stratified) | 0.3437 | 0.9638 | 0.6423 | 14.33 | histgb |
> | 101,766 (full) | 0.3095 | 0.9639 | 0.6550 | 35.56 | histgb |

B-2 `fl` by sample size: 10k = 0.4008, 40k = 0.3437, 100k = 0.3095.
The 10k → 100k movement is `0.4008 - 0.3095 = 0.0913` (~0.0913).

**Condition:** if the CI-A bootstrap width at 100k is materially narrower
than 0.0913, then CI-A measures evaluation-set sampling precision **only**
and does **not** capture n-sensitivity.

**Consequence if it fires:** CI-A ships with an explicit scope disclaimer
and **must not** be surfaced as a stability, transferability, or
n-transfer claim.

**Stated plainly, in advance:** this condition is **expected to fire**,
because `fl`'s n-sensitivity (documented in `TEST_B_RESULTS.md`'s own
"Scale comparison — the fl-vs-n drift" section, lines 81-118) is a change
in the *estimand itself* — `deployable_auc` (C) genuinely improves as the
safe-feature model gets more data to learn from, shrinking the B−C gap —
not variance around a fixed estimand. A precision interval, by
construction, cannot answer a question about the estimand changing.
Recording this expectation in advance, before CI-A is implemented or
measured, exists specifically to prevent a precision statistic later being
misread or mis-marketed as a magnitude/stability statistic once it ships.

## 6. Open question deferred to implementation

Fate of the existing `fold_ci` JSON block: keep as a renamed diagnostic
(e.g. surfaced alongside CI-A rather than replaced by it) vs. leave as-is
unchanged. This is a **public schema change either way** (`fold_ci` is
already a documented top-level JSON key per `zekan/reports/json_export.py`)
and is **not decided in this document**.

## 7. Falsification integrity clause

If measurement contradicts any expectation recorded here, the finding is
recorded as an addendum. Thresholds and gates are never tuned to produce a
desired outcome.
