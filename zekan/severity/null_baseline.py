"""Permutation null distribution for fixable_leakage (B−C pooled OOF).

NULL HYPOTHESIS: the declared forbidden features carry no genuine future information.
The null is estimated by breaking the leakage signal while holding everything else
(fold structure, class balance, safe-feature signal, temporal protocol) constant.

Three methods:

  within_entity (recommended)
    Shuffle each forbidden column's values within each entity's rows.
    Preserves:  entity marginal distributions of each forbidden value.
    Destroys:   the row-level temporal ordering (the future-information link).
    Invariant:  AUC_C is unchanged — C drops forbidden features, so permuting them
                has no effect.  AUC_C_pool is pre-computed once and reused across
                all permutations; only AUC_B is refit per draw.  ~2× faster than
                a naive two-pass null.
    Blind spot: a forbidden column that is CONSTANT WITHIN each entity (e.g. an
                entity-level aggregate) is a no-op under this permutation — it
                cannot detect leakage carried by between-entity structure.  See
                across_entity below.

  across_entity (spec 1 — closes the within-entity blind spot)
    Shuffle each forbidden column's values across ALL rows, ignoring entity
    boundaries entirely (no groupby).
    Preserves:  the column's overall marginal distribution.
    Destroys:   both the row-level temporal ordering AND the row-entity
                association — including entity-level aggregates that are
                constant within each entity and therefore invisible to
                within_entity.
    Invariant:  same AUC_C invariant as within_entity, for the same reason (C
                never reads forbidden columns) — reuses the identical
                precompute-once/y_pool-reuse structure.
    No-op guard: meaningless (mathematically identical to within_entity) when
                the dataset has fewer than 2 entities; in that case the null is
                NOT RUN (n_permutations=0) rather than silently reporting a
                clean result.

  target_within_period
    Shuffle target values within each time period.
    Preserves:  period-level base rates; fold structure; all feature values.
    Destroys:   all entity-level signal (safe features AND forbidden features).
    The null answers "what is B−C when nothing predicts?" rather than "what is
    B−C when the forbidden feature is noise?"  Less precise: variance is higher
    because both B and C regress to ~0.5 AUC.

The within_entity method isolates the forbidden feature's contribution, giving a
tighter null and a more interpretable p-value.  The across_entity method trades
some of that precision to additionally catch between-entity leakage channels.

USAGE
-----
>>> null = estimate_fixable_leakage_null(df, contract, config, clf_factory,
...     observed_fixable_leakage=0.087, n_permutations=100, seed=0)
>>> null.p_value    # p(null >= 0.087)
>>> null.null_95th  # 95th percentile of permutation distribution

PERMUTATION COUNT
-----------------
Default n_permutations=100.  Stability analysis across the benchmark DGP shows
the null 95th percentile from two independent batches of 100 agrees within ~0.003
(measured via the original calibration benchmark, since removed).  100 is the minimum defensible count; 200 adds robustness.
The null refits only AUC_B per permutation (AUC_C is invariant), so the marginal
cost is one temporal CV pass per permutation (~4 model fits on the benchmark DGP).
NOTE (F2a): the ~0.003 figure above was measured under the now-retired serial_v1
stream (see SEEDING below); it has not been re-measured under spawn_v2.

SEEDING (F2a) — within_entity and across_entity only
-----------------------------------------------------
Each permutation draws from an INDEPENDENT child stream:
    children = np.random.SeedSequence(seed).spawn(n_permutations)
Permutation i always uses children[i], regardless of n_jobs or worker scheduling
order — null_samples is therefore identical for n_jobs=1 and n_jobs>1 (see
_null_permutation_once).  Scheme name: "spawn_v2" (NullResult.scheme).  This
replaced a single shared `np.random.default_rng(seed)` mutated serially across
all permutations ("serial_v1") — that scheme could not be parallelized safely
because permutation i's draw depended on exactly how much entropy permutations
0..i-1 had already consumed from the same Generator.  target_within_period is
unchanged and still uses the single shared-stream serial_v1 scheme (deferred;
comparison-only method, not on the production call path).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm
from sklearn.metrics import roc_auc_score

from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.metrics import _feature_matrix, evaluate_folds
from zekan.severity.splitters import temporal_expanding_folds


# ── Tier 2: sequential/adaptive permutation stopping (Besag-Clifford style) ───
# Parameters are pre-registered and locked -- not caller-adjustable, matching
# the "spec is locked" discipline used for the NSL ladder thresholds in
# engine.py. Mirrors engine._NULL_ALPHA (0.01) as a local constant rather than
# importing it, to avoid a circular import (engine.py imports FROM this
# module) -- the same pattern already used for the feature-exclusion set in
# estimate_fixable_leakage_null below.
_SEQ_H: int = 10                    # Besag-Clifford exceedance-count stopping threshold
_SEQ_N_MIN: int = 30                # minimum draws before any early-stop check runs
_SEQ_N_MAX: int = 500               # hard ceiling; spawn(N_MAX) up front (see module docstring)
_SEQ_ALPHA: float = 0.01            # mirrors engine._NULL_ALPHA. Historically also gated when
                                     # the decision-stability check below was allowed to run;
                                     # Tier 2b FIX A removed that gate (it was the defect: it
                                     # forced ~100 draws before the check could even be attempted
                                     # when count_gte==0, regardless of NSL's distance from the
                                     # 1.0 boundary). Kept here as the documented alpha this
                                     # module's design is calibrated against (see the h/N_max
                                     # ratio note in estimate_fixable_leakage_null's docstring).
_SEQ_IQR_CONFIDENCE: float = 0.99   # conservative (wide) two-sided CI for the IQR-stability check


def _derive_alpha_floor_draws(alpha: float) -> int:
    """Smallest n such that the zero-exceedance Laplace-corrected p-value,
    1 / (n + 1), is STRICTLY less than alpha -- the honest information-
    theoretic minimum number of permutation draws before a DETECTED
    conclusion is possible at all, regardless of how the stopping rule is
    written (Tier 2b-final; see Addendum 5 and TIER2B_CALIBRATION.md).

    Derived by direct search over the actual formula, not assumed
    algebraically: for alpha=0.01 this is 100, not 99. At n=99,
    1/(99+1) = 0.01 exactly, which is NOT strictly less than 0.01 (engine.py's
    gate is `if p_value >= _NULL_ALPHA: not-detected`, i.e. detection needs
    strict p < alpha). At n=100, 1/(100+1) ~= 0.00990099, which is. Addendum 5
    stated ceil(1/alpha)-1 = 99 for this floor; that was off by one, corrected
    here rather than edited into the (append-only) addendum.
    """
    n = 1
    while 1.0 / (n + 1) >= alpha:
        n += 1
    return n


_ALPHA_FLOOR_DRAWS: int = _derive_alpha_floor_draws(_SEQ_ALPHA)
"""The honest minimum draws before a DETECTED (p < alpha) conclusion is
possible with zero exceedances. 100 for alpha=0.01 -- see
_derive_alpha_floor_draws. Used only to gate the DETECTED-direction early
stop (Tier 2b-final Part 2); the NOT-DETECTED direction has no floor."""


def _quantile_order_stat_indices(n: int, p: float, confidence: float) -> tuple[int, int]:
    """Distribution-free (normal-approximation) two-sided CI for the p-th
    population quantile, expressed as 0-indexed order-statistic bounds in a
    sorted sample of size n.

    Standard nonparametric method (e.g. Conover, "Practical Nonparametric
    Statistics"): the index of the p-th sample quantile is asymptotically
    Normal(n*p, n*p*(1-p)) under the binomial count of observations below the
    true quantile. Deterministic given n -- no resampling, no extra RNG state,
    which matters here because the sequential stopping rule must not add a
    new randomness source on top of the permutation draws themselves.
    """
    z = float(_scipy_norm.ppf((1 + confidence) / 2))
    se = float(np.sqrt(n * p * (1 - p)))
    lo = int(np.floor(n * p - z * se))
    hi = int(np.ceil(n * p + z * se))
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    return lo, hi


def _conservative_iqr_bound(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Conservative (widest plausible) range for the population IQR given
    `samples`, using order-statistic CIs for Q25 and Q75 independently.

    iqr_low  = (low end of Q75's CI)  - (high end of Q25's CI)  -- smallest plausible IQR
    iqr_high = (high end of Q75's CI) - (low end of Q25's CI)   -- largest plausible IQR

    This is intentionally conservative (wider than either quantile's own CI
    alone): it does not assume Q25 and Q75's estimation errors are correlated
    or cancel, so [iqr_low, iqr_high] is a safe (if loose) bound on where the
    true IQR could plausibly sit given only `samples`.
    """
    n = len(samples)
    sorted_samples = np.sort(samples)
    q25_lo, q25_hi = _quantile_order_stat_indices(n, 0.25, confidence)
    q75_lo, q75_hi = _quantile_order_stat_indices(n, 0.75, confidence)
    iqr_low = float(sorted_samples[q75_lo] - sorted_samples[q25_hi])
    iqr_high = float(sorted_samples[q75_hi] - sorted_samples[q25_lo])
    return iqr_low, iqr_high


def _conservative_quantile_bound(
    samples: np.ndarray, p: float, confidence: float
) -> tuple[float, float]:
    """Conservative order-statistic CI [lo, hi] for the p-th population
    quantile of `samples` -- same normal-approximation method as
    _conservative_iqr_bound's Q25/Q75 bounds (see _quantile_order_stat_indices).

    Factored out (Tier 2b FIX A) so null_99th's own sampling uncertainty can
    be bounded the same way null_iqr's is, instead of being held at its point
    estimate -- see _nsl_decision_stable.
    """
    n = len(samples)
    sorted_samples = np.sort(samples)
    lo_idx, hi_idx = _quantile_order_stat_indices(n, p, confidence)
    return float(sorted_samples[lo_idx]), float(sorted_samples[hi_idx])


def _nsl_interval_bound(
    observed_fixable_leakage: float,
    null_99th_lo: float,
    null_99th_hi: float,
    iqr_low: float,
    iqr_high: float,
    nsl_eps: float = 1e-4,
) -> tuple[float, float]:
    """Conservative [nsl_min, nsl_max] bound on NSL given BOTH the
    [iqr_low, iqr_high] bound on the null IQR AND the [null_99th_lo,
    null_99th_hi] bound on null_99th itself (Tier 2b FIX A -- previously
    null_99th was held at its point estimate and only null_iqr was bounded).

    NSL(x, y) = (observed - x) / y is monotonically decreasing in x for any
    y > 0 (derivative -1/y < 0 always), so the global max over the
    [null_99th_lo, null_99th_hi] x [iqr_low, iqr_high] box is attained at
    x=null_99th_lo, and the global min at x=null_99th_hi -- regardless of the
    numerator's sign. For each of those two x-corners, whether increasing or
    decreasing y increases the quotient depends on the numerator's sign at
    that corner (dividing a positive numerator by a smaller y increases it;
    dividing a negative numerator by a smaller y decreases it further), so
    the y-corner is chosen per-corner rather than assumed fixed.

    Factored out of _nsl_decision_stable (Tier 2b-final) so callers can also
    tell WHICH side of 1.0 a stable interval landed on -- needed to decide
    whether the DETECTED-direction alpha floor applies (see the sequential_v1
    loop in estimate_fixable_leakage_null).
    """
    num_hi = observed_fixable_leakage - null_99th_lo  # candidate for the global max
    denom_for_max = iqr_low if num_hi >= 0 else iqr_high
    nsl_max = num_hi / max(denom_for_max, nsl_eps)

    num_lo = observed_fixable_leakage - null_99th_hi  # candidate for the global min
    denom_for_min = iqr_high if num_lo >= 0 else iqr_low
    nsl_min = num_lo / max(denom_for_min, nsl_eps)

    return nsl_min, nsl_max


def _nsl_decision_stable(
    observed_fixable_leakage: float,
    null_99th_lo: float,
    null_99th_hi: float,
    iqr_low: float,
    iqr_high: float,
    nsl_eps: float = 1e-4,
) -> bool:
    """True iff the NSL>=1.0 vs <1.0 ladder-boundary decision is unchanged
    across the conservative NSL interval from _nsl_interval_bound.
    """
    nsl_min, nsl_max = _nsl_interval_bound(
        observed_fixable_leakage, null_99th_lo, null_99th_hi, iqr_low, iqr_high, nsl_eps
    )
    return (nsl_min >= 1.0) == (nsl_max >= 1.0)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class NullResult:
    """Permutation null distribution for fixable_leakage."""

    observed: float              # fixable_leakage on the unpermuted data
    null_samples: np.ndarray     # shape (n_permutations,); one draw per permutation
    null_median: float
    null_95th: float
    null_99th: float             # quantile(null, 0.99) — boundary consistent with alpha=0.01
    null_iqr: float              # q75 - q25; stable spread for the NSL denominator
    p_value: float               # (count(null >= observed) + 1) / (N + 1); Laplace-corrected
    method: str
    n_permutations: int
    elapsed_seconds: float = 0.0
    scheme: str = "spawn_v2"     # seeding scheme: "spawn_v2" (within/across_entity) or
                                  # "serial_v1" (target_within_period, unchanged)
    stopping: str = "fixed_v1"   # "fixed_v1" (draw exactly n_permutations, unchanged default)
                                  # or "sequential_v1" (Tier 2: Besag-Clifford + decision-stability)
    stopped_early: bool = False  # True iff sequential_v1 stopped before _SEQ_N_MAX
    p_is_upper_bound: bool = False  # Tier 2b-final: True iff p_value == 1/(n+1) because zero
                                      # null draws reached observed -- the Laplace formula's floor,
                                      # not a precise estimate. False when count_gte > 0 (a real
                                      # count backs the p-value) or when the Besag-Clifford exact
                                      # h/n formula applies (_p_value_override; count_gte >= h > 0).


# ── Permutation strategies ────────────────────────────────────────────────────

def _permute_column_within_entity(
    values: np.ndarray,
    entity_codes: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle `values` within each entity group; one rng.permutation() call
    per group, iterated in group-code order (0..n_groups-1).

    `entity_codes`/`n_groups` come from pd.factorize(entity_values, sort=False)
    -- group-code order under sort=False is each entity's FIRST-OCCURRENCE
    order in the original row sequence, which is exactly the iteration order
    pandas' groupby(sort=False) uses. This was verified empirically (matching
    call sequence, matching output, and matching downstream RNG state after
    the call) against the groupby(sort=False).transform(lambda x:
    rng.permutation(x.values)) pattern this replaces, before this change was
    made -- see scratch/verify_rng_order.py.
    """
    result = np.empty_like(values)
    for group_id in range(n_groups):
        idx = np.where(entity_codes == group_id)[0]
        result[idx] = rng.permutation(values[idx])
    return result


def _permute_within_entity(
    df: pd.DataFrame,
    forbidden_cols: list[str],
    entity_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle each forbidden column's values within entity rows.

    Row indices and all other columns are unchanged.
    Returns a copy; never mutates the input.

    Implementation note: this used to call
    df.groupby(entity_col, sort=False)[col].transform(lambda x: rng.permutation(x.values))
    directly. That is replaced here by _permute_column_within_entity, which
    does the same per-group rng.permutation() calls in the same order (see its
    docstring) via a vectorized pd.factorize grouping instead of pandas'
    groupby/transform/lambda dispatch machinery -- profiling showed the latter
    costs roughly 260x more per call than an equivalent-sized vectorized
    global shuffle, almost entirely in per-group Python-level dispatch
    overhead unrelated to the actual permutation work.
    """
    df_perm = df.copy()
    entity_codes, entity_uniques = pd.factorize(df[entity_col].to_numpy(), sort=False)
    n_groups = len(entity_uniques)
    for col in forbidden_cols:
        values = df_perm[col].to_numpy()
        df_perm[col] = _permute_column_within_entity(values, entity_codes, n_groups, rng)
    return df_perm


def _permute_across_entity(
    df: pd.DataFrame,
    forbidden_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle each forbidden column's values across ALL rows, ignoring entity.

    Unlike _permute_within_entity (which shuffles inside each entity's rows
    and so is a no-op on a column that is constant within entity), this
    breaks the row<->entity association entirely — the mechanism needed to
    detect leakage carried by between-entity structure (e.g. an entity-level
    aggregate that is constant within each entity).

    Row indices and all other columns are unchanged.
    Returns a copy; never mutates the input.
    """
    df_perm = df.copy()
    for col in forbidden_cols:
        df_perm[col] = rng.permutation(df_perm[col].values)
    return df_perm


def _permute_target_within_period(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle target values within each time period.

    Preserves period-level base rates; destroys all entity-level target signal.
    Returns a copy; never mutates the input.
    """
    df_perm = df.copy()
    df_perm[target_col] = (
        df_perm.groupby(time_col, sort=False)[target_col]
        .transform(lambda x: rng.permutation(x.values))
    )
    return df_perm


# ── Per-permutation work unit (F2a: parallel-safe, spawn_v2 seeding) ───────────

def _null_permutation_once(
    child_seed: np.random.SeedSequence,
    df: pd.DataFrame,
    forbidden_cols: list[str],
    entity_col: str,
    method: str,
    all_feature_cols: list[str],
    target_col: str,
    temp_folds: list,
    model_factory: Callable[[], Any],
    interior_fold_idxs: set[int],
    auc_c_pool: float,
    y_pool: np.ndarray,
    X_base: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    forbidden_col_positions: Optional[list[int]] = None,
    entity_codes: Optional[np.ndarray] = None,
    entity_n_groups: Optional[int] = None,
) -> Optional[float]:
    """Compute one permutation draw: auc_b_perm - auc_c_pool, or None when no
    interior fold produced OOF predictions (mirrors the original `continue`
    skip semantics exactly — a None is filtered out by the caller before the
    null_samples array is built, never counted as a zero draw).

    `child_seed` must be an independent np.random.SeedSequence (one entry from
    np.random.SeedSequence(seed).spawn(n_permutations)) — never a live
    np.random.Generator shared across permutations.  This is what makes the
    result independent of n_jobs / worker scheduling: permutation i's draw
    depends only on child_seed, never on any other permutation's draws.

    Module-level so loky can pickle it by (module, qualname) — mirrors
    zekan.severity.ablation._ablate_one.  method must be "within_entity" or
    "across_entity"; auc_c_pool/y_pool are read-only constants shared (by
    value) across every call, computed once by the caller before dispatch.

    X_base / y_all / forbidden_col_positions / entity_codes / entity_n_groups
        Optional fast path (F2a-perf): when X_base is given, the feature
        matrix for ALL rows is already built (once, by the caller, shared
        read-only across every permutation) and this call only copies it and
        overwrites the forbidden column(s)' positions with freshly-permuted
        values, instead of copying the whole dataframe and re-deriving the
        matrix from scratch via evaluate_folds. The rng call sequence is
        identical to the df_perm path below either way (see
        _permute_column_within_entity's docstring for the within_entity case;
        across_entity is a single rng.permutation() per forbidden column in
        both paths). When X_base is None (the default), falls back to the
        original df_perm-based path unchanged -- this keeps every existing
        direct call to this function (bypassing the real caller's precompute)
        correct without passing the new arguments.
    """
    rng = np.random.default_rng(child_seed)

    if X_base is not None:
        if method not in ("within_entity", "across_entity"):
            raise ValueError(f"_null_permutation_once: unsupported method {method!r}")
        X_perm = X_base.copy()
        for pos in forbidden_col_positions:
            if method == "within_entity":
                X_perm[:, pos] = _permute_column_within_entity(
                    X_base[:, pos], entity_codes, entity_n_groups, rng
                )
            else:
                X_perm[:, pos] = rng.permutation(X_base[:, pos])
        eval_b_perm = evaluate_folds(
            df, all_feature_cols, target_col, temp_folds, model_factory,
            return_predictions=True, X_all=X_perm, y_all=y_all,
        )
    else:
        if method == "within_entity":
            df_perm = _permute_within_entity(df, forbidden_cols, entity_col, rng)
        elif method == "across_entity":
            df_perm = _permute_across_entity(df, forbidden_cols, rng)
        else:
            raise ValueError(f"_null_permutation_once: unsupported method {method!r}")

        eval_b_perm = evaluate_folds(
            df_perm, all_feature_cols, target_col, temp_folds, model_factory,
            return_predictions=True,
        )
    b_fe_by_idx = {fe.meta.fold_idx: fe for fe in eval_b_perm.fold_evals}
    proba_b_parts = [
        b_fe_by_idx[idx].proba
        for idx in sorted(interior_fold_idxs)
        if idx in b_fe_by_idx and b_fe_by_idx[idx].proba is not None
    ]
    if not proba_b_parts:
        return None
    auc_b_perm = float(roc_auc_score(y_pool, np.concatenate(proba_b_parts)))
    return auc_b_perm - auc_c_pool


# ── Interior fold context ─────────────────────────────────────────────────────

def _build_interior_fold_set(
    df: pd.DataFrame,
    contract: PredictionContract,
    config: ZekanConfig,
    temp_folds: list,
) -> set[int]:
    """Return the fold indices that are interior (non-terminal).

    Mirrors the boundary-aware logic in engine.py exactly:
    a fold is terminal when its test_time_max rank > n_periods - 1 - leak_lookahead.
    """
    sorted_periods = sorted(
        df[contract.prediction_time].unique(),
        key=lambda x: pd.to_datetime(x),
    )
    period_rank = {
        pd.Timestamp(p).strftime("%Y-%m-%d"): i
        for i, p in enumerate(sorted_periods)
    }
    n_periods = len(period_rank)
    cutoff = n_periods - 1 - config.split_policy.leak_lookahead

    interior: set[int] = set()
    for fold in temp_folds:
        if fold.meta.skipped:
            continue
        ttm = fold.meta.test_time_max
        ttm_rank = period_rank.get(ttm) if ttm is not None else None
        is_terminal = ttm_rank is not None and ttm_rank > cutoff
        if not is_terminal:
            interior.add(fold.meta.fold_idx)
    return interior


def _pool_oof_predictions(
    fold_evals: list,
    interior_fold_idxs: set[int],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Collect OOF predictions from interior folds; return (y_true, proba) arrays."""
    fe_by_idx = {fe.meta.fold_idx: fe for fe in fold_evals}
    ys, ps = [], []
    for idx in sorted(interior_fold_idxs):
        fe = fe_by_idx.get(idx)
        if fe is not None and fe.y_true is not None and fe.proba is not None:
            ys.append(fe.y_true)
            ps.append(fe.proba)
    if not ys:
        return None, None
    return np.concatenate(ys), np.concatenate(ps)


def _pool_oof_auc(fold_evals: list, interior_fold_idxs: set[int]) -> Optional[float]:
    y, p = _pool_oof_predictions(fold_evals, interior_fold_idxs)
    if y is None:
        return None
    return float(roc_auc_score(y, p))


# ── Main estimation function ──────────────────────────────────────────────────

def estimate_fixable_leakage_null(
    df: pd.DataFrame,
    contract: PredictionContract,
    config: ZekanConfig,
    model_factory: Callable[[], Any],
    observed_fixable_leakage: float,
    n_permutations: int = 100,
    seed: int = 0,
    method: str = "within_entity",
    verbose: bool = False,
    n_jobs: int = 1,
    stopping: str = "fixed_v1",
    categorical_map: Optional[dict[str, dict]] = None,
) -> NullResult:
    """Estimate the permutation null distribution for fixable_leakage.

    Parameters
    ----------
    categorical_map
        Optional {column: {"codes": ..., "nan_sentinel": ...}} mapping (see
        contract_checks.build_categorical_mapping), the SAME object engine.py
        built once for the whole audit and used for X_all_full -- passed
        through here (not re-derived) so every feature-matrix rebuild this
        function does internally (X_base, and every evaluate_folds call that
        doesn't reuse a pre-built matrix) uses the identical encoding.
        Default None preserves every existing caller's behavior exactly.
    observed_fixable_leakage
        The fixable_leakage value from the unmodified engine run.
        Used only to compute the p-value; does not affect null sampling.
    n_permutations
        Number of permutation draws when stopping="fixed_v1" (the default).
        100 gives stable 95th percentile on the benchmark DGP (two batches
        agree within ~0.003). Increase to 200 for higher resolution. IGNORED
        when stopping="sequential_v1" -- that mode uses the pre-registered,
        locked ceiling _SEQ_N_MAX (500) instead; see `stopping` below.
    seed
        RNG seed for reproducibility.  Different seeds give different null samples
        but convergent 95th percentile at n=100.  For within_entity/across_entity
        (spawn_v2), the seed feeds np.random.SeedSequence(seed).spawn(n_permutations)
        — permutation i always uses child i, so results are independent of n_jobs.
    method
        "within_entity": shuffle forbidden columns within entity rows (recommended).
        "across_entity": shuffle forbidden columns across all rows, ignoring entity
            boundaries — catches leakage carried by between-entity structure that
            within_entity cannot see.  Returns a NOT-RUN NullResult (n_permutations=0)
            when the dataset has fewer than 2 entities.
        "target_within_period": shuffle target within time periods (comparison only);
            still uses the single shared-stream serial_v1 scheme (deferred, not
            parallelized).
    n_jobs
        Parallel workers for within_entity/across_entity permutations (loky backend
        via joblib).  Default 1 = serial, using the SAME spawn_v2 child-stream
        scheme as the parallel path — null_samples is byte-identical regardless of
        n_jobs.  Has no effect on target_within_period (always serial).
    stopping
        "fixed_v1" (default): draw exactly n_permutations, unchanged from the
            original behavior -- byte-identical to every pre-Tier-2 call.
        "sequential_v1" (Tier 2, within_entity/across_entity only): draw in
            batches (batch size = n_jobs when n_jobs > 1, else 1) up to
            _SEQ_N_MAX=500, checking after each batch (once at least
            _SEQ_N_MIN=30 draws have landed) whether to stop early:
              - Besag-Clifford exact rule: stop as soon as _SEQ_H=10 draws
                are >= observed_fixable_leakage. p_value = h/n_drawn exactly
                (this is what makes it exact, not an approximation: the
                stopping rule itself waited for the h-th success, so h/n is
                an unbiased estimator under that specific sampling scheme).
                Since _SEQ_H/_SEQ_N_MAX = 10/500 = 0.02 > _SEQ_ALPHA=0.01,
                reaching h ANYWHERE in [30, 500] already guarantees the final
                p_value cannot be significant at alpha=0.01 -- the NSL ladder
                is never entered in this branch, so no further check is
                needed before stopping.
              - Decision-stability rule (Tier 2b FIX A + Tier 2b-final,
                asymmetric): checked on EVERY batch once n_valid >= _SEQ_N_MIN,
                no longer gated behind "running p already looks significant"
                (that was the Tier 2 defect: for the common zero-exceedance
                case it required n>=100 before the check was even attempted,
                regardless of how far NSL already sat from the 1.0 boundary).
                Computes a conservative NSL interval [nsl_min, nsl_max] from a
                99% distribution-free bound on BOTH the null IQR and
                null_99th itself (see _conservative_iqr_bound /
                _conservative_quantile_bound / _nsl_interval_bound), and asks
                whether that whole interval sits on one side of 1.0:
                  * NOT-DETECTED side (nsl_max < 1.0): stop immediately, no
                    floor. "This isn't a leak" is a claim the data supports at
                    low n regardless of exceedance count -- this is the common
                    clean-data case and gets the full speedup.
                  * DETECTED side (nsl_min >= 1.0): NSL alone is NOT enough to
                    stop. engine.py's reality gate requires p_value < alpha
                    before NSL is even consulted, and the Laplace-corrected
                    p_value = (count_gte+1)/(n+1) has a hard floor that cannot
                    cross alpha before _ALPHA_FLOOR_DRAWS draws when
                    count_gte==0 (more, if count_gte>0) -- no matter how
                    stable the NSL interval looks. Stopping on NSL stability
                    alone here would flip a genuine FAIL to
                    UNCONFIRMED_HIGH_DAMAGE purely because p hadn't run long
                    enough (see Addendum 5 -- this is exactly the bug it
                    recorded). So this side only stops once n_valid >=
                    _ALPHA_FLOOR_DRAWS AND the actual running p_value has
                    itself crossed alpha at the current exceedance count;
                    otherwise it keeps drawing (Besag-Clifford's h=10 rule or
                    the N_MAX backstop resolve it if this persists).
                p_value here uses the standard Laplace-corrected formula
                (count+1)/(n+1) -- the Besag-Clifford h/n formula is only
                valid for the exact h-exceedance stopping rule above.
              - Otherwise: continue to the next batch, up to _SEQ_N_MAX. If
                _SEQ_N_MAX is reached without either rule firing, p_value
                also uses (count+1)/(n+1).
            p_is_upper_bound (NullResult field): True whenever p_value ==
            1/(n+1) because zero null draws reached observed_fixable_leakage
            -- the Laplace formula's floor value, not a precise estimate of a
            small p. False when count_gte > 0, or when the Besag-Clifford
            exact h/n formula applies. Threads through to
            SeverityResult/EngineDetection/JSON as an additive field.
            RNG: children are spawned once for _SEQ_N_MAX up front
            (np.random.SeedSequence(seed).spawn(_SEQ_N_MAX)), so draw i is
            byte-identical to the fixed_v1 scheme's draw i, and to whatever
            draw i would be regardless of where/whether stopping occurs --
            verified empirically (spawn(N)[:k] == spawn(k) for any k <= N)
            before this was implemented.
            Tier 2b FIX B: the loky worker pool is constructed ONCE for the
            whole sequential run (a persistent `with Parallel(...) as
            parallel:` context reused across every batch dispatch) instead of
            once per batch -- see _dispatch_batch. Draw i's value is
            unaffected either way; only wall-clock changes.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    _scheme = "spawn_v2" if method in ("within_entity", "across_entity") else "serial_v1"

    policy = config.split_policy
    all_feature_cols = [
        c for c in df.columns
        if c not in {contract.entity_id, contract.prediction_time,
                     contract.available_features_until, contract.target}
    ]
    forbidden_cols = [
        c for c in (contract.forbidden_after_prediction or []) if c in df.columns
    ]
    safe_feature_cols = [f for f in all_feature_cols if f not in set(forbidden_cols)]

    # If no forbidden features, null is degenerate (B=C always → fixable_leakage=0)
    if not forbidden_cols:
        _degen_n = 0 if stopping == "sequential_v1" else n_permutations
        null_samples = np.zeros(_degen_n)
        # Laplace-corrected p: observed <= 0 → all N zeros >= observed → p = (N+1)/(N+1) = 1
        _p_degen = 1.0 if observed_fixable_leakage <= 0.0 else 1.0 / (_degen_n + 1)
        # Same zero-exceedance-floor logic as the main return path below: when
        # observed > 0, every degenerate (all-zero) draw is < observed, so
        # count_gte == 0 exactly and _p_degen is the same Laplace floor value.
        # When observed <= 0, every draw >= observed instead -- a saturated,
        # exact p=1.0, not a floor artifact.
        _p_degen_is_upper_bound = observed_fixable_leakage > 0.0
        return NullResult(
            observed=observed_fixable_leakage,
            null_samples=null_samples,
            null_median=0.0,
            null_95th=0.0,
            null_99th=0.0,
            null_iqr=0.0,
            p_value=_p_degen,
            method=method,
            n_permutations=_degen_n,
            elapsed_seconds=time.perf_counter() - t0,
            scheme=_scheme,
            stopping=stopping,
            stopped_early=False,
            p_is_upper_bound=_p_degen_is_upper_bound,
        )

    # No-op guard (fail-safe epistemics): across-entity permutation needs >=2
    # entities to differ at all from within-entity permutation — with a single
    # entity the two are mathematically identical, so an across-entity null
    # would be redundant, not diagnostic.  Report NOT-RUN (n_permutations=0)
    # rather than a misleading "ran and found nothing."  NaN sentinels make
    # NOT-RUN visually unmistakable from a genuine clean result (p_value=1.0-ish).
    if method == "across_entity" and df[contract.entity_id].nunique() < 2:
        return NullResult(
            observed=observed_fixable_leakage,
            null_samples=np.array([]),
            null_median=float("nan"),
            null_95th=float("nan"),
            null_99th=float("nan"),
            null_iqr=float("nan"),
            p_value=float("nan"),
            method=method,
            n_permutations=0,
            elapsed_seconds=time.perf_counter() - t0,
            scheme=_scheme,
            stopping=stopping,
            stopped_early=False,
        )

    # Build temporal folds (same structure as the engine uses)
    temp_folds = temporal_expanding_folds(
        df,
        time_col=contract.prediction_time,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )
    interior_fold_idxs = _build_interior_fold_set(df, contract, config, temp_folds)

    null_samples: list[float] = []
    _stopped_early = False
    _p_value_override: Optional[float] = None

    if method in ("within_entity", "across_entity"):
        # Invariant: AUC_C_pool doesn't change when forbidden features are permuted
        # (C drops forbidden features entirely, for both permutation strategies).
        # Pre-compute once and reuse across every permutation draw.
        eval_c = evaluate_folds(
            df, safe_feature_cols, contract.target, temp_folds, model_factory,
            return_predictions=True, categorical_map=categorical_map,
        )
        y_pool, proba_c_pool = _pool_oof_predictions(eval_c.fold_evals, interior_fold_idxs)
        if y_pool is None:
            raise RuntimeError("No interior folds produced OOF predictions for AUC_C.")
        auc_c_pool = float(roc_auc_score(y_pool, proba_c_pool))

        # F2a-perf fast path: the feature matrix is identical across every
        # permutation except the forbidden column(s), so build it once here
        # (float32, via the same guarded cast evaluate_folds itself uses) and
        # let _null_permutation_once patch just those columns per draw,
        # instead of copying the whole dataframe and re-deriving the matrix
        # from scratch on every one of the n_permutations calls below.
        X_base = _feature_matrix(df, all_feature_cols, categorical_map=categorical_map)
        y_all = df[contract.target].to_numpy()
        forbidden_col_positions = [all_feature_cols.index(c) for c in forbidden_cols]
        if method == "within_entity":
            entity_codes, entity_uniques = pd.factorize(
                df[contract.entity_id].to_numpy(), sort=False
            )
            entity_n_groups = len(entity_uniques)
        else:
            entity_codes, entity_n_groups = None, None

        def _dispatch_batch(child_batch: list, parallel: Optional[Any] = None) -> list:
            """Run one batch of permutation draws, serial or parallel.

            `parallel`, when given, is a live joblib Parallel context (opened
            once by the caller via `with Parallel(...) as parallel:`) reused
            across every batch dispatch -- Tier 2b FIX B: constructing a fresh
            Parallel() per batch reintroduced the loky pool-startup overhead
            this project already profiled away in Tier 1 (Addendum 4). When
            None (fixed_v1's n_jobs>1 path, a single one-shot dispatch, and
            the n_jobs==1 serial path which never touches Parallel at all), a
            Parallel is constructed here exactly as before.
            """
            if n_jobs == 1:
                return [
                    _null_permutation_once(
                        c, df, forbidden_cols, contract.entity_id, method,
                        all_feature_cols, contract.target, temp_folds, model_factory,
                        interior_fold_idxs, auc_c_pool, y_pool,
                        X_base=X_base, y_all=y_all,
                        forbidden_col_positions=forbidden_col_positions,
                        entity_codes=entity_codes, entity_n_groups=entity_n_groups,
                    )
                    for c in child_batch
                ]
            from joblib import delayed  # noqa: PLC0415
            tasks = (
                delayed(_null_permutation_once)(
                    c, df, forbidden_cols, contract.entity_id, method,
                    all_feature_cols, contract.target, temp_folds, model_factory,
                    interior_fold_idxs, auc_c_pool, y_pool,
                    X_base=X_base, y_all=y_all,
                    forbidden_col_positions=forbidden_col_positions,
                    entity_codes=entity_codes, entity_n_groups=entity_n_groups,
                )
                for c in child_batch
            )
            if parallel is not None:
                return parallel(tasks)
            from joblib import Parallel  # noqa: PLC0415
            return Parallel(
                n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0,
            )(tasks)

        if stopping == "fixed_v1":
            # spawn_v2 seeding: permutation i always draws from children[i], an
            # independent child stream — never a shared, order-dependent Generator.
            # This is what makes null_samples independent of n_jobs / scheduling.
            children = np.random.SeedSequence(seed).spawn(n_permutations)

            if n_jobs == 1:
                raw_samples: list = []
                for i in range(n_permutations):
                    raw_samples.extend(_dispatch_batch([children[i]]))
                    if verbose and (i + 1) % 10 == 0:
                        print(f"  [{i+1}/{n_permutations}] running...", flush=True)
            else:
                raw_samples = _dispatch_batch(list(children))

            null_samples = [s for s in raw_samples if s is not None]

        elif stopping == "sequential_v1":
            # Tier 2b: Besag-Clifford exceedance-count rule (unchanged) +
            # distance-aware decision-stability early stop (FIX A -- see
            # _nsl_decision_stable). See estimate_fixable_leakage_null's
            # docstring for the full rule. Spawn once for _SEQ_N_MAX so draw i
            # is byte-identical regardless of where (or whether) stopping
            # occurs.
            children = np.random.SeedSequence(seed).spawn(_SEQ_N_MAX)
            batch_size = n_jobs if n_jobs > 1 else 1

            raw_index = 0
            stop_reason = "n_max_reached"

            # FIX B: one persistent loky pool for the whole sequential run,
            # not one per batch. nullcontext when n_jobs==1 keeps the loop
            # below single-path -- _dispatch_batch never touches `parallel`
            # in that case anyway (it takes the serial list-comp branch).
            from contextlib import nullcontext  # noqa: PLC0415
            if n_jobs > 1:
                from joblib import Parallel  # noqa: PLC0415
                pool_ctx = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)
            else:
                pool_ctx = nullcontext()

            with pool_ctx as parallel:
                while raw_index < _SEQ_N_MAX:
                    batch_end = min(raw_index + batch_size, _SEQ_N_MAX)
                    batch_results = _dispatch_batch(
                        list(children[raw_index:batch_end]), parallel
                    )
                    null_samples.extend(s for s in batch_results if s is not None)
                    raw_index = batch_end
                    if verbose:
                        print(f"  [{raw_index}/{_SEQ_N_MAX}] drawn, "
                              f"{len(null_samples)} valid...", flush=True)

                    n_valid = len(null_samples)
                    if n_valid < _SEQ_N_MIN:
                        continue
                    arr_running = np.array(null_samples)
                    count_gte = int(np.sum(arr_running >= observed_fixable_leakage))

                    if count_gte >= _SEQ_H:
                        stop_reason = "h_exceedance"
                        _stopped_early = True
                        break

                    # FIX A: checked on EVERY batch once n_valid >= N_MIN, no
                    # longer gated behind "running p already looks
                    # significant" -- that gate was the Tier 2 defect (it
                    # forced ~100 draws before this could even be attempted
                    # when count_gte==0, regardless of how far NSL already
                    # sat from the 1.0 boundary).
                    iqr_lo, iqr_hi = _conservative_iqr_bound(arr_running, _SEQ_IQR_CONFIDENCE)
                    null_99th_lo, null_99th_hi = _conservative_quantile_bound(
                        arr_running, 0.99, _SEQ_IQR_CONFIDENCE
                    )
                    # Stability check goes through _nsl_decision_stable (not
                    # inlined) so it stays independently mockable -- existing
                    # tests patch this exact name to force "never resolves".
                    # _nsl_interval_bound is called again just below, only
                    # when stable, purely to read off WHICH side of 1.0 the
                    # (already-established) stable interval sits on.
                    stable = _nsl_decision_stable(
                        observed_fixable_leakage, null_99th_lo, null_99th_hi, iqr_lo, iqr_hi
                    )

                    # Tier 2b-final (Addendum 5): asymmetric by design, per
                    # Besag-Clifford semantics. A stable NOT-DETECTED
                    # conclusion (NSL provably below 1.0) needs no floor --
                    # "this isn't a leak" is supportable at low n. A stable
                    # DETECTED conclusion (NSL provably above 1.0) is NOT
                    # enough to stop on its own: engine.py's reality gate
                    # requires p_value < _SEQ_ALPHA before NSL is even
                    # consulted, and the Laplace-corrected p_value has a hard
                    # floor of 1/(n+1) that cannot cross alpha before
                    # _ALPHA_FLOOR_DRAWS draws when count_gte==0 (more, if
                    # count_gte>0) -- no matter how stable the NSL interval
                    # already looks. Stopping on NSL stability alone in this
                    # direction is exactly the bug Addendum 5 recorded: a
                    # verdict flip from FAIL to UNCONFIRMED_HIGH_DAMAGE on an
                    # unmistakable leak, purely because p hadn't run long
                    # enough to be entitled to an opinion.
                    if stable:
                        _, nsl_max = _nsl_interval_bound(
                            observed_fixable_leakage, null_99th_lo, null_99th_hi, iqr_lo, iqr_hi
                        )
                        detected_side = nsl_max >= 1.0
                        if not detected_side:
                            stop_reason = "decision_stable_not_detected"
                            _stopped_early = True
                            break
                        if n_valid >= _ALPHA_FLOOR_DRAWS:
                            p_running = (count_gte + 1) / (n_valid + 1)
                            if p_running < _SEQ_ALPHA:
                                stop_reason = "decision_stable_detected"
                                _stopped_early = True
                                break
                        # else: NSL already looks like a leak, but the honest
                        # p-value floor for the CURRENT exceedance count
                        # hasn't been reached -- keep drawing. Besag-Clifford's
                        # h=10 rule or the N_MAX backstop resolves it if this
                        # persists.

            if stop_reason == "h_exceedance":
                _p_value_override = _SEQ_H / len(null_samples)

        else:
            raise ValueError(
                f"estimate_fixable_leakage_null: unsupported stopping {stopping!r}. "
                "Use 'fixed_v1' or 'sequential_v1'."
            )

    elif method == "target_within_period":
        # Target is permuted → both B and C change every iteration; no invariant to exploit
        for i in range(n_permutations):
            df_perm = _permute_target_within_period(
                df, contract.target, contract.prediction_time, rng
            )
            eval_b_perm = evaluate_folds(
                df_perm, all_feature_cols, contract.target, temp_folds, model_factory,
                return_predictions=True, categorical_map=categorical_map,
            )
            eval_c_perm = evaluate_folds(
                df_perm, safe_feature_cols, contract.target, temp_folds, model_factory,
                return_predictions=True, categorical_map=categorical_map,
            )
            auc_b = _pool_oof_auc(eval_b_perm.fold_evals, interior_fold_idxs)
            auc_c = _pool_oof_auc(eval_c_perm.fold_evals, interior_fold_idxs)
            if auc_b is not None and auc_c is not None:
                null_samples.append(auc_b - auc_c)
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_permutations}] running...", flush=True)

    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'within_entity', 'across_entity', "
            "or 'target_within_period'."
        )

    arr = np.array(null_samples)
    n_draws = len(arr)
    if _p_value_override is not None:
        # Besag-Clifford exact rule (sequential_v1, stopped on h exceedances):
        # p = h / n_drawn. Valid specifically because the stopping rule itself
        # waited for the h-th exceedance -- not the generic Laplace estimator.
        # count_gte >= _SEQ_H > 0 by construction here, so this is a real
        # count-backed p-value, never a zero-exceedance floor artifact.
        p_value = _p_value_override
        p_is_upper_bound = False
    else:
        # Laplace-corrected p-value: (count + 1) / (N + 1).
        # Prevents p = 0.0 when no null sample reaches the observed value (which would
        # make the minimum representable p depend on N rather than reality).
        # For N=100: minimum p ≈ 1/101 ≈ 0.0099, consistent with alpha=0.01.
        count_gte = int(np.sum(arr >= observed_fixable_leakage))
        p_value = (count_gte + 1) / (n_draws + 1)
        # Tier 2b-final: True iff this p_value IS the Laplace floor (zero
        # null draws reached observed) rather than a count-backed estimate --
        # see NullResult.p_is_upper_bound and Addendum 5.
        p_is_upper_bound = count_gte == 0

    return NullResult(
        observed=observed_fixable_leakage,
        null_samples=arr,
        null_median=float(np.median(arr)),
        null_95th=float(np.percentile(arr, 95)),
        null_99th=float(np.percentile(arr, 99)),
        null_iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        p_value=p_value,
        method=method,
        n_permutations=n_draws,
        stopping=stopping,
        stopped_early=_stopped_early,
        elapsed_seconds=time.perf_counter() - t0,
        scheme=_scheme,
        p_is_upper_bound=p_is_upper_bound,
    )
