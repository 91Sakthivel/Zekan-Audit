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
(see diag_null.py).  100 is the minimum defensible count; 200 adds robustness.
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
from sklearn.metrics import roc_auc_score

from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.metrics import _feature_matrix, evaluate_folds
from zekan.severity.splitters import temporal_expanding_folds


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
) -> NullResult:
    """Estimate the permutation null distribution for fixable_leakage.

    Parameters
    ----------
    observed_fixable_leakage
        The fixable_leakage value from the unmodified engine run.
        Used only to compute the p-value; does not affect null sampling.
    n_permutations
        Number of permutation draws.  100 gives stable 95th percentile on the
        benchmark DGP (two batches agree within ~0.003).  Increase to 200 for
        higher resolution.
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
        null_samples = np.zeros(n_permutations)
        # Laplace-corrected p: observed <= 0 → all N zeros >= observed → p = (N+1)/(N+1) = 1
        _p_degen = 1.0 if observed_fixable_leakage <= 0.0 else 1.0 / (n_permutations + 1)
        return NullResult(
            observed=observed_fixable_leakage,
            null_samples=null_samples,
            null_median=0.0,
            null_95th=0.0,
            null_99th=0.0,
            null_iqr=0.0,
            p_value=_p_degen,
            method=method,
            n_permutations=n_permutations,
            elapsed_seconds=time.perf_counter() - t0,
            scheme=_scheme,
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

    if method in ("within_entity", "across_entity"):
        # Invariant: AUC_C_pool doesn't change when forbidden features are permuted
        # (C drops forbidden features entirely, for both permutation strategies).
        # Pre-compute once and reuse across every permutation draw.
        eval_c = evaluate_folds(
            df, safe_feature_cols, contract.target, temp_folds, model_factory,
            return_predictions=True,
        )
        y_pool, proba_c_pool = _pool_oof_predictions(eval_c.fold_evals, interior_fold_idxs)
        if y_pool is None:
            raise RuntimeError("No interior folds produced OOF predictions for AUC_C.")
        auc_c_pool = float(roc_auc_score(y_pool, proba_c_pool))

        # spawn_v2 seeding: permutation i always draws from children[i], an
        # independent child stream — never a shared, order-dependent Generator.
        # This is what makes null_samples independent of n_jobs / scheduling.
        children = np.random.SeedSequence(seed).spawn(n_permutations)

        # F2a-perf fast path: the feature matrix is identical across every
        # permutation except the forbidden column(s), so build it once here
        # (float32, via the same guarded cast evaluate_folds itself uses) and
        # let _null_permutation_once patch just those columns per draw,
        # instead of copying the whole dataframe and re-deriving the matrix
        # from scratch on every one of the n_permutations calls below.
        X_base = _feature_matrix(df, all_feature_cols)
        y_all = df[contract.target].to_numpy()
        forbidden_col_positions = [all_feature_cols.index(c) for c in forbidden_cols]
        if method == "within_entity":
            entity_codes, entity_uniques = pd.factorize(
                df[contract.entity_id].to_numpy(), sort=False
            )
            entity_n_groups = len(entity_uniques)
        else:
            entity_codes, entity_n_groups = None, None

        if n_jobs == 1:
            raw_samples: list = []
            for i in range(n_permutations):
                raw_samples.append(
                    _null_permutation_once(
                        children[i], df, forbidden_cols, contract.entity_id, method,
                        all_feature_cols, contract.target, temp_folds, model_factory,
                        interior_fold_idxs, auc_c_pool, y_pool,
                        X_base=X_base, y_all=y_all,
                        forbidden_col_positions=forbidden_col_positions,
                        entity_codes=entity_codes, entity_n_groups=entity_n_groups,
                    )
                )
                if verbose and (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{n_permutations}] running...", flush=True)
        else:
            from joblib import Parallel, delayed  # noqa: PLC0415
            raw_samples = Parallel(
                n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0,
            )(
                delayed(_null_permutation_once)(
                    children[i], df, forbidden_cols, contract.entity_id, method,
                    all_feature_cols, contract.target, temp_folds, model_factory,
                    interior_fold_idxs, auc_c_pool, y_pool,
                    X_base=X_base, y_all=y_all,
                    forbidden_col_positions=forbidden_col_positions,
                    entity_codes=entity_codes, entity_n_groups=entity_n_groups,
                )
                for i in range(n_permutations)
            )

        null_samples = [s for s in raw_samples if s is not None]

    elif method == "target_within_period":
        # Target is permuted → both B and C change every iteration; no invariant to exploit
        for i in range(n_permutations):
            df_perm = _permute_target_within_period(
                df, contract.target, contract.prediction_time, rng
            )
            eval_b_perm = evaluate_folds(
                df_perm, all_feature_cols, contract.target, temp_folds, model_factory,
                return_predictions=True,
            )
            eval_c_perm = evaluate_folds(
                df_perm, safe_feature_cols, contract.target, temp_folds, model_factory,
                return_predictions=True,
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
    # Laplace-corrected p-value: (count + 1) / (N + 1).
    # Prevents p = 0.0 when no null sample reaches the observed value (which would
    # make the minimum representable p depend on N rather than reality).
    # For N=100: minimum p ≈ 1/101 ≈ 0.0099, consistent with alpha=0.01.
    count_gte = int(np.sum(arr >= observed_fixable_leakage))
    p_value = (count_gte + 1) / (n_draws + 1)

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
        elapsed_seconds=time.perf_counter() - t0,
        scheme=_scheme,
    )
