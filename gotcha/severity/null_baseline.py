"""Permutation null distribution for fixable_leakage (B−C pooled OOF).

NULL HYPOTHESIS: the declared forbidden features carry no genuine future information.
The null is estimated by breaking the leakage signal while holding everything else
(fold structure, class balance, safe-feature signal, temporal protocol) constant.

Two methods:

  within_entity (recommended)
    Shuffle each forbidden column's values within each entity's rows.
    Preserves:  entity marginal distributions of each forbidden value.
    Destroys:   the row-level temporal ordering (the future-information link).
    Invariant:  AUC_C is unchanged — C drops forbidden features, so permuting them
                has no effect.  AUC_C_pool is pre-computed once and reused across
                all permutations; only AUC_B is refit per draw.  ~2× faster than
                a naive two-pass null.

  target_within_period
    Shuffle target values within each time period.
    Preserves:  period-level base rates; fold structure; all feature values.
    Destroys:   all entity-level signal (safe features AND forbidden features).
    The null answers "what is B−C when nothing predicts?" rather than "what is
    B−C when the forbidden feature is noise?"  Less precise: variance is higher
    because both B and C regress to ~0.5 AUC.

The within_entity method isolates the forbidden feature's contribution, giving a
tighter null and a more interpretable p-value.

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
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gotcha.config.schema import GotchaConfig
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.severity.metrics import evaluate_folds
from gotcha.severity.splitters import temporal_expanding_folds


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


# ── Permutation strategies ────────────────────────────────────────────────────

def _permute_within_entity(
    df: pd.DataFrame,
    forbidden_cols: list[str],
    entity_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle each forbidden column's values within entity rows.

    Row indices and all other columns are unchanged.
    Returns a copy; never mutates the input.
    """
    df_perm = df.copy()
    for col in forbidden_cols:
        df_perm[col] = (
            df_perm.groupby(entity_col, sort=False)[col]
            .transform(lambda x: rng.permutation(x.values))
        )
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


# ── Interior fold context ─────────────────────────────────────────────────────

def _build_interior_fold_set(
    df: pd.DataFrame,
    contract: PredictionContract,
    config: GotchaConfig,
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
    config: GotchaConfig,
    model_factory: Callable[[], Any],
    observed_fixable_leakage: float,
    n_permutations: int = 100,
    seed: int = 0,
    method: str = "within_entity",
    verbose: bool = False,
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
        but convergent 95th percentile at n=100.
    method
        "within_entity": shuffle forbidden columns within entity rows (recommended).
        "target_within_period": shuffle target within time periods (comparison only).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

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

    if method == "within_entity":
        # Invariant: AUC_C_pool doesn't change when forbidden features are permuted
        # (C drops forbidden features entirely).  Pre-compute once and reuse.
        eval_c = evaluate_folds(
            df, safe_feature_cols, contract.target, temp_folds, model_factory,
            return_predictions=True,
        )
        y_pool, proba_c_pool = _pool_oof_predictions(eval_c.fold_evals, interior_fold_idxs)
        if y_pool is None:
            raise RuntimeError("No interior folds produced OOF predictions for AUC_C.")
        auc_c_pool = float(roc_auc_score(y_pool, proba_c_pool))

        for i in range(n_permutations):
            df_perm = _permute_within_entity(
                df, forbidden_cols, contract.entity_id, rng
            )
            eval_b_perm = evaluate_folds(
                df_perm, all_feature_cols, contract.target, temp_folds, model_factory,
                return_predictions=True,
            )
            # Use the SAME y_pool (target unchanged by within-entity feature permutation)
            b_fe_by_idx = {fe.meta.fold_idx: fe for fe in eval_b_perm.fold_evals}
            proba_b_parts = [
                b_fe_by_idx[idx].proba
                for idx in sorted(interior_fold_idxs)
                if idx in b_fe_by_idx and b_fe_by_idx[idx].proba is not None
            ]
            if not proba_b_parts:
                continue
            auc_b_perm = float(roc_auc_score(y_pool, np.concatenate(proba_b_parts)))
            null_samples.append(auc_b_perm - auc_c_pool)
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_permutations}] running...", flush=True)

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
        raise ValueError(f"Unknown method {method!r}. Use 'within_entity' or 'target_within_period'.")

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
    )
