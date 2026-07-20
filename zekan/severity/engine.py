"""Severity engine: reconciles naive offline AUC into deployable performance.

A/B/C decomposition triangle:
  A = random grouped CV, all features          -> naive_auc (optimistic upper bound)
  B = temporal CV, all features                -> reveals temporal gap
  C = temporal CV, forbidden features dropped  -> estimated_deployable_auc

Derived quantities:
  total_optimism      = A - mean(C)             (total gap)
  fixable_leakage     = median(B_i - C_i)       (paired per-fold difference)
  nonfixable_optimism = A - mean(B)             (temporal evaluation gap)

Approximate invariant (within float tolerance):
  total_optimism ~= fixable_leakage + nonfixable_optimism

Honest limitation: this decomposition measures ONLY DECLARED leakage -- features
listed in forbidden_after_prediction.  Undeclared leaky features appear in A, B,
AND C equally and cancel out, remaining invisible to this ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from zekan.config.schema import ZekanConfig
from zekan.contract.contract_checks import validate_contract
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.metrics import _feature_matrix, evaluate_folds
from zekan.severity.splitters import random_grouped_folds, temporal_expanding_folds


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class PerFoldSeverity:
    """Paired B/C AUC for one temporal fold."""

    fold_idx: int
    auc_with_forbidden: float
    auc_without_forbidden: float
    test_time_max: Optional[str] = None
    is_terminal: bool = False  # True when this fold's test window reaches the dataset's last period

    @property
    def fold_leakage(self) -> float:
        return self.auc_with_forbidden - self.auc_without_forbidden


@dataclass
class SeverityResult:
    """Output of the A/B/C decomposition."""

    status: str                                   # pass / warn / fail / unavailable
    metric: str                                   # always "roc_auc" for now
    naive_auc: float                              # evaluation A (grouped CV, all features)
    estimated_deployable_auc: float               # evaluation C mean (temporal, no forbidden)
    total_optimism: float                         # A - C
    fixable_leakage: float                        # pooled OOF AUC_B - AUC_C (interior folds)
    fixable_leakage_range: tuple[float, float]    # (min, max) of per-fold leakages
    nonfixable_optimism: float                    # A - mean(B)
    per_fold: list[PerFoldSeverity] = field(default_factory=list)
    caveat: str = ""
    unavailable_reason: Optional[str] = None
    # Permutation null baseline — populated when run_severity_analysis is called
    # with n_permutations > 0.  None when the null was not run.
    null_95th: Optional[float] = None             # 95th percentile of null (kept for backward compat)
    null_99th: Optional[float] = None             # 99th percentile = boundary for alpha=0.01 gate
    null_median: Optional[float] = None           # median of null distribution (for NSL_med comparison)
    null_iqr: Optional[float] = None              # IQR of null distribution (q75 - q25)
    p_value: Optional[float] = None               # (count+1)/(N+1) Laplace-corrected; one-tailed
    nsl: Optional[float] = None                   # null-standardized leakage = (obs - q99)/IQR
    n_permutations_run: int = 0                   # 0 when null was not run
    # Across-entity permutation null (spec 1) — additive second channel, closes the
    # within-entity blind spot (leakage carried by between-entity structure, e.g. an
    # entity-level aggregate constant within each entity).  None when not run: either
    # n_permutations==0 (null skipped entirely) or the no-op guard fired (<2 entities).
    # Uses the SAME method-agnostic live gate as within-entity (_NULL_ALPHA, NSL>=1.0);
    # does NOT share power.py's within-entity-scoped 0.176/0.352 calibration constants.
    null_95th_across: Optional[float] = None
    null_99th_across: Optional[float] = None
    null_median_across: Optional[float] = None
    null_iqr_across: Optional[float] = None
    p_value_across: Optional[float] = None
    nsl_across: Optional[float] = None
    n_permutations_across: int = 0                # 0 when the across-entity null was not run
    # Tier 2 (sequential/adaptive stopping) — additive; "fixed_v1" + stopped_early=False
    # for every pre-Tier-2 caller (the default run_severity_analysis(null_stopping=...)
    # value). n_permutations_run/n_permutations_across above already ARE n_drawn under
    # either stopping scheme -- no separate n_drawn field is needed.
    null_stopping: str = "fixed_v1"               # "fixed_v1" or "sequential_v1"
    null_stopped_early: bool = False              # within-entity null stopped before _SEQ_N_MAX
    null_stopped_early_across: bool = False       # across-entity null stopped before _SEQ_N_MAX
    # Tier 2b-final -- additive; False for every pre-Tier-2b-final caller (the
    # default run_severity_analysis() value). True iff p_value is the Laplace
    # formula's floor (1/(n+1), zero null draws reached observed) rather than a
    # count-backed estimate -- see null_baseline.NullResult.p_is_upper_bound.
    p_is_upper_bound: bool = False
    p_is_upper_bound_across: bool = False
    feature_attribution: Optional[Any] = None     # AblationSummary when ablation ran; None otherwise
    folds: list = field(default_factory=list)     # temporal FoldIndices used for B/C eval; internal only
    # Upgrade 1 step 1b -- additive; internal only, same as `folds` above. The
    # float32 all-features matrix + target array this run already built once
    # (this session's matrix-hoist work), exposed so a structural probe that
    # declares needs_matrix=True (audit._run_structural_probes) can slice a
    # column by POSITION instead of re-deriving the cast from df. all_features
    # gives the column order X_all's columns are in -- never assume any other
    # order. None only for the status="unavailable" early-return path, where
    # no matrix was ever built.
    all_features: Optional[list[str]] = None
    X_all: Optional[np.ndarray] = None
    y_all: Optional[np.ndarray] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

_CAVEAT = (
    "This decomposition measures only DECLARED leakage (features listed in "
    "forbidden_after_prediction). Undeclared leaky features appear in A, B, "
    "and C equally and cancel out -- invisible to this ablation."
)

# ── Null-standardized verdict parameters ─────────────────────────────────────
# The verdict is null-standardized: constant thresholds on fixable_leakage are
# replaced by per-dataset null calibration (see _leakage_status).
#
# NSL (null-standardized leakage) = (observed - null_99th) / null_unit, where
# null_99th is the 99th percentile of the permutation null (boundary for alpha=0.01)
# and null_unit = max(null_iqr, _NSL_EPS) is the IQR of the null distribution.
# IQR is used instead of (null_99th - null_median) because the 99th percentile
# of N=100 samples is near-max and has high sampling variance; IQR is stable.
#
# NSL ladder (only entered when p < _NULL_ALPHA):
#   NSL < 1.0  -> NOTE   (statistically significant but small effect size)
#   1.0 <= NSL < 2.0 -> WARN
#   NSL >= 2.0 -> FAIL
#
# These constants are retained as named exports for reference in reporting strings
# and external callers, but they are NOT gate thresholds in the verdict.
_NULL_ALPHA: float = 0.01          # gate: p >= alpha -> PASS
_NSL_NOTE_THRESHOLD: float = 1.0   # NSL < 1 -> NOTE
_NSL_WARN_THRESHOLD: float = 2.0   # NSL < 2 -> WARN; >= 2 -> FAIL
_NSL_EPS: float = 1e-4             # floor for IQR denominator (prevents zero-div when all draws identical)

FIXABLE_LEAKAGE_NOISE_FLOOR: float = 0.02   # retained for reference / reporting only
FIXABLE_LEAKAGE_CLEAR_LEAK: float = 0.04    # retained for reference / reporting only
FIXABLE_LEAKAGE_STRONG_LEAK: float = 0.10   # retained for reference / reporting only


def _feature_cols(df: pd.DataFrame, contract: PredictionContract) -> list[str]:
    """All columns except metadata identifiers and target."""
    excluded = {
        contract.entity_id,
        contract.prediction_time,
        contract.available_features_until,
        contract.target,
    }
    return [c for c in df.columns if c not in excluded]


def _status_from_optimism(total_optimism: float) -> str:
    if total_optimism <= 0.03:
        return "pass"
    if total_optimism <= 0.05:
        return "warn"
    return "fail"


# ── Public API ────────────────────────────────────────────────────────────────

def run_severity_analysis(
    df: pd.DataFrame,
    contract: PredictionContract,
    config: ZekanConfig,
    model_factory: Optional[Callable[[], Any]] = None,
    n_permutations: int = 0,
    null_seed: int = 0,
    ablation_warn_floor: Optional[float] = None,
    n_jobs: int = 1,
    null_stopping: str = "fixed_v1",
) -> SeverityResult:
    """Run the A/B/C performance decomposition and return a SeverityResult.

    Returns status='unavailable' when contract validation blocks computation.
    B and C share the same temporal fold objects so per-fold differences cancel
    fold-level variance exactly.

    n_permutations
        When > 0, runs BOTH the within-entity permutation null (stores null_95th,
        p_value, nsl, n_permutations_run) AND the across-entity permutation null
        (spec 1; stores the *_across fields, n_permutations_across) after computing
        fixable_leakage.  The across-entity null shares n_permutations and null_seed
        with the within-entity null and closes the within-entity blind spot: leakage
        carried by between-entity structure (e.g. an entity-level aggregate constant
        within each entity) that a within-entity permutation cannot see.  It is
        additive — the within-entity fields and calibration are unaffected — and is
        NOT RUN (fields stay None) when the dataset has fewer than 2 entities.
        Default 0 skips both nulls (backward-compatible). IGNORED for both nulls
        when null_stopping="sequential_v1" -- see null_stopping below.
    null_seed
        RNG seed for the permutation null.  Different seeds give convergent null_95th
        at n_permutations >= 100.  Under spawn_v2 (F2a), seeds the permutation's
        SeedSequence — results are identical for any n_jobs.
    n_jobs
        Parallel workers, shared by per-feature ablation AND both permutation nulls
        (loky backend via joblib).  Default 1 = serial.  Both nulls use spawn_v2
        seeding regardless of n_jobs, so null_samples/null_iqr/p_value/nsl are
        byte-identical whether n_jobs=1 or n_jobs>1 — only wall-clock time changes.
    null_stopping
        "fixed_v1" (default): both nulls draw exactly n_permutations, unchanged
        from pre-Tier-2 behavior. "sequential_v1" (Tier 2): both nulls use the
        Besag-Clifford + decision-stability adaptive stopping rule instead (see
        null_baseline.estimate_fixable_leakage_null's docstring) -- changes HOW
        MANY permutations are drawn, never the verdict logic itself. Result
        surfaces as null_stopping / null_stopped_early / null_stopped_early_across
        on the returned SeverityResult.
    """
    policy = config.split_policy

    val = validate_contract(contract, df)
    if not val.can_compute_severity:
        bad = [c.message for c in val.checks if c.status.value in ("fail", "warn")]
        return SeverityResult(
            status="unavailable",
            metric="roc_auc",
            naive_auc=float("nan"),
            estimated_deployable_auc=float("nan"),
            total_optimism=float("nan"),
            fixable_leakage=float("nan"),
            fixable_leakage_range=(float("nan"), float("nan")),
            nonfixable_optimism=float("nan"),
            per_fold=[],
            caveat=_CAVEAT,
            unavailable_reason="; ".join(bad) if bad else "Contract validation failed.",
        )

    all_features = _feature_cols(df, contract)
    forbidden = set(contract.forbidden_after_prediction) & set(df.columns)
    safe_features = [f for f in all_features if f not in forbidden]

    # ── Efficiency: build the float32 feature matrix + target ONCE per audit ──
    # (mirrors the Tier 1 X_all/y_all reuse null_baseline.py already does for the
    # permutation null). eval_a and eval_b both use `all_features` -- same column
    # set, same row set (rand_folds/temp_folds only differ in which index arrays
    # each fold uses into this same matrix) -- so both share X_all_full directly.
    # eval_c uses `safe_features`, a strict order-preserving subset of
    # all_features (safe_features = [f for f in all_features if f not in
    # forbidden]), so its matrix is sliced from X_all_full BY POSITION rather
    # than re-cast from df. Position-based (not boolean-mask) slicing is used
    # throughout so this generalizes correctly to callers (ablation) that
    # reorder columns, not just filter them -- column order is NOT
    # interchangeable for every allowlisted estimator: verified empirically that
    # RandomForestClassifier's fitted predictions change when input columns are
    # reordered even with a fixed random_state (index-based feature subsampling
    # per split), while HistGradientBoostingClassifier's do not. Slicing by an
    # explicit position list (not a mask) reproduces the exact column order any
    # feature_cols list specifies, so this is safe for every estimator, not just
    # the ones that happen to be order-invariant.
    X_all_full = _feature_matrix(df, all_features)
    y_all = df[contract.target].to_numpy()
    _col_pos = {f: i for i, f in enumerate(all_features)}
    safe_positions = [_col_pos[f] for f in safe_features]
    X_safe = X_all_full[:, safe_positions]

    # ── A: random grouped CV, all features ────────────────────────────────────
    rand_folds = random_grouped_folds(
        df,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )
    eval_a = evaluate_folds(
        df, all_features, contract.target, rand_folds, model_factory,
        X_all=X_all_full, y_all=y_all,
    )
    naive_auc = eval_a.mean_auc

    # ── Shared temporal folds for B and C ─────────────────────────────────────
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

    # Build fold-index → FoldMeta for partial-fold detection.
    fold_meta_by_idx = {
        fold.meta.fold_idx: fold.meta
        for fold in temp_folds
        if not fold.meta.skipped and fold.meta.test_time_max is not None
    }

    # Rank-based interior cutoff: a fold is excluded from fixable_leakage when its
    # test_time_max rank > n_periods - 1 - leak_lookahead.  For leak_lookahead=1
    # this reproduces the old terminal-only exclusion (rank > n-2, i.e. the last
    # period only).  For leak_lookahead=k it additionally excludes folds whose test
    # window is within k periods of the end, where future rows cannot carry a full
    # k-step signal.
    sorted_all_periods = sorted(
        df[contract.prediction_time].unique(),
        key=lambda x: pd.to_datetime(x),
    )
    period_rank: dict[str, int] = {
        pd.Timestamp(p).strftime("%Y-%m-%d"): i
        for i, p in enumerate(sorted_all_periods)
    }
    n_time_periods = len(period_rank)
    interior_rank_cutoff = n_time_periods - 1 - policy.leak_lookahead

    # ── B: temporal CV, all features ──────────────────────────────────────────
    eval_b = evaluate_folds(
        df, all_features, contract.target, temp_folds, model_factory,
        return_predictions=True, X_all=X_all_full, y_all=y_all,
    )

    # ── C: temporal CV, forbidden features dropped and retrained ──────────────
    eval_c = evaluate_folds(
        df, safe_features, contract.target, temp_folds, model_factory,
        return_predictions=True, X_all=X_safe, y_all=y_all,
    )
    estimated_deployable_auc = eval_c.mean_auc

    # ── Paired per-fold differences (B and C share same fold objects) ──────────
    b_by_idx = {fe.meta.fold_idx: fe.auc for fe in eval_b.fold_evals}
    c_by_idx = {fe.meta.fold_idx: fe.auc for fe in eval_c.fold_evals}

    per_fold: list[PerFoldSeverity] = []
    for idx in sorted(b_by_idx):
        if idx not in c_by_idx:
            continue
        meta = fold_meta_by_idx.get(idx)
        ttm = meta.test_time_max if meta else None
        ttm_rank = period_rank.get(ttm) if ttm is not None else None
        is_terminal = (
            ttm_rank is not None and ttm_rank > interior_rank_cutoff
        )
        per_fold.append(PerFoldSeverity(
            fold_idx=idx,
            auc_with_forbidden=b_by_idx[idx],
            auc_without_forbidden=c_by_idx[idx],
            test_time_max=ttm,
            is_terminal=is_terminal,
        ))

    fold_leakages = [pf.fold_leakage for pf in per_fold]
    fixable_leakage_range = (
        (float(min(fold_leakages)), float(max(fold_leakages)))
        if fold_leakages else (0.0, 0.0)
    )

    # ── Pooled OOF fixable_leakage (replaces per-fold median) ─────────────────
    # Collect out-of-fold predictions from interior folds only and compute a
    # single AUC for B and C on the pooled set.  This collapses fold-sampling
    # variance: instead of taking the median of 4 noisy AUC differences, we get
    # one stable AUC computed on ~4× as many rows.  Terminal-boundary folds are
    # already excluded via is_terminal so their rows never enter the pool.
    interior_fold_idxs = {pf.fold_idx for pf in per_fold if not pf.is_terminal}
    b_fe_by_idx = {fe.meta.fold_idx: fe for fe in eval_b.fold_evals}
    c_fe_by_idx = {fe.meta.fold_idx: fe for fe in eval_c.fold_evals}

    pooled_y: list[np.ndarray] = []
    pooled_proba_b: list[np.ndarray] = []
    pooled_proba_c: list[np.ndarray] = []
    for idx in sorted(interior_fold_idxs):
        fe_b = b_fe_by_idx.get(idx)
        fe_c = c_fe_by_idx.get(idx)
        if (fe_b is not None and fe_b.y_true is not None
                and fe_c is not None and fe_c.proba is not None):
            pooled_y.append(fe_b.y_true)
            pooled_proba_b.append(fe_b.proba)
            pooled_proba_c.append(fe_c.proba)

    if pooled_y:
        y_pool = np.concatenate(pooled_y)
        auc_b_pool = float(roc_auc_score(y_pool, np.concatenate(pooled_proba_b)))
        auc_c_pool = float(roc_auc_score(y_pool, np.concatenate(pooled_proba_c)))
        fixable_leakage = auc_b_pool - auc_c_pool
    else:
        # Fallback: interior median (should not be reached in normal operation)
        interior_leakages = [pf.fold_leakage for pf in per_fold if not pf.is_terminal]
        leakages_for_median = interior_leakages if interior_leakages else fold_leakages
        fixable_leakage = float(np.median(leakages_for_median)) if leakages_for_median else 0.0

    feature_attribution = None
    if ablation_warn_floor is not None and fixable_leakage >= ablation_warn_floor:
        from zekan.severity.ablation import run_ablation
        feature_attribution = run_ablation(
            df, contract, baseline_auc=naive_auc, folds=temp_folds,
            model_factory=model_factory, n_jobs=n_jobs,
            all_features=all_features, X_all=X_all_full, y_all=y_all,
        )

    total_optimism = naive_auc - estimated_deployable_auc
    nonfixable_optimism = naive_auc - eval_b.mean_auc

    # ── Permutation null (optional) ────────────────────────────────────────────
    null_95th: Optional[float] = None
    null_99th: Optional[float] = None
    null_median: Optional[float] = None
    null_iqr: Optional[float] = None
    p_value: Optional[float] = None
    nsl: Optional[float] = None
    n_permutations_run = 0
    null_stopped_early = False
    p_is_upper_bound = False

    # Across-entity null (spec 1) — additive second channel; stays None unless it
    # actually runs (see the no-op guard in estimate_fixable_leakage_null).
    null_95th_across: Optional[float] = None
    null_99th_across: Optional[float] = None
    null_median_across: Optional[float] = None
    null_iqr_across: Optional[float] = None
    p_value_across: Optional[float] = None
    nsl_across: Optional[float] = None
    n_permutations_across = 0
    null_stopped_early_across = False
    p_is_upper_bound_across = False

    if n_permutations > 0:
        from zekan.severity.null_baseline import estimate_fixable_leakage_null
        from zekan.severity.metrics import _default_model_factory
        _null_factory = model_factory if model_factory is not None else _default_model_factory
        _null = estimate_fixable_leakage_null(
            df, contract, config, _null_factory,
            observed_fixable_leakage=fixable_leakage,
            n_permutations=n_permutations,
            seed=null_seed,
            method="within_entity",
            n_jobs=n_jobs,
            stopping=null_stopping,
        )
        null_95th = _null.null_95th
        null_99th = _null.null_99th
        null_median = _null.null_median
        null_iqr = _null.null_iqr
        p_value = _null.p_value
        n_permutations_run = _null.n_permutations
        null_stopped_early = _null.stopped_early
        p_is_upper_bound = _null.p_is_upper_bound
        # NSL denominator: IQR of null distribution (stable at N=100; the
        # 99th-percentile−median spread near the tail is noisy at N=100 because
        # q99 ≈ max, whose sampling variance dominates).
        _null_unit = max(_null.null_iqr, _NSL_EPS)
        nsl = (fixable_leakage - _null.null_99th) / _null_unit

        # Second null: across-entity permutation.  Shares n_permutations, null_seed,
        # and n_jobs with the within-entity null.  Both use the spawn_v2 seeding
        # scheme (F2a): np.random.SeedSequence(seed).spawn(n_permutations), so
        # results are identical regardless of n_jobs/scheduling — determinism no
        # longer depends on a single shared serial rng.  Uses ONLY the
        # method-agnostic live gate (_NULL_ALPHA, _NSL_NOTE_THRESHOLD) below —
        # deliberately does NOT route through power.py's estimate_n_required or its
        # 0.176/0.352 constants, which are within-entity-scoped by measurement (see
        # power.py docstring).  An across-entity power estimate would need its own
        # calibration sweep (not built here).
        _null_across = estimate_fixable_leakage_null(
            df, contract, config, _null_factory,
            observed_fixable_leakage=fixable_leakage,
            n_permutations=n_permutations,
            seed=null_seed,
            method="across_entity",
            n_jobs=n_jobs,
            stopping=null_stopping,
        )
        if _null_across.n_permutations > 0:
            null_95th_across = _null_across.null_95th
            null_99th_across = _null_across.null_99th
            null_median_across = _null_across.null_median
            null_iqr_across = _null_across.null_iqr
            p_value_across = _null_across.p_value
            n_permutations_across = _null_across.n_permutations
            null_stopped_early_across = _null_across.stopped_early
            p_is_upper_bound_across = _null_across.p_is_upper_bound
            _null_unit_across = max(_null_across.null_iqr, _NSL_EPS)
            nsl_across = (fixable_leakage - _null_across.null_99th) / _null_unit_across

    return SeverityResult(
        status=_status_from_optimism(total_optimism),
        metric="roc_auc",
        naive_auc=naive_auc,
        estimated_deployable_auc=estimated_deployable_auc,
        total_optimism=total_optimism,
        fixable_leakage=fixable_leakage,
        fixable_leakage_range=fixable_leakage_range,
        nonfixable_optimism=nonfixable_optimism,
        per_fold=per_fold,
        caveat=_CAVEAT,
        null_95th=null_95th,
        null_99th=null_99th,
        null_median=null_median,
        null_iqr=null_iqr,
        p_value=p_value,
        nsl=nsl,
        n_permutations_run=n_permutations_run,
        null_95th_across=null_95th_across,
        null_99th_across=null_99th_across,
        null_median_across=null_median_across,
        null_iqr_across=null_iqr_across,
        p_value_across=p_value_across,
        nsl_across=nsl_across,
        n_permutations_across=n_permutations_across,
        null_stopping=null_stopping,
        null_stopped_early=null_stopped_early,
        null_stopped_early_across=null_stopped_early_across,
        p_is_upper_bound=p_is_upper_bound,
        p_is_upper_bound_across=p_is_upper_bound_across,
        feature_attribution=feature_attribution,
        folds=temp_folds,
        all_features=all_features,
        X_all=X_all_full,
        y_all=y_all,
    )


# ── Null-standardized verdict helpers ────────────────────────────────────────

def _leakage_status(
    observed: float,
    p_value: Optional[float],
    nsl: Optional[float],
) -> str:
    """Null-standardized verdict for the TEMPORAL_LEAKAGE IssueRecord.

    Constant bands (0.02, 0.04) are GONE.  The verdict is entirely determined
    by the permutation null:

      Reality gate  p >= _NULL_ALPHA (0.01)  ->  PASS
                    p <  _NULL_ALPHA          ->  enter NSL ladder

      NSL ladder    NSL < _NSL_NOTE_THRESHOLD (1.0)  ->  NOTE
                    NSL < _NSL_WARN_THRESHOLD (2.0)  ->  WARN
                    NSL >= _NSL_WARN_THRESHOLD        ->  FAIL

    When the null was not run (p_value=None), returns 'unavailable'.
    This is intentionally separate from _status_from_optimism(), which gates
    on total_optimism (A-C) for the overall SeverityResult.status.
    """
    if p_value is None:
        return "unavailable"
    if p_value >= _NULL_ALPHA:
        return "pass"
    # Outside null: NSL grades effect size
    _nsl = nsl if nsl is not None else 0.0
    if _nsl < _NSL_NOTE_THRESHOLD:
        return "note"
    if _nsl < _NSL_WARN_THRESHOLD:
        return "warn"
    return "fail"


def leakage_issue_record(result: SeverityResult) -> "IssueRecord":
    """Convert a SeverityResult into an IssueRecord for TEMPORAL_LEAKAGE.

    Populates Evidence.null_95th, Evidence.p_value (reserved in v1).
    Status is fully null-standardized (see _leakage_status):
      pass / note / warn / fail / unavailable.

    'unavailable' is returned for two distinct reasons:
      1. SeverityResult.status == 'unavailable' (contract validation blocked the engine).
      2. n_permutations_run == 0 (null was not run; no null-standardized verdict possible).

    Example
    -------
    >>> result = run_severity_analysis(df, contract, config, clf, n_permutations=100)
    >>> issue = leakage_issue_record(result)
    >>> issue.status          # 'pass' / 'note' / 'warn' / 'fail' / 'unavailable'
    >>> issue.evidence.p_value
    """
    from zekan.detectors.schema import Evidence, IssueRecord, IssueType

    _why = (
        "A feature that encodes future information inflates measured AUC and "
        "causes deployed model performance to fall short of offline estimates."
    )

    if result.status == "unavailable":
        return IssueRecord(
            issue_type=IssueType.TEMPORAL_LEAKAGE,
            status="unavailable",
            what="Temporal leakage analysis could not be completed.",
            why=_why,
            how_much=result.unavailable_reason or "Contract validation failed.",
            next_fix="Resolve the contract validation errors listed above.",
            evidence=Evidence(metric_name="fixable_leakage"),
        )

    if result.n_permutations_run == 0:
        return IssueRecord(
            issue_type=IssueType.TEMPORAL_LEAKAGE,
            status="unavailable",
            what="No permutation null was run; a null-standardized verdict requires n_permutations >= 100.",
            why=_why,
            how_much=(
                f"fixable_leakage = {result.fixable_leakage:+.4f} (null not run; "
                "constant bands are no longer used — run with n_permutations=100)"
            ),
            next_fix="Rerun run_severity_analysis() with n_permutations=100 to get a verdict.",
            evidence=Evidence(
                measured_value=result.fixable_leakage,
                metric_name="fixable_leakage",
            ),
        )

    fl = result.fixable_leakage
    pv = result.p_value
    _nsl = result.nsl
    status = _leakage_status(fl, pv, _nsl)

    # ── how_much ──────────────────────────────────────────────────────────────
    if pv is not None and pv >= _NULL_ALPHA:
        null_str = f"p={pv:.4f} >= {_NULL_ALPHA} (inside null — PASS)"
    elif pv is not None:
        nsl_str = f"NSL={_nsl:.2f}" if _nsl is not None else "NSL=n/a"
        null_str = f"p={pv:.4f} < {_NULL_ALPHA} (outside null); {nsl_str}"
    else:
        null_str = "null not run"

    how_much = (
        f"fixable_leakage = {fl:+.4f}; "
        f"null_99th = {result.null_99th:+.4f}; "
        f"{null_str}"
        if result.null_99th is not None
        else f"fixable_leakage = {fl:+.4f}; {null_str}"
    )

    # ── narrative ─────────────────────────────────────────────────────────────
    if status == "pass":
        what = (
            "Fixable leakage is indistinguishable from the permutation null — "
            "no confirmed temporal leakage from the declared forbidden features."
        )
        next_fix = "No action required."

    elif status == "note":
        what = (
            "Temporal leakage is statistically confirmed (p < 0.01) but the "
            "effect size is small (NSL < 1.0) — the forbidden feature carries "
            "genuine future signal but the AUC impact is modest."
        )
        next_fix = (
            "Monitor for drift; consider removing or lagging the forbidden feature "
            "if operational constraints require it."
        )

    elif status == "warn":
        what = (
            "Temporal leakage confirmed with moderate effect size (1.0 <= NSL < 2.0); "
            "the forbidden feature meaningfully inflates temporal-CV AUC."
        )
        next_fix = (
            "Remove the forbidden feature from training, or replace it with a "
            "lagged version available at prediction time."
        )

    else:  # fail
        what = (
            "Temporal leakage confirmed with large effect size (NSL >= 2.0); "
            "the forbidden feature substantially inflates AUC in temporal cross-validation."
        )
        next_fix = (
            "Remove the forbidden feature from training immediately, or replace it "
            "with a lagged version that is genuinely available at prediction time."
        )

    return IssueRecord(
        issue_type=IssueType.TEMPORAL_LEAKAGE,
        status=status,
        what=what,
        why=_why,
        how_much=how_much,
        next_fix=next_fix,
        evidence=Evidence(
            measured_value=fl,
            threshold=result.null_99th,
            metric_name="fixable_leakage",
            null_95th=result.null_95th,
            p_value=pv,
        ),
    )
