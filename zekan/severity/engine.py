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
from zekan.severity.metrics import evaluate_folds
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
    feature_attribution: Optional[Any] = None     # AblationSummary when ablation ran; None otherwise
    folds: list = field(default_factory=list)     # temporal FoldIndices used for B/C eval; internal only


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
) -> SeverityResult:
    """Run the A/B/C performance decomposition and return a SeverityResult.

    Returns status='unavailable' when contract validation blocks computation.
    B and C share the same temporal fold objects so per-fold differences cancel
    fold-level variance exactly.

    n_permutations
        When > 0, runs the within-entity permutation null (method a) after computing
        fixable_leakage and stores null_95th, p_value, n_permutations_run on the result.
        Default 0 skips the null (backward-compatible).
    null_seed
        RNG seed for the permutation null.  Different seeds give convergent null_95th
        at n_permutations >= 100.
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
    eval_a = evaluate_folds(df, all_features, contract.target, rand_folds, model_factory)
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
        return_predictions=True,
    )

    # ── C: temporal CV, forbidden features dropped and retrained ──────────────
    eval_c = evaluate_folds(
        df, safe_features, contract.target, temp_folds, model_factory,
        return_predictions=True,
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
            model_factory=model_factory,
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
        )
        null_95th = _null.null_95th
        null_99th = _null.null_99th
        null_median = _null.null_median
        null_iqr = _null.null_iqr
        p_value = _null.p_value
        n_permutations_run = _null.n_permutations
        # NSL denominator: IQR of null distribution (stable at N=100; the
        # 99th-percentile−median spread near the tail is noisy at N=100 because
        # q99 ≈ max, whose sampling variance dominates).
        _null_unit = max(_null.null_iqr, _NSL_EPS)
        nsl = (fixable_leakage - _null.null_99th) / _null_unit

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
        feature_attribution=feature_attribution,
        folds=temp_folds,
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
