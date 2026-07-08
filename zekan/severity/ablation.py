"""Per-issue ranked ablation for forbidden feature candidates.

Ranking criteria (in priority order):
  1. Contract-declared forbidden status (boolean flag)
  2. Univariate AUC association with target on temporal folds
  3. Suspicious name-pattern score (final_*, days_to_*, future_*, *_after_*)

Ablation method: retrain_without -- drop the feature and retrain on same folds.
Only top_k candidates are ablated; the rest are marked not_ablated with a reason.

Correlated leaks: when two features share the same underlying signal, removing
either one alone barely changes AUC (the other compensates).  A cumulative
ablation drops ALL ablated features at once; if cumulative leakage substantially
exceeds the max individual leakage, a warning is issued.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.metrics import evaluate_folds
from zekan.severity.splitters import FoldIndices


_SUSPICIOUS_PATTERNS = re.compile(
    r"(?:^(?:final_|days_to_|future_|next_))"
    r"|(?:(?:_after_|_future|_next_period|_lag0|_t0)$)",
    re.IGNORECASE,
)


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class AblationEntry:
    """Result of ablating one forbidden feature."""

    feature: str
    is_contract_forbidden: bool
    univariate_auc: float
    name_pattern_score: float           # 0.0 or 1.0
    rank_score: float                   # composite; higher = ablate first
    auc_without: float                  # NaN when not ablated
    leakage_estimate: float             # baseline_auc - auc_without; NaN when not ablated
    ablated: bool
    not_ablated_reason: Optional[str] = None


@dataclass
class CumulativeAblation:
    """Result of dropping all ablated features simultaneously."""

    features_dropped: list[str]
    auc_without: float
    cumulative_leakage: float


@dataclass
class AblationSummary:
    """Full ablation report for a set of forbidden candidates."""

    baseline_auc: float
    individual: list[AblationEntry] = field(default_factory=list)
    cumulative: Optional[CumulativeAblation] = None
    one_at_a_time_understates: bool = False
    understatement_warning: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_factory() -> Any:
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)


def _univariate_auc(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    folds: list[FoldIndices],
    model_factory: Callable[[], Any],
) -> float:
    """Mean temporal AUC when model is trained on a single feature only."""
    try:
        result = evaluate_folds(df, [feature], target_col, folds, model_factory)
        return result.mean_auc if result.n_valid_folds > 0 else 0.5
    except Exception:
        return 0.5


def _name_score(name: str) -> float:
    return 1.0 if _SUSPICIOUS_PATTERNS.search(name) else 0.0


def _ablate_one(
    feature: str,
    all_features: list[str],
    df: pd.DataFrame,
    target_col: str,
    folds: list,
    model_factory: Callable[[], Any],
    baseline_auc: float,
) -> tuple:
    """Compute ablation for one feature; safe to call in a joblib worker.

    Returns (feature, auc_without, leakage_estimate, ablated, not_ablated_reason).
    Module-level so loky can pickle it by (module, qualname).
    """
    features_minus_one = [f for f in all_features if f != feature]
    result = evaluate_folds(df, features_minus_one, target_col, folds, model_factory)
    if result.n_valid_folds == 0:
        return (feature, float("nan"), float("nan"), False,
                "no valid folds after dropping feature")
    return (feature, result.mean_auc, baseline_auc - result.mean_auc, True, None)


# ── Public API ────────────────────────────────────────────────────────────────

def run_ablation(
    df: pd.DataFrame,
    contract: PredictionContract,
    baseline_auc: float,
    folds: list[FoldIndices],
    top_k: int = 10,
    model_factory: Optional[Callable[[], Any]] = None,
    n_jobs: int = 1,
) -> AblationSummary:
    """Run per-feature ablation on forbidden candidates.

    Each candidate is ranked then ablated one-at-a-time (retrain_without).
    A cumulative ablation is always run when >= 2 features are ablated so that
    correlated leakage sources can be detected.
    """
    if model_factory is None:
        model_factory = _default_factory

    excluded = {
        contract.entity_id,
        contract.prediction_time,
        contract.available_features_until,
        contract.target,
    }
    all_features = [c for c in df.columns if c not in excluded]
    forbidden_set = set(contract.forbidden_after_prediction) & set(df.columns)

    candidates = [f for f in all_features if f in forbidden_set]
    if not candidates:
        return AblationSummary(baseline_auc=baseline_auc)

    # Rank: contract-forbidden (always 1.0) + univariate AUC scaled 0..1 + name pattern
    entries: list[AblationEntry] = []
    for feat in candidates:
        uni = _univariate_auc(df, feat, contract.target, folds, model_factory)
        name = _name_score(feat)
        rank = 1.0 + (uni - 0.5) * 2.0 + name * 0.5
        entries.append(AblationEntry(
            feature=feat,
            is_contract_forbidden=True,
            univariate_auc=uni,
            name_pattern_score=name,
            rank_score=rank,
            auc_without=float("nan"),
            leakage_estimate=float("nan"),
            ablated=False,
        ))

    entries.sort(key=lambda e: e.rank_score, reverse=True)

    # Mark budget-exceeded entries (beyond top_k) — no computation needed.
    to_ablate = entries[:top_k]
    for entry in entries[top_k:]:
        entry.not_ablated_reason = f"budget exceeded (top_k={top_k})"

    # One-at-a-time ablation: serial (n_jobs=1) or parallel (n_jobs != 1).
    if n_jobs == 1:
        for entry in to_ablate:
            features_minus_one = [f for f in all_features if f != entry.feature]
            result = evaluate_folds(
                df, features_minus_one, contract.target, folds, model_factory
            )
            if result.n_valid_folds == 0:
                entry.not_ablated_reason = "no valid folds after dropping feature"
                continue
            entry.auc_without = result.mean_auc
            entry.leakage_estimate = baseline_auc - result.mean_auc
            entry.ablated = True
    else:
        from joblib import Parallel, delayed  # noqa: PLC0415
        raw = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_ablate_one)(
                entry.feature, all_features, df, contract.target, folds,
                model_factory, baseline_auc,
            )
            for entry in to_ablate
        )
        # Write results back by feature name — independent of completion order.
        by_feat: dict[str, tuple] = {
            feat: (auc_w, leak, abl, reason)
            for feat, auc_w, leak, abl, reason in raw
        }
        for entry in to_ablate:
            auc_w, leak, abl, reason = by_feat[entry.feature]
            entry.auc_without = auc_w
            entry.leakage_estimate = leak
            entry.ablated = abl
            if reason is not None:
                entry.not_ablated_reason = reason

    # Cumulative ablation: drop ALL ablated features simultaneously
    ablated = [e for e in entries if e.ablated]
    cumulative: Optional[CumulativeAblation] = None
    understates = False
    warning: Optional[str] = None

    if len(ablated) >= 2:
        drop_set = {e.feature for e in ablated}
        features_drop_all = [f for f in all_features if f not in drop_set]
        cum_result = evaluate_folds(
            df, features_drop_all, contract.target, folds, model_factory
        )
        cum_leakage = baseline_auc - cum_result.mean_auc
        cumulative = CumulativeAblation(
            features_dropped=[e.feature for e in ablated],
            auc_without=cum_result.mean_auc,
            cumulative_leakage=cum_leakage,
        )

        max_individual = max(e.leakage_estimate for e in ablated)
        # Correlated leaks: cumulative leakage >> max individual
        if cum_leakage > max_individual + 0.02 and cum_leakage > max_individual * 1.3:
            understates = True
            warning = (
                f"Cumulative leakage ({cum_leakage:.3f}) substantially exceeds "
                f"max individual leakage ({max_individual:.3f}). "
                "Features share a common underlying signal; "
                "one-at-a-time ablation understates true leakage."
            )

    return AblationSummary(
        baseline_auc=baseline_auc,
        individual=entries,
        cumulative=cumulative,
        one_at_a_time_understates=understates,
        understatement_warning=warning,
    )
