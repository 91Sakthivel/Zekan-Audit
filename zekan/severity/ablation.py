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

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

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
    apportioned_leakage: float = float("nan")  # measured marginal given other leaks removed; NaN when not computed


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
    max_correlation_among_ablated: float = float("nan")  # max abs off-diagonal Pearson; NaN when <2 ablated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_factory() -> Any:
    # Tier 3 Phase C: default estimator is histgb (was rf). Single source of
    # truth is estimators.DEFAULT_ESTIMATOR_NAME -- see its docstring.
    from zekan.severity.estimators import DEFAULT_ESTIMATOR_NAME, _build_factory
    return _build_factory(DEFAULT_ESTIMATOR_NAME)()


def _univariate_auc(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    folds: list[FoldIndices],
    model_factory: Callable[[], Any],
    X_sub: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
) -> float:
    """Mean temporal AUC when model is trained on a single feature only.

    X_sub / y_all: pre-sliced feature/target arrays for just this one column,
    row-aligned to `df` (see run_ablation's X_all threading). When None (every
    pre-existing caller), evaluate_folds rebuilds them from df, unchanged.
    """
    try:
        result = evaluate_folds(df, [feature], target_col, folds, model_factory,
                                 X_all=X_sub, y_all=y_all)
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
    X_sub: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
) -> tuple:
    """Compute ablation for one feature; safe to call in a joblib worker.

    X_sub / y_all: pre-sliced (by the caller, in the SAME column order as
    features_minus_one below) feature/target arrays. When None (every
    pre-existing caller), evaluate_folds rebuilds them from df, unchanged.

    Returns (feature, auc_without, leakage_estimate, ablated, not_ablated_reason).
    Module-level so loky can pickle it by (module, qualname).
    """
    features_minus_one = [f for f in all_features if f != feature]
    result = evaluate_folds(df, features_minus_one, target_col, folds, model_factory,
                             X_all=X_sub, y_all=y_all)
    if result.n_valid_folds == 0:
        return (feature, float("nan"), float("nan"), False,
                "no valid folds after dropping feature")
    return (feature, result.mean_auc, baseline_auc - result.mean_auc, True, None)


def _marginal_one(
    feature: str,
    drop_all_set: list[str],
    cum_auc_without: float,
    df: pd.DataFrame,
    target_col: str,
    folds: list,
    model_factory: Callable[[], Any],
    X_sub: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
) -> tuple[str, float]:
    """Measure feature's marginal leakage given all other ablated features are already removed.

    feature_set_f = drop_all_set + [feature]  (everything dropped except this one)
    marginal = evaluate_folds(feature_set_f).mean_auc - cum_auc_without

    X_sub / y_all: pre-sliced arrays in feature_set_f's EXACT column order
    (drop_all_set's order, then `feature` appended last) -- this reorders
    columns relative to all_features, so X_sub must be built with a position
    list matching that exact order, never a boolean mask (column order is not
    interchangeable for every allowlisted estimator -- e.g. RandomForest's
    fitted predictions change under a column reorder even with a fixed
    random_state, since its per-split feature subsampling is index-based).
    When None (every pre-existing caller), evaluate_folds rebuilds from df.

    Returns (feature, marginal_leakage).  NaN when no valid folds.
    Module-level so loky can pickle it by (module, qualname).
    """
    feature_set_f = drop_all_set + [feature]
    result = evaluate_folds(df, feature_set_f, target_col, folds, model_factory,
                             X_all=X_sub, y_all=y_all)
    if result.n_valid_folds == 0:
        return (feature, float("nan"))
    return (feature, result.mean_auc - cum_auc_without)


# ── Public API ────────────────────────────────────────────────────────────────

def run_ablation(
    df: pd.DataFrame,
    contract: PredictionContract,
    baseline_auc: float,
    folds: list[FoldIndices],
    top_k: int = 10,
    model_factory: Optional[Callable[[], Any]] = None,
    n_jobs: int = 1,
    all_features: Optional[list[str]] = None,
    X_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
) -> AblationSummary:
    """Run per-feature ablation on forbidden candidates.

    Each candidate is ranked then ablated one-at-a-time (retrain_without).
    A cumulative ablation is always run when >= 2 features are ablated so that
    correlated leakage sources can be detected.

    all_features / X_all / y_all
        Efficiency hoist: when the caller (engine.run_severity_analysis)
        already built the float32 feature matrix once for eval_a/eval_b,
        `all_features` is that exact column-order list and `X_all` is that
        exact matrix (`y_all` the target array) -- every subset this function
        needs (single feature, minus-one, cumulative-drop, marginal) is then
        sliced from X_all by an explicit COLUMN POSITION LIST (never a
        boolean mask -- some subsets, e.g. the marginal path, reorder columns
        relative to all_features, and column order is not interchangeable for
        every allowlisted estimator: RandomForest's fitted predictions change
        under a column reorder even with a fixed random_state, since its
        per-split feature subsampling is index-based). When None (every
        pre-existing direct caller, e.g. tests), all_features is recomputed
        exactly as before and every evaluate_folds call rebuilds its own
        matrix from df, unchanged.
    """
    if model_factory is None:
        model_factory = _default_factory

    excluded = {
        contract.entity_id,
        contract.prediction_time,
        contract.available_features_until,
        contract.target,
    }
    if all_features is None:
        all_features = [c for c in df.columns if c not in excluded]
    forbidden_set = set(contract.forbidden_after_prediction) & set(df.columns)

    candidates = [f for f in all_features if f in forbidden_set]
    if not candidates:
        return AblationSummary(baseline_auc=baseline_auc)

    _col_pos = {f: i for i, f in enumerate(all_features)} if X_all is not None else {}

    def _slice(cols: list[str]) -> Optional[np.ndarray]:
        """Pre-slice X_all by an explicit column-POSITION list (never a
        boolean mask -- see run_ablation's docstring). Returns None when
        X_all wasn't provided, so evaluate_folds falls back to rebuilding
        from df exactly as before."""
        if X_all is None:
            return None
        return X_all[:, [_col_pos[c] for c in cols]]

    # Rank: contract-forbidden (always 1.0) + univariate AUC scaled 0..1 + name pattern
    entries: list[AblationEntry] = []
    for feat in candidates:
        uni = _univariate_auc(df, feat, contract.target, folds, model_factory,
                               X_sub=_slice([feat]), y_all=y_all)
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
                df, features_minus_one, contract.target, folds, model_factory,
                X_all=_slice(features_minus_one), y_all=y_all,
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
                X_sub=_slice([f for f in all_features if f != entry.feature]),
                y_all=y_all,
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
            df, features_drop_all, contract.target, folds, model_factory,
            X_all=_slice(features_drop_all), y_all=y_all,
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

    # ── Marginal leakage (Upgrade G) ─────────────────────────────────────────
    # For each ablated feature, measure its contribution GIVEN the other ablated
    # features are already removed (start from the fully-cleaned baseline).
    # single-feature: apportioned == leakage_estimate by definition, no extra retrain.
    # multi-feature: requires the cumulative block's drop_all_set and auc_without.
    max_corr: float = float("nan")

    if len(ablated) == 1:
        ablated[0].apportioned_leakage = ablated[0].leakage_estimate

    elif len(ablated) >= 2 and cumulative is not None and not math.isnan(cumulative.auc_without):
        drop_all_set = [f for f in all_features if f not in {e.feature for e in ablated}]
        cum_auc = cumulative.auc_without

        # Max absolute off-diagonal Pearson — explanatory metadata, no threshold.
        ablated_cols = [e.feature for e in ablated]
        corr_vals = df[ablated_cols].corr().to_numpy()
        n_abl = len(ablated_cols)
        off_diag = [abs(corr_vals[i, j]) for i in range(n_abl) for j in range(n_abl) if i != j]
        if off_diag:
            max_corr = float(max(off_diag))

        if n_jobs == 1:
            for entry in ablated:
                feature_set_f = drop_all_set + [entry.feature]
                r = evaluate_folds(df, feature_set_f, contract.target, folds, model_factory,
                                    X_all=_slice(feature_set_f), y_all=y_all)
                entry.apportioned_leakage = (
                    r.mean_auc - cum_auc if r.n_valid_folds > 0 else float("nan")
                )
        else:
            from joblib import Parallel, delayed  # noqa: PLC0415
            raw_m = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_marginal_one)(
                    entry.feature, drop_all_set, cum_auc,
                    df, contract.target, folds, model_factory,
                    X_sub=_slice(drop_all_set + [entry.feature]),
                    y_all=y_all,
                )
                for entry in ablated
            )
            by_feat_m: dict[str, float] = {feat: marg for feat, marg in raw_m}
            for entry in ablated:
                entry.apportioned_leakage = by_feat_m[entry.feature]

    return AblationSummary(
        baseline_auc=baseline_auc,
        individual=entries,
        cumulative=cumulative,
        one_at_a_time_understates=understates,
        understatement_warning=warning,
        max_correlation_among_ablated=max_corr,
    )
