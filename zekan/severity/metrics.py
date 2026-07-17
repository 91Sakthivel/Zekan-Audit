"""Matched evaluation harness: identical code path for both split protocols.

Swapping the fold iterator is the only difference between random-grouped
and temporal evaluation — this is what makes the comparison valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from zekan.severity.splitters import FoldIndices, FoldMeta


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class FoldEval:
    """AUC plus fold metadata for one evaluated fold."""

    meta: FoldMeta
    auc: float
    y_true: Optional[np.ndarray] = None   # populated only when return_predictions=True
    proba: Optional[np.ndarray] = None    # predicted probabilities for class 1


@dataclass
class EvaluationResult:
    """Aggregate result from running evaluate_folds across a fold list."""

    mean_auc: float
    fold_evals: list[FoldEval] = field(default_factory=list)
    n_valid_folds: int = 0
    n_skipped_folds: int = 0

    @property
    def fold_aucs(self) -> list[float]:
        return [fe.auc for fe in self.fold_evals]


# ── Harness ───────────────────────────────────────────────────────────────────

def _default_model_factory() -> Any:
    return RandomForestClassifier(n_estimators=200, random_state=42)


def evaluate_folds(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: list[FoldIndices],
    model_factory: Optional[Callable[[], Any]] = None,
    return_predictions: bool = False,
) -> EvaluationResult:
    """Evaluate a model across pre-built folds and return per-fold + mean AUC.

    model_factory is called fresh for every valid fold — equivalent to clone().
    Skipped folds are excluded from the mean. The code path is identical
    regardless of whether folds came from random_grouped_folds or
    temporal_expanding_folds; only the index arrays differ.

    Defense in depth: contract_checks._check_feature_columns_numeric should
    already have rejected any non-numeric feature column before this runs.
    The try/except below is a should-never-happen guard for callers that reach
    evaluate_folds directly, bypassing that gate -- it converts a raw pandas
    cast error into a clear, typed message instead of letting an internal
    exception type leak to the user.
    """
    if model_factory is None:
        model_factory = _default_model_factory

    fold_evals: list[FoldEval] = []
    n_skipped = sum(1 for f in folds if f.meta.skipped)

    for fold in folds:
        if fold.meta.skipped:
            continue

        try:
            X_train = df.iloc[fold.train_idx][feature_cols].to_numpy(dtype=float)
            X_test = df.iloc[fold.test_idx][feature_cols].to_numpy(dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not cast feature columns to numeric ({type(e).__name__}: {e}). "
                f"This should never happen -- contract_checks._check_feature_columns_numeric "
                f"is the intended gate for this and should have caught it before evaluate_folds "
                f"ever ran. If you're calling evaluate_folds directly, bypassing contract "
                f"validation, run validate_contract first. Zekan's models require every "
                f"feature column to be numeric -- encode categorical columns (e.g. ordinal "
                f"or one-hot encoding) before auditing this data."
            ) from e
        y_train = df.iloc[fold.train_idx][target_col].to_numpy()
        y_test = df.iloc[fold.test_idx][target_col].to_numpy()

        estimator = model_factory()
        estimator.fit(X_train, y_train)
        probs = estimator.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, probs))

        fe = FoldEval(meta=fold.meta, auc=auc)
        if return_predictions:
            fe.y_true = y_test
            fe.proba = probs
        fold_evals.append(fe)

    aucs = [fe.auc for fe in fold_evals]
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    return EvaluationResult(
        mean_auc=mean_auc,
        fold_evals=fold_evals,
        n_valid_folds=len(fold_evals),
        n_skipped_folds=n_skipped,
    )
