"""Matched evaluation harness: identical code path for both split protocols.

Swapping the fold iterator is the only difference between random-grouped
and temporal evaluation — this is what makes the comparison valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
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
    # Tier 3 Phase C: default estimator is histgb (was rf). Single source of
    # truth is estimators.DEFAULT_ESTIMATOR_NAME -- see its docstring.
    from zekan.severity.estimators import DEFAULT_ESTIMATOR_NAME, _build_factory
    return _build_factory(DEFAULT_ESTIMATOR_NAME)()


def _feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_map: Optional[dict[str, dict]] = None,
    unseen_counts: Optional[dict[str, int]] = None,
) -> np.ndarray:
    """Cast feature columns to a single float32 array, once.

    float32 because sklearn's tree-based estimators internally convert X to
    float32 on every fit/predict call (via check_array) regardless of the
    input dtype -- casting once here, instead of passing float64 and letting
    sklearn downcast it repeatedly, removes a redundant per-fit conversion.
    It does not change the values the trees split on: float64->float32
    narrowing is a deterministic, element-wise operation, so casting once up
    front and casting on every internal sklearn call produce bit-identical
    float32 values either way.

    Defense in depth: contract_checks._check_feature_columns_numeric should
    already have rejected any non-numeric feature column before this runs.
    This guard exists for callers that reach evaluate_folds (or this helper)
    directly, bypassing that gate -- it converts a raw pandas cast error into
    a clear, typed message instead of letting an internal exception type leak
    to the user.

    categorical_map
        Optional {column: {raw_value: code}} mapping, built once by
        contract_checks.build_categorical_mapping and passed in here rather
        than derived from `df` on every call -- this function only APPLIES a
        given map (contract_checks.apply_categorical_mapping), it never
        builds one, so calling it repeatedly (e.g. once from engine.py's
        X_all_full and again from null_baseline.py's X_base within the same
        audit) never redoes the sorted-unique derivation. Default None
        preserves every existing caller's behavior exactly -- step 3 wires
        engine.py's own call to build and pass a real map.
    unseen_counts
        Passed straight through to apply_categorical_mapping (see its own
        unseen_counts docstring) when categorical_map is given. Ignored
        (nothing to count) when categorical_map is empty/None.
    """
    if categorical_map:
        from zekan.contract.contract_checks import apply_categorical_mapping
        df = apply_categorical_mapping(df, categorical_map, unseen_counts=unseen_counts)
    try:
        return df[feature_cols].to_numpy(dtype=np.float32)
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


def evaluate_folds(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: list[FoldIndices],
    model_factory: Optional[Callable[[], Any]] = None,
    return_predictions: bool = False,
    X_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    categorical_map: Optional[dict[str, dict]] = None,
) -> EvaluationResult:
    """Evaluate a model across pre-built folds and return per-fold + mean AUC.

    model_factory is called fresh for every valid fold — equivalent to clone().
    Skipped folds are excluded from the mean. The code path is identical
    regardless of whether folds came from random_grouped_folds or
    temporal_expanding_folds; only the index arrays differ.

    Defense in depth: contract_checks._check_feature_columns_numeric should
    already have rejected any non-numeric feature column before this runs.
    _feature_matrix's try/except is a should-never-happen guard for callers
    that reach evaluate_folds directly, bypassing that gate -- it converts a
    raw pandas cast error into a clear, typed message instead of letting an
    internal exception type leak to the user.

    X_all / y_all
        Pre-built feature/target arrays, row-aligned to `df` (X_all's columns
        must match `feature_cols`'s order). When given, evaluate_folds skips
        rebuilding them from `df` -- lets a caller that already has the
        matrix (e.g. the permutation null, which reuses the same base matrix
        across every draw, patching only the permuted forbidden column(s))
        avoid re-deriving it on every call. When None (the default, used by
        every pre-existing caller), built here exactly as before, once per
        call instead of once per fold.
    categorical_map
        Passed straight through to _feature_matrix when X_all is None (see
        that function's docstring). Ignored when X_all is given -- the
        caller already decided what encoding, if any, went into it.
    """
    if model_factory is None:
        model_factory = _default_model_factory

    if X_all is None:
        X_all = _feature_matrix(df, feature_cols, categorical_map=categorical_map)
    if y_all is None:
        y_all = df[target_col].to_numpy()

    fold_evals: list[FoldEval] = []
    n_skipped = sum(1 for f in folds if f.meta.skipped)

    for fold in folds:
        if fold.meta.skipped:
            continue

        X_train = X_all[fold.train_idx]
        X_test = X_all[fold.test_idx]
        y_train = y_all[fold.train_idx]
        y_test = y_all[fold.test_idx]

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
