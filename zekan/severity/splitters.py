"""Temporal and grouped cross-validation splitters for the zekan severity engine.

Guiding principle: Temporal same-entity overlap is NOT automatically leakage;
random same-entity overlap usually IS. So the random baseline groups by entity
to prevent memorization, while temporal folds split purely by time and allow
the same entity across train/test at different times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FoldMeta:
    """Per-fold diagnostics returned alongside the index arrays."""

    fold_idx: int
    strategy: str
    train_rows: int
    test_rows: int
    train_base_rate: float
    test_base_rate: float
    # Entity overlap: always 0 for grouped CV; non-zero is expected (not a failure) for temporal.
    entity_overlap_count: int
    entity_overlap_pct: float
    # Time ranges: populated for temporal folds, None for grouped CV.
    train_time_min: Optional[str] = None
    train_time_max: Optional[str] = None
    test_time_min: Optional[str] = None
    test_time_max: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class FoldIndices:
    """Index arrays plus metadata for one fold."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: FoldMeta


# ── Shared helpers ────────────────────────────────────────────────────────────

def _skip_reason(
    test_idx: np.ndarray,
    df: pd.DataFrame,
    target_col: str,
    min_rows: int,
    min_pos: int,
    min_neg: int,
) -> Optional[str]:
    """Return a human-readable reason string if this fold should be skipped, else None."""
    n = len(test_idx)
    y_test = df.iloc[test_idx][target_col]
    n_pos = int(y_test.sum())
    n_neg = n - n_pos
    reasons = []
    if n < min_rows:
        reasons.append(f"test_rows={n} < {min_rows}")
    if n_pos < min_pos:
        reasons.append(f"test_positives={n_pos} < {min_pos}")
    if n_neg < min_neg:
        reasons.append(f"test_negatives={n_neg} < {min_neg}")
    return "; ".join(reasons) if reasons else None


# ── Splitters ─────────────────────────────────────────────────────────────────

def random_grouped_folds(
    df: pd.DataFrame,
    entity_col: str,
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
    min_test_rows: int = 100,
    min_pos: int = 20,
    min_neg: int = 20,
) -> list[FoldIndices]:
    """Stratified group k-fold: groups on entity, stratified on target.

    The same entity never appears in both train and test within a fold.
    This is the random baseline; it prevents entity-level memorization but
    ignores temporal ordering entirely.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(df), 1))
    y = df[target_col].values
    groups = df[entity_col].values

    folds: list[FoldIndices] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_dummy, y, groups)
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_entities = set(train_df[entity_col])
        test_entities = set(test_df[entity_col])
        overlap = train_entities & test_entities  # expected to be empty

        meta = FoldMeta(
            fold_idx=fold_idx,
            strategy="grouped_cv",
            train_rows=len(train_idx),
            test_rows=len(test_idx),
            train_base_rate=float(train_df[target_col].mean()),
            test_base_rate=float(test_df[target_col].mean()),
            entity_overlap_count=len(overlap),
            entity_overlap_pct=len(overlap) / max(1, len(test_entities)) * 100.0,
        )

        reason = _skip_reason(test_idx, df, target_col, min_test_rows, min_pos, min_neg)
        if reason:
            meta.skipped = True
            meta.skip_reason = reason

        folds.append(FoldIndices(train_idx=train_idx, test_idx=test_idx, meta=meta))

    return folds


def temporal_expanding_folds(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    target_col: str,
    n_splits: int = 5,
    min_test_rows: int = 100,
    min_pos: int = 20,
    min_neg: int = 20,
) -> list[FoldIndices]:
    """Rolling-origin expanding-window temporal split.

    Each fold trains on ALL rows strictly before the cut, tests on the next
    time block. The same entity is allowed in both train and test at different
    times — this is not leakage. Entity-overlap % is computed and attached to
    metadata as an informational diagnostic, not a failure criterion.

    Defense in depth: contract_checks._check_prediction_time should already
    have rejected an unparseable time_col before this runs. The try/except
    below exists for callers that reach this splitter directly, bypassing
    that gate -- it converts a raw pandas parse error into a clear, typed
    message instead of letting an internal exception type leak to the user.
    """
    try:
        times = pd.to_datetime(df[time_col])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"prediction_time column '{time_col}' could not be parsed as a time signal "
            f"({type(e).__name__}: {e}). Zekan audits leakage over time and needs an "
            f"ordered, parseable date/time column -- provide a real date/time column, or "
            f"derive one (e.g. an ordinal period column) from your data."
        ) from e
    sorted_periods: list = sorted(times.unique())
    n_periods = len(sorted_periods)

    if n_periods < 2:
        raise ValueError(
            f"temporal_expanding_folds requires >= 2 distinct time periods; got {n_periods}"
        )

    # Distribute n_periods across n_splits+1 blocks using n_splits+2 edge points.
    # np.unique removes duplicate edges that appear when n_periods < n_splits+1.
    raw_edges = np.round(np.linspace(0, n_periods, n_splits + 2)).astype(int)
    edges = list(map(int, np.unique(raw_edges)))

    folds: list[FoldIndices] = []
    fold_idx = 0

    for i in range(len(edges) - 2):
        train_end = edges[i + 1]   # exclusive index into sorted_periods
        test_start = edges[i + 1]
        test_end = edges[i + 2]

        if train_end == 0 or test_start >= n_periods or test_end > n_periods:
            continue

        train_periods = set(sorted_periods[:train_end])
        test_periods = set(sorted_periods[test_start:test_end])

        if not train_periods or not test_periods:
            continue

        train_mask = times.isin(train_periods)
        test_mask = times.isin(test_periods)
        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_entities = set(train_df[entity_col])
        test_entities = set(test_df[entity_col])
        overlap = train_entities & test_entities

        def _fmt(ts: object) -> str:
            return pd.Timestamp(ts).strftime("%Y-%m-%d")

        meta = FoldMeta(
            fold_idx=fold_idx,
            strategy="temporal_expanding",
            train_rows=len(train_idx),
            test_rows=len(test_idx),
            train_base_rate=float(train_df[target_col].mean()),
            test_base_rate=float(test_df[target_col].mean()),
            entity_overlap_count=len(overlap),
            entity_overlap_pct=len(overlap) / max(1, len(test_entities)) * 100.0,
            train_time_min=_fmt(min(train_periods)),
            train_time_max=_fmt(max(train_periods)),
            test_time_min=_fmt(min(test_periods)),
            test_time_max=_fmt(max(test_periods)),
        )

        reason = _skip_reason(test_idx, df, target_col, min_test_rows, min_pos, min_neg)
        if reason:
            meta.skipped = True
            meta.skip_reason = reason

        folds.append(FoldIndices(train_idx=train_idx, test_idx=test_idx, meta=meta))
        fold_idx += 1

    return folds
