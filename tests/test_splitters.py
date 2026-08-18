"""Tests for splitters, evaluation harness, synthetic fixtures, and injectors (Phase 2a)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import (
    inject_covariate_drift,
    inject_concept_drift,
    inject_correlated_leaks,
    inject_future_feature,
    inject_label_proxy,
    inject_presplit_artifact,
)
from zekan.severity.metrics import compute_fold_active_positions, evaluate_folds
from zekan.severity.splitters import (
    FoldIndices,
    FoldMeta,
    build_period_rank,
    random_grouped_folds,
    temporal_expanding_folds,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """200 entities x 10 periods = 2 000 rows — well above fold minimums."""
    return make_clean_dataset(n_entities=200, snapshots_per_entity=5, seed=99)


@pytest.fixture(scope="module")
def rand_folds(clean_df: pd.DataFrame) -> list[FoldIndices]:
    return random_grouped_folds(
        clean_df, entity_col="entity_id", target_col="target", n_splits=5
    )


@pytest.fixture(scope="module")
def temp_folds(clean_df: pd.DataFrame) -> list[FoldIndices]:
    return temporal_expanding_folds(
        clean_df,
        time_col="prediction_time",
        entity_col="entity_id",
        target_col="target",
        n_splits=5,
    )


def _fast_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


# ── Random grouped splitter ───────────────────────────────────────────────────

def test_grouped_no_entity_overlap(
    clean_df: pd.DataFrame, rand_folds: list[FoldIndices]
) -> None:
    """Grouped CV must never put the same entity in both train and test."""
    for fold in rand_folds:
        if fold.meta.skipped:
            continue
        train_entities = set(clean_df.iloc[fold.train_idx]["entity_id"])
        test_entities = set(clean_df.iloc[fold.test_idx]["entity_id"])
        assert len(train_entities & test_entities) == 0
        assert fold.meta.entity_overlap_count == 0
        assert fold.meta.entity_overlap_pct == 0.0


def test_grouped_class_balance_preserved(rand_folds: list[FoldIndices]) -> None:
    """Stratification should keep per-fold base rates within 10 pp of 0.3."""
    for fold in rand_folds:
        if fold.meta.skipped:
            continue
        assert abs(fold.meta.train_base_rate - 0.3) < 0.10
        assert abs(fold.meta.test_base_rate - 0.3) < 0.10


# ── Temporal splitter ─────────────────────────────────────────────────────────

def test_temporal_test_is_strictly_after_train(temp_folds: list[FoldIndices]) -> None:
    """Every test window must start AFTER the training window ends."""
    valid = [f for f in temp_folds if not f.meta.skipped]
    assert valid, "No valid temporal folds produced"
    for fold in valid:
        assert fold.meta.train_time_max is not None
        assert fold.meta.test_time_min is not None
        # ISO dates compare correctly as strings
        assert fold.meta.train_time_max < fold.meta.test_time_min, (
            f"fold {fold.meta.fold_idx}: train_max={fold.meta.train_time_max} "
            f">= test_min={fold.meta.test_time_min}"
        )


def test_temporal_entity_overlap_diagnostics_populated(
    temp_folds: list[FoldIndices],
) -> None:
    """Entity-overlap diagnostics must be present and non-negative for temporal folds."""
    valid = [f for f in temp_folds if not f.meta.skipped]
    for fold in valid:
        assert fold.meta.entity_overlap_count >= 0
        assert 0.0 <= fold.meta.entity_overlap_pct <= 100.0


def test_temporal_entity_overlap_is_nonzero(
    clean_df: pd.DataFrame, temp_folds: list[FoldIndices]
) -> None:
    """With all entities present at all time periods, some overlap is expected."""
    valid = [f for f in temp_folds if not f.meta.skipped]
    # At least one fold should show overlap (same entities in train+test at diff times)
    assert any(f.meta.entity_overlap_count > 0 for f in valid)


# ── Fold metadata completeness ────────────────────────────────────────────────

def test_fold_metadata_complete(
    rand_folds: list[FoldIndices], temp_folds: list[FoldIndices]
) -> None:
    """All metadata fields must be populated for every non-skipped fold."""
    for fold in rand_folds:
        if fold.meta.skipped:
            continue
        assert fold.meta.train_rows > 0
        assert fold.meta.test_rows > 0
        assert 0.0 <= fold.meta.train_base_rate <= 1.0
        assert 0.0 <= fold.meta.test_base_rate <= 1.0

    for fold in temp_folds:
        if fold.meta.skipped:
            continue
        assert fold.meta.train_time_min is not None
        assert fold.meta.train_time_max is not None
        assert fold.meta.test_time_min is not None
        assert fold.meta.test_time_max is not None
        assert fold.meta.train_rows > 0
        assert fold.meta.test_rows > 0


# ── Evaluation harness ────────────────────────────────────────────────────────

def test_harness_both_protocols_auc_in_range(
    clean_df: pd.DataFrame,
    rand_folds: list[FoldIndices],
    temp_folds: list[FoldIndices],
) -> None:
    """Both protocols must produce a mean AUC strictly in (0.5, 1.0) on clean data."""
    feature_cols = [c for c in clean_df.columns if c.startswith("feature_")]

    rand_result = evaluate_folds(
        clean_df, feature_cols, "target", rand_folds, model_factory=_fast_clf
    )
    temp_result = evaluate_folds(
        clean_df, feature_cols, "target", temp_folds, model_factory=_fast_clf
    )

    assert rand_result.n_valid_folds >= 1
    assert temp_result.n_valid_folds >= 1
    assert 0.5 < rand_result.mean_auc < 1.0, f"rand mean AUC = {rand_result.mean_auc}"
    assert 0.5 < temp_result.mean_auc < 1.0, f"temp mean AUC = {temp_result.mean_auc}"


# ── Injectors ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=7)


def test_inject_label_proxy_adds_leak_column(small_df: pd.DataFrame) -> None:
    modified, record = inject_label_proxy(small_df)
    assert record.is_leak
    assert record.planted_columns
    assert all(c in modified.columns for c in record.planted_columns)
    # Verify content: 5% flip-rate gives corr ≈ 0.89 with target.
    # Measured min across 10 seeds = 0.8892; gate = 0.85 gives +0.039 margin.
    col = record.planted_columns[0]
    corr = float(modified[col].corr(modified["target"]))
    assert corr > 0.85, (
        f"label_proxy corr with target={corr:.4f}, expected >0.85 "
        "(5% flip-rate should produce corr ~0.89)"
    )


def test_inject_future_feature_adds_leak_column(small_df: pd.DataFrame) -> None:
    modified, record = inject_future_feature(small_df)
    assert record.is_leak
    assert all(c in modified.columns for c in record.planted_columns)


def test_inject_presplit_artifact_adds_leak_column(small_df: pd.DataFrame) -> None:
    modified, record = inject_presplit_artifact(small_df)
    assert record.is_leak
    assert all(c in modified.columns for c in record.planted_columns)
    # Verify content: entity-mean target encoding must correlate with individual targets.
    # Measured min across 10 seeds = 0.6241; gate = 0.55 gives +0.074 margin.
    col = record.planted_columns[0]
    corr = float(modified[col].corr(modified["target"]))
    assert corr > 0.55, (
        f"presplit_artifact corr with target={corr:.4f}, expected >0.55 "
        "(entity target-mean encoding should correlate ~0.62–0.72 with individual targets)"
    )


def test_inject_correlated_leaks_adds_two_columns(small_df: pd.DataFrame) -> None:
    modified, record = inject_correlated_leaks(small_df)
    assert record.is_leak
    assert len(record.planted_columns) == 2
    assert all(c in modified.columns for c in record.planted_columns)
    # Both columns should be highly correlated (design target corr ~ 0.90 via
    # sigma_noise = sigma_z/3). Assert >= 0.80 for sampling headroom on small_df.
    col_a, col_b = record.planted_columns
    corr = float(modified[col_a].corr(modified[col_b]))
    assert corr > 0.80


def test_inject_covariate_drift_not_a_leak(small_df: pd.DataFrame) -> None:
    modified, record = inject_covariate_drift(small_df)
    assert not record.is_leak
    assert all(c in modified.columns for c in record.planted_columns)


def test_inject_concept_drift_not_a_leak(small_df: pd.DataFrame) -> None:
    modified, record = inject_concept_drift(small_df)
    assert not record.is_leak
    assert all(c in modified.columns for c in record.planted_columns)


def test_inject_concept_drift_early_positive_late_negative(small_df: pd.DataFrame) -> None:
    """concept_drift_feat must be positively correlated with target in early periods
    and negatively correlated in late periods — confirming genuine relationship reversal."""
    modified, record = inject_concept_drift(small_df)
    col = record.planted_columns[0]

    times = modified["prediction_time"]
    sorted_periods = sorted(times.unique(), key=lambda x: pd.to_datetime(x))
    n = len(sorted_periods)

    # Use the outermost quarter of periods for maximum signal strength
    n_slice = max(1, n // 4)
    early_set = set(sorted_periods[:n_slice])
    late_set = set(sorted_periods[-n_slice:])

    early = modified[times.isin(early_set)]
    late = modified[times.isin(late_set)]

    early_corr = early[col].corr(early["target"].astype(float))
    late_corr = late[col].corr(late["target"].astype(float))

    assert early_corr > 0.1, (
        f"Expected positive early-period correlation, got {early_corr:.4f}"
    )
    assert late_corr < -0.1, (
        f"Expected negative late-period correlation, got {late_corr:.4f}"
    )


def test_drift_injectors_add_no_target_column(small_df: pd.DataFrame) -> None:
    """Drift injectors must not add or overwrite the target column."""
    for inject_fn in (inject_covariate_drift, inject_concept_drift):
        modified, record = inject_fn(small_df)
        assert "target" not in record.planted_columns
        # Original target values must be unchanged
        pd.testing.assert_series_equal(small_df["target"], modified["target"])


def test_temporal_expanding_folds_unparseable_time_col_raises_clean_error() -> None:
    """Defense in depth: called directly (bypassing contract validation), an
    unparseable time_col must raise a clear, typed ValueError naming the
    column and explaining what Zekan needs -- not a raw pandas parse error
    leaking from deep in the stack."""
    df = pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(20)],
        "prediction_time": [str(197661240 + i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    with pytest.raises(ValueError, match="prediction_time.*could not be parsed as a time signal"):
        temporal_expanding_folds(
            df, time_col="prediction_time", entity_col="entity_id", target_col="target",
        )


def test_build_period_rank_integer_yyyymm_column() -> None:
    """PERIOD_RANK_ADDENDUM_01_ORDINAL_RANK.md: an integer YYYYMM period
    column (e.g. 201801, 201802, ...) must produce one distinct rank per
    distinct period, keyed on the RAW column value -- not collapse every
    period onto a single day-formatted string key."""
    periods = [201801, 201802, 201803, 201804, 201805]
    df = pd.DataFrame({"period": periods * 4})
    period_rank = build_period_rank(df, "period")
    assert len(period_rank) == len(periods), (
        f"expected {len(periods)} distinct period ranks, got {len(period_rank)}: {period_rank}"
    )
    assert period_rank == {p: i for i, p in enumerate(periods)}


def test_build_period_rank_string_date_column() -> None:
    """Same helper, string date column -- the case that already worked."""
    periods = ["2018-01-01", "2018-02-01", "2018-03-01"]
    df = pd.DataFrame({"period": periods * 4})
    period_rank = build_period_rank(df, "period")
    assert len(period_rank) == len(periods)
    assert period_rank["2018-01-01"] == 0
    assert period_rank["2018-02-01"] == 1
    assert period_rank["2018-03-01"] == 2


def test_evaluate_folds_non_numeric_feature_raises_clean_error(
    clean_df: pd.DataFrame, rand_folds: list[FoldIndices]
) -> None:
    """Defense in depth: called directly (bypassing contract validation), a
    non-numeric feature column must raise a clear, typed ValueError that names
    the problem and points back to the pre-flight check as the intended gate
    -- not a raw pandas 'could not convert string to float' error leaking
    from deep in the stack."""
    df = clean_df.copy()
    df["race"] = "Caucasian"
    with pytest.raises(ValueError, match="Could not cast feature columns to numeric"):
        evaluate_folds(df, ["feature_0", "race"], "target", rand_folds)


def _fake_fold_meta(fold_idx: int) -> FoldMeta:
    return FoldMeta(
        fold_idx=fold_idx, strategy="temporal_expanding",
        train_rows=2, test_rows=0,
        train_base_rate=0.0, test_base_rate=0.0,
        entity_overlap_count=0, entity_overlap_pct=0.0,
    )


def test_compute_fold_active_positions_raises_on_non_nested_folds() -> None:
    """FOLD_INERT_FEATURES_PREREGISTRATION.md section 7: expanding-window
    train sets are nested, so a column active in an earlier fold's train
    slice must never be inert in a later one. Two folds whose train slices
    are NOT nested -- fold 1's rows are disjoint from fold 0's, not a
    superset -- must raise, naming it as a monotonicity violation."""
    X_all = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [np.nan, 30.0],
        [np.nan, 40.0],
    ], dtype=np.float32)

    fold0 = FoldIndices(
        train_idx=np.array([0, 1]), test_idx=np.array([], dtype=int),
        meta=_fake_fold_meta(0),
    )
    fold1 = FoldIndices(
        train_idx=np.array([2, 3]), test_idx=np.array([], dtype=int),
        meta=_fake_fold_meta(1),
    )

    with pytest.raises(ValueError, match="monotonicity"):
        compute_fold_active_positions(X_all, ["col_a", "col_b"], [fold0, fold1])


def test_original_df_not_mutated(small_df: pd.DataFrame) -> None:
    """Every injector must return a copy; the input df is untouched."""
    original = small_df.copy()
    for inject_fn in (
        inject_label_proxy,
        inject_future_feature,
        inject_presplit_artifact,
        inject_correlated_leaks,
        inject_covariate_drift,
        inject_concept_drift,
    ):
        inject_fn(small_df)
    pd.testing.assert_frame_equal(original, small_df)
