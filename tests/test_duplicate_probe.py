"""Tests for the Phase-3 content-hash duplicate probes.

probe_raw_duplicates  -> IssueType.ROW_DUPLICATION
probe_cross_fold_duplicates  -> IssueType.CROSS_FOLD_DUPLICATE

Design decisions under test
---------------------------
Hash content: all columns except entity_id, prediction_time, available_features_until.
RAW threshold: any excess copy > 0 -> 'warn'; no size-based exemption.
CROSS_FOLD threshold: any contaminating row -> 'fail'; binary gate.
NaN handling: NaN == NaN in hash comparison (sentinel replacement).

Robustness cases (labelled as test_robustness_*):
  (a) Longitudinal repeat: entity_id and prediction_time repeat but feature
      content differs -> must NOT trigger RAW or CROSS_FOLD probe.
  (b) Degenerate all-identical dataset -> must flag without crashing.
  (c) Categorical/object columns and NaN values -> deterministic hashing, no error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.duplicate_probe import probe_cross_fold_duplicates, probe_raw_duplicates
from zekan.detectors.schema import (
    CrossFoldDuplicateDetail,
    IssueSeverity,
    IssueType,
    RowDuplicationDetail,
    SourceLayer,
)
from zekan.severity.splitters import FoldIndices, FoldMeta


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="probe-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _make_fold(
    fold_idx: int,
    train_idx: list[int],
    test_idx: list[int],
    skipped: bool = False,
) -> FoldIndices:
    meta = FoldMeta(
        fold_idx=fold_idx,
        strategy="temporal_expanding",
        train_rows=len(train_idx),
        test_rows=len(test_idx),
        train_base_rate=0.5,
        test_base_rate=0.5,
        entity_overlap_count=0,
        entity_overlap_pct=0.0,
        skipped=skipped,
    )
    return FoldIndices(
        train_idx=np.array(train_idx, dtype=int),
        test_idx=np.array(test_idx, dtype=int),
        meta=meta,
    )


def _clean_small() -> pd.DataFrame:
    """400-row longitudinal panel (40 entities x 10 periods) with unique feature vectors per row."""
    return make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=0)


def _planted_raw_df(n_unique: int = 10, n_dup_groups: int = 2, group_size: int = 3) -> pd.DataFrame:
    """Unique rows followed by duplicate groups.

    n_unique  : rows with distinct content (no duplicates)
    n_dup_groups: number of identical groups to plant
    group_size: rows per group (group_size - 1 excess copies per group)
    """
    rng = np.random.default_rng(1)
    total = n_unique + n_dup_groups * group_size
    df = pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(total)],
        "prediction_time": ["2022-01-01"] * total,
        "feature_0": rng.normal(0, 1, total),
        "feature_1": rng.normal(0, 1, total),
        "target": rng.integers(0, 2, total).astype(int),
    })
    # Overwrite duplicate groups with identical content
    for g in range(n_dup_groups):
        start = n_unique + g * group_size
        # All rows in group g get the same feature+target as the first row of the group
        anchor = start
        for j in range(1, group_size):
            df.loc[start + j, "feature_0"] = df.loc[anchor, "feature_0"]
            df.loc[start + j, "feature_1"] = df.loc[anchor, "feature_1"]
            df.loc[start + j, "target"] = df.loc[anchor, "target"]
    return df


def _planted_cross_fold_df() -> tuple[pd.DataFrame, FoldIndices]:
    """20-row df where 3 test rows (indices 10-12) match 3 train rows (indices 0-2).

    Returns the DataFrame and a fold with train=0..9, test=10..19.
    """
    rng = np.random.default_rng(2)
    n = 20
    df = pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(n)],
        "prediction_time": ["2022-01-01"] * 10 + ["2022-02-01"] * 10,
        "feature_0": rng.normal(0, 1, n),
        "feature_1": rng.normal(0, 1, n),
        "target": rng.integers(0, 2, n).astype(int),
    })
    # Plant: rows 10, 11, 12 get same feature+target as rows 0, 1, 2
    for i in range(3):
        df.loc[10 + i, "feature_0"] = df.loc[i, "feature_0"]
        df.loc[10 + i, "feature_1"] = df.loc[i, "feature_1"]
        df.loc[10 + i, "target"] = df.loc[i, "target"]
    fold = _make_fold(0, train_idx=list(range(10)), test_idx=list(range(10, 20)))
    return df, fold


# ═══════════════════════════════════════════════════════════════════════════════
# RAW_DUPLICATE_ROWS tests  (probe_raw_duplicates -> IssueType.ROW_DUPLICATION)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProbeRawDuplicates:

    # ── Clean data: no flag ──────────────────────────────────────────────────

    def test_clean_data_passes(self):
        df = _clean_small()
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "pass"

    def test_clean_data_zero_count(self):
        df = _clean_small()
        result = probe_raw_duplicates(df, _contract())
        assert result.evidence.structural_detail.duplicate_rows == 0
        assert result.evidence.structural_detail.duplicate_fraction == 0.0

    # ── Planted duplicates: flag ─────────────────────────────────────────────

    def test_planted_duplicates_warns(self):
        # 2 groups of 3 identical rows -> 2 excess copies per group -> 4 total
        df = _planted_raw_df(n_unique=10, n_dup_groups=2, group_size=3)
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"

    def test_excess_copies_count_two_groups(self):
        # group_size=3 -> 2 excess per group; 2 groups -> 4 total excess
        df = _planted_raw_df(n_unique=10, n_dup_groups=2, group_size=3)
        result = probe_raw_duplicates(df, _contract())
        detail = result.evidence.structural_detail
        assert detail.duplicate_rows == 4

    def test_excess_copies_count_single_pair(self):
        # Single pair of identical rows -> 1 excess copy
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e2"],
            "prediction_time": ["2022-01-01"] * 3,
            "feature_0": [1.0, 1.0, 2.0],
            "target": [0, 0, 1],
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.evidence.structural_detail.duplicate_rows == 1

    def test_excess_copies_multi_group_sizes(self):
        # 5-row group (4 excess) + 4-row group (3 excess) + 3-row group (2 excess) = 9 total
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(15)],
            "prediction_time": ["2022-01-01"] * 15,
            "feature_0": [1.0]*5 + [2.0]*4 + [3.0]*3 + [4.0, 5.0, 6.0],
            "target": [0]*5 + [1]*4 + [0]*3 + [0, 0, 1],
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.evidence.structural_detail.duplicate_rows == 9

    def test_duplicate_fraction_correct(self):
        # 10 rows total: 1 pair of identical rows -> 1 excess, fraction = 1/10
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(10)],
            "prediction_time": ["2022-01-01"] * 10,
            "feature_0": [1.0, 1.0] + list(range(2, 10, 1, )),
            "target": [0, 0] + [1] * 8,
        })
        result = probe_raw_duplicates(df, _contract())
        detail = result.evidence.structural_detail
        assert abs(detail.duplicate_fraction - 1 / 10) < 1e-12

    # ── Schema and evidence ──────────────────────────────────────────────────

    def test_issue_type_row_duplication(self):
        df = _planted_raw_df()
        result = probe_raw_duplicates(df, _contract())
        assert result.issue_type == IssueType.ROW_DUPLICATION

    def test_source_layer_detected_structural(self):
        df = _planted_raw_df()
        result = probe_raw_duplicates(df, _contract())
        assert result.source_layer == SourceLayer.DETECTED_STRUCTURAL

    def test_severity_high(self):
        df = _planted_raw_df()
        result = probe_raw_duplicates(df, _contract())
        assert result.severity == IssueSeverity.HIGH

    def test_evidence_detail_is_row_duplication(self):
        df = _planted_raw_df()
        result = probe_raw_duplicates(df, _contract())
        assert isinstance(result.evidence.structural_detail, RowDuplicationDetail)

    def test_pass_has_correct_issue_type(self):
        df = _clean_small()
        result = probe_raw_duplicates(df, _contract())
        assert result.issue_type == IssueType.ROW_DUPLICATION

    def test_metric_name_populated(self):
        df = _planted_raw_df()
        result = probe_raw_duplicates(df, _contract())
        assert result.evidence.metric_name == "duplicate_fraction"

    # ── Robustness (a): longitudinal repeat not flagged ──────────────────────

    def test_robustness_longitudinal_no_false_positive(self):
        """entity_id and prediction_time repeat across rows; feature content
        differs per row -> must NOT flag ROW_DUPLICATION."""
        df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=42)
        # Verify the panel repeats
        assert df["entity_id"].value_counts().max() > 1   # entity_id repeats
        assert df["prediction_time"].value_counts().max() > 1  # time repeats
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "pass", (
            f"False positive: longitudinal dataset flagged as RAW_DUPLICATE_ROWS. "
            f"duplicate_rows={result.evidence.structural_detail.duplicate_rows}"
        )

    # ── Robustness (b): all-identical rows ────────────────────────────────────

    def test_robustness_all_identical_rows(self):
        """Degenerate dataset: every row has the same feature+target.
        Must flag without crashing."""
        n = 50
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": ["2022-01-01"] * n,
            "feature_0": [1.0] * n,
            "target": [1] * n,
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"
        detail = result.evidence.structural_detail
        assert detail.duplicate_rows == n - 1   # 49 excess copies
        assert abs(detail.duplicate_fraction - (n - 1) / n) < 1e-12

    # ── Robustness (c): categorical/object columns and NaN values ─────────────

    def test_robustness_categorical_columns_no_error(self):
        """Object/string columns must hash without error."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e2", "e3"],
            "prediction_time": ["2022-01-01"] * 4,
            "feature_cat": ["cat_A", "cat_B", "cat_C", "cat_A"],
            "feature_num": [1.0, 2.0, 3.0, 1.0],
            "target": [0, 1, 0, 0],
        })
        # Rows 0 and 3: same feature_cat, feature_num, target -> duplicate
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"
        assert result.evidence.structural_detail.duplicate_rows == 1

    def test_robustness_none_in_object_column_no_error(self):
        """None in an object column must be handled without error."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e2"],
            "prediction_time": ["2022-01-01"] * 3,
            "feature_cat": ["A", None, "A"],  # row 0 and row 2 both have "A"
            "target": [0, 1, 0],
        })
        # Rows 0 and 2 are identical (feature_cat="A", target=0)
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"
        assert result.evidence.structural_detail.duplicate_rows == 1

    def test_robustness_nan_equals_nan_in_hash(self):
        """Two rows where a numeric column is NaN must be detected as duplicates."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1"],
            "prediction_time": ["2022-01-01", "2022-02-01"],
            "feature_0": [float("nan"), float("nan")],  # both NaN
            "target": [0, 0],                            # same target
        })
        result = probe_raw_duplicates(df, _contract())
        # The two rows are content-identical (NaN==NaN by sentinel)
        assert result.status == "warn"
        assert result.evidence.structural_detail.duplicate_rows == 1

    def test_robustness_nan_not_equal_to_value(self):
        """A row with NaN and a row with a numeric value must NOT collide."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1"],
            "prediction_time": ["2022-01-01", "2022-02-01"],
            "feature_0": [float("nan"), 1.0],  # different content
            "target": [0, 0],
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "pass"

    # ── Fail-loud cases ───────────────────────────────────────────────────────

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["entity_id", "prediction_time", "feature_0", "target"])
        with pytest.raises(ValueError, match="empty"):
            probe_raw_duplicates(df, _contract())

    def test_no_hash_columns_raises(self):
        """When all DataFrame columns are excluded as identity cols, raise ValueError."""
        df = pd.DataFrame({
            "entity_id": ["e0"],
            "prediction_time": ["2022-01-01"],
        })
        contract = _contract()  # excludes entity_id, prediction_time, available_features_until
        # available_features_until == "prediction_time" (already excluded)
        # No other columns exist -> must raise
        with pytest.raises(ValueError, match="No hashable columns"):
            probe_raw_duplicates(df, contract)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS_FOLD_DUPLICATE tests  (probe_cross_fold_duplicates)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProbeCrossFoldDuplicates:

    # ── Clean folds: no flag ─────────────────────────────────────────────────

    def test_clean_folds_passes(self):
        """End-to-end: temporal folds on clean data -> no content overlap -> PASS."""
        from zekan.config.schema import ZekanConfig, SplitPolicy
        from zekan.severity.splitters import temporal_expanding_folds

        df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=0)
        contract = _contract()
        folds = temporal_expanding_folds(
            df, time_col="prediction_time", entity_col="entity_id",
            target_col="target", n_splits=5,
            min_test_rows=20, min_pos=5, min_neg=5,
        )
        result = probe_cross_fold_duplicates(df, contract, folds)
        assert result.status == "pass"

    def test_clean_folds_zero_count(self):
        from zekan.severity.splitters import temporal_expanding_folds
        df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=1)
        contract = _contract()
        folds = temporal_expanding_folds(
            df, time_col="prediction_time", entity_col="entity_id",
            target_col="target", n_splits=5,
            min_test_rows=20, min_pos=5, min_neg=5,
        )
        result = probe_cross_fold_duplicates(df, contract, folds)
        detail = result.evidence.structural_detail
        assert detail.duplicate_rows == 0
        assert detail.affected_folds == []
        assert detail.rows_per_fold == {}

    # ── Planted overlap: flag ─────────────────────────────────────────────────

    def test_planted_overlap_fails(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.status == "fail"

    def test_contamination_count_correct(self):
        """3 planted duplicate rows -> duplicate_rows == 3."""
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.evidence.structural_detail.duplicate_rows == 3

    def test_affected_folds_correct(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.evidence.structural_detail.affected_folds == [0]

    def test_rows_per_fold_correct(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.evidence.structural_detail.rows_per_fold == {0: 3}

    def test_multiple_folds_accumulate(self):
        """Two folds, each with 2 contaminating rows -> total == 4."""
        rng = np.random.default_rng(3)
        n = 12
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": (
                ["2022-01-01"] * 4 + ["2022-02-01"] * 4 + ["2022-03-01"] * 4
            ),
            "feature_0": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n).astype(int),
        })
        # Fold 0: train=0..3, test=4..7; plant rows 4 and 5 == rows 0 and 1
        df.loc[4, "feature_0"] = df.loc[0, "feature_0"]
        df.loc[4, "target"] = df.loc[0, "target"]
        df.loc[5, "feature_0"] = df.loc[1, "feature_0"]
        df.loc[5, "target"] = df.loc[1, "target"]
        fold0 = _make_fold(0, train_idx=[0, 1, 2, 3], test_idx=[4, 5, 6, 7])

        # Fold 1: train=0..7, test=8..11; plant rows 8 and 9 == rows 0 and 1
        df.loc[8, "feature_0"] = df.loc[0, "feature_0"]
        df.loc[8, "target"] = df.loc[0, "target"]
        df.loc[9, "feature_0"] = df.loc[1, "feature_0"]
        df.loc[9, "target"] = df.loc[1, "target"]
        fold1 = _make_fold(1, train_idx=list(range(8)), test_idx=[8, 9, 10, 11])

        result = probe_cross_fold_duplicates(df, _contract(), [fold0, fold1])
        assert result.status == "fail"
        detail = result.evidence.structural_detail
        assert detail.duplicate_rows == 4
        assert sorted(detail.affected_folds) == [0, 1]
        assert detail.rows_per_fold[0] == 2
        assert detail.rows_per_fold[1] == 2

    def test_duplicate_fraction_correct(self):
        """fold has 10 test rows; 3 contaminated -> fraction == 0.3."""
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        detail = result.evidence.structural_detail
        assert abs(detail.duplicate_fraction - 3 / 10) < 1e-12

    # ── Schema and evidence ──────────────────────────────────────────────────

    def test_issue_type_cross_fold(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.issue_type == IssueType.CROSS_FOLD_DUPLICATE

    def test_source_layer_detected_structural(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.source_layer == SourceLayer.DETECTED_STRUCTURAL

    def test_severity_high(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.severity == IssueSeverity.HIGH

    def test_evidence_detail_is_cross_fold(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert isinstance(result.evidence.structural_detail, CrossFoldDuplicateDetail)

    def test_pass_has_correct_issue_type(self):
        from zekan.severity.splitters import temporal_expanding_folds
        df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=2)
        folds = temporal_expanding_folds(
            df, time_col="prediction_time", entity_col="entity_id",
            target_col="target", n_splits=5,
            min_test_rows=20, min_pos=5, min_neg=5,
        )
        result = probe_cross_fold_duplicates(df, _contract(), folds)
        assert result.issue_type == IssueType.CROSS_FOLD_DUPLICATE

    def test_metric_name_populated(self):
        df, fold = _planted_cross_fold_df()
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.evidence.metric_name == "cross_fold_duplicate_fraction"

    # ── Robustness (a): longitudinal entity overlap not flagged ──────────────

    def test_robustness_longitudinal_entity_overlap_not_flagged(self):
        """Same entity appears in both train and test at different time periods
        with different feature values -> must NOT trigger CROSS_FOLD_DUPLICATE."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e0", "e1"],   # e0 and e1 in both periods
            "prediction_time": ["2022-01-01", "2022-01-01", "2022-02-01", "2022-02-01"],
            "feature_0": [1.0, 2.0, 3.0, 4.0],       # DIFFERENT content per row
            "target": [0, 1, 1, 0],
        })
        contract = _contract()
        # Train: Jan rows (0, 1); Test: Feb rows (2, 3)
        # Same entity e0 in both train and test, but with different feature values
        fold = _make_fold(0, train_idx=[0, 1], test_idx=[2, 3])
        result = probe_cross_fold_duplicates(df, contract, [fold])
        assert result.status == "pass", (
            "False positive: same entity at different times with different "
            "features should not trigger CROSS_FOLD_DUPLICATE."
        )

    # ── Robustness (b): all-identical rows ────────────────────────────────────

    def test_robustness_all_identical_rows_fails(self):
        """Every row has identical feature+target -> all test rows contaminate -> FAIL."""
        n = 10
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": ["2022-01-01"] * 5 + ["2022-02-01"] * 5,
            "feature_0": [1.0] * n,
            "target": [1] * n,
        })
        fold = _make_fold(0, train_idx=list(range(5)), test_idx=list(range(5, 10)))
        result = probe_cross_fold_duplicates(df, _contract(), [fold])
        assert result.status == "fail"
        assert result.evidence.structural_detail.duplicate_rows == 5
        assert result.evidence.structural_detail.duplicate_fraction == pytest.approx(1.0)

    # ── Robustness (c): NaN values in columns ─────────────────────────────────

    def test_robustness_nan_columns_no_error(self):
        """NaN in feature columns must hash consistently without error."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e2", "e3"],
            "prediction_time": ["2022-01-01"] * 2 + ["2022-02-01"] * 2,
            "feature_0": [float("nan"), 2.0, float("nan"), 3.0],
            "target": [0, 1, 0, 1],
        })
        contract = _contract()
        # Train: rows 0, 1; Test: rows 2, 3
        # Row 2 has same content as row 0 (feature_0=NaN, target=0) -> contamination
        fold = _make_fold(0, train_idx=[0, 1], test_idx=[2, 3])
        result = probe_cross_fold_duplicates(df, contract, [fold])
        assert result.status == "fail"
        assert result.evidence.structural_detail.duplicate_rows == 1

    def test_robustness_categorical_columns_no_error(self):
        """Object/string columns must hash without error across folds."""
        df = pd.DataFrame({
            "entity_id": ["e0", "e1", "e2", "e3"],
            "prediction_time": ["2022-01-01"] * 2 + ["2022-02-01"] * 2,
            "feature_cat": ["A", "B", "A", "C"],   # row 2 matches row 0 in content
            "target": [0, 1, 0, 1],
        })
        contract = _contract()
        fold = _make_fold(0, train_idx=[0, 1], test_idx=[2, 3])
        result = probe_cross_fold_duplicates(df, contract, [fold])
        assert result.status == "fail"
        assert result.evidence.structural_detail.duplicate_rows == 1

    # ── Skipped folds excluded ────────────────────────────────────────────────

    def test_skipped_folds_not_counted(self):
        """A fold marked as skipped must not contribute to contamination counts."""
        rng = np.random.default_rng(9)
        n = 15
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": (
                ["2022-01-01"] * 5 + ["2022-02-01"] * 5 + ["2022-03-01"] * 5
            ),
            "feature_0": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n).astype(int),
        })
        # Plant contamination: row 5 matches row 0
        df.loc[5, "feature_0"] = df.loc[0, "feature_0"]
        df.loc[5, "target"] = df.loc[0, "target"]

        contract = _contract()

        # Fold 0 has contamination but is SKIPPED
        skipped = _make_fold(0, train_idx=[0, 1, 2, 3, 4], test_idx=[5, 6, 7, 8, 9], skipped=True)
        # Fold 1 has no contamination (rows 10-14 have distinct features)
        # Ensure rows 10-14 are distinct from everything in 0-9
        df.loc[10:14, "feature_0"] = [100.0, 200.0, 300.0, 400.0, 500.0]
        clean = _make_fold(1, train_idx=list(range(10)), test_idx=[10, 11, 12, 13, 14])

        result = probe_cross_fold_duplicates(df, contract, [skipped, clean])
        assert result.status == "pass"
        assert result.evidence.structural_detail.duplicate_rows == 0
        assert result.evidence.structural_detail.affected_folds == []

    # ── Fail-loud cases ───────────────────────────────────────────────────────

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["entity_id", "prediction_time", "feature_0", "target"])
        fold = _make_fold(0, train_idx=[0, 1], test_idx=[2, 3])
        with pytest.raises(ValueError, match="empty"):
            probe_cross_fold_duplicates(df, _contract(), [fold])

    def test_empty_folds_raises(self):
        df = _clean_small()
        with pytest.raises(ValueError, match="empty"):
            probe_cross_fold_duplicates(df, _contract(), [])
