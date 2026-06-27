"""Tests for probe_splitter_contract_violation (SPLITTER_CONTRACT_VIOLATION).

Key invariants under test
--------------------------
Properly grouped folds (no entity in both train+test) -> status='pass'.
Any entity spanning both partitions of a fold -> status='internal_fail', confirmed=True.
Wording frames Zekan as the source of a violation; no user-blame language.
Robustness: empty folds, all-skipped folds, missing column -> 'unavailable'; no crash.
Skipped folds never contribute to the violation check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import (
    EvidenceScope,
    ImpactType,
    IssueSeverity,
    IssueType,
    SourceLayer,
    SplitterContractViolationDetail,
)
from zekan.detectors.splitter_contract_probe import probe_splitter_contract_violation
from zekan.severity.splitters import FoldIndices, FoldMeta


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="scv-test",
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
        strategy="grouped_cv",
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


def _entity_df() -> pd.DataFrame:
    """6-row df: 3 entities (e0, e1, e2), each with 2 observations.

    Row layout:
        0,1  → e0 (period 2022-01, 2022-02)
        2,3  → e1 (period 2022-01, 2022-02)
        4,5  → e2 (period 2022-01, 2022-02)
    """
    return pd.DataFrame({
        "entity_id": ["e0", "e0", "e1", "e1", "e2", "e2"],
        "prediction_time": ["2022-01", "2022-02", "2022-01", "2022-02",
                            "2022-01", "2022-02"],
        "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0, 1, 0, 1, 0, 1],
    })


def _clean_fold() -> FoldIndices:
    """Properly grouped: e0+e1 in train (rows 0-3), e2 in test (rows 4-5)."""
    return _make_fold(0, train_idx=[0, 1, 2, 3], test_idx=[4, 5])


def _violated_fold() -> FoldIndices:
    """e0 appears in both train (row 0) and test (row 1) — contract violation."""
    # train: row 0 (e0), rows 2,3 (e1)  →  {e0, e1}
    # test:  row 1 (e0), rows 4,5 (e2)  →  {e0, e2}
    # spanning: {e0}
    return _make_fold(0, train_idx=[0, 2, 3], test_idx=[1, 4, 5])


# ═══════════════════════════════════════════════════════════════════════════════
# Pass path — contract upheld
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass:

    def test_clean_fold_status_pass(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.status == "pass"

    def test_clean_fold_no_affected_folds(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.affected_folds == []

    def test_clean_fold_spanning_dict_empty(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.spanning_entities_per_fold == {}

    def test_clean_fold_total_folds_checked(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.total_folds_checked == 1

    def test_clean_fold_total_entities_checked(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.total_entities_checked == 3

    def test_multiple_clean_folds_pass(self):
        """Two properly grouped folds — both must pass."""
        df = _entity_df()
        fold0 = _make_fold(0, train_idx=[0, 1], test_idx=[2, 3])  # e0 train, e1 test
        fold1 = _make_fold(1, train_idx=[2, 3], test_idx=[4, 5])  # e1 train, e2 test
        result = probe_splitter_contract_violation([fold0, fold1], df, _contract())
        assert result.status == "pass"
        assert result.evidence.structural_detail.total_folds_checked == 2

    def test_degenerate_single_entity_per_side_passes(self):
        """Each side has one distinct entity — zero spanning."""
        df = pd.DataFrame({
            "entity_id": ["only_train", "only_test"],
            "prediction_time": ["2022-01", "2022-02"],
            "target": [0, 1],
        })
        fold = _make_fold(0, train_idx=[0], test_idx=[1])
        result = probe_splitter_contract_violation(
            [fold], df, _contract()
        )
        assert result.status == "pass"


# ═══════════════════════════════════════════════════════════════════════════════
# Internal-fail path — contract violated
# ═══════════════════════════════════════════════════════════════════════════════

class TestInternalFail:

    def test_violated_fold_status_internal_fail(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.status == "internal_fail"

    def test_confirmed_true(self):
        """SPLITTER_CONTRACT_VIOLATION is confirmed=True in the registry."""
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.confirmed is True

    def test_detail_type(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert isinstance(
            result.evidence.structural_detail, SplitterContractViolationDetail
        )

    def test_affected_folds_correct(self):
        """Violated fold index must appear in affected_folds."""
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.affected_folds == [0]

    def test_spanning_entities_per_fold_count(self):
        """e0 spans both partitions → count 1 for fold 0."""
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.spanning_entities_per_fold == {0: 1}

    def test_total_folds_checked(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.total_folds_checked == 1

    def test_total_entities_checked(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.structural_detail.total_entities_checked == 3

    def test_measured_value_equals_total_violations(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.measured_value == pytest.approx(1.0)

    def test_metric_name(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence.metric_name == "spanning_entity_count"

    def test_multiple_folds_only_one_violated(self):
        """Two folds: fold 0 clean, fold 1 violated — only fold 1 in affected_folds."""
        df = _entity_df()
        clean = _make_fold(0, train_idx=[0, 1, 2, 3], test_idx=[4, 5])
        # fold 1: e0 in both (row 0 in train, row 1 in test)
        violated = _make_fold(1, train_idx=[0, 2, 3], test_idx=[1, 4, 5])
        result = probe_splitter_contract_violation([clean, violated], df, _contract())
        assert result.status == "internal_fail"
        detail = result.evidence.structural_detail
        assert detail.affected_folds == [1]
        assert detail.spanning_entities_per_fold == {1: 1}
        assert detail.total_folds_checked == 2

    def test_multiple_folds_both_violated(self):
        """Both folds violated — both appear in affected_folds."""
        df = _entity_df()
        # fold 0: e0 spans (row 0 train, row 1 test)
        # fold 1: e2 spans (row 4 train, row 5 test)
        fold0 = _make_fold(0, train_idx=[0, 2, 3], test_idx=[1, 4, 5])
        fold1 = _make_fold(1, train_idx=[0, 1, 2, 3, 4], test_idx=[5])
        # fold1 test=[5]→e2; train contains row 4→e2 — e2 spans
        result = probe_splitter_contract_violation([fold0, fold1], df, _contract())
        assert result.status == "internal_fail"
        detail = result.evidence.structural_detail
        assert sorted(detail.affected_folds) == [0, 1]
        assert detail.total_folds_checked == 2

    def test_all_entities_spanning_single_fold(self):
        """If the splitter randomly shuffled rows, all entities span — count = 3."""
        df = _entity_df()
        # Alternate rows: train gets even indices, test gets odd
        # row 0 (e0) train, row 1 (e0) test → e0 spans
        # row 2 (e1) train, row 3 (e1) test → e1 spans
        # row 4 (e2) train, row 5 (e2) test → e2 spans
        fold = _make_fold(0, train_idx=[0, 2, 4], test_idx=[1, 3, 5])
        result = probe_splitter_contract_violation([fold], df, _contract())
        assert result.status == "internal_fail"
        detail = result.evidence.structural_detail
        assert detail.spanning_entities_per_fold == {0: 3}
        assert detail.affected_folds == [0]


# ═══════════════════════════════════════════════════════════════════════════════
# Schema computed fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaFields:

    def test_issue_type(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.issue_type == IssueType.SPLITTER_CONTRACT_VIOLATION

    def test_source_layer_zekan_integrity(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.source_layer == SourceLayer.ZEKAN_INTEGRITY

    def test_severity_critical(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.severity == IssueSeverity.CRITICAL

    def test_evidence_scope_self_check(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.evidence_scope == EvidenceScope.SELF_CHECK

    def test_impact_type_measurement_error(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert result.impact_type == ImpactType.MEASUREMENT_ERROR

    def test_pass_record_has_correct_issue_type(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.issue_type == IssueType.SPLITTER_CONTRACT_VIOLATION

    def test_pass_confirmed_true(self):
        """Confirmed is a registry property; it is True for pass records too."""
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        assert result.confirmed is True


# ═══════════════════════════════════════════════════════════════════════════════
# Wording tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWording:
    """The probe accuses Zekan, not the user.  Verified against all narrative fields."""

    def _narrative(self, result) -> str:
        return " ".join([result.what, result.why, result.how_much, result.next_fix])

    def test_fail_mentions_zekan(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert "Zekan" in self._narrative(result)

    def test_fail_no_user_blame_language(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        narrative = self._narrative(result).lower()
        for phrase in ("your split", "you used", "wrong split", "you leaked"):
            assert phrase not in narrative, (
                f"Internal-fail record contains user-blame phrase {phrase!r}"
            )

    def test_fail_frames_as_internal_violation(self):
        result = probe_splitter_contract_violation(
            [_violated_fold()], _entity_df(), _contract()
        )
        assert "internal" in result.what.lower() or "Zekan" in result.what

    def test_pass_no_blame_language(self):
        result = probe_splitter_contract_violation(
            [_clean_fold()], _entity_df(), _contract()
        )
        narrative = self._narrative(result).lower()
        for phrase in ("your split", "you used", "wrong split", "you leaked"):
            assert phrase not in narrative


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:

    # ── Empty folds → unavailable, never silent pass ──────────────────────────

    def test_empty_folds_unavailable(self):
        result = probe_splitter_contract_violation([], _entity_df(), _contract())
        assert result.status == "unavailable"

    def test_empty_folds_issue_type_preserved(self):
        result = probe_splitter_contract_violation([], _entity_df(), _contract())
        assert result.issue_type == IssueType.SPLITTER_CONTRACT_VIOLATION

    def test_empty_folds_not_pass(self):
        result = probe_splitter_contract_violation([], _entity_df(), _contract())
        assert result.status != "pass"

    # ── All folds skipped → unavailable ──────────────────────────────────────

    def test_all_skipped_unavailable(self):
        """Skipped folds are excluded; if all are skipped there is nothing to check."""
        skipped = _make_fold(0, train_idx=[0, 1, 2, 3], test_idx=[4, 5], skipped=True)
        result = probe_splitter_contract_violation(
            [skipped], _entity_df(), _contract()
        )
        assert result.status == "unavailable"

    def test_all_skipped_with_violation_still_unavailable(self):
        """A violated-but-skipped fold must not fire internal_fail."""
        # Build a violated fold pattern but mark it skipped
        skipped_violated = _make_fold(
            0, train_idx=[0, 2, 3], test_idx=[1, 4, 5], skipped=True
        )
        result = probe_splitter_contract_violation(
            [skipped_violated], _entity_df(), _contract()
        )
        assert result.status == "unavailable"

    # ── Skipped fold mixed with clean fold → only clean fold checked ──────────

    def test_skipped_fold_not_counted_in_total_folds_checked(self):
        """Only non-skipped folds count toward total_folds_checked."""
        clean = _clean_fold()
        skipped = _make_fold(1, train_idx=[0, 1], test_idx=[2, 3], skipped=True)
        result = probe_splitter_contract_violation(
            [clean, skipped], _entity_df(), _contract()
        )
        assert result.status == "pass"
        assert result.evidence.structural_detail.total_folds_checked == 1

    def test_skipped_violated_fold_does_not_trigger(self):
        """A skipped fold with entity overlap must not appear in affected_folds."""
        clean = _clean_fold()
        # This fold has a violation pattern but is skipped
        skipped_violated = _make_fold(
            1, train_idx=[0, 2, 3], test_idx=[1, 4, 5], skipped=True
        )
        result = probe_splitter_contract_violation(
            [clean, skipped_violated], _entity_df(), _contract()
        )
        assert result.status == "pass"
        assert result.evidence.structural_detail.affected_folds == []

    # ── Missing entity_id column → unavailable, no crash ─────────────────────

    def test_no_entity_id_column_unavailable(self):
        df = pd.DataFrame({
            "not_entity": ["a", "b", "c"],
            "prediction_time": ["2022-01"] * 3,
            "target": [0, 1, 0],
        })
        result = probe_splitter_contract_violation(
            [_make_fold(0, [0, 1], [2])], df, _contract()
        )
        assert result.status == "unavailable"

    def test_no_entity_id_column_not_pass(self):
        df = pd.DataFrame({
            "not_entity": ["a", "b"],
            "prediction_time": ["2022-01", "2022-02"],
            "target": [0, 1],
        })
        result = probe_splitter_contract_violation(
            [_make_fold(0, [0], [1])], df, _contract()
        )
        assert result.status != "pass"

    # ── Single fold — correct binary check ───────────────────────────────────

    def test_single_clean_fold_passes(self):
        df = pd.DataFrame({
            "entity_id": ["only_a", "only_b"],
            "prediction_time": ["2022-01", "2022-02"],
            "target": [0, 1],
        })
        fold = _make_fold(0, train_idx=[0], test_idx=[1])
        result = probe_splitter_contract_violation([fold], df, _contract())
        assert result.status == "pass"

    def test_single_violated_fold_fails(self):
        df = pd.DataFrame({
            "entity_id": ["same", "same"],
            "prediction_time": ["2022-01", "2022-02"],
            "target": [0, 1],
        })
        fold = _make_fold(0, train_idx=[0], test_idx=[1])
        result = probe_splitter_contract_violation([fold], df, _contract())
        assert result.status == "internal_fail"
        assert result.evidence.structural_detail.spanning_entities_per_fold == {0: 1}
