"""Tests for probe_entity_contamination_risk (ENTITY_CONTAMINATION_RISK).

Key invariants under test
--------------------------
Cross-sectional data (every entity in exactly one period) -> status='pass'; must NOT fire.
Longitudinal data (entity recurs across >1 period) -> status='warn', confirmed=False.
Wording is conditional throughout: "WOULD" / "random" / "Gotcha evaluates under".
Messages must NOT imply user fault: "wrong split", "you leaked", "contaminated",
"split strategy detected".
Robustness: missing column -> 'unavailable'; nulls dropped deterministically; no crash.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.detectors.entity_contamination_risk import probe_entity_contamination_risk
from gotcha.detectors.schema import (
    EntityContaminationRiskDetail,
    EvidenceScope,
    ImpactType,
    IssueSeverity,
    IssueType,
    SourceLayer,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="ecr-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _cross_sectional_df() -> pd.DataFrame:
    """3 entities, each at a distinct prediction period — no recurrence."""
    return pd.DataFrame({
        "entity_id": ["e0", "e1", "e2"],
        "prediction_time": ["2022-01", "2022-02", "2022-03"],
        "feature_0": [1.0, 2.0, 3.0],
        "target": [0, 1, 0],
    })


def _longitudinal_df() -> pd.DataFrame:
    """3 entities; e0 and e1 span 2 periods, e2 appears in 1 period.

    Expected panel statistics (computed by hand):
        recurring_entities           = 2   (e0, e1)
        total_entities               = 3
        entity_recurrence_fraction   = 2/3
        row counts: e0->2, e1->2, e2->1  -> median=2.0, max=2
        distinct_periods             = 2
    """
    return pd.DataFrame({
        "entity_id":       ["e0",     "e1",     "e2",     "e0",     "e1"],
        "prediction_time": ["2022-01", "2022-01", "2022-01", "2022-02", "2022-02"],
        "feature_0":       [1.0,      2.0,      3.0,      4.0,      5.0],
        "target":          [0,        1,        0,        1,        0],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — pass path (cross-sectional)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSectional:

    def test_cross_sectional_status_pass(self):
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        assert result.status == "pass"

    def test_cross_sectional_recurring_entities_zero(self):
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        detail = result.evidence.structural_detail
        assert detail.recurring_entities == 0

    def test_cross_sectional_entity_recurrence_fraction_zero(self):
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        assert result.evidence.structural_detail.entity_recurrence_fraction == 0.0

    def test_cross_sectional_total_entities_correct(self):
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        assert result.evidence.structural_detail.total_entities == 3

    def test_cross_sectional_distinct_periods_correct(self):
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        assert result.evidence.structural_detail.distinct_periods == 3

    def test_same_entity_multiple_rows_same_period_passes(self):
        """Entity with >1 row in the same period is NOT longitudinal — must pass.

        nunique(prediction_time) for e0 = 1 even though it has 2 rows.
        """
        df = pd.DataFrame({
            "entity_id":       ["e0", "e0", "e1"],
            "prediction_time": ["2022-01", "2022-01", "2022-02"],
            "feature_0":       [1.0, 2.0, 3.0],
            "target":          [0, 1, 0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "pass"

    def test_clean_dataset_longitudinal_warns(self):
        """make_clean_dataset produces a full panel (n_entities × n_periods).

        Every entity appears at all 5 snapshot periods, so all 40 entities are
        recurring.  Must fire WARN, not pass.
        """
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=0)
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "warn"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — warn path (longitudinal)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLongitudinalWarn:

    def test_longitudinal_status_warn(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.status == "warn"

    def test_confirmed_false(self):
        """ENTITY_CONTAMINATION_RISK is an advisory probe: confirmed must be False."""
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.confirmed is False

    def test_detail_type(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert isinstance(result.evidence.structural_detail, EntityContaminationRiskDetail)

    def test_recurring_entities_count(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.recurring_entities == 2

    def test_total_entities_count(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.total_entities == 3

    def test_entity_recurrence_fraction(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.entity_recurrence_fraction == pytest.approx(2 / 3)

    def test_median_obs_per_entity(self):
        # row counts: e0->2, e1->2, e2->1  ->  median of [2, 2, 1] sorted = [1, 2, 2] -> 2.0
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.median_obs_per_entity == pytest.approx(2.0)

    def test_max_obs_per_entity(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.max_obs_per_entity == 2

    def test_distinct_periods(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.structural_detail.distinct_periods == 2

    def test_measured_value_is_recurrence_fraction(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.measured_value == pytest.approx(2 / 3)

    def test_metric_name(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence.metric_name == "entity_recurrence_fraction"

    # ── Schema computed fields ───────────────────────────────────────────────

    def test_issue_type(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.issue_type == IssueType.ENTITY_CONTAMINATION_RISK

    def test_source_layer_detected_structural(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.source_layer == SourceLayer.DETECTED_STRUCTURAL

    def test_severity_high(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.severity == IssueSeverity.HIGH

    def test_evidence_scope_data_only(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.evidence_scope == EvidenceScope.DATA_ONLY

    def test_impact_type_structural_risk(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert result.impact_type == ImpactType.STRUCTURAL_RISK

    def test_full_panel_evidence_fields(self):
        """End-to-end: make_clean_dataset full panel — all entities recur, verify counts.

        Expected values are derived from the dataframe itself so this test is not
        coupled to how many rows the fixture decides to generate per entity.
        """
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=0)
        # Derive ground truth from the dataframe directly.
        row_counts = df.groupby("entity_id").size()
        expected_max = int(row_counts.max())
        expected_median = float(row_counts.median())
        expected_periods = int(df["prediction_time"].nunique())

        result = probe_entity_contamination_risk(df, _contract())
        detail = result.evidence.structural_detail
        assert detail.recurring_entities == 40
        assert detail.total_entities == 40
        assert detail.entity_recurrence_fraction == pytest.approx(1.0)
        assert detail.max_obs_per_entity == expected_max
        assert detail.median_obs_per_entity == pytest.approx(expected_median)
        assert detail.distinct_periods == expected_periods


# ═══════════════════════════════════════════════════════════════════════════════
# Wording tests — the probe's correctness contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestWording:
    """The ENTITY_CONTAMINATION_RISK wording rule is load-bearing correctness.

    The probe must be conditional (risk advisory), not accusatory.
    Every narrative field is checked against the forbidden-phrase list.
    """

    def _all_narrative(self, result) -> str:
        """Concatenate all four narrative fields for phrase scanning."""
        return " ".join([result.what, result.why, result.how_much, result.next_fix])

    def test_warn_does_not_say_wrong_split(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "wrong split" not in self._all_narrative(result).lower()

    def test_warn_does_not_say_you_leaked(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "you leaked" not in self._all_narrative(result).lower()

    def test_warn_does_not_say_contaminated(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "contaminated" not in self._all_narrative(result).lower()

    def test_warn_does_not_say_split_strategy_detected(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "split strategy detected" not in self._all_narrative(result).lower()

    def test_warn_contains_would(self):
        """Conditional framing: a random split WOULD leak (not 'did leak')."""
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "WOULD" in result.what

    def test_warn_contains_random(self):
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "random" in self._all_narrative(result).lower()

    def test_warn_references_gotcha_contract(self):
        """Must acknowledge that Gotcha's own evaluation avoids this risk."""
        result = probe_entity_contamination_risk(_longitudinal_df(), _contract())
        assert "Gotcha evaluates under" in result.why

    def test_pass_narrative_is_clean(self):
        """Even the pass record must not contain accusatory language."""
        result = probe_entity_contamination_risk(_cross_sectional_df(), _contract())
        narrative = self._all_narrative(result)
        for phrase in ("wrong split", "you leaked", "contaminated", "split strategy detected"):
            assert phrase not in narrative.lower(), (
                f"Pass record narrative contains forbidden phrase {phrase!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:

    # ── (a) Missing column → unavailable, not a silent pass, no crash ─────────

    def test_no_entity_id_column_is_unavailable(self):
        """entity_id column absent -> status='unavailable', not a crash or silent pass."""
        df = pd.DataFrame({
            "date": ["2022-01-01"],
            "prediction_time": ["2022-01-01"],
            "feature_0": [1.0],
            "target": [0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "unavailable"

    def test_no_entity_id_column_issue_type_preserved(self):
        df = pd.DataFrame({
            "date": ["2022-01-01"],
            "prediction_time": ["2022-01-01"],
            "feature_0": [1.0],
            "target": [0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.issue_type == IssueType.ENTITY_CONTAMINATION_RISK

    def test_no_entity_id_column_is_not_pass(self):
        """Missing column must not silently produce status='pass'."""
        df = pd.DataFrame({
            "date": ["2022-01-01"],
            "prediction_time": ["2022-01-01"],
            "feature_0": [1.0],
            "target": [0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status != "pass"

    def test_no_prediction_time_column_is_unavailable(self):
        """prediction_time column absent -> status='unavailable'."""
        df = pd.DataFrame({
            "entity_id": ["e0"],
            "date": ["2022-01-01"],
            "feature_0": [1.0],
            "target": [0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "unavailable"

    # ── (b) Single entity across many periods → sensible warn ────────────────

    def test_single_entity_many_periods_warns(self):
        """A single entity spanning 5 periods is longitudinal: must fire WARN."""
        df = pd.DataFrame({
            "entity_id": ["e0"] * 5,
            "prediction_time": ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05"],
            "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "warn"

    def test_single_entity_evidence_fields(self):
        df = pd.DataFrame({
            "entity_id": ["e0"] * 5,
            "prediction_time": ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05"],
            "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        detail = result.evidence.structural_detail
        assert detail.recurring_entities == 1
        assert detail.total_entities == 1
        assert detail.entity_recurrence_fraction == pytest.approx(1.0)
        assert detail.max_obs_per_entity == 5
        assert detail.distinct_periods == 5

    # ── (c) Many entities at a single period → must NOT fire ─────────────────

    def test_many_entities_single_period_passes(self):
        """100 entities all at the same prediction period — cross-sectional; must NOT fire."""
        n = 100
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": ["2022-01-01"] * n,
            "feature_0": list(range(n)),
            "target": [0] * n,
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "pass"

    def test_many_entities_single_period_zero_recurring(self):
        n = 50
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": ["2022-01-01"] * n,
            "feature_0": list(range(n)),
            "target": [0] * n,
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.evidence.structural_detail.recurring_entities == 0

    # ── (d) Nulls in entity_id or prediction_time → deterministic, no crash ──

    def test_null_entity_id_rows_dropped_deterministically(self):
        """Rows with null entity_id are dropped before counts; surviving rows drive result."""
        df = pd.DataFrame({
            "entity_id":       ["e0",     None,     "e1",     "e0"],
            "prediction_time": ["2022-01", "2022-01", "2022-02", "2022-02"],
            "feature_0":       [1.0,      2.0,      3.0,      4.0],
            "target":          [0,        1,        0,        1],
        })
        # After dropping null entity_id row (row 1):
        #   e0: 2022-01, 2022-02  -> recurring
        #   e1: 2022-02 only      -> not recurring
        # Expect: warn, recurring_entities=1, total_entities=2
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "warn"
        detail = result.evidence.structural_detail
        assert detail.recurring_entities == 1
        assert detail.total_entities == 2

    def test_null_prediction_time_rows_dropped_deterministically(self):
        """Rows with null prediction_time are dropped; remaining rows drive result."""
        df = pd.DataFrame({
            "entity_id":       ["e0",     "e1",     "e0"],
            "prediction_time": ["2022-01", None,     "2022-02"],
            "feature_0":       [1.0,      2.0,      3.0],
            "target":          [0,        1,        0],
        })
        # After dropping null prediction_time row (row 1, entity e1):
        #   e0: 2022-01, 2022-02  -> recurring
        # total_entities=1, recurring_entities=1  -> warn
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "warn"
        detail = result.evidence.structural_detail
        assert detail.recurring_entities == 1
        assert detail.total_entities == 1

    def test_all_nulls_in_entity_id_is_unavailable(self):
        """All entity_id values null -> no valid rows -> status='unavailable'."""
        df = pd.DataFrame({
            "entity_id":       [None, None],
            "prediction_time": ["2022-01", "2022-02"],
            "feature_0":       [1.0, 2.0],
            "target":          [0, 1],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "unavailable"

    def test_nulls_in_both_columns_no_crash(self):
        """Mixed nulls in both columns must not crash."""
        df = pd.DataFrame({
            "entity_id":       [None,     "e0",     "e1",     None],
            "prediction_time": ["2022-01", None,     "2022-01", "2022-02"],
            "feature_0":       [1.0,      2.0,      3.0,      4.0],
            "target":          [0,        1,        0,        1],
        })
        # Only row 2 (e1, 2022-01) survives the null drop.
        # total_entities=1, recurring_entities=0  -> pass (single period)
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "pass"
        assert result.evidence.structural_detail.recurring_entities == 0
