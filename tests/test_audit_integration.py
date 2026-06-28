"""Integration tests for run_audit() — chains run_severity_analysis → build_verdict.

Three verdict paths are tested end-to-end:
  PASS                   : clean dataset, null not run
  UNCONFIRMED_HIGH_DAMAGE: strong label-proxy leak, null not run (n_permutations=0)
  FAIL                   : strong label-proxy leak, 100 permutations confirm detection

Each path runs the engine exactly once (module-scoped fixture) to keep the suite fast.
The FAIL path uses RF(n_estimators=5) to minimise permutation wall time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import IssueType
from zekan.severity.audit import run_audit
from zekan.severity.verdict import VerdictReport


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


def _noise_contract():
    return PredictionContract(
        prediction_problem="integration-test-pass",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["noise_forbidden"],
    )


def _proxy_contract():
    return PredictionContract(
        prediction_problem="integration-test-proxy",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )


# ── Module-scoped fixtures — each engine run executes once ────────────────────

@pytest.fixture(scope="module")
def pass_report():
    df = make_clean_dataset(n_entities=100, seed=0)
    df["noise_forbidden"] = np.random.default_rng(42).standard_normal(size=len(df))
    contract = _noise_contract()
    return run_audit(
        df, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=0,
    )


@pytest.fixture(scope="module")
def unconfirmed_report():
    """Label-proxy + n_permutations=0 → detected=False + fl>>warn_floor → UNCONFIRMED_HIGH_DAMAGE."""
    df_clean = make_clean_dataset(n_entities=100, seed=0)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = _proxy_contract()
    return run_audit(
        df_leaky, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=0,
    )


@pytest.fixture(scope="module")
def fail_report():
    """Label-proxy + n_permutations=100 → p=1/101<0.01, NSL>>1, fl>>0.15 → FAIL."""
    df_clean = make_clean_dataset(n_entities=100, seed=0)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = _proxy_contract()
    return run_audit(
        df_leaky, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=100,
    )


# ── PASS path ─────────────────────────────────────────────────────────────────

class TestPassPath:
    def test_returns_verdict_report(self, pass_report):
        assert isinstance(pass_report, VerdictReport)

    def test_verdict_is_pass(self, pass_report):
        assert pass_report.policy_decision.verdict == "PASS"

    def test_not_detected(self, pass_report):
        assert pass_report.engine_detection.detected is False

    def test_fixable_leakage_near_zero(self, pass_report):
        assert pass_report.measured_damage.fixable_leakage < 0.10

    def test_all_three_blocks_present(self, pass_report):
        assert pass_report.engine_detection is not None
        assert pass_report.measured_damage is not None
        assert pass_report.policy_decision is not None


# ── UNCONFIRMED_HIGH_DAMAGE path ──────────────────────────────────────────────

class TestUnconfirmedHighDamagePath:
    def test_verdict_is_unconfirmed(self, unconfirmed_report):
        assert unconfirmed_report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"

    def test_not_detected(self, unconfirmed_report):
        """p=None when null not run — detection gate cannot fire."""
        assert unconfirmed_report.engine_detection.detected is False

    def test_large_fixable_leakage(self, unconfirmed_report):
        """Label-proxy fl is well above warn_floor=0.10 (temporal ceiling ~0.096)."""
        assert unconfirmed_report.measured_damage.fixable_leakage >= 0.10

    def test_interpretation_not_clean_pass(self, unconfirmed_report):
        assert "not a clean PASS" in unconfirmed_report.policy_decision.interpretation


# ── FAIL path ─────────────────────────────────────────────────────────────────

class TestFailPath:
    def test_verdict_is_fail(self, fail_report):
        assert fail_report.policy_decision.verdict == "FAIL"

    def test_detected(self, fail_report):
        """With 100 permutations and strong proxy, best p=1/101≈0.0099 < 0.01."""
        assert fail_report.engine_detection.detected is True

    def test_p_value_below_alpha(self, fail_report):
        assert fail_report.engine_detection.p_value is not None
        assert fail_report.engine_detection.p_value < 0.01

    def test_fixable_leakage_above_fail_floor(self, fail_report):
        assert fail_report.measured_damage.fixable_leakage >= 0.15


# ── Structural probe fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_dup_report():
    """Dataset with 4 planted content-duplicate rows → ROW_DUPLICATION annotation."""
    df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=7)
    # Copy 4 rows; change only identity columns so feature+target content is identical.
    extra = df.iloc[:4].copy()
    extra = extra.assign(
        entity_id=["dup_e0", "dup_e1", "dup_e2", "dup_e3"],
        prediction_time="2099-01",
    )
    df = pd.concat([df, extra], ignore_index=True)
    contract = PredictionContract(
        prediction_problem="raw-dup-integration",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    return run_audit(df, contract, ZekanConfig(contract=contract),
                     model_factory=_fast_clf, n_permutations=0)


@pytest.fixture(scope="module")
def cross_fold_dup_report():
    """Dataset with planted cross-fold content duplicates → CROSS_FOLD_DUPLICATE annotation.

    3 rows from the earliest period are copied into the latest period with new
    entity_ids.  The temporal probe sees them in the training set (early period)
    and also in the test set (late period) for the last fold, causing a FAIL.
    """
    df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=0)
    periods = sorted(df["prediction_time"].unique())
    early_period, late_period = periods[0], periods[-1]
    extra = df[df["prediction_time"] == early_period].head(3).copy()
    extra = extra.assign(
        entity_id=["xf_e0", "xf_e1", "xf_e2"],
        prediction_time=late_period,
    )
    df = pd.concat([df, extra], ignore_index=True)
    contract = PredictionContract(
        prediction_problem="cross-fold-dup-integration",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    return run_audit(df, contract, ZekanConfig(contract=contract),
                     model_factory=_fast_clf, n_permutations=0)


@pytest.fixture(scope="module")
def entity_agg_report():
    """Dataset with entity-level aggregate forbidden feature → FORBIDDEN_ENTITY_LEVEL_AGGREGATE."""
    df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=5)
    # Each entity gets the mean of its target values — constant within entity, varies across.
    df["agg_forbidden"] = df.groupby("entity_id")["target"].transform("mean")
    contract = PredictionContract(
        prediction_problem="entity-agg-integration",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["agg_forbidden"],
    )
    return run_audit(df, contract, ZekanConfig(contract=contract),
                     model_factory=_fast_clf, n_permutations=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Structural probe loop — integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralProbes:
    """probe_raw_duplicates + entity_aggregate wired into run_audit via _run_structural_probes."""

    # ── Regression: clean audit stays clean ──────────────────────────────────

    def test_clean_no_structural_annotations(self, pass_report):
        """Clean data with a random forbidden feature → no structural annotations."""
        assert pass_report.structural_annotations == []

    # ── Entity-aggregate: existing behavior unchanged ─────────────────────────

    def test_entity_agg_annotation_present(self, entity_agg_report):
        assert len(entity_agg_report.structural_annotations) >= 1

    def test_entity_agg_issue_type(self, entity_agg_report):
        types = [a.issue_type for a in entity_agg_report.structural_annotations]
        assert IssueType.FORBIDDEN_ENTITY_LEVEL_AGGREGATE in types

    def test_entity_agg_annotation_serializes(self, entity_agg_report):
        from zekan.reports.json_export import verdict_to_dict
        d = verdict_to_dict(entity_agg_report)
        assert len(d["structural_annotations"]) >= 1

    # ── Raw-duplicate: new annotation ─────────────────────────────────────────

    def test_raw_dup_annotation_present(self, raw_dup_report):
        dup_annotations = [
            a for a in raw_dup_report.structural_annotations
            if a.issue_type == IssueType.ROW_DUPLICATION
        ]
        assert len(dup_annotations) == 1

    def test_raw_dup_status_warn(self, raw_dup_report):
        ann = next(
            a for a in raw_dup_report.structural_annotations
            if a.issue_type == IssueType.ROW_DUPLICATION
        )
        assert ann.status == "warn"

    def test_raw_dup_excess_copies_count(self, raw_dup_report):
        """4 planted rows → 4 excess copies reported."""
        ann = next(
            a for a in raw_dup_report.structural_annotations
            if a.issue_type == IssueType.ROW_DUPLICATION
        )
        assert ann.evidence.structural_detail.duplicate_rows == 4

    def test_raw_dup_annotation_serializes(self, raw_dup_report):
        """IssueRecord with RowDuplicationDetail survives verdict_to_dict without error."""
        from zekan.reports.json_export import verdict_to_dict
        d = verdict_to_dict(raw_dup_report)
        dup_annotations = [
            a for a in d["structural_annotations"]
            if a.get("issue_type") == IssueType.ROW_DUPLICATION.value
        ]
        assert len(dup_annotations) == 1

    def test_raw_dup_annotation_in_json(self, raw_dup_report):
        """ROW_DUPLICATION annotation present in the JSON output."""
        import json
        from zekan.reports.json_export import verdict_to_json
        parsed = json.loads(verdict_to_json(raw_dup_report))
        types_in_json = [a.get("issue_type") for a in parsed["structural_annotations"]]
        assert IssueType.ROW_DUPLICATION.value in types_in_json


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-fold duplicate probe wired into run_audit
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossFoldProbe:

    def test_cross_fold_annotation_present(self, cross_fold_dup_report):
        types = [a.issue_type for a in cross_fold_dup_report.structural_annotations]
        assert IssueType.CROSS_FOLD_DUPLICATE in types

    def test_cross_fold_annotation_status_fail(self, cross_fold_dup_report):
        ann = next(
            a for a in cross_fold_dup_report.structural_annotations
            if a.issue_type == IssueType.CROSS_FOLD_DUPLICATE
        )
        assert ann.status == "fail"

    def test_cross_fold_excess_copies_count(self, cross_fold_dup_report):
        """3 planted cross-fold rows → duplicate_rows == 3 in the cross-fold record."""
        ann = next(
            a for a in cross_fold_dup_report.structural_annotations
            if a.issue_type == IssueType.CROSS_FOLD_DUPLICATE
        )
        assert ann.evidence.structural_detail.duplicate_rows == 3

    def test_cross_fold_annotation_in_json(self, cross_fold_dup_report):
        import json
        from zekan.reports.json_export import verdict_to_json
        parsed = json.loads(verdict_to_json(cross_fold_dup_report))
        types = [a.get("issue_type") for a in parsed["structural_annotations"]]
        assert IssueType.CROSS_FOLD_DUPLICATE.value in types


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation rendering gap closed: TRUSTED + annotation → text + html show it
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnotationRendering:

    def test_trusted_raw_dup_text_shows_structural_finding(self, raw_dup_report):
        from zekan.reports.text_view import render_verdict
        out = render_verdict(raw_dup_report)
        assert "STRUCTURAL FINDING" in out

    def test_trusted_raw_dup_text_shows_also_noticed(self, raw_dup_report):
        from zekan.reports.text_view import render_verdict
        out = render_verdict(raw_dup_report)
        assert "also noticed" in out.lower()

    def test_trusted_raw_dup_html_shows_structural_finding(self, raw_dup_report):
        from zekan.reports.html_view import render_verdict_html
        out = render_verdict_html(raw_dup_report)
        assert "STRUCTURAL FINDING" in out

    def test_trusted_raw_dup_html_shows_also_noticed(self, raw_dup_report):
        from zekan.reports.html_view import render_verdict_html
        out = render_verdict_html(raw_dup_report)
        assert "also noticed" in out.lower()

    def test_folds_not_in_json_output(self, pass_report):
        """SeverityResult.folds must not appear in verdict_to_dict output."""
        import json
        from zekan.reports.json_export import verdict_to_json
        serialized = verdict_to_json(pass_report)
        assert '"folds"' not in serialized
