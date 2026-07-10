"""Tests for text_view formatter and feature_attribution wiring.

(a) Attribution-lock: end-to-end run_audit calls verify that ablation fires
    when fl >= warn_floor and stays None when fl < warn_floor.
(b) Formatter: one synthetic VerdictReport per verdict state; assert markers
    and structural properties of the rendered string.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.audit import run_audit
from zekan.severity.engine import PerFoldSeverity, SeverityResult
from zekan.severity.verdict import build_verdict
from zekan.reports.text_view import render_verdict


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


def _proxy_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="text-view-test-leak",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )


def _noise_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="text-view-test-clean",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["noise_forbidden"],
    )


def _interior_folds(leakages: list[float]) -> list[PerFoldSeverity]:
    return [
        PerFoldSeverity(
            fold_idx=i,
            auc_with_forbidden=0.5 + fl,
            auc_without_forbidden=0.5,
            is_terminal=False,
        )
        for i, fl in enumerate(leakages)
    ]


def _severity_result(
    fixable_leakage: float = 0.05,
    p_value: float | None = 0.50,
    nsl: float | None = -1.0,
    null_iqr: float | None = 0.010,
    null_99th: float | None = 0.040,
    p_value_across: float | None = None,
    nsl_across: float | None = None,
    naive_auc: float = 0.80,
    estimated_deployable_auc: float = 0.75,
    nonfixable_optimism: float = 0.02,
    n_permutations_run: int = 100,
    per_fold: list | None = None,
    feature_attribution: AblationSummary | None = None,
) -> SeverityResult:
    if per_fold is None:
        per_fold = []
    return SeverityResult(
        status="pass",
        metric="roc_auc",
        naive_auc=naive_auc,
        estimated_deployable_auc=estimated_deployable_auc,
        total_optimism=naive_auc - estimated_deployable_auc,
        fixable_leakage=fixable_leakage,
        fixable_leakage_range=(fixable_leakage * 0.8, fixable_leakage * 1.2),
        nonfixable_optimism=nonfixable_optimism,
        per_fold=per_fold,
        caveat="test",
        null_iqr=null_iqr,
        null_99th=null_99th,
        p_value=p_value,
        nsl=nsl,
        null_iqr_across=0.010 if p_value_across is not None else None,
        null_99th_across=0.040 if p_value_across is not None else None,
        p_value_across=p_value_across,
        nsl_across=nsl_across,
        n_permutations_across=100 if p_value_across is not None else 0,
        n_permutations_run=n_permutations_run,
        feature_attribution=feature_attribution,
    )


def _fake_attribution(feature: str = "bad_feature", leakage: float = 0.15) -> AblationSummary:
    entry = AblationEntry(
        feature=feature,
        is_contract_forbidden=True,
        univariate_auc=0.85,
        name_pattern_score=0.0,
        rank_score=1.7,
        auc_without=0.65,
        leakage_estimate=leakage,
        ablated=True,
    )
    return AblationSummary(baseline_auc=0.80, individual=[entry])


# ── (a) Attribution-lock: integration tests ───────────────────────────────────

@pytest.fixture(scope="module")
def leaky_report():
    """Label-proxy + 100 permutations → fl >> warn_floor → ablation must run."""
    df_clean = make_clean_dataset(n_entities=100, seed=0)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = _proxy_contract()
    return run_audit(
        df_leaky, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=100,
    )


@pytest.fixture(scope="module")
def clean_report():
    """Clean dataset + noise forbidden → fl < warn_floor → no ablation."""
    df = make_clean_dataset(n_entities=100, seed=0)
    df = df.copy()
    df["noise_forbidden"] = np.random.default_rng(42).standard_normal(size=len(df))
    contract = _noise_contract()
    return run_audit(
        df, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=0,
    )


def test_attribution_present_when_leaky(leaky_report):
    attr = leaky_report.measured_damage.feature_attribution
    assert attr is not None, "expected AblationSummary when fl >= warn_floor"
    assert len(attr.individual) > 0, "expected at least one ablated entry"


def test_attribution_absent_when_clean(clean_report):
    assert clean_report.measured_damage.feature_attribution is None


# ── Fake UTF-8 stream — pins the glyph path for formatter tests ───────────────

class _Utf8Stream:
    encoding = "utf-8"


# ── (b) Formatter: one test per verdict state ─────────────────────────────────

def test_pass_renders_trusted():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    assert report.policy_decision.verdict == "PASS"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "✓ TRUSTED" in out
    assert "WHAT TO FIX" not in out


def test_warn_renders_risky_with_fix_list():
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12,
        p_value=0.001,
        nsl=1.5,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    assert report.policy_decision.verdict == "WARN"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "⚠ RISKY" in out
    assert "WHAT TO FIX" in out
    assert "top_offender" in out


def test_fail_renders_failed_with_fix_list():
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20,
        p_value=0.001,
        nsl=2.5,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    assert report.policy_decision.verdict == "FAIL"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "✗ FAILED" in out
    assert "WHAT TO FIX" in out
    assert "top_offender" in out


def test_unconfirmed_renders_inconclusive():
    sr = _severity_result(
        fixable_leakage=0.20,
        p_value=None,
        nsl=None,
        null_iqr=None,
        null_99th=None,
        n_permutations_run=0,
    )
    report = build_verdict(sr)
    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "⚠ INCONCLUSIVE" in out
    assert "✓ TRUSTED" not in out


# ── Per-verdict translation lines ─────────────────────────────────────────────

def test_trusted_translation():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "no evidence of leakage" in out


def test_risky_translation():
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "noticeably worse in the real world" in out


def test_failed_translation():
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "cannot be trusted as a guide to real-world performance" in out


def test_inconclusive_translation():
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "could not certify this result" in out


# ── Action lines ──────────────────────────────────────────────────────────────

def test_risky_action_names_top_feature():
    attr = _fake_attribution("bad_col", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Start by removing or rebuilding" in out
    assert "'bad_col'" in out


def test_failed_action_names_top_feature():
    attr = _fake_attribution("evil_feat", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Start by removing or rebuilding" in out
    assert "'evil_feat'" in out


def test_inconclusive_action_is_firm():
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Do not treat this as trusted" in out


def test_trusted_has_no_action_line():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Start by removing or rebuilding" not in out
    assert "Do not treat this as trusted" not in out


# ── Per-worst-feature translation ─────────────────────────────────────────────

def test_risky_per_feature_translation():
    attr = _fake_attribution("toxic_col", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "already contain information about the answer" in out
    assert "'toxic_col'" in out


def test_failed_per_feature_translation():
    attr = _fake_attribution("leak_col", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "already contain information about the answer" in out
    assert "'leak_col'" in out


def test_inconclusive_per_feature_translation():
    attr = _fake_attribution("suspect_col", leakage=0.18)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "could not be statistically confirmed" in out
    assert "'suspect_col'" in out


def test_inconclusive_no_confident_tail():
    attr = _fake_attribution("suspect_col", leakage=0.18)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "that is why the results look better than they really are" not in out


def test_trusted_no_per_feature_translation():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "already contain information about the answer" not in out


# ── Scope footer always present ───────────────────────────────────────────────

def test_trusted_footer():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Scope note" in out
    assert "not that the whole pipeline is guaranteed clean" in out


def test_risky_footer():
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Scope note" in out
    assert "Additional issues may exist" in out


def test_failed_footer():
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Scope note" in out
    assert "Additional issues may exist" in out


def test_inconclusive_footer_not_a_pass():
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Scope note" in out
    assert "not a pass" in out


# ── Cause-appropriate INCONCLUSIVE guidance ───────────────────────────────────

class _FakeAnnotation:
    what = "Structural pattern detected: entity-level aggregate."


def _unconfirmed_statistical_report():
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
    )
    return build_verdict(sr)


def _unconfirmed_structural_report():
    report = _unconfirmed_statistical_report()
    return report.model_copy(update={"structural_annotations": [_FakeAnnotation()]})


def test_inconclusive_statistical_guidance_when_no_annotations():
    out = render_verdict(_unconfirmed_statistical_report(), stream=_Utf8Stream())
    assert "Add more time periods" in out
    assert "Adding more data will not resolve this" not in out


def test_inconclusive_structural_guidance_when_annotations_present():
    out = render_verdict(_unconfirmed_structural_report(), stream=_Utf8Stream())
    assert "Adding more data will not resolve this" in out
    assert "Add more time periods" not in out


def test_no_sample_count_in_statistical_inconclusive():
    out = render_verdict(_unconfirmed_statistical_report(), stream=_Utf8Stream())
    assert "samples" not in out


def test_no_sample_count_in_structural_inconclusive():
    out = render_verdict(_unconfirmed_structural_report(), stream=_Utf8Stream())
    assert "samples" not in out


def test_gather_more_data_absent_in_structural_render():
    out = render_verdict(_unconfirmed_structural_report(), stream=_Utf8Stream())
    assert "Gather more data" not in out


def test_gather_more_data_absent_in_statistical_render():
    out = render_verdict(_unconfirmed_statistical_report(), stream=_Utf8Stream())
    assert "Gather more data" not in out


# ── Structural annotations in TRUSTED / RISKY / FAILED states ────────────────

def _annotated_pass_report():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    return report.model_copy(update={"structural_annotations": [_FakeAnnotation()]})


def _annotated_warn_report():
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    return report.model_copy(update={"structural_annotations": [_FakeAnnotation()]})


def _annotated_fail_report():
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    return report.model_copy(update={"structural_annotations": [_FakeAnnotation()]})


def test_trusted_with_annotation_shows_structural_heading():
    out = render_verdict(_annotated_pass_report(), stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" in out


def test_trusted_with_annotation_shows_also_noticed():
    out = render_verdict(_annotated_pass_report(), stream=_Utf8Stream())
    assert "also noticed" in out.lower()


def test_trusted_with_annotation_shows_what():
    out = render_verdict(_annotated_pass_report(), stream=_Utf8Stream())
    assert _FakeAnnotation.what in out


def test_trusted_with_annotation_still_shows_trusted_marker():
    out = render_verdict(_annotated_pass_report(), stream=_Utf8Stream())
    assert "✓ TRUSTED" in out


def test_trusted_without_annotation_no_structural_block():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" not in out


def test_risky_with_annotation_shows_structural_heading():
    out = render_verdict(_annotated_warn_report(), stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" in out


def test_risky_with_annotation_shows_what():
    out = render_verdict(_annotated_warn_report(), stream=_Utf8Stream())
    assert _FakeAnnotation.what in out


def test_risky_without_annotation_no_structural_block():
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(
        fixable_leakage=0.12, p_value=0.001, nsl=1.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" not in out


def test_failed_with_annotation_shows_structural_heading():
    out = render_verdict(_annotated_fail_report(), stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" in out


def test_failed_with_annotation_shows_what():
    out = render_verdict(_annotated_fail_report(), stream=_Utf8Stream())
    assert _FakeAnnotation.what in out


def test_failed_without_annotation_no_structural_block():
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=0.001, nsl=2.5,
        null_iqr=0.01, null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict(report, stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" not in out


def test_inconclusive_annotation_behavior_unchanged():
    """INCONCLUSIVE + annotation shows STRUCTURAL FINDING block (byte-identical)."""
    out = render_verdict(_unconfirmed_structural_report(), stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" in out
    assert _FakeAnnotation.what in out


# ── Spec 1: across-entity attribution line ─────────────────────────────────────
# Pure synthetic SeverityResult inputs — fast, no real permutation run.

def test_across_entity_line_shown_on_warn_when_channel_across():
    sr = _severity_result(
        fixable_leakage=0.12,
        p_value=0.5, nsl=-1.0,               # within-entity did NOT fire
        p_value_across=0.005, nsl_across=2.0,  # across-entity fired
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
    )
    report = build_verdict(sr)
    assert report.engine_detection.detection_channel == "across_entity"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "across-entity" in out.lower()
    assert "⚠ RISKY" in out


def test_across_entity_line_shown_on_note_when_channel_both():
    sr = _severity_result(
        fixable_leakage=0.03,                 # below warn_floor -> NOTE
        p_value=0.005, nsl=2.0,
        p_value_across=0.005, nsl_across=2.0,
        per_fold=_interior_folds([0.03, 0.03, 0.03]),
    )
    report = build_verdict(sr)
    assert report.engine_detection.detection_channel == "both"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "across-entity" in out.lower()


def test_across_entity_line_absent_when_within_only():
    """Within-entity-only detection must NOT show the across-entity line (no clutter)."""
    sr = _severity_result(
        fixable_leakage=0.12,
        p_value=0.001, nsl=2.5,
        per_fold=_interior_folds([0.12, 0.11, 0.13]),
    )
    report = build_verdict(sr)
    assert report.engine_detection.detection_channel == "within_entity"
    out = render_verdict(report, stream=_Utf8Stream())
    assert "across-entity" not in out.lower()


def test_across_entity_line_absent_when_pass():
    sr = _severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5)
    report = build_verdict(sr)
    assert report.engine_detection.detection_channel == ""
    out = render_verdict(report, stream=_Utf8Stream())
    assert "across-entity" not in out.lower()
