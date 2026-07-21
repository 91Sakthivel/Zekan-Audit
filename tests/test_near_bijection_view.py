"""Tests for surfacing NEAR_BIJECTION_UNDECLARED_LEAK (Upgrade H) beside the
verdict, and its cross-reference with NEAR_CERTAIN_UNDECLARED_LEAK (Upgrade 1)
when both fire on the same feature.

Covers:
  - NEAR_BIJECTION renders prominently, BEFORE the damage/fix sections, in
    every verdict state including TRUSTED, with at least equal prominence to
    NEAR_CERTAIN (confirmed=True is the stronger claim)
  - the feature name, Theil's U value, and criterion appear in the rendered
    text
  - when NEAR_BIJECTION and NEAR_CERTAIN name the SAME feature, the output is
    a single primary statement (NEAR_BIJECTION) plus a short corroboration
    line -- NOT two separate restatements of the same claim
  - when they name DIFFERENT features, both render standalone, unmerged
  - the verdict line itself is unchanged in every case (annotate-only)
  - text/HTML parity (drift-lock)

Does not touch the probe, the criterion, or any threshold -- pure rendering.
"""

from __future__ import annotations

from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.engine import PerFoldSeverity, SeverityResult
from zekan.severity.verdict import VerdictReport, build_verdict
from zekan.reports.html_view import render_verdict_html
from zekan.reports.text_view import render_verdict
from zekan.detectors.schema import (
    Evidence,
    IssueRecord,
    IssueType,
    NearBijectionUndeclaredLeakDetail,
    NearCertainUndeclaredLeakDetail,
)


# ── Shared fixtures (mirrors test_undeclared_screen_view.py's own builders) ──

class _Utf8Stream:
    encoding = "utf-8"


def _interior_folds(leakages: list[float]) -> list[PerFoldSeverity]:
    return [
        PerFoldSeverity(fold_idx=i, auc_with_forbidden=0.5 + fl, auc_without_forbidden=0.5,
                         is_terminal=False)
        for i, fl in enumerate(leakages)
    ]


def _severity_result(
    fixable_leakage: float = 0.05,
    p_value: float | None = 0.50,
    nsl: float | None = -1.0,
    null_iqr: float | None = 0.010,
    null_99th: float | None = 0.040,
    n_permutations_run: int = 100,
    per_fold: list | None = None,
    feature_attribution: AblationSummary | None = None,
) -> SeverityResult:
    return SeverityResult(
        status="pass", metric="roc_auc", naive_auc=0.80, estimated_deployable_auc=0.75,
        total_optimism=0.05, fixable_leakage=fixable_leakage,
        fixable_leakage_range=(fixable_leakage * 0.8, fixable_leakage * 1.2),
        nonfixable_optimism=0.02, per_fold=per_fold or [], caveat="test",
        null_iqr=null_iqr, null_99th=null_99th, p_value=p_value, nsl=nsl,
        n_permutations_run=n_permutations_run, feature_attribution=feature_attribution,
    )


def _fake_attribution(feature: str = "bad_feature", leakage: float = 0.15) -> AblationSummary:
    entry = AblationEntry(
        feature=feature, is_contract_forbidden=True, univariate_auc=0.85,
        name_pattern_score=0.0, rank_score=1.7, auc_without=0.65,
        leakage_estimate=leakage, ablated=True,
    )
    return AblationSummary(baseline_auc=0.80, individual=[entry])


def _pass_report() -> VerdictReport:
    return build_verdict(_severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5))


def _warn_report() -> VerdictReport:
    attr = _fake_attribution("top_offender", leakage=0.12)
    sr = _severity_result(fixable_leakage=0.12, p_value=0.001, nsl=1.5, null_iqr=0.01,
                           null_99th=0.09, per_fold=_interior_folds([0.12, 0.11, 0.13]),
                           feature_attribution=attr)
    return build_verdict(sr)


def _fail_report() -> VerdictReport:
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity_result(fixable_leakage=0.20, p_value=0.001, nsl=2.5, null_iqr=0.01,
                           null_99th=0.09, per_fold=_interior_folds([0.20, 0.19, 0.21]),
                           feature_attribution=attr)
    return build_verdict(sr)


def _unconfirmed_report() -> VerdictReport:
    sr = _severity_result(fixable_leakage=0.20, p_value=None, nsl=None, null_iqr=None,
                           null_99th=None, n_permutations_run=0)
    return build_verdict(sr)


def _near_bijection_annotation(feature: str = "readmitted", theil_u: float = 1.0) -> IssueRecord:
    return IssueRecord(
        issue_type=IssueType.NEAR_BIJECTION_UNDECLARED_LEAK, status="fail",
        what="w", why="w", how_much="w", next_fix="w",
        evidence=Evidence(structural_detail=NearBijectionUndeclaredLeakDetail(
            feature=feature, theil_u=theil_u, support_floor=2,
            threshold_compared_against=0.99, n_distinct_raw=2,
            n_distinct_after_pooling=2, n_rows_pooled=0,
            screened_count=3, total_features=3,
        )),
    )


def _near_certain_annotation(feature: str = "readmitted", auc: float = 1.0) -> IssueRecord:
    return IssueRecord(
        issue_type=IssueType.NEAR_CERTAIN_UNDECLARED_LEAK, status="fail",
        what="w", why="w", how_much="w", next_fix="w",
        evidence=Evidence(structural_detail=NearCertainUndeclaredLeakDetail(
            feature=feature, univariate_auc=auc, threshold_compared_against=0.99,
            n_folds_evaluated=5, screened_count=49, total_features=49,
            name_pattern_score=0.0,
        )),
    )


# ── NEAR_BIJECTION alone: prominent, ordered before damage/fix, every state ──

def test_near_bijection_renders_in_trusted_state():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "NEAR-BIJECTION" in out
    assert "readmitted" in out


def test_near_bijection_before_damage_section_warn():
    report = _warn_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    idx_nb = out.find("NEAR-BIJECTION")
    idx_damage = out.find("THE DAMAGE")
    assert idx_nb != -1 and idx_damage != -1
    assert idx_nb < idx_damage, "NEAR_BIJECTION must render before THE DAMAGE, not after"


def test_near_bijection_before_damage_section_fail():
    report = _fail_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.find("NEAR-BIJECTION") < out.find("THE DAMAGE")


def test_near_bijection_before_damage_section_inconclusive():
    report = _unconfirmed_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.find("NEAR-BIJECTION") < out.find("THE DAMAGE")


def test_near_bijection_theil_u_and_criterion_in_rendered_text():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation(theil_u=0.9950)]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "0.9950" in out
    assert "0.99" in out


def test_near_bijection_not_in_trailing_structural_finding_block():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.count("NEAR-BIJECTION") == 1
    assert "STRUCTURAL FINDING" not in out


def test_near_bijection_names_feature_and_theil_u_html():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation(theil_u=0.9950)]}
    )
    out = render_verdict_html(report)
    assert "readmitted" in out
    assert "0.9950" in out
    assert "NEAR-BIJECTION" in out


def test_near_bijection_html_before_damage_section():
    report = _warn_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict_html(report)
    assert out.find("NEAR-BIJECTION") < out.find("THE DAMAGE")


def test_near_bijection_states_meaning_before_statistic():
    """Wording must be legible to a non-builder: state what the finding
    MEANS before any statistic (per UPGRADE_H_PREREGISTRATION.md)."""
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    meaning_idx = out.find("determine the target almost exactly")
    stat_idx = out.find("Theil's U")
    assert meaning_idx != -1 and stat_idx != -1
    assert meaning_idx < stat_idx


# ── BOTH fire on the SAME feature: merged, cross-referenced, not restated ────

def test_both_fired_same_feature_renders_bijection_as_primary():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "NEAR-BIJECTION" in out
    assert out.find("NEAR-BIJECTION") < out.find("THE DAMAGE") if "THE DAMAGE" in out else True


def test_both_fired_same_feature_does_not_restate_near_certain_lead():
    """CORRECTNESS REQUIREMENT: when the merged form renders, NEAR_CERTAIN
    must NOT also render its own separate block for that feature -- the
    NEAR_CERTAIN lead sentence must appear ZERO times."""
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "predicts the answer almost perfectly" not in out
    assert out.count("NEAR-CERTAIN") == 0


def test_both_fired_same_feature_includes_corroboration_line():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "two independent checks agree" in out
    assert "model-based screen flagged" in out


def test_both_fired_same_feature_not_duplicated_verbatim():
    """Not literally two restatements: the merged output must not contain
    the feature's headline claim twice."""
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.count("determine the target almost exactly") == 1


def test_both_fired_same_feature_html_no_separate_near_certain_box():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    out = render_verdict_html(report)
    assert out.count("NEAR-CERTAIN") == 0
    assert "NEAR-BIJECTION" in out
    assert "two independent checks agree" in out


# ── Multi-feature edge case: different features, both render, unmerged ──────

def test_different_features_both_render_unmerged():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_bijection_annotation("feature_a", 1.0),
            _near_certain_annotation("feature_b", 0.995),
        ]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "NEAR-BIJECTION" in out
    assert "NEAR-CERTAIN" in out
    assert "feature_a" in out
    assert "feature_b" in out
    assert "two independent checks agree" not in out


def test_different_features_both_render_unmerged_html():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_bijection_annotation("feature_a", 1.0),
            _near_certain_annotation("feature_b", 0.995),
        ]}
    )
    out = render_verdict_html(report)
    assert out.count("NEAR-BIJECTION") == 1
    assert out.count("NEAR-CERTAIN") == 1
    assert "feature_a" in out
    assert "feature_b" in out


# ── Verdict line unchanged (annotate-only) ───────────────────────────────────

def test_verdict_line_unchanged_solo():
    baseline = render_verdict(_pass_report(), stream=_Utf8Stream()).splitlines()[0]
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    annotated = render_verdict(report, stream=_Utf8Stream()).splitlines()[0]
    assert baseline == annotated


def test_verdict_line_unchanged_both_fired():
    baseline = render_verdict(_pass_report(), stream=_Utf8Stream()).splitlines()[0]
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    annotated = render_verdict(report, stream=_Utf8Stream()).splitlines()[0]
    assert baseline == annotated


# ── Footer honesty extends to NEAR_BIJECTION-only runs ───────────────────────

def test_footer_corrected_when_only_near_bijection_fired():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "does not inspect undeclared features" not in out


# ── Text/HTML parity (drift-lock) ────────────────────────────────────────────

def test_parity_near_bijection_heading():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "NEAR-BIJECTION UNDECLARED LEAK" in text_out
    assert "NEAR-BIJECTION UNDECLARED LEAK" in html_out


def test_parity_near_bijection_lead_sentence():
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_bijection_annotation()]}
    )
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "determine the target almost exactly" in text_out
    assert "determine the target almost exactly" in html_out
    assert "was NOT declared forbidden" in text_out
    assert "was NOT declared forbidden" in html_out


def test_parity_both_fired_corroboration_line():
    report = _pass_report().model_copy(
        update={"structural_annotations": [
            _near_certain_annotation("readmitted", 1.0),
            _near_bijection_annotation("readmitted", 1.0),
        ]}
    )
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "two independent checks agree" in text_out
    assert "two independent checks agree" in html_out
