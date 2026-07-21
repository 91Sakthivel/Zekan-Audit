"""Tests for the undeclared-feature screen's SURFACE work (Upgrade 1 step 1f).

Covers:
  - NEAR_CERTAIN renders prominently, BEFORE the damage/fix sections, in
    every verdict state including TRUSTED (the B-3 motivating case)
  - the feature name and univariate AUC number appear in the rendered text
  - the ranked panel renders with no threshold/suspicion language, includes
    the "screened X of Y" line and the not-screenable count
  - the scope footer is corrected (no longer claims Zekan doesn't inspect
    undeclared features) whenever the screen ran
  - cp1252 safety of all new wording
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
    NearCertainUndeclaredLeakDetail,
)
from zekan.detectors.undeclared_feature_probe import (
    NotScreenableEntry,
    PanelEntry,
    UndeclaredFeaturePanel,
)


# ── Shared fixtures (mirrors test_html_view.py's own builders) ───────────────

class _Utf8Stream:
    encoding = "utf-8"


class _Cp1252Stream:
    encoding = "cp1252"


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


def _panel(
    entries: list[tuple[str, float]] | None = None,
    screened: int = 49, total: int = 49,
    not_screenable: list[str] | None = None,
) -> UndeclaredFeaturePanel:
    entries = entries or [("readmitted", 1.0), ("number_inpatient", 0.61)]
    return UndeclaredFeaturePanel(
        entries=[
            PanelEntry(feature=f, univariate_auc=a, name_pattern_score=0.0, n_folds_evaluated=5)
            for f, a in entries
        ],
        screened_count=screened, total_features=total,
        not_screenable=[NotScreenableEntry(feature=f, reason="insufficient usable data")
                         for f in (not_screenable or [])],
    )


def _with_screen(report: VerdictReport, near_certain=True, panel=True) -> VerdictReport:
    updates: dict = {}
    if near_certain:
        updates["structural_annotations"] = [_near_certain_annotation()]
    if panel:
        updates["undeclared_feature_panel"] = _panel()
    return report.model_copy(update=updates)


# ── NEAR_CERTAIN: prominent, ordered before damage/fix, in every state ──────

def test_near_certain_renders_in_trusted_state():
    """B-3's own case: a NEAR_CERTAIN finding must render even when the
    verdict is TRUSTED/PASS."""
    report = _with_screen(_pass_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert "NEAR-CERTAIN" in out
    assert "readmitted" in out


def test_near_certain_before_damage_section_warn():
    report = _with_screen(_warn_report())
    out = render_verdict(report, stream=_Utf8Stream())
    idx_nc = out.find("NEAR-CERTAIN")
    idx_damage = out.find("THE DAMAGE")
    assert idx_nc != -1 and idx_damage != -1
    assert idx_nc < idx_damage, "NEAR_CERTAIN must render before THE DAMAGE, not after"


def test_near_certain_before_damage_section_fail():
    report = _with_screen(_fail_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.find("NEAR-CERTAIN") < out.find("THE DAMAGE")


def test_near_certain_before_damage_section_inconclusive():
    report = _with_screen(_unconfirmed_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert out.find("NEAR-CERTAIN") < out.find("THE DAMAGE")


def test_near_certain_not_in_trailing_structural_finding_block():
    """The prominent NEAR_CERTAIN block replaces the trailing generic
    rendering for this specific finding -- it must not ALSO appear as a bare
    `ann.what` line in the STRUCTURAL FINDING section (no duplication)."""
    report = _pass_report().model_copy(
        update={"structural_annotations": [_near_certain_annotation()]}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    # The prominent block is present...
    assert out.count("NEAR-CERTAIN") == 1
    # ...and the generic annotation loop's raw `.what` text ("w") does not
    # ALSO appear as a standalone structural-finding line.
    assert "STRUCTURAL FINDING" not in out


def test_near_certain_auc_number_in_rendered_text():
    report = _with_screen(_pass_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert "1.0000" in out


def test_near_certain_names_feature_and_auc_html():
    report = _with_screen(_pass_report())
    out = render_verdict_html(report)
    assert "readmitted" in out
    assert "1.0000" in out
    assert "NEAR-CERTAIN" in out


def test_near_certain_html_before_damage_section():
    report = _with_screen(_warn_report())
    out = render_verdict_html(report)
    assert out.find("NEAR-CERTAIN") < out.find("THE DAMAGE")


def test_near_certain_tie_multiple_features_all_rendered():
    ann_a = _near_certain_annotation("copy_a", 1.0)
    ann_b = _near_certain_annotation("copy_b", 1.0)
    report = _pass_report().model_copy(
        update={"structural_annotations": [ann_a, ann_b], "undeclared_feature_panel": _panel()}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "copy_a" in out
    assert "copy_b" in out


def test_non_near_certain_annotations_still_render_in_trailing_position():
    """Other structural findings (not NEAR_CERTAIN) must be unaffected --
    still render in the original trailing STRUCTURAL FINDING position."""
    class _OtherFinding:
        issue_type = IssueType.ROW_DUPLICATION
        what = "3 duplicate rows found."

    report = _pass_report().model_copy(update={"structural_annotations": [_OtherFinding()]})
    out = render_verdict(report, stream=_Utf8Stream())
    assert "STRUCTURAL FINDING" in out
    assert "3 duplicate rows found." in out
    assert "NEAR-CERTAIN" not in out


# ── Ranked panel: informational, no threshold/suspicion language ────────────

def test_panel_renders_screened_x_of_y():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel(screened=49, total=51)})
    out = render_verdict(report, stream=_Utf8Stream())
    assert "Screened 49 of 51" in out


def test_panel_renders_not_screenable_count():
    report = _pass_report().model_copy(
        update={"undeclared_feature_panel": _panel(not_screenable=["weight"])}
    )
    out = render_verdict(report, stream=_Utf8Stream())
    assert "1 feature(s) could not be screened" in out


def test_panel_no_not_screenable_line_when_none():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel(not_screenable=[])})
    out = render_verdict(report, stream=_Utf8Stream())
    assert "could not be screened" not in out


def test_panel_wording_has_no_suspicion_or_threshold_language():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel()})
    out = render_verdict(report, stream=_Utf8Stream())
    # Isolate the panel section (between its heading and the divider) so this
    # check is about the panel specifically, not the whole document.
    start = out.find("FEATURES RANKED BY ASSOCIATION")
    end = out.find("─" * 10) if "─" * 10 in out else len(out)
    panel_text = out[start:end].lower()
    for banned in ("suspicious", "flagged", "flag", "threshold", "cutoff", "exceeds"):
        assert banned not in panel_text, f"banned word {banned!r} found in panel text"


def test_panel_includes_next_action():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel()})
    out = render_verdict(report, stream=_Utf8Stream())
    assert "forbidden_after_prediction" in out
    assert "re-run" in out


def test_panel_renders_html_without_suspicion_language():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel()})
    out = render_verdict_html(report)
    start = out.find("FEATURES RANKED BY ASSOCIATION")
    lowered = out[start:].lower()
    for banned in ("suspicious", "flagged", "threshold"):
        assert banned not in lowered


def test_panel_none_renders_nothing():
    """When the screen didn't run (panel is None), no panel section appears
    and the original footer wording is unchanged."""
    out = render_verdict(_pass_report(), stream=_Utf8Stream())
    assert "FEATURES RANKED" not in out
    assert "does not inspect undeclared features" in out


# ── Scope footer honesty ──────────────────────────────────────────────────────

def test_footer_drops_false_claim_when_screen_ran():
    report = _with_screen(_pass_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert "does not inspect undeclared features" not in out
    assert "screens non-forbidden features" in out


def test_footer_unchanged_when_screen_did_not_run():
    out = render_verdict(_pass_report(), stream=_Utf8Stream())
    assert "does not inspect undeclared features" in out


def test_footer_honest_for_warn_and_fail_with_screen():
    for report in (_with_screen(_warn_report()), _with_screen(_fail_report())):
        out = render_verdict(report, stream=_Utf8Stream())
        assert "Additional issues may exist" in out
        assert "univariate screen" in out


def test_footer_honest_for_inconclusive_with_screen():
    report = _with_screen(_unconfirmed_report())
    out = render_verdict(report, stream=_Utf8Stream())
    assert "not a pass" in out
    assert "does not inspect undeclared features" not in out


def test_footer_panel_only_no_near_certain_still_corrected():
    """Footer correction is keyed on panel-or-near-certain presence, not
    near-certain alone -- a clean screen (panel present, nothing flagged)
    still ran and inspected undeclared features."""
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel()})
    out = render_verdict(report, stream=_Utf8Stream())
    assert "does not inspect undeclared features" not in out


# ── cp1252 safety (explicit encode check, not assumed) ───────────────────────

def test_cp1252_encodes_cleanly_with_screen_all_states():
    for report in (
        _with_screen(_pass_report()),
        _with_screen(_warn_report()),
        _with_screen(_fail_report()),
        _with_screen(_unconfirmed_report()),
    ):
        out = render_verdict(report, stream=_Cp1252Stream())
        out.encode("cp1252")  # must not raise


def test_cp1252_panel_only_encodes_cleanly():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel(not_screenable=["weight"])})
    out = render_verdict(report, stream=_Cp1252Stream())
    out.encode("cp1252")


# ── Text/HTML parity (drift-lock) ────────────────────────────────────────────

def test_parity_near_certain_heading():
    report = _with_screen(_pass_report())
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "NEAR-CERTAIN UNDECLARED LEAK" in text_out
    assert "NEAR-CERTAIN UNDECLARED LEAK" in html_out


def test_parity_near_certain_lead_sentence():
    report = _with_screen(_pass_report())
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "predicts the answer almost perfectly" in text_out
    assert "predicts the answer almost perfectly" in html_out
    assert "was NOT declared forbidden" in text_out
    assert "was NOT declared forbidden" in html_out


def test_parity_panel_screened_line():
    report = _pass_report().model_copy(update={"undeclared_feature_panel": _panel(screened=10, total=20)})
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "Screened 10 of 20" in text_out
    assert "Screened 10 of 20" in html_out


def test_parity_footer_with_screen():
    report = _with_screen(_pass_report())
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "TRUSTED means no issue was found within this scope" in text_out
    assert "TRUSTED means no issue was found within this scope" in html_out
    assert "not that the whole pipeline is guaranteed clean" in text_out
    assert "not that the whole pipeline is guaranteed clean" in html_out
