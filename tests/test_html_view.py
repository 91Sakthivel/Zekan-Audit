"""Tests for the HTML formatter and VerdictReport repr hooks.

Covers:
  - Four state banner words (TRUSTED / RISKY / FAILED / INCONCLUSIVE)
  - TRUSTED: no WHAT TO FIX block
  - RISKY / FAILED: WHAT TO FIX present, top feature name present
  - INCONCLUSIVE: TRUSTED word absent
  - HTML escaping: <script> feature name must not appear unescaped
  - Drift-lock: text_view and html_view agree on same state
  - _repr_html_ and __str__ hooks on VerdictReport
"""

from __future__ import annotations

import pytest

from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.engine import PerFoldSeverity, SeverityResult
from zekan.severity.verdict import VerdictReport, build_verdict
from zekan.reports.html_view import render_verdict_html
from zekan.reports.text_view import render_verdict


# ── Shared test helpers (mirrored from test_text_view.py) ────────────────────

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
        n_permutations_run=n_permutations_run,
        feature_attribution=feature_attribution,
    )


def _fake_attribution(
    feature: str = "bad_feature",
    leakage: float = 0.15,
) -> AblationSummary:
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


def _pass_report() -> VerdictReport:
    return build_verdict(_severity_result(fixable_leakage=0.03, p_value=0.50, nsl=-0.5))


def _warn_report() -> VerdictReport:
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
    return build_verdict(sr)


def _fail_report() -> VerdictReport:
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
    return build_verdict(sr)


def _unconfirmed_report() -> VerdictReport:
    sr = _severity_result(
        fixable_leakage=0.20,
        p_value=None,
        nsl=None,
        null_iqr=None,
        null_99th=None,
        n_permutations_run=0,
    )
    return build_verdict(sr)


# ── State banner tests ────────────────────────────────────────────────────────

def test_trusted_banner():
    out = render_verdict_html(_pass_report())
    assert "TRUSTED" in out


def test_risky_banner():
    out = render_verdict_html(_warn_report())
    assert "RISKY" in out


def test_failed_banner():
    out = render_verdict_html(_fail_report())
    assert "FAILED" in out


def test_inconclusive_banner():
    out = render_verdict_html(_unconfirmed_report())
    assert "INCONCLUSIVE" in out


# ── Structural content tests ──────────────────────────────────────────────────

def test_trusted_has_no_fix_block():
    out = render_verdict_html(_pass_report())
    assert "WHAT TO FIX" not in out


def test_risky_has_fix_block_and_feature():
    out = render_verdict_html(_warn_report())
    assert "WHAT TO FIX" in out
    assert "top_offender" in out


def test_failed_has_fix_block_and_feature():
    out = render_verdict_html(_fail_report())
    assert "WHAT TO FIX" in out
    assert "top_offender" in out


def test_inconclusive_does_not_say_trusted():
    out = render_verdict_html(_unconfirmed_report())
    assert "TRUSTED" not in out


# ── HTML escaping ─────────────────────────────────────────────────────────────

def test_feature_name_is_escaped():
    """A column named '<script>x' must not inject raw markup."""
    attr = _fake_attribution(feature="<script>x", leakage=0.18)
    sr = _severity_result(
        fixable_leakage=0.18,
        p_value=0.001,
        nsl=2.0,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.18, 0.17, 0.19]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict_html(report)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_ampersand_in_feature_name_is_escaped():
    attr = _fake_attribution(feature="a & b", leakage=0.18)
    sr = _severity_result(
        fixable_leakage=0.18,
        p_value=0.001,
        nsl=2.0,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.18, 0.17, 0.19]),
        feature_attribution=attr,
    )
    report = build_verdict(sr)
    out = render_verdict_html(report)
    assert "a & b" not in out
    assert "a &amp; b" in out


# ── Fake UTF-8 stream — pins the glyph path ──────────────────────────────────

class _Utf8Stream:
    encoding = "utf-8"


# ── Drift-lock: text and HTML agree on state ──────────────────────────────────

def test_text_and_html_agree_on_risky_state():
    report = _warn_report()
    assert "⚠ RISKY" in render_verdict(report, stream=_Utf8Stream())
    assert "RISKY" in render_verdict_html(report)


# ── VerdictReport repr hooks ──────────────────────────────────────────────────

def test_repr_html_returns_html_string():
    report = _warn_report()
    html_out = report._repr_html_()
    assert isinstance(html_out, str)
    assert "RISKY" in html_out
    assert "<div" in html_out


def test_str_equals_render_verdict():
    report = _pass_report()
    assert str(report) == render_verdict(report)


def test_str_on_fail_report():
    report = _fail_report()
    assert str(report) == render_verdict(report)


# ── Per-verdict translation lines ─────────────────────────────────────────────

def test_html_trusted_translation():
    out = render_verdict_html(_pass_report())
    assert "no evidence of leakage" in out


def test_html_risky_translation():
    out = render_verdict_html(_warn_report())
    assert "noticeably worse in the real world" in out


def test_html_failed_translation():
    out = render_verdict_html(_fail_report())
    assert "cannot be trusted as a guide to real-world performance" in out


def test_html_inconclusive_translation():
    out = render_verdict_html(_unconfirmed_report())
    assert "could not certify this result" in out


# ── Action lines ──────────────────────────────────────────────────────────────

def test_html_risky_action_names_top_feature():
    out = render_verdict_html(_warn_report())
    assert "Start by removing or rebuilding" in out
    assert "top_offender" in out


def test_html_failed_action_names_top_feature():
    out = render_verdict_html(_fail_report())
    assert "Start by removing or rebuilding" in out
    assert "top_offender" in out


def test_html_inconclusive_action_is_firm():
    out = render_verdict_html(_unconfirmed_report())
    assert "Do not treat this as trusted" in out


def test_html_trusted_has_no_action_line():
    out = render_verdict_html(_pass_report())
    assert "Start by removing or rebuilding" not in out
    assert "Do not treat this as trusted" not in out


# ── Per-worst-feature translation ─────────────────────────────────────────────

def test_html_risky_per_feature_translation():
    out = render_verdict_html(_warn_report())
    assert "already contain information about the answer" in out
    assert "top_offender" in out


def test_html_failed_per_feature_translation():
    out = render_verdict_html(_fail_report())
    assert "already contain information about the answer" in out
    assert "top_offender" in out


def _unconfirmed_with_attr_report() -> VerdictReport:
    attr = _fake_attribution("suspect_col", leakage=0.18)
    sr = _severity_result(
        fixable_leakage=0.20, p_value=None, nsl=None,
        null_iqr=None, null_99th=None, n_permutations_run=0,
        feature_attribution=attr,
    )
    return build_verdict(sr)


def test_html_inconclusive_per_feature_translation():
    out = render_verdict_html(_unconfirmed_with_attr_report())
    assert "could not be statistically confirmed" in out
    assert "suspect_col" in out


def test_html_trusted_no_per_feature_translation():
    out = render_verdict_html(_pass_report())
    assert "already contain information about the answer" not in out


# ── Scope footer always present ───────────────────────────────────────────────

def test_html_trusted_footer():
    out = render_verdict_html(_pass_report())
    assert "Scope note" in out
    assert "not that the whole pipeline is guaranteed clean" in out


def test_html_risky_footer():
    out = render_verdict_html(_warn_report())
    assert "Scope note" in out
    assert "Additional issues may exist" in out


def test_html_failed_footer():
    out = render_verdict_html(_fail_report())
    assert "Scope note" in out
    assert "Additional issues may exist" in out


def test_html_inconclusive_footer_not_a_pass():
    out = render_verdict_html(_unconfirmed_report())
    assert "Scope note" in out
    assert "not a pass" in out


# ── Footer/messages parity drift-lock ────────────────────────────────────────

def test_footer_parity_trusted():
    report = _pass_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "not that the whole pipeline is guaranteed clean" in text_out
    assert "not that the whole pipeline is guaranteed clean" in html_out


def test_footer_parity_risky():
    report = _warn_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "Additional issues may exist" in text_out
    assert "Additional issues may exist" in html_out


def test_footer_parity_failed():
    report = _fail_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "Additional issues may exist" in text_out
    assert "Additional issues may exist" in html_out


def test_footer_parity_inconclusive():
    report = _unconfirmed_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "not a pass" in text_out
    assert "not a pass" in html_out


def test_translation_parity_risky():
    report = _warn_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "noticeably worse in the real world" in text_out
    assert "noticeably worse in the real world" in html_out


def test_translation_parity_inconclusive():
    report = _unconfirmed_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "could not certify this result" in text_out
    assert "could not certify this result" in html_out


def test_feature_translation_parity_risky():
    report = _warn_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "that is why the results look better than they really are" in text_out
    assert "that is why the results look better than they really are" in html_out


def test_feature_translation_parity_inconclusive_unconfirmed():
    report = _unconfirmed_with_attr_report()
    text_out = render_verdict(report, stream=_Utf8Stream())
    html_out = render_verdict_html(report)
    assert "could not be statistically confirmed" in text_out
    assert "could not be statistically confirmed" in html_out
    assert "that is why the results look better than they really are" not in text_out
    assert "that is why the results look better than they really are" not in html_out
