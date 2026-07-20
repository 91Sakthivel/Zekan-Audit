"""Unit tests for zekan/reports/json_export.py.

All tests build VerdictReport objects directly — no training, no CLI invocation.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.engine import PerFoldSeverity, SeverityResult
from zekan.severity.verdict import VerdictReport, build_verdict
from zekan.reports.json_export import verdict_to_dict, verdict_to_json


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _severity(
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
    return SeverityResult(
        status="pass",
        metric="roc_auc",
        naive_auc=naive_auc,
        estimated_deployable_auc=estimated_deployable_auc,
        total_optimism=naive_auc - estimated_deployable_auc,
        fixable_leakage=fixable_leakage,
        fixable_leakage_range=(fixable_leakage * 0.8, fixable_leakage * 1.2),
        nonfixable_optimism=nonfixable_optimism,
        per_fold=per_fold or [],
        caveat="test",
        null_iqr=null_iqr,
        null_99th=null_99th,
        p_value=p_value,
        nsl=nsl,
        n_permutations_run=n_permutations_run,
        feature_attribution=feature_attribution,
    )


def _fake_attribution(feature: str = "bad_col", leakage: float = 0.20) -> AblationSummary:
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


@pytest.fixture
def trusted_report() -> VerdictReport:
    return build_verdict(_severity(fixable_leakage=0.03, p_value=0.50, nsl=-0.5))


@pytest.fixture
def failed_report() -> VerdictReport:
    attr = _fake_attribution("top_offender", leakage=0.20)
    sr = _severity(
        fixable_leakage=0.20,
        p_value=0.001,
        nsl=2.5,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    return build_verdict(sr)


@pytest.fixture
def nan_report() -> VerdictReport:
    """Report whose AblationEntry has NaN auc_without / leakage_estimate."""
    nan_entry = AblationEntry(
        feature="suspect_col",
        is_contract_forbidden=True,
        univariate_auc=0.80,
        name_pattern_score=0.0,
        rank_score=1.5,
        auc_without=float("nan"),
        leakage_estimate=float("nan"),
        ablated=False,
        not_ablated_reason="beyond_top_k",
    )
    attr = AblationSummary(baseline_auc=0.80, individual=[nan_entry])
    sr = _severity(
        fixable_leakage=0.20,
        p_value=0.001,
        nsl=2.5,
        null_iqr=0.01,
        null_99th=0.09,
        per_fold=_interior_folds([0.20, 0.19, 0.21]),
        feature_attribution=attr,
    )
    return build_verdict(sr)


# ── Helpers for walking the dict ──────────────────────────────────────────────

def _all_scalars(obj: Any):
    """Recursively yield all non-container leaf values."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _all_scalars(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_scalars(v)
    else:
        yield obj


# ── Round-trip ────────────────────────────────────────────────────────────────

def test_round_trip_trusted(trusted_report):
    text = verdict_to_json(trusted_report)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


def test_round_trip_failed(failed_report):
    text = verdict_to_json(failed_report)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_determinism(failed_report):
    out1 = verdict_to_json(failed_report, indent=2)
    out2 = verdict_to_json(failed_report, indent=2)
    assert out1 == out2


def test_determinism_trusted(trusted_report):
    assert verdict_to_json(trusted_report) == verdict_to_json(trusted_report)


# ── NaN coercion ──────────────────────────────────────────────────────────────

def test_nan_coercion_does_not_raise(nan_report):
    text = verdict_to_json(nan_report)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


def test_nan_becomes_null(nan_report):
    d = verdict_to_dict(nan_report)
    individual = d["measured_damage"]["feature_attribution"]["individual"]
    assert len(individual) == 1
    assert individual[0]["auc_without"] is None
    assert individual[0]["leakage_estimate"] is None


def test_nan_in_unavailable_engine_result():
    from zekan.severity.engine import SeverityResult as _SR
    sr = _SR(
        status="unavailable",
        metric="roc_auc",
        naive_auc=float("nan"),
        estimated_deployable_auc=float("nan"),
        total_optimism=float("nan"),
        fixable_leakage=float("nan"),
        fixable_leakage_range=(float("nan"), float("nan")),
        nonfixable_optimism=float("nan"),
        per_fold=[],
        caveat="unavailable",
        null_iqr=None,
        null_99th=None,
        p_value=None,
        nsl=None,
        n_permutations_run=0,
        feature_attribution=None,
        unavailable_reason="test unavailable",
    )
    report = build_verdict(sr)
    text = verdict_to_json(report)
    parsed = json.loads(text)
    assert parsed["summary"]["fixable_leakage"] is None


# ── No numpy types ────────────────────────────────────────────────────────────

_JSON_NATIVE = (bool, int, float, str, type(None))


def test_no_numpy_types_trusted(trusted_report):
    d = verdict_to_dict(trusted_report)
    for scalar in _all_scalars(d):
        assert type(scalar) in _JSON_NATIVE, f"non-native: {type(scalar)} {scalar!r}"


def test_no_numpy_types_failed(failed_report):
    d = verdict_to_dict(failed_report)
    for scalar in _all_scalars(d):
        assert type(scalar) in _JSON_NATIVE, f"non-native: {type(scalar)} {scalar!r}"


# ── Every key always present ──────────────────────────────────────────────────

_TOP_LEVEL_KEYS = {
    "engine_detection",
    "fold_ci",
    "gate",           # added by CLI, not by verdict_to_dict — tested separately
    "measured_damage",
    "policy_decision",
    "schema_version",
    "structural_annotations",
    "summary",
    "undeclared_feature_panel",  # Upgrade 1 step 1e; None when the screen didn't fire
}

_TOP_LEVEL_KEYS_EXPORT = _TOP_LEVEL_KEYS - {"gate"}  # gate is CLI-only


def test_trusted_all_export_keys_present(trusted_report):
    d = verdict_to_dict(trusted_report)
    assert _TOP_LEVEL_KEYS_EXPORT == set(d.keys())


def test_failed_all_export_keys_present(failed_report):
    d = verdict_to_dict(failed_report)
    assert _TOP_LEVEL_KEYS_EXPORT == set(d.keys())


def test_trusted_top_feature_is_null(trusted_report):
    d = verdict_to_dict(trusted_report)
    assert d["summary"]["top_feature"] is None


def test_trusted_structural_annotations_empty_list(trusted_report):
    d = verdict_to_dict(trusted_report)
    assert d["structural_annotations"] == []


def test_failed_feature_attribution_under_measured_damage(failed_report):
    """feature_attribution stays nested under measured_damage — not hoisted."""
    d = verdict_to_dict(failed_report)
    assert "feature_attribution" in d["measured_damage"]
    assert "feature_attribution" not in d  # not at top level


# ── summary block ─────────────────────────────────────────────────────────────

def test_summary_verdict_matches_policy_decision(failed_report):
    d = verdict_to_dict(failed_report)
    assert d["summary"]["verdict"] == d["policy_decision"]["verdict"]


def test_summary_fixable_leakage_matches_measured_damage(failed_report):
    d = verdict_to_dict(failed_report)
    assert d["summary"]["fixable_leakage"] == d["measured_damage"]["fixable_leakage"]


def test_summary_top_feature_matches_text_view_sort(failed_report):
    """top_feature uses same _sorted_ablated sort as text_view — same-source proof."""
    from zekan.reports.text_view import _sorted_ablated
    d = verdict_to_dict(failed_report)
    ablated = _sorted_ablated(failed_report.measured_damage.feature_attribution)
    expected = ablated[0].feature if ablated else None
    assert d["summary"]["top_feature"] == expected


def test_summary_headline_trusted(trusted_report):
    import zekan.reports.messages as _MSG
    d = verdict_to_dict(trusted_report)
    assert d["summary"]["headline"] == _MSG.TRANSLATION_TRUSTED


def test_summary_headline_failed(failed_report):
    import zekan.reports.messages as _MSG
    d = verdict_to_dict(failed_report)
    assert d["summary"]["headline"] == _MSG.TRANSLATION_FAILED


# ── schema_version ────────────────────────────────────────────────────────────

def test_schema_version_trusted(trusted_report):
    d = verdict_to_dict(trusted_report)
    assert d["schema_version"] == "1"


def test_schema_version_failed(failed_report):
    d = verdict_to_dict(failed_report)
    assert d["schema_version"] == "1"


# ── verdict_to_json indent ────────────────────────────────────────────────────

def test_no_indent_is_compact(trusted_report):
    text = verdict_to_json(trusted_report)
    assert "\n" not in text


def test_indent_2_is_pretty(trusted_report):
    text = verdict_to_json(trusted_report, indent=2)
    assert "\n" in text


# ── VerdictReport.to_dict / to_json convenience methods ──────────────────────

def test_to_dict_method(failed_report):
    assert failed_report.to_dict() == verdict_to_dict(failed_report)


def test_to_json_method(failed_report):
    assert failed_report.to_json() == verdict_to_json(failed_report)


def test_to_json_method_indent(trusted_report):
    assert trusted_report.to_json(indent=2) == verdict_to_json(trusted_report, indent=2)


# ── structural_annotations serialized correctly ───────────────────────────────

def test_folds_not_in_json_output(trusted_report):
    """SeverityResult.folds is an internal artifact and must not appear in the JSON."""
    text = verdict_to_json(trusted_report)
    assert '"folds"' not in text


def test_structural_annotations_serialized(trusted_report):
    class _FakeAnnotation:
        def model_dump(self, *, mode="python"):
            return {"issue_type": "test_issue", "what": "test what"}

    report = trusted_report.model_copy(update={"structural_annotations": [_FakeAnnotation()]})
    d = verdict_to_dict(report)
    assert len(d["structural_annotations"]) == 1
    assert d["structural_annotations"][0]["issue_type"] == "test_issue"
