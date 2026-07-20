"""Registry-level tests for zekan.detectors.schema.

No probe exists for SUSPECTED_UNDECLARED_LEAK / NEAR_CERTAIN_UNDECLARED_LEAK
yet (that's Upgrade 1 step 1e) and PROBE_FAILED isn't emitted by a "detector"
in the usual sense -- so unlike the other IssueTypes (each covered by its own
probe's test file, e.g. test_entity_aggregate_probe.py, via a real probe
call), these three are tested here by constructing IssueRecord directly
against the registry. This is the arbiter that the new _REGISTRY rows and
detail structs added in Upgrade 1 step 1c are well-formed BEFORE any probe
logic is written on top of them.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from zekan.detectors.schema import (
    Evidence,
    EvidenceScope,
    ImpactType,
    IssueRecord,
    IssueSeverity,
    IssueType,
    NearCertainUndeclaredLeakDetail,
    ProbeDetail,
    ProbeFailedDetail,
    SourceLayer,
    SuspectedUndeclaredLeakDetail,
)


# ── Registry coherence: every IssueType has exactly one _REGISTRY row ────────

def test_every_issue_type_has_a_registry_row():
    from zekan.detectors.schema import _REGISTRY
    assert set(IssueType) == set(_REGISTRY.keys())


# ── SUSPECTED_UNDECLARED_LEAK ──────────────────────────────────────────────────

def _suspected_record(**detail_overrides) -> IssueRecord:
    detail_kwargs = dict(
        feature="some_feature", univariate_auc=0.85, threshold_compared_against=0.80,
        n_folds_evaluated=5, screened_count=48, total_features=48,
        suppressed_by_known_strong_features=False,
    )
    detail_kwargs.update(detail_overrides)
    return IssueRecord(
        issue_type=IssueType.SUSPECTED_UNDECLARED_LEAK,
        status="warn",
        what="w", why="w", how_much="w", next_fix="w",
        evidence=Evidence(structural_detail=SuspectedUndeclaredLeakDetail(**detail_kwargs)),
    )


def test_suspected_undeclared_leak_computed_fields():
    rec = _suspected_record()
    assert rec.source_layer == SourceLayer.FLAGGED_SUSPICIOUS
    assert rec.severity == IssueSeverity.MEDIUM
    assert rec.evidence_scope == EvidenceScope.ENGINE_MEASURED
    assert rec.impact_type == ImpactType.HEURISTIC
    assert rec.confirmed is False


def test_suspected_undeclared_leak_detail_kind_matches_issue_type():
    rec = _suspected_record()
    assert rec.evidence.structural_detail.kind == "suspected_undeclared_leak"


def test_suspected_undeclared_leak_suppression_marker_settable():
    rec = _suspected_record(suppressed_by_known_strong_features=True)
    assert rec.evidence.structural_detail.suppressed_by_known_strong_features is True


def test_suspected_undeclared_leak_round_trips_through_json():
    rec = _suspected_record()
    dumped = json.loads(rec.model_dump_json())
    restored = IssueRecord.model_validate(dumped)
    assert restored.issue_type == IssueType.SUSPECTED_UNDECLARED_LEAK
    assert restored.source_layer == SourceLayer.FLAGGED_SUSPICIOUS
    assert restored.evidence.structural_detail.kind == "suspected_undeclared_leak"
    assert isinstance(restored.evidence.structural_detail, SuspectedUndeclaredLeakDetail)
    assert restored.evidence.structural_detail.feature == "some_feature"


# ── NEAR_CERTAIN_UNDECLARED_LEAK ───────────────────────────────────────────────

def _near_certain_record(**detail_overrides) -> IssueRecord:
    detail_kwargs = dict(
        feature="readmitted", univariate_auc=1.0, threshold_compared_against=0.999,
        n_folds_evaluated=5, screened_count=48, total_features=48,
    )
    detail_kwargs.update(detail_overrides)
    return IssueRecord(
        issue_type=IssueType.NEAR_CERTAIN_UNDECLARED_LEAK,
        status="fail",
        what="w", why="w", how_much="w", next_fix="w",
        evidence=Evidence(structural_detail=NearCertainUndeclaredLeakDetail(**detail_kwargs)),
    )


def test_near_certain_undeclared_leak_computed_fields():
    rec = _near_certain_record()
    assert rec.source_layer == SourceLayer.FLAGGED_SUSPICIOUS
    assert rec.severity == IssueSeverity.HIGH
    assert rec.evidence_scope == EvidenceScope.ENGINE_MEASURED
    assert rec.impact_type == ImpactType.HEURISTIC
    assert rec.confirmed is False


def test_near_certain_severity_exceeds_suspected_severity():
    """Pre-registration's explicit design requirement: NEAR_CERTAIN carries
    the higher severity, mirroring ENTITY_CONTAMINATION > ENTITY_CONTAMINATION_RISK."""
    severity_order = {
        IssueSeverity.LOW: 0, IssueSeverity.MEDIUM: 1,
        IssueSeverity.HIGH: 2, IssueSeverity.CRITICAL: 3,
    }
    suspected = _suspected_record()
    near_certain = _near_certain_record()
    assert severity_order[near_certain.severity] > severity_order[suspected.severity]


def test_near_certain_undeclared_leak_never_critical():
    """CRITICAL is reserved for silent-corruption-class findings (see
    schema.py's module docstring) -- an annotate-only heuristic flag must
    never claim that reserved meaning, no matter how high the AUC."""
    rec = _near_certain_record(univariate_auc=1.0)
    assert rec.severity != IssueSeverity.CRITICAL


def test_near_certain_undeclared_leak_detail_has_no_suppression_field():
    """Never suppressible by known_strong_features -- there is deliberately
    no waiver field on this detail struct at all (unlike SUSPECTED's)."""
    rec = _near_certain_record()
    assert not hasattr(rec.evidence.structural_detail, "suppressed_by_known_strong_features")


def test_near_certain_undeclared_leak_round_trips_through_json():
    rec = _near_certain_record()
    dumped = json.loads(rec.model_dump_json())
    restored = IssueRecord.model_validate(dumped)
    assert restored.issue_type == IssueType.NEAR_CERTAIN_UNDECLARED_LEAK
    assert restored.evidence.structural_detail.kind == "near_certain_undeclared_leak"
    assert isinstance(restored.evidence.structural_detail, NearCertainUndeclaredLeakDetail)
    assert restored.evidence.structural_detail.univariate_auc == pytest.approx(1.0)


# ── PROBE_FAILED ────────────────────────────────────────────────────────────────

def _probe_failed_record() -> IssueRecord:
    return IssueRecord(
        issue_type=IssueType.PROBE_FAILED,
        status="internal_fail",
        what="w", why="w", how_much="w", next_fix="w",
        evidence=Evidence(structural_detail=ProbeFailedDetail(
            probe_name="probe_x", exception_type="ValueError", message="boom",
        )),
    )


def test_probe_failed_computed_fields():
    """Mirrors SPLITTER_CONTRACT_VIOLATION on source_layer/evidence_scope/
    confirmed (Zekan reporting its own failure); diverges on severity (HIGH,
    not CRITICAL -- CRITICAL stays reserved) and impact_type (STRUCTURAL_RISK,
    not MEASUREMENT_ERROR -- a probe crash never touches engine-computed
    performance numbers). See schema.py's _REGISTRY row for the full reasoning."""
    rec = _probe_failed_record()
    assert rec.source_layer == SourceLayer.ZEKAN_INTEGRITY
    assert rec.severity == IssueSeverity.HIGH
    assert rec.evidence_scope == EvidenceScope.SELF_CHECK
    assert rec.impact_type == ImpactType.STRUCTURAL_RISK
    assert rec.confirmed is True


def test_probe_failed_status_is_internal_fail():
    """internal_fail is the same status SPLITTER_CONTRACT_VIOLATION uses for
    exactly this class of finding (Zekan's own machinery, not a data property)."""
    rec = _probe_failed_record()
    assert rec.status == "internal_fail"


def test_probe_failed_round_trips_through_json():
    rec = _probe_failed_record()
    dumped = json.loads(rec.model_dump_json())
    restored = IssueRecord.model_validate(dumped)
    assert restored.issue_type == IssueType.PROBE_FAILED
    assert restored.source_layer == SourceLayer.ZEKAN_INTEGRITY
    assert isinstance(restored.evidence.structural_detail, ProbeFailedDetail)
    assert restored.evidence.structural_detail.probe_name == "probe_x"
    assert restored.evidence.structural_detail.exception_type == "ValueError"


def test_probe_failed_detail_has_no_raw_traceback_field():
    """Deliberate: no existing detail struct in this schema carries an
    unstructured diagnostic blob (the Case-4-pattern the module docstring
    warns against) -- confirms that principle held here too."""
    rec = _probe_failed_record()
    assert not hasattr(rec.evidence.structural_detail, "traceback")
    assert not hasattr(rec.evidence.structural_detail, "traceback_text")


# ── Discriminated-union parsing (ProbeDetail) ────────────────────────────────

@pytest.mark.parametrize("kind,cls", [
    ("suspected_undeclared_leak", SuspectedUndeclaredLeakDetail),
    ("near_certain_undeclared_leak", NearCertainUndeclaredLeakDetail),
    ("probe_failed", ProbeFailedDetail),
])
def test_probe_detail_union_discriminates_new_kinds_from_bare_dict(kind, cls):
    """Pydantic must resolve the correct concrete class from `kind` alone,
    parsing a plain dict (no Python object involved) -- the exact mechanism
    the report layer relies on when loading a JSON artifact back in."""
    base = {"suspected_undeclared_leak": dict(
                feature="f", univariate_auc=0.9, threshold_compared_against=0.8,
                n_folds_evaluated=5, screened_count=10, total_features=10,
                suppressed_by_known_strong_features=False,
            ),
            "near_certain_undeclared_leak": dict(
                feature="f", univariate_auc=1.0, threshold_compared_against=0.999,
                n_folds_evaluated=5, screened_count=10, total_features=10,
            ),
            "probe_failed": dict(
                probe_name="p", exception_type="ValueError", message="m",
            )}[kind]
    raw = {"kind": kind, **base}
    parsed = TypeAdapter(ProbeDetail).validate_python(raw)
    assert isinstance(parsed, cls)
    assert parsed.kind == kind


def test_probe_detail_union_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        TypeAdapter(ProbeDetail).validate_python({"kind": "not_a_real_kind"})


# ── Computed fields cannot be overridden by the caller ────────────────────────

def test_suspected_undeclared_leak_computed_fields_not_settable():
    """Passing source_layer/severity/etc as kwargs is silently ignored --
    the registry value always wins (IssueRecord._strip_computed)."""
    rec = IssueRecord(
        issue_type=IssueType.SUSPECTED_UNDECLARED_LEAK,
        status="warn",
        what="w", why="w", how_much="w", next_fix="w",
        source_layer="detected_structural",  # bogus attempted override
        severity="critical",                  # bogus attempted override
        evidence=Evidence(structural_detail=SuspectedUndeclaredLeakDetail(
            feature="f", univariate_auc=0.9, threshold_compared_against=0.8,
            n_folds_evaluated=5, screened_count=10, total_features=10,
            suppressed_by_known_strong_features=False,
        )),
    )
    assert rec.source_layer == SourceLayer.FLAGGED_SUSPICIOUS
    assert rec.severity == IssueSeverity.MEDIUM
