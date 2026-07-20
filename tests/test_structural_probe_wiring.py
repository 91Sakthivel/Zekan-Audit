"""Tests for Upgrade 1 steps 1b/1c: the widened structural-probe calling
convention and its exception-isolation guarantee.

Covers _run_structural_probes's capability flags (needs_model, needs_matrix,
needs_budget), exception isolation (now producing a registered PROBE_FAILED
IssueRecord as of step 1c -- 1b's duck-typed _ProbeFailure placeholder was
retired), and that the three currently registered probes are unaffected
(backward compatibility -- their own dedicated test files, e.g.
test_entity_aggregate_probe.py, cover their actual behavior; this file is
about the calling convention around them, not the probes themselves).

Fake probes are injected via mock.patch("zekan.severity.audit._build_probe_registry",
...) -- the registry-construction seam step 1b introduced specifically so the
needs_model/needs_matrix/needs_budget/exception-isolation wiring could be
exercised directly without a real model-fitting probe existing yet (that's
step 1e).
"""

from __future__ import annotations

import time as _time
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import Evidence, IssueRecord, IssueType, ProbeFailedDetail
from zekan.severity.audit import _ProbeSpec, _run_structural_probes


def _contract(forbidden=None) -> PredictionContract:
    return PredictionContract(
        prediction_problem="probe-wiring-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=forbidden or [],
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": [1, 2, 3],
        "prediction_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "target": [0, 1, 0],
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [4.0, 5.0, 6.0],
    })


def _pass_record() -> IssueRecord:
    return IssueRecord(
        issue_type=IssueType.CORRELATED_LEAK_PAIR,
        status="pass",
        what="ok", why="ok", how_much="ok", next_fix="No action required.",
    )


# ── Backward compatibility ────────────────────────────────────────────────────

def test_existing_probes_unaffected_by_widened_signature():
    """The three registered probes take no new args -- the real registry must
    still run cleanly through the widened _run_structural_probes signature."""
    result = _run_structural_probes(_df(), _contract(), folds=None)
    assert isinstance(result, list)


# ── needs_model ────────────────────────────────────────────────────────────────

def test_needs_model_probe_receives_usable_factory():
    received: dict = {}

    def _fake_probe(df, contract, model_factory, n_jobs):
        received["model_factory"] = model_factory
        received["n_jobs"] = n_jobs
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_model=True)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(_df(), _contract(), n_jobs=4)

    assert received["model_factory"] is not None
    assert callable(received["model_factory"])
    est = received["model_factory"]()
    assert hasattr(est, "fit")  # a real, usable sklearn-compatible estimator
    assert received["n_jobs"] == 4


def test_needs_model_probe_receives_caller_supplied_factory_unchanged():
    """When the caller DID pass a model_factory, it's threaded through as-is
    -- no silent substitution of the default."""
    def _sentinel_factory():
        return "not a real estimator, just a sentinel"

    received: dict = {}

    def _fake_probe(df, contract, model_factory, n_jobs):
        received["model_factory"] = model_factory
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_model=True)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(_df(), _contract(), model_factory=_sentinel_factory)

    assert received["model_factory"] is _sentinel_factory


# ── needs_matrix: column slicing by position ────────────────────────────────

def test_needs_matrix_probe_receives_matrix_and_correct_col_pos():
    X_all = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    y_all = np.array([0, 1, 0])
    received: dict = {}

    def _fake_probe(df, contract, X_all, y_all, col_pos):
        received["X_all"] = X_all
        received["y_all"] = y_all
        received["col_pos"] = col_pos
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_matrix=True)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(
            _df(), _contract(), X_all=X_all, y_all=y_all,
            all_features=["feature_a", "feature_b"],
        )

    assert received["col_pos"] == {"feature_a": 0, "feature_b": 1}
    # Slicing feature_b by its reported position must return feature_b's column
    # -- by POSITION, never a boolean mask (see _run_structural_probes docstring).
    pos = received["col_pos"]["feature_b"]
    sliced = received["X_all"][:, [pos]]
    np.testing.assert_array_equal(sliced.ravel(), X_all[:, 1])
    np.testing.assert_array_equal(received["y_all"], y_all)


def test_needs_matrix_probe_gets_none_when_no_matrix_was_built():
    """A needs_matrix probe must not be silently skipped when X_all is None --
    unlike needs_folds, the None itself is passed through; the probe decides
    what to do with it."""
    received: dict = {}

    def _fake_probe(df, contract, X_all, y_all, col_pos):
        received["X_all"] = X_all
        received["col_pos"] = col_pos
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_matrix=True)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(_df(), _contract())

    assert received["X_all"] is None
    assert received["col_pos"] == {}


# ── needs_budget ──────────────────────────────────────────────────────────────

def test_needs_budget_probe_receives_monotonic_deadline():
    received: dict = {}

    def _fake_probe(df, contract, deadline):
        received["deadline"] = deadline
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_budget=True)
    before = _time.monotonic()
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(_df(), _contract(), time_budget_seconds=5.0)
    after = _time.monotonic()

    assert received["deadline"] is not None
    assert before + 5.0 <= received["deadline"] <= after + 5.0


def test_needs_budget_probe_receives_none_when_no_budget_given():
    received: dict = {}

    def _fake_probe(df, contract, deadline):
        received["deadline"] = deadline
        return _pass_record()

    fake_spec = _ProbeSpec(fn=_fake_probe, needs_budget=True)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        _run_structural_probes(_df(), _contract())

    assert received["deadline"] is None


# ── Exception isolation ──────────────────────────────────────────────────────
# Step 1c: a probe that raises now produces a REGISTERED IssueType.PROBE_FAILED
# IssueRecord (source_layer=ZEKAN_INTEGRITY, status="internal_fail", confirmed=
# True -- see schema.py's _REGISTRY row and its reasoning comment), not the
# step-1b duck-typed _ProbeFailure placeholder.

def test_probe_exception_is_isolated_and_surfaced():
    def _raising_probe(df, contract):
        raise ValueError("synthetic failure for test coverage")

    fake_spec = _ProbeSpec(fn=_raising_probe)
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        result = _run_structural_probes(_df(), _contract())

    assert len(result) == 1
    failure = result[0]
    assert isinstance(failure, IssueRecord)
    assert failure.issue_type == IssueType.PROBE_FAILED
    assert failure.status == "internal_fail"
    # Computed fields correctly derived from the registry, not hand-set.
    assert failure.source_layer.value == "zekan_integrity"
    assert failure.severity.value == "high"
    assert failure.confirmed is True

    detail = failure.evidence.structural_detail
    assert isinstance(detail, ProbeFailedDetail)
    assert detail.probe_name == "_raising_probe"
    assert detail.exception_type == "ValueError"
    assert "synthetic failure for test coverage" in detail.message
    assert "synthetic failure for test coverage" in failure.what


def test_one_probe_raising_does_not_stop_other_probes_from_running():
    calls: list[str] = []

    def _raising_probe(df, contract):
        calls.append("raising")
        raise RuntimeError("boom")

    def _ok_probe(df, contract):
        calls.append("ok")
        return _pass_record()

    specs = [_ProbeSpec(fn=_raising_probe), _ProbeSpec(fn=_ok_probe)]
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=specs):
        result = _run_structural_probes(_df(), _contract())

    # Both probes actually ran (order preserved) -- the exception in the first
    # did not prevent the second from executing.
    assert calls == ["raising", "ok"]
    # _ok_probe returns status="pass" so it contributes nothing to `found`;
    # only the failure record appears -- critically, nothing propagated.
    assert len(result) == 1
    assert result[0].issue_type == IssueType.PROBE_FAILED


def test_probe_failed_record_survives_json_export_and_text_render():
    """The exact resilience claim: a probe raising must never crash the
    surfaces that consume structural_annotations (JSON export, text render)."""
    from zekan.reports.json_export import verdict_to_dict
    from zekan.severity.verdict import (
        EngineDetection, FoldCI, MeasuredDamage, OptimismDecomposition,
        PolicyDecision, VerdictReport,
    )

    failure = IssueRecord(
        issue_type=IssueType.PROBE_FAILED,
        status="internal_fail",
        what="Structural probe 'x' failed to run (ValueError: boom).",
        why="A probe that cannot run means one structural-risk category went unchecked.",
        how_much="ValueError: boom",
        next_fix="Report it so the probe can be fixed.",
        evidence=Evidence(structural_detail=ProbeFailedDetail(
            probe_name="x", exception_type="ValueError", message="boom",
        )),
    )
    report = VerdictReport(
        engine_detection=EngineDetection(
            detected=False, p_value=None, nsl=None, alpha=0.01,
            confidence="high", interpretation="n/a",
        ),
        measured_damage=MeasuredDamage(
            metric="roc_auc", fixable_leakage=0.0,
            optimism_decomposition=OptimismDecomposition(
                naive_auc=0.5, temporal_all_auc=0.5, deployable_auc=0.5,
            ),
            interpretation="n/a",
        ),
        policy_decision=PolicyDecision(
            policy_profile="default_auc", warn_floor=0.1, fail_floor=0.15,
            user_overridable=True, verdict="PASS", policy_default_used=True,
            interpretation="n/a",
        ),
        fold_ci=FoldCI(
            fixable_leakage_pooled=0.0, fixable_leakage_fold_mean=None,
            fl_fold_std=None, fl_fold_se=None, ci_center="fl_mean",
            ci_half=None, ci_low=None, ci_high=None, ci_ratio=None,
            confidence_tier="unavailable",
            confidence_calibration_status="unavailable: test",
            pooled_vs_fold_gap=None, instability_note="",
        ),
        structural_annotations=[failure],
    )

    d = verdict_to_dict(report)
    assert any(a.get("what") == failure.what for a in d["structural_annotations"])
    # The registered record round-trips its typed detail too, not just `what`.
    assert d["structural_annotations"][0]["evidence"]["structural_detail"]["kind"] == "probe_failed"

    text = str(report)
    assert failure.what in text


def test_run_audit_still_returns_a_verdict_when_a_probe_raises():
    """End-to-end: run_audit() must return a VerdictReport, not propagate,
    when a registered structural probe raises."""
    from zekan.config.schema import ZekanConfig
    from zekan.severity.audit import run_audit
    from zekan.severity.verdict import VerdictReport

    def _raising_probe(df, contract):
        raise RuntimeError("boom")

    fake_spec = _ProbeSpec(fn=_raising_probe)
    contract = _contract()
    df = _df()
    with mock.patch("zekan.severity.audit._build_probe_registry", return_value=[fake_spec]):
        report = run_audit(df, contract, ZekanConfig(contract=contract), n_permutations=0)

    assert isinstance(report, VerdictReport)
    assert any(
        isinstance(a, IssueRecord) and a.issue_type == IssueType.PROBE_FAILED
        for a in report.structural_annotations
    )
