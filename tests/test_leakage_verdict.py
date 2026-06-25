"""Tests for _leakage_status() and leakage_issue_record() — Step-2 null-standardized verdict.

The verdict is now fully null-standardized: constant bands (0.02, 0.04) are NOT
used as gate thresholds.  Gate logic:
  p >= 0.01 (alpha)            -> PASS
  p <  0.01, NSL < 1.0        -> NOTE
  p <  0.01, 1.0 <= NSL < 2.0 -> WARN
  p <  0.01, NSL >= 2.0       -> FAIL
  p is None (null not run)    -> unavailable

leakage_issue_record() returns 'unavailable' for two reasons:
  1. result.status == 'unavailable' (contract validation blocked the engine)
  2. result.n_permutations_run == 0 (null not run; no null-std verdict possible)
"""
from dataclasses import replace

import pytest

from gotcha.detectors.schema import IssueType, SourceLayer
from gotcha.severity.engine import (
    _NULL_ALPHA, _NSL_NOTE_THRESHOLD, _NSL_WARN_THRESHOLD,
    SeverityResult,
    _leakage_status,
    leakage_issue_record,
)


# ── Minimal SeverityResult factory ────────────────────────────────────────────

def _result(
    fixable_leakage: float = 0.0,
    p_value=None,
    nsl=None,
    null_99th=None,
    null_median=None,
    null_iqr=None,
    null_95th=None,
    n_permutations_run: int = 0,
    engine_status: str = "pass",
) -> SeverityResult:
    return SeverityResult(
        status=engine_status,
        metric="roc_auc",
        naive_auc=0.80,
        estimated_deployable_auc=0.78,
        total_optimism=0.02,
        fixable_leakage=fixable_leakage,
        fixable_leakage_range=(fixable_leakage, fixable_leakage),
        nonfixable_optimism=0.01,
        per_fold=[],
        caveat="test",
        null_95th=null_95th,
        null_99th=null_99th,
        null_median=null_median,
        null_iqr=null_iqr,
        p_value=p_value,
        nsl=nsl,
        n_permutations_run=n_permutations_run,
    )


# ── _leakage_status tests ─────────────────────────────────────────────────────

class TestLeakageStatus:
    """Unit tests for the null-standardized _leakage_status() gate."""

    # Reality gate
    def test_null_not_run_returns_unavailable(self):
        assert _leakage_status(0.09, p_value=None, nsl=5.0) == "unavailable"

    def test_inside_null_returns_pass(self):
        assert _leakage_status(0.09, p_value=0.12, nsl=5.0) == "pass"

    def test_at_alpha_boundary_returns_pass(self):
        # p == _NULL_ALPHA → inside null (>= threshold)
        assert _leakage_status(0.09, p_value=_NULL_ALPHA, nsl=5.0) == "pass"

    def test_just_below_alpha_enters_ladder(self):
        # p just below 0.01 → outside null → use NSL
        p = _NULL_ALPHA - 1e-9
        assert _leakage_status(0.09, p_value=p, nsl=5.0) == "fail"

    # NSL ladder (p < alpha)
    def test_nsl_none_gives_note(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=None) == "note"

    def test_nsl_negative_gives_note(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=-0.5) == "note"

    def test_nsl_zero_gives_note(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=0.0) == "note"

    def test_nsl_below_note_threshold_gives_note(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=0.99) == "note"

    def test_nsl_at_note_threshold_gives_warn(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=_NSL_NOTE_THRESHOLD) == "warn"

    def test_nsl_between_note_and_warn_thresholds(self):
        mid = (_NSL_NOTE_THRESHOLD + _NSL_WARN_THRESHOLD) / 2
        assert _leakage_status(0.09, p_value=0.005, nsl=mid) == "warn"

    def test_nsl_at_warn_threshold_gives_fail(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=_NSL_WARN_THRESHOLD) == "fail"

    def test_nsl_large_gives_fail(self):
        assert _leakage_status(0.09, p_value=0.005, nsl=15.0) == "fail"

    # Old constant bands must NOT gate (these were the wrong thresholds)
    def test_small_observed_inside_null_is_pass_not_band_driven(self):
        # even tiny observed with p >= alpha -> PASS (no band gate)
        assert _leakage_status(0.001, p_value=0.5, nsl=-10.0) == "pass"

    def test_large_observed_inside_null_is_still_pass(self):
        # 0.09 > 0.04 old threshold but inside null -> PASS
        assert _leakage_status(0.09, p_value=0.20, nsl=3.0) == "pass"


# ── leakage_issue_record tests ────────────────────────────────────────────────

class TestLeakageIssueRecord:

    def test_issue_type_is_temporal_leakage(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=3.0, n_permutations_run=100)
        assert leakage_issue_record(r).issue_type == IssueType.TEMPORAL_LEAKAGE

    def test_source_layer_is_engine_measured(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=3.0, n_permutations_run=100)
        assert leakage_issue_record(r).source_layer == SourceLayer.ENGINE_MEASURED

    # Unavailable paths
    def test_null_not_run_returns_unavailable(self):
        r = _result(fixable_leakage=0.09, n_permutations_run=0)
        issue = leakage_issue_record(r)
        assert issue.status == "unavailable"
        assert "n_permutations" in issue.what.lower() or "null" in issue.what.lower()

    def test_contract_unavailable_returns_unavailable(self):
        r = _result(fixable_leakage=float("nan"), engine_status="unavailable")
        r = replace(r, unavailable_reason="Not enough time periods.")
        issue = leakage_issue_record(r)
        assert issue.status == "unavailable"
        assert "Not enough time periods" in issue.how_much

    # Four-rung ladder
    def test_pass_when_inside_null(self):
        r = _result(fixable_leakage=0.09, p_value=0.15, nsl=4.0, n_permutations_run=100)
        assert leakage_issue_record(r).status == "pass"

    def test_note_when_outside_null_small_nsl(self):
        r = _result(fixable_leakage=0.05, p_value=0.005, nsl=0.4, n_permutations_run=100)
        assert leakage_issue_record(r).status == "note"

    def test_warn_when_outside_null_moderate_nsl(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=1.5, n_permutations_run=100)
        assert leakage_issue_record(r).status == "warn"

    def test_fail_when_outside_null_large_nsl(self):
        r = _result(fixable_leakage=0.09, p_value=0.005, nsl=5.0, n_permutations_run=100)
        assert leakage_issue_record(r).status == "fail"

    # Evidence fields
    def test_evidence_p_value_populated(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=3.0,
                    null_99th=0.02, n_permutations_run=100)
        issue = leakage_issue_record(r)
        assert issue.evidence.p_value == pytest.approx(0.005)

    def test_evidence_null_95th_populated(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=3.0,
                    null_95th=0.015, null_99th=0.02, n_permutations_run=100)
        issue = leakage_issue_record(r)
        assert issue.evidence.null_95th == pytest.approx(0.015)

    def test_evidence_threshold_is_null_99th(self):
        r = _result(fixable_leakage=0.07, p_value=0.005, nsl=3.0,
                    null_99th=0.025, n_permutations_run=100)
        issue = leakage_issue_record(r)
        assert issue.evidence.threshold == pytest.approx(0.025)

    def test_evidence_null_fields_none_when_not_run(self):
        r = _result(fixable_leakage=0.09, n_permutations_run=0)
        issue = leakage_issue_record(r)
        assert issue.evidence.null_95th is None
        assert issue.evidence.p_value is None

    def test_metric_name_is_fixable_leakage(self):
        r = _result(fixable_leakage=0.05, p_value=0.3, nsl=0.0, n_permutations_run=100)
        assert leakage_issue_record(r).evidence.metric_name == "fixable_leakage"

    def test_how_much_contains_p_value(self):
        r = _result(fixable_leakage=0.09, p_value=0.005, nsl=3.0,
                    null_99th=0.02, n_permutations_run=100)
        issue = leakage_issue_record(r)
        assert "0.0050" in issue.how_much or "p=" in issue.how_much

    def test_how_much_mentions_inside_null_for_pass(self):
        r = _result(fixable_leakage=0.05, p_value=0.20, nsl=-1.0,
                    null_99th=0.06, n_permutations_run=100)
        issue = leakage_issue_record(r)
        assert "inside null" in issue.how_much.lower() or "pass" in issue.how_much.lower()

    # SeverityResult null field defaults
    def test_severity_result_null_fields_default_none(self):
        r = _result(fixable_leakage=0.07)
        assert r.p_value is None
        assert r.null_95th is None
        assert r.null_99th is None
        assert r.null_median is None
        assert r.null_iqr is None
        assert r.nsl is None
        assert r.n_permutations_run == 0

    def test_severity_result_accepts_all_null_fields(self):
        r = _result(
            fixable_leakage=0.07, p_value=0.003, nsl=4.5,
            null_99th=0.02, null_median=0.005, null_iqr=0.008,
            null_95th=0.015, n_permutations_run=100,
        )
        assert r.p_value == pytest.approx(0.003)
        assert r.nsl == pytest.approx(4.5)
        assert r.null_99th == pytest.approx(0.02)
        assert r.null_median == pytest.approx(0.005)
        assert r.null_iqr == pytest.approx(0.008)
        assert r.n_permutations_run == 100

    # Constant bands must NOT control the verdict anymore
    def test_old_noise_floor_constant_not_a_gate(self):
        # observed = 0.001 < 0.02 (old noise floor), but inside null -> PASS
        r = _result(fixable_leakage=0.001, p_value=0.40, nsl=-5.0, n_permutations_run=100)
        assert leakage_issue_record(r).status == "pass"

    def test_old_clear_leak_constant_not_a_gate(self):
        # observed = 0.05 > 0.04 (old clear-leak), but inside null -> PASS (not FAIL)
        r = _result(fixable_leakage=0.05, p_value=0.08, nsl=2.5, n_permutations_run=100)
        assert leakage_issue_record(r).status == "pass"
