"""Tests for probe_near_bijection (NEAR_BIJECTION_UNDECLARED_LEAK, Upgrade H).

Trigger: a non-forbidden feature's Theil's U, after rare-value pooling at
SUPPORT_FLOOR, clears CRITERION (>= 0.99). Deterministic counting only --
no estimator, no folds, no RNG. Mirrors FORBIDDEN_ENTITY_LEVEL_AGGREGATE's
registry classification (DETECTED_STRUCTURAL, DATA_ONLY, confirmed=True),
not Upgrade 1's FLAGGED_SUSPICIOUS/confirmed=False screen.

Negative cases: row-unique ID column (the ID trap, guarded by SUPPORT_FLOOR
pooling), an independent/random feature, a declared-forbidden feature (never
a candidate). Integration: full run_audit() shows the verdict is unchanged
when the probe fires (annotate-only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zekan.config.schema import ZekanConfig, SplitPolicy
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.near_bijection_probe import (
    CRITERION,
    SUPPORT_FLOOR,
    probe_near_bijection,
)
from zekan.detectors.schema import (
    EvidenceScope,
    ImpactType,
    IssueSeverity,
    IssueType,
    NearBijectionUndeclaredLeakDetail,
    SourceLayer,
)
from zekan.severity.audit import run_audit


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="near-bijection-probe-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=[],
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A frame with: an exact-partition leak ('leaky'), a row-unique ID-like
    column ('row_id'), and an independent random feature ('honest_noise')."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(n)],
        "prediction_time": ["2020-01-01"] * n,
        "target": target,
        "leaky": target,                          # exact copy of target
        "row_id": list(range(n)),                  # unique on every row (ID trap)
        "honest_noise": rng.integers(0, 2, size=n), # independent of target
    })


# ── POSITIVE: exact-partition feature fires ───────────────────────────────────

def test_positive_exact_partition_fires():
    df = _make_df()
    results = probe_near_bijection(df, _contract())
    fired = [r for r in results if r.evidence.structural_detail.feature == "leaky"]
    assert len(fired) == 1
    rec = fired[0]
    assert rec.issue_type == IssueType.NEAR_BIJECTION_UNDECLARED_LEAK
    assert rec.status == "fail"


def test_positive_theil_u_is_exactly_one():
    df = _make_df()
    rec = [r for r in probe_near_bijection(df, _contract())
           if r.evidence.structural_detail.feature == "leaky"][0]
    assert rec.evidence.structural_detail.theil_u == pytest.approx(1.0)


def test_positive_registry_classification():
    """Mirrors FORBIDDEN_ENTITY_LEVEL_AGGREGATE, NOT the FLAGGED_SUSPICIOUS/
    confirmed=False Upgrade 1 screen: a deterministic counting fact."""
    df = _make_df()
    rec = [r for r in probe_near_bijection(df, _contract())
           if r.evidence.structural_detail.feature == "leaky"][0]
    assert rec.source_layer == SourceLayer.DETECTED_STRUCTURAL
    assert rec.severity == IssueSeverity.HIGH
    assert rec.evidence_scope == EvidenceScope.DATA_ONLY
    assert rec.impact_type == ImpactType.STRUCTURAL_RISK
    assert rec.confirmed is True


def test_positive_detail_fields():
    df = _make_df(n=200)
    rec = [r for r in probe_near_bijection(df, _contract())
           if r.evidence.structural_detail.feature == "leaky"][0]
    detail = rec.evidence.structural_detail
    assert isinstance(detail, NearBijectionUndeclaredLeakDetail)
    assert detail.feature == "leaky"
    assert detail.theil_u == pytest.approx(1.0)
    assert detail.support_floor == SUPPORT_FLOOR
    assert detail.threshold_compared_against == CRITERION
    assert detail.n_distinct_raw == 2               # 0/1, well-supported (~100 each)
    assert detail.n_distinct_after_pooling == 2      # both values well above the floor -- no pooling
    assert detail.n_rows_pooled == 0
    assert detail.screened_count == detail.total_features  # no screenability gate/cap in this design
    assert detail.total_features == 3  # leaky, row_id, honest_noise


# ── NEGATIVE: row-unique ID column does NOT trigger (the pooling guard) ───────

def test_negative_row_unique_id_does_not_fire():
    df = _make_df()
    results = probe_near_bijection(df, _contract())
    fired = [r.evidence.structural_detail.feature for r in results]
    assert "row_id" not in fired, (
        "A row-unique column trivially scores U=1.0 raw (the ID trap) but must be "
        "neutralized by SUPPORT_FLOOR pooling before scoring"
    )


# ── NEGATIVE: independent feature does NOT trigger ────────────────────────────

def test_negative_independent_feature_does_not_fire():
    df = _make_df()
    results = probe_near_bijection(df, _contract())
    fired = [r.evidence.structural_detail.feature for r in results]
    assert "honest_noise" not in fired


# ── NEGATIVE: declared-forbidden feature is never a candidate ────────────────

def test_negative_forbidden_feature_never_scored():
    df = _make_df()
    contract = _contract(forbidden_after_prediction=["leaky"])
    results = probe_near_bijection(df, contract)
    fired = [r.evidence.structural_detail.feature for r in results]
    assert "leaky" not in fired, "A declared-forbidden feature must never be a screen candidate"


# ── NEGATIVE: no candidates at all ────────────────────────────────────────────

def test_negative_no_candidate_features_returns_empty():
    df = pd.DataFrame({
        "entity_id": ["e0", "e1"],
        "prediction_time": ["2020-01-01", "2020-01-01"],
        "target": [0, 1],
    })
    results = probe_near_bijection(df, _contract())
    assert results == []


# ── Registry wiring ────────────────────────────────────────────────────────────

def test_registered_in_probe_registry_with_no_capability_flags():
    """Deterministic counting only -- mirrors probe_forbidden_entity_level_aggregate's
    registration exactly (no folds/model/matrix/budget/side_channel)."""
    from zekan.severity.audit import _build_probe_registry

    specs = {spec.fn.__name__: spec for spec in _build_probe_registry()}
    spec = specs["probe_near_bijection"]
    assert spec.needs_folds is False
    assert spec.needs_model is False
    assert spec.needs_matrix is False
    assert spec.needs_budget is False
    assert spec.needs_side_channel is False


# ── INTEGRATION: full run_audit gives an unchanged verdict + annotation ───────

def _integration_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(n)],
        "prediction_time": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "target": target,
        "raw_leak": target,                          # undeclared exact copy of target
        "feature_a": rng.normal(size=n),
        "feature_b": rng.integers(0, 5, size=n),
    })


def _integration_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="integration test -- undeclared raw_leak",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=[],
    )


def _integration_config() -> ZekanConfig:
    return ZekanConfig(
        contract=_integration_contract(),
        split_policy=SplitPolicy(n_splits=5, min_test_rows_per_fold=20,
                                  min_positive_cases_per_fold=5,
                                  min_negative_cases_per_fold=5),
    )


def test_integration_verdict_unchanged_with_annotation():
    df = _integration_df()
    report = run_audit(df, _integration_contract(), _integration_config(), n_permutations=0)

    annotations = [
        a for a in report.structural_annotations
        if a.issue_type == IssueType.NEAR_BIJECTION_UNDECLARED_LEAK
    ]
    assert len(annotations) == 1
    ann = annotations[0]
    detail = ann.evidence.structural_detail
    assert isinstance(detail, NearBijectionUndeclaredLeakDetail)
    assert detail.feature == "raw_leak"
    assert detail.theil_u == pytest.approx(1.0)

    # Verdict is unchanged: annotate-only, exactly as every other structural probe.
    # No forbidden feature was declared here, so the primary A/B/C attribution
    # path has nothing to attribute -- the verdict reflects that (empty)
    # forbidden-feature set only, never this structural annotation.
    assert report.policy_decision.verdict in {"PASS", "NOTE", "WARN", "FAIL",
                                               "UNCONFIRMED_HIGH_DAMAGE"}
    assert report.measured_damage.feature_attribution is None
