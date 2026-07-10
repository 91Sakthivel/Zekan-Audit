"""Tests for the across-entity permutation null (spec 1).

Closes the within-entity blind spot: a forbidden ENTITY-LEVEL AGGREGATE
(constant within each entity, varies across entities) is a no-op under
within-entity permutation and so is invisible to it, even when it carries
real signal about the target.  The across-entity null permutes forbidden
columns globally (ignoring entity boundaries) and catches this class.

Layout:
  (a) unit tests on null_baseline.py: _permute_across_entity, the
      method=="across_entity" dispatch branch, and the no-op guard.
  (b) integration tests on the full run_severity_analysis / run_audit /
      build_verdict chain: the flagship blind-spot case, the within-entity
      -only case (unchanged), the clean case, determinism, additive JSON,
      and the power.py calibration boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.config.schema import SplitPolicy, ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.severity.audit import run_audit
from zekan.severity.engine import run_severity_analysis
from zekan.severity.null_baseline import (
    NullResult,
    _permute_across_entity,
    estimate_fixable_leakage_null,
)
from zekan.severity.verdict import build_verdict

from tests.test_entity_aggregate_probe import (
    _audit_config,
    _audit_contract,
    _make_entity_churn_rate_dataset,
)


def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


# ── (a) Unit: _permute_across_entity ───────────────────────────────────────────

def test_permute_across_entity_shuffles_globally_not_by_group():
    """Unlike within-entity, the same row can receive a value that originated
    at a DIFFERENT entity — proof the shuffle ignores entity boundaries."""
    df = pd.DataFrame({
        "entity_id": ["e0"] * 5 + ["e1"] * 5,
        "forbidden_col": list(range(10)),
    })
    rng = np.random.default_rng(0)
    df_perm = _permute_across_entity(df, ["forbidden_col"], rng)

    # Marginal distribution of values is preserved (a permutation, not a resample).
    assert sorted(df_perm["forbidden_col"].tolist()) == list(range(10))
    # With a global shuffle, at least one row's original entity group boundary
    # must be crossed (values 0-4 originally all sat in e0's rows) — assert the
    # post-shuffle e0 rows are NOT simply {0,1,2,3,4} (which within-entity would
    # guarantee, since it never moves a value out of its entity).
    e0_values = set(df_perm.loc[df_perm["entity_id"] == "e0", "forbidden_col"])
    assert e0_values != {0, 1, 2, 3, 4}, (
        "across-entity shuffle should not (with overwhelming probability) "
        "reproduce the within-entity partition"
    )


def test_permute_across_entity_does_not_mutate_input():
    df = pd.DataFrame({"entity_id": ["e0", "e1"], "forbidden_col": [1.0, 2.0]})
    original = df.copy()
    _permute_across_entity(df, ["forbidden_col"], np.random.default_rng(0))
    pd.testing.assert_frame_equal(df, original)


# ── (a) Unit: no-op guard ───────────────────────────────────────────────────────

def _single_entity_df(n_rows: int = 40, n_periods: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        rows.append({
            "entity_id": "only_entity",
            "t": f"2022-{(i % n_periods) + 1:02d}-01",
            "y": i % 2,
            "forbidden_col": float(i),
            "feature_x": float(i % 3),
        })
    return pd.DataFrame(rows)


def _single_entity_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="single-entity-guard-test",
        entity_id="entity_id",
        prediction_time="t",
        target="y",
        available_features_until="t",
        forbidden_after_prediction=["forbidden_col"],
    )


def _single_entity_config() -> ZekanConfig:
    return ZekanConfig(
        contract=_single_entity_contract(),
        split_policy=SplitPolicy(
            n_splits=3, min_test_rows_per_fold=2,
            min_positive_cases_per_fold=1, min_negative_cases_per_fold=1,
        ),
    )


def test_across_entity_not_run_on_single_entity():
    """<2 entities -> NOT-RUN NullResult (n_permutations=0), not a misleading
    'ran and found nothing' clean result."""
    df = _single_entity_df()
    result = estimate_fixable_leakage_null(
        df, _single_entity_contract(), _single_entity_config(), _fast_clf,
        observed_fixable_leakage=0.05, n_permutations=10, seed=0,
        method="across_entity",
    )
    assert isinstance(result, NullResult)
    assert result.method == "across_entity"
    assert result.n_permutations == 0
    assert np.isnan(result.p_value), "NOT-RUN must carry a NaN sentinel, not a real p-value"


def test_within_entity_unaffected_by_across_entity_guard():
    """The guard is method-scoped: within_entity on the SAME single-entity data
    must not be short-circuited by the across-entity no-op guard."""
    df = _single_entity_df()
    result = estimate_fixable_leakage_null(
        df, _single_entity_contract(), _single_entity_config(), _fast_clf,
        observed_fixable_leakage=0.05, n_permutations=10, seed=0,
        method="within_entity",
    )
    assert result.method == "within_entity"
    assert result.n_permutations == 10, (
        "within_entity must run its full permutation loop on single-entity data "
        "(temporal folds are time-based, not entity-group based)"
    )
    assert not np.isnan(result.p_value)


def test_unknown_method_error_lists_across_entity():
    df = _single_entity_df()
    with pytest.raises(ValueError, match="across_entity"):
        estimate_fixable_leakage_null(
            df, _single_entity_contract(), _single_entity_config(), _fast_clf,
            observed_fixable_leakage=0.05, n_permutations=10, seed=0,
            method="not_a_real_method",
        )


# ── (b) Integration: flagship blind-spot case ───────────────────────────────────
# entity_churn_rate (from test_entity_aggregate_probe.py): constant within entity,
# varies across entities, genuinely correlated with the target (it IS the
# historical churn mean). Within-entity permutation is a no-op on it;
# across-entity permutation is not.

@pytest.fixture(scope="module")
def _blind_spot_result():
    df = _make_entity_churn_rate_dataset()
    return run_severity_analysis(
        df, _audit_contract(), _audit_config(), _fast_clf,
        n_permutations=100, null_seed=0,
    )


def test_within_entity_null_misses_blind_spot(_blind_spot_result):
    result = _blind_spot_result
    assert result.n_permutations_run > 0
    within_detected = (
        result.p_value is not None and result.p_value < 0.01
        and result.nsl is not None and result.nsl >= 1.0
    )
    assert within_detected is False, (
        f"expected the within-entity blind spot: p={result.p_value}, nsl={result.nsl}"
    )


def test_across_entity_null_catches_blind_spot(_blind_spot_result):
    result = _blind_spot_result
    assert result.n_permutations_across > 0, "across-entity null did not run"
    assert result.p_value_across is not None
    assert result.nsl_across is not None
    assert result.p_value_across < 0.01
    assert result.nsl_across >= 1.0


def test_or_combined_detection_fires_via_across_only(_blind_spot_result):
    report = build_verdict(_blind_spot_result)
    assert report.engine_detection.detected is True
    assert report.engine_detection.detection_channel == "across_entity"


def test_text_view_names_across_entity_channel(_blind_spot_result):
    from zekan.reports.text_view import render_verdict
    rendered = render_verdict(build_verdict(_blind_spot_result))
    assert "across-entity" in rendered.lower()


# ── (b) Integration: within-entity-only case (unchanged from pre-spec-1) ───────

@pytest.fixture(scope="module")
def _within_entity_leak_result():
    """A normal within-entity temporal leak (label proxy varying per row) —
    within-entity permutation should catch it as before spec 1."""
    df_clean = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=1)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = PredictionContract(
        prediction_problem="within-only-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )
    config = ZekanConfig(contract=contract)
    return run_severity_analysis(
        df_leaky, contract, config, _fast_clf, n_permutations=100, null_seed=0,
    )


def test_within_entity_only_case_still_fires(_within_entity_leak_result):
    r = _within_entity_leak_result
    assert r.p_value is not None and r.p_value < 0.01
    assert r.nsl is not None and r.nsl >= 1.0


def test_within_entity_only_case_detected_and_channel(_within_entity_leak_result):
    report = build_verdict(_within_entity_leak_result)
    assert report.engine_detection.detected is True
    assert report.engine_detection.detection_channel in ("within_entity", "both")


def test_within_entity_calibration_fields_unchanged_shape(_within_entity_leak_result):
    """The within-entity fields are populated exactly as before spec 1."""
    r = _within_entity_leak_result
    assert r.null_iqr is not None and r.null_iqr > 0
    assert r.null_99th is not None
    assert r.null_median is not None
    assert r.null_95th is not None


# ── (b) Integration: clean case (neither channel fires) ────────────────────────

@pytest.fixture(scope="module")
def _clean_result():
    rng = np.random.default_rng(999)
    df = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=1)
    df["noise_forbidden"] = rng.normal(size=len(df))
    contract = PredictionContract(
        prediction_problem="clean-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["noise_forbidden"],
    )
    config = ZekanConfig(contract=contract)
    return run_severity_analysis(
        df, contract, config, _fast_clf, n_permutations=100, null_seed=0,
    )


def test_clean_case_neither_channel_fires(_clean_result):
    report = build_verdict(_clean_result)
    assert report.engine_detection.detected is False
    assert report.engine_detection.detection_channel == ""


def test_clean_case_verdict_pass(_clean_result):
    report = build_verdict(_clean_result)
    assert report.policy_decision.verdict == "PASS", report.policy_decision.interpretation


# ── (b) Integration: determinism ────────────────────────────────────────────────

def test_two_runs_identical_within_and_across_scalars():
    df = _make_entity_churn_rate_dataset()
    contract = _audit_contract()
    config = _audit_config()
    r1 = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=100, null_seed=0)
    r2 = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=100, null_seed=0)

    assert r1.p_value == r2.p_value
    assert r1.nsl == r2.nsl
    assert r1.null_iqr == r2.null_iqr
    assert r1.null_99th == r2.null_99th

    assert r1.n_permutations_across == r2.n_permutations_across
    assert r1.p_value_across == r2.p_value_across
    assert r1.nsl_across == r2.nsl_across
    assert r1.null_iqr_across == r2.null_iqr_across
    assert r1.null_99th_across == r2.null_99th_across


def test_json_byte_identical_across_two_runs():
    df = _make_entity_churn_rate_dataset()
    contract = _audit_contract()
    config = _audit_config()
    r1 = run_audit(df, contract, config, model_factory=_fast_clf, n_permutations=100, null_seed=0)
    r2 = run_audit(df, contract, config, model_factory=_fast_clf, n_permutations=100, null_seed=0)
    assert r1.to_json() == r2.to_json()


# ── (b) Integration: additive JSON ──────────────────────────────────────────────

def test_json_additive_detection_channel_present_schema_unchanged():
    df = _make_entity_churn_rate_dataset()
    report = run_audit(
        df, _audit_contract(), _audit_config(), model_factory=_fast_clf,
        n_permutations=100, null_seed=0,
    )
    d = report.to_dict()

    assert d["schema_version"] == "1"

    ed = d["engine_detection"]
    assert "detection_channel" in ed
    assert ed["detection_channel"] == "across_entity"
    # Pre-existing keys still present and unchanged in shape.
    for key in ("detected", "p_value", "nsl", "alpha", "confidence", "interpretation"):
        assert key in ed

    # Existing top-level shape untouched.
    for key in ("engine_detection", "fold_ci", "measured_damage", "policy_decision",
                "schema_version", "structural_annotations", "summary"):
        assert key in d


# ── (b) Integration: power.py calibration boundary ──────────────────────────────

def test_power_py_estimate_n_required_never_called_by_across_entity_path(monkeypatch):
    """The across-entity null must use ONLY the method-agnostic live gate
    (_NULL_ALPHA, NSL>=1.0) — never power.py's within-entity-scoped
    0.176/0.352 constants (estimate_n_required needs its own calibration
    sweep before an across-entity power estimate would be valid)."""
    import zekan.severity.power as power_mod

    calls: list[float] = []
    original = power_mod.estimate_n_required

    def _spy(fl: float):
        calls.append(fl)
        return original(fl)

    monkeypatch.setattr(power_mod, "estimate_n_required", _spy)

    # Small/fast dataset; only need n_permutations > 0 to exercise both null
    # branches, detection outcome is irrelevant to this assertion.
    df = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=2)
    contract = PredictionContract(
        prediction_problem="power-boundary-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["feature_0"],
    )
    config = ZekanConfig(contract=contract, split_policy=SplitPolicy(n_splits=3))

    run_severity_analysis(df, contract, config, _fast_clf, n_permutations=20, null_seed=0)

    assert calls == [], (
        f"estimate_n_required was called {len(calls)} time(s) during a run that "
        "includes the across-entity null; power.py must stay within-entity-only"
    )
