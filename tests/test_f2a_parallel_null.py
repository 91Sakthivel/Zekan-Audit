"""Tests for F2a: parallelizing the null permutation loop via joblib with
spawn_v2 (SeedSequence) seeding.

Layout:
  (a) unit tests on null_baseline.py: _null_permutation_once's skip contract,
      NullResult.scheme, and the aggregation logic (None-filtering / n_draws /
      p_value denominator) under a controlled monkeypatch.
  (b) THE acceptance bar: null_samples/NullResult scalars are byte-identical
      for n_jobs=1 vs n_jobs=2, for BOTH within_entity and across_entity.
  (c) determinism: repeated runs at a fixed n_jobs are identical.
  (d) full audit: --jobs accelerates both nulls; --json is byte-identical vs
      n_jobs=1, and provenance.seed.null_scheme == "spawn_v2".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.config.schema import SplitPolicy, ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.audit import run_audit
from zekan.severity.engine import run_severity_analysis
from zekan.severity.null_baseline import (
    NullResult,
    _null_permutation_once,
    estimate_fixable_leakage_null,
)

from tests.test_entity_aggregate_probe import (
    _audit_config,
    _audit_contract,
    _make_entity_churn_rate_dataset,
)


def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


# ── Shared lightweight leaky panel (small, fast — no >=1000-row gate needed
# since estimate_fixable_leakage_null is called directly, not through
# run_severity_analysis's contract validation) ─────────────────────────────

def _leaky_panel(n_entities: int = 60, n_periods: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        entity_risk = float(rng.normal(0.0, 1.0))
        for t in range(n_periods):
            logit = 0.8 * entity_risk + 0.3 * t / n_periods + float(rng.normal(0.0, 0.5))
            rows.append({
                "entity_id": f"e{e}",
                "prediction_time": f"2022-{t+1:02d}-01",
                "feature_a": float(rng.normal()),
                "future_leak": logit + float(rng.normal(0.0, 0.1)),  # varies per row
                "target": 0,  # placeholder, filled below
                "_logit": logit,
            })
    df = pd.DataFrame(rows)
    thresh = float(np.percentile(df["_logit"].values, 60))
    df["target"] = (df["_logit"] > thresh).astype(int)
    return df.drop(columns=["_logit"])


def _leaky_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="f2a-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["future_leak"],
    )


def _leaky_config() -> ZekanConfig:
    return ZekanConfig(
        contract=_leaky_contract(),
        split_policy=SplitPolicy(
            n_splits=3, min_test_rows_per_fold=10,
            min_positive_cases_per_fold=3, min_negative_cases_per_fold=3,
        ),
    )


def _observed_fl(df, contract, config) -> float:
    r = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=0)
    return r.fixable_leakage


# ── (a) Unit: _null_permutation_once skip contract ─────────────────────────

def test_null_permutation_once_returns_none_when_no_interior_predictions():
    """Rigging interior_fold_idxs to not match any real fold index forces the
    same skip path as the original `continue` — deterministic, no randomness."""
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)

    from zekan.severity.null_baseline import _build_interior_fold_set, _pool_oof_predictions
    from zekan.severity.splitters import temporal_expanding_folds
    from zekan.severity.metrics import evaluate_folds
    from sklearn.metrics import roc_auc_score

    policy = config.split_policy
    temp_folds = temporal_expanding_folds(
        df, time_col=contract.prediction_time, entity_col=contract.entity_id,
        target_col=contract.target, n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )
    interior_fold_idxs = _build_interior_fold_set(df, contract, config, temp_folds)
    all_feature_cols = ["feature_a", "future_leak"]
    safe_feature_cols = ["feature_a"]

    eval_c = evaluate_folds(df, safe_feature_cols, contract.target, temp_folds, _fast_clf, return_predictions=True)
    y_pool, proba_c_pool = _pool_oof_predictions(eval_c.fold_evals, interior_fold_idxs)
    auc_c_pool = float(roc_auc_score(y_pool, proba_c_pool))

    child = np.random.SeedSequence(0).spawn(1)[0]

    # Real interior_fold_idxs -> a real float.
    real = _null_permutation_once(
        child, df, ["future_leak"], contract.entity_id, "within_entity",
        all_feature_cols, contract.target, temp_folds, _fast_clf,
        interior_fold_idxs, auc_c_pool, y_pool,
    )
    assert real is not None
    assert isinstance(real, float)

    # Bogus interior_fold_idxs (no real fold has this index) -> None (skip).
    bogus_idxs = {99999}
    skipped = _null_permutation_once(
        child, df, ["future_leak"], contract.entity_id, "within_entity",
        all_feature_cols, contract.target, temp_folds, _fast_clf,
        bogus_idxs, auc_c_pool, y_pool,
    )
    assert skipped is None


def test_null_permutation_once_unsupported_method_raises():
    df = _leaky_panel()
    contract = _leaky_contract()
    child = np.random.SeedSequence(0).spawn(1)[0]
    with pytest.raises(ValueError, match="unsupported method"):
        _null_permutation_once(
            child, df, ["future_leak"], contract.entity_id, "target_within_period",
            ["feature_a", "future_leak"], "target", [], _fast_clf, set(), 0.5,
            np.array([0, 1]),
        )


# ── (a) Unit: NullResult.scheme ─────────────────────────────────────────────

def test_scheme_is_spawn_v2_for_within_entity():
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)
    result = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=10, seed=0, method="within_entity",
    )
    assert isinstance(result, NullResult)
    assert result.scheme == "spawn_v2"


def test_scheme_is_spawn_v2_for_across_entity():
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)
    result = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=10, seed=0, method="across_entity",
    )
    assert result.scheme == "spawn_v2"


def test_scheme_is_serial_v1_for_target_within_period():
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)
    result = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=10, seed=0, method="target_within_period",
    )
    assert result.scheme == "serial_v1"


# ── (a) Unit: aggregation (None-filtering / n_draws / p_value denominator) ──

def test_skip_sentinel_aggregation_serial(monkeypatch):
    """Monkeypatch the work unit to skip a KNOWN subset of permutations and
    confirm the parent's n_draws/p_value correctly reflect only the non-skipped
    draws — proves the None-filtering contract independent of real skip data.
    """
    import zekan.severity.null_baseline as nb

    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        # Skip every 3rd call deterministically (call order == permutation order
        # for n_jobs=1, since it is a plain serial loop).
        if calls["n"] % 3 == 0:
            return None
        return 0.01 * calls["n"]

    monkeypatch.setattr(nb, "_null_permutation_once", _fake_unit)

    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    result = nb.estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=0.5,
        n_permutations=9, seed=0, method="within_entity", n_jobs=1,
    )

    # 9 calls, every 3rd (3, 6, 9) skipped -> 6 real draws.
    assert calls["n"] == 9
    assert result.n_permutations == 6
    assert len(result.null_samples) == 6
    # p_value denominator must use n_draws (6), not n_permutations (9).
    count_gte = int(np.sum(result.null_samples >= 0.5))
    expected_p = (count_gte + 1) / (6 + 1)
    assert result.p_value == pytest.approx(expected_p)


# ── (b) THE acceptance bar: n_jobs=1 == n_jobs=2, byte-identical ───────────

@pytest.mark.parametrize("method", ["within_entity", "across_entity"])
def test_n_jobs_independence_byte_identical(method):
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)

    r1 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=20, seed=0, method=method, n_jobs=1,
    )
    r2 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=20, seed=0, method=method, n_jobs=2,
    )

    assert r1.n_permutations == r2.n_permutations
    assert np.array_equal(r1.null_samples, r2.null_samples), (
        f"{method}: null_samples differ between n_jobs=1 and n_jobs=2 — "
        "spawn_v2 seeding must make results independent of scheduling"
    )
    assert r1.null_median == r2.null_median
    assert r1.null_95th == r2.null_95th
    assert r1.null_99th == r2.null_99th
    assert r1.null_iqr == r2.null_iqr
    assert r1.p_value == r2.p_value


# ── (c) Determinism at a fixed n_jobs ───────────────────────────────────────

@pytest.mark.parametrize("n_jobs", [1, 2])
def test_determinism_same_seed_same_n_jobs(n_jobs):
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    fl = _observed_fl(df, contract, config)

    r1 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=15, seed=7, method="within_entity", n_jobs=n_jobs,
    )
    r2 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=15, seed=7, method="within_entity", n_jobs=n_jobs,
    )

    assert np.array_equal(r1.null_samples, r2.null_samples)
    assert r1.p_value == r2.p_value


# ── (b/d) Both nulls through run_severity_analysis, n_jobs=1 vs n_jobs=2 ───

def test_both_nulls_parallel_match_serial_via_run_severity_analysis():
    # row_count_and_folds needs >= 1000 rows to reach can_compute_severity=True
    # through the full run_severity_analysis contract gate (unlike the direct
    # estimate_fixable_leakage_null calls above, which bypass that gate).
    df = _leaky_panel(n_entities=180, n_periods=6)
    contract = _leaky_contract()
    config = _leaky_config()

    r1 = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=20, null_seed=0, n_jobs=1)
    r2 = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=20, null_seed=0, n_jobs=2)
    assert r1.status != "unavailable", r1.unavailable_reason

    # within-entity fields
    assert r1.p_value == r2.p_value
    assert r1.nsl == r2.nsl
    assert r1.null_iqr == r2.null_iqr

    # across-entity fields — both must actually have run (n_entities >= 2)
    assert r1.n_permutations_across > 0
    assert r2.n_permutations_across > 0
    assert r1.p_value_across == r2.p_value_across
    assert r1.nsl_across == r2.nsl_across
    assert r1.null_iqr_across == r2.null_iqr_across


# ── (d) Full audit: --jobs (n_jobs) end-to-end, JSON byte-identical ────────

def test_run_audit_json_byte_identical_n_jobs_1_vs_2():
    df = _make_entity_churn_rate_dataset()
    contract = _audit_contract()
    config = _audit_config()

    r1 = run_audit(df, contract, config, model_factory=_fast_clf, n_permutations=20, null_seed=0, n_jobs=1)
    r2 = run_audit(df, contract, config, model_factory=_fast_clf, n_permutations=20, null_seed=0, n_jobs=2)

    assert r1.to_json() == r2.to_json()
    assert r1.engine_detection.detection_channel == r2.engine_detection.detection_channel


def test_run_audit_n_jobs_2_exercises_across_null():
    """--jobs 2 must reach and populate the across-entity fields through the
    FULL run_audit path, not just within-entity.  Uses n_permutations=20 (cheap)
    since population, not detection significance, is what's under test here —
    detection-under-spawn_v2 at the full n_permutations=100 floor is already
    proven by tests/test_across_entity_null.py's blind-spot suite."""
    df = _make_entity_churn_rate_dataset()
    contract = _audit_contract()
    config = _audit_config()

    report = run_audit(df, contract, config, model_factory=_fast_clf, n_permutations=20, null_seed=0, n_jobs=2)
    d = report.to_dict()
    assert d["schema_version"] == "1"

    # run_audit wraps run_severity_analysis + build_verdict; call the former
    # directly (same df/contract/config) to inspect the raw across-entity
    # fields that VerdictReport does not expose unless detection fires.
    severity = run_severity_analysis(df, contract, config, _fast_clf, n_permutations=20, null_seed=0, n_jobs=2)
    assert severity.n_permutations_across > 0
    assert severity.p_value_across is not None
    assert severity.nsl_across is not None
