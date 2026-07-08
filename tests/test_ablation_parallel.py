"""Tests for joblib-parallel ablation (Upgrade F1).

Verifies that n_jobs=2 (loky) produces byte-identical output to n_jobs=1 (serial)
for every quantity that flows into the --json verdict: same feature order, same
auc_without / leakage_estimate per feature, same not_ablated_reason on
budget-excluded entries, and identical verdict_to_dict JSON.

_fast_factory is defined at module level so loky can pickle it by (module, qualname).
"""

from __future__ import annotations

import json

import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.severity.ablation import run_ablation

# ── Fast estimator (module-level so loky workers can pickle by reference) ──────

def _fast_factory():
    """5-tree RF; n_jobs=1 prevents nested sklearn parallelism inside loky workers."""
    return RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1)


# ── Shared fixture ────────────────────────────────────────────────────────────

_N_ENTITIES = 200
_SNAPSHOTS = 6


@pytest.fixture(scope="module")
def ablation_base():
    """1200-row panel + contract (2 forbidden features) + folds + baseline_auc.

    Computed once per module; all tests share the same immutable inputs.
    """
    from zekan.config.schema import ZekanConfig
    from zekan.contract.prediction_contract import PredictionContract
    from zekan.severity.metrics import evaluate_folds
    from zekan.severity.splitters import random_grouped_folds

    df = make_clean_dataset(n_entities=_N_ENTITIES, snapshots_per_entity=_SNAPSHOTS, seed=1)

    contract = PredictionContract(
        prediction_problem="parallel-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["feature_0", "feature_1"],
    )
    cfg = ZekanConfig(contract=contract)
    policy = cfg.split_policy

    from zekan.severity.splitters import temporal_expanding_folds
    folds = temporal_expanding_folds(
        df,
        time_col=contract.prediction_time,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )

    # baseline_auc: grouped-CV AUC over all features (what the engine passes to run_ablation)
    all_features = [
        c for c in df.columns
        if c not in {contract.entity_id, contract.prediction_time,
                     contract.available_features_until, contract.target}
    ]
    rand_folds = random_grouped_folds(
        df,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )
    eval_a = evaluate_folds(df, all_features, contract.target, rand_folds, _fast_factory)
    baseline_auc = eval_a.mean_auc

    return df, contract, cfg, folds, baseline_auc


# ── Unit: run_ablation determinism ────────────────────────────────────────────

def test_serial_parallel_entries_identical(ablation_base):
    """n_jobs=1 and n_jobs=2 produce the same AblationEntry list (order + values)."""
    df, contract, _cfg, folds, baseline_auc = ablation_base

    serial = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=1
    )
    parallel = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=2
    )

    assert len(serial.individual) == len(parallel.individual)
    for s, p in zip(serial.individual, parallel.individual):
        assert s.feature == p.feature, "feature order differs"
        assert s.ablated == p.ablated
        assert s.auc_without == p.auc_without, f"auc_without differs for {s.feature}"
        assert s.leakage_estimate == p.leakage_estimate, f"leakage_estimate differs for {s.feature}"
        assert s.not_ablated_reason == p.not_ablated_reason


def test_serial_parallel_cumulative_identical(ablation_base):
    """Cumulative ablation block is identical between serial and parallel runs."""
    df, contract, _cfg, folds, baseline_auc = ablation_base

    serial = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=1
    )
    parallel = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=2
    )

    assert (serial.cumulative is None) == (parallel.cumulative is None)
    if serial.cumulative is not None:
        assert serial.cumulative.features_dropped == parallel.cumulative.features_dropped
        assert serial.cumulative.auc_without == parallel.cumulative.auc_without
        assert serial.cumulative.cumulative_leakage == parallel.cumulative.cumulative_leakage
    assert serial.one_at_a_time_understates == parallel.one_at_a_time_understates


def test_budget_excluded_reason_identical(ablation_base):
    """With top_k=1 and 2 forbidden features, the second is budget-excluded in both paths."""
    df, contract, _cfg, folds, baseline_auc = ablation_base

    serial = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=1, top_k=1
    )
    parallel = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=2, top_k=1
    )

    assert len(serial.individual) == 2
    assert len(parallel.individual) == 2

    # Top-ranked entry: ablated in both paths
    assert serial.individual[0].ablated
    assert parallel.individual[0].ablated
    assert serial.individual[0].feature == parallel.individual[0].feature

    # Second entry: budget-exceeded in both paths, with identical reason string
    assert not serial.individual[1].ablated
    assert not parallel.individual[1].ablated
    assert serial.individual[1].not_ablated_reason == "budget exceeded (top_k=1)"
    assert parallel.individual[1].not_ablated_reason == "budget exceeded (top_k=1)"


def test_parallel_same_ablated_set(ablation_base):
    """n_jobs=2 ablates the same feature set as n_jobs=1 (default top_k=10)."""
    df, contract, _cfg, folds, baseline_auc = ablation_base

    serial = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=1
    )
    parallel = run_ablation(
        df, contract, baseline_auc, folds, model_factory=_fast_factory, n_jobs=2
    )

    s_ablated = [e.feature for e in serial.individual if e.ablated]
    p_ablated = [e.feature for e in parallel.individual if e.ablated]
    assert s_ablated == p_ablated


# ── Byte-identical JSON proof ─────────────────────────────────────────────────

def test_parallel_verdict_json_byte_identical(ablation_base):
    """Full audit with n_jobs=2 produces byte-identical verdict_to_dict JSON as n_jobs=1.

    warn_floor=-1.0 forces ablation to run regardless of fixable_leakage magnitude.
    n_permutations=0 skips the null loop (out of scope for F1) for speed.
    """
    from zekan.reports.json_export import verdict_to_dict
    from zekan.severity.audit import run_audit

    df, contract, cfg, _folds, _baseline = ablation_base

    common = dict(
        model_factory=_fast_factory,
        n_permutations=0,
        warn_floor=-1.0,   # force ablation regardless of leakage magnitude
    )
    report_serial = run_audit(df, contract, cfg, **common, n_jobs=1)
    report_parallel = run_audit(df, contract, cfg, **common, n_jobs=2)

    json_serial = json.dumps(verdict_to_dict(report_serial), sort_keys=True, indent=2)
    json_parallel = json.dumps(verdict_to_dict(report_parallel), sort_keys=True, indent=2)

    assert json_serial == json_parallel, (
        "parallel-path JSON differs from serial:\n"
        + _first_diff(json_serial, json_parallel)
    )


def _first_diff(a: str, b: str) -> str:
    """Return the first line that differs between two multi-line strings."""
    for i, (la, lb) in enumerate(zip(a.splitlines(), b.splitlines())):
        if la != lb:
            return f"line {i + 1}: serial={la!r}  parallel={lb!r}"
    return f"lengths differ: {len(a)} vs {len(b)}"
