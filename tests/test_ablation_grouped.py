"""Tests for correlation-aware grouped ablation (Upgrade G).

Verifies the measured marginal contribution metric (apportioned_leakage) and
explanatory correlation metadata (max_correlation_among_ablated).

Flagship suppressor case: two correlated forbidden features (corr ≈ 0.90) each
show near-zero individual leakage but a large measured marginal — the partner was
compensating, masking the true contribution.

_fast_factory / _corr_factory are module-level so loky can pickle them by (module, qualname).
"""

from __future__ import annotations

import json
import math

import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.severity.ablation import run_ablation

# ── Estimator factories (module-level for loky pickling) ─────────────────────

def _fast_factory():
    """5-tree RF; n_jobs=1 prevents nested sklearn parallelism inside loky workers."""
    return RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1)


def _corr_factory():
    """20-tree RF for correlated-leak tests: reduces fold-level variance."""
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_folds_and_baseline(df, contract, factory):
    from zekan.config.schema import ZekanConfig
    from zekan.severity.metrics import evaluate_folds
    from zekan.severity.splitters import temporal_expanding_folds, random_grouped_folds

    cfg = ZekanConfig(contract=contract)
    policy = cfg.split_policy

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
    excluded = {contract.entity_id, contract.prediction_time,
                contract.available_features_until, contract.target}
    all_features = [c for c in df.columns if c not in excluded]
    rand_folds = random_grouped_folds(
        df,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=policy.n_splits,
        min_test_rows=policy.min_test_rows_per_fold,
        min_pos=policy.min_positive_cases_per_fold,
        min_neg=policy.min_negative_cases_per_fold,
    )
    baseline_auc = evaluate_folds(df, all_features, contract.target, rand_folds, factory).mean_auc
    return cfg, folds, baseline_auc


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def correlated_base():
    """Correlated leaky pair (corr ≈ 0.90): flagship suppressor case.

    Both features share z[entity, T+1] as a latent source.  Dropping either
    alone barely changes AUC (partner compensates); dropping both reveals the
    full shared-source leakage.
    """
    from zekan.benchmark.injectors import inject_correlated_leaks
    from zekan.contract.prediction_contract import PredictionContract

    df_clean = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=42)
    df, _ = inject_correlated_leaks(df_clean, seed=5, dataset_seed=42)

    contract = PredictionContract(
        prediction_problem="suppressor-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["corr_leak_alpha", "corr_leak_beta"],
    )
    cfg, folds, baseline_auc = _make_folds_and_baseline(df, contract, _corr_factory)
    return df, contract, cfg, folds, baseline_auc


@pytest.fixture(scope="module")
def independent_base():
    """Two independent leaky features: label proxy + future covariate.

    leaky_label_proxy  — near-copy of the target label (row-level signal)
    future_z_latent    — z[entity, T+1] plus noise (temporal signal)

    These derive from different underlying signals, so they carry largely
    independent information.  apportioned_leakage should ≈ leakage_estimate.
    """
    from zekan.benchmark.injectors import inject_label_proxy, inject_future_feature
    from zekan.contract.prediction_contract import PredictionContract

    df_clean = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=42)
    df, _ = inject_label_proxy(df_clean, seed=1)
    df, _ = inject_future_feature(df, seed=2, dataset_seed=42)

    contract = PredictionContract(
        prediction_problem="independent-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy", "future_z_latent"],
    )
    cfg, folds, baseline_auc = _make_folds_and_baseline(df, contract, _fast_factory)
    return df, contract, cfg, folds, baseline_auc


@pytest.fixture(scope="module")
def single_base():
    """Single forbidden feature: only one feature ablated."""
    from zekan.benchmark.injectors import inject_label_proxy
    from zekan.contract.prediction_contract import PredictionContract

    df_clean = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=42)
    df, _ = inject_label_proxy(df_clean, seed=1)

    contract = PredictionContract(
        prediction_problem="single-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )
    cfg, folds, baseline_auc = _make_folds_and_baseline(df, contract, _fast_factory)
    return df, contract, cfg, folds, baseline_auc


# ── Suppressor case ───────────────────────────────────────────────────────────

def test_suppressor_apportioned_exceeds_individual(correlated_base):
    """Correlated pair: each feature's apportioned_leakage >> leakage_estimate.

    When the partner is present, it compensates and individual leakage is near-zero.
    Once the partner is removed (the marginal measurement starting point), each
    feature's true contribution becomes visible.
    """
    df, contract, _cfg, folds, baseline_auc = correlated_base

    summary = run_ablation(df, contract, baseline_auc, folds, model_factory=_corr_factory)

    ablated = [e for e in summary.individual if e.ablated]
    assert len(ablated) == 2, "both corr_leak features must be ablated"

    for entry in ablated:
        assert not math.isnan(entry.apportioned_leakage), (
            f"{entry.feature}: apportioned_leakage is NaN"
        )
        assert entry.apportioned_leakage > entry.leakage_estimate + 0.03, (
            f"{entry.feature}: apportioned={entry.apportioned_leakage:.4f} should "
            f"substantially exceed individual={entry.leakage_estimate:.4f} "
            "(suppressor signature not detected)"
        )


# ── Independent-features case ─────────────────────────────────────────────────

def test_independent_apportioned_approx_individual(independent_base):
    """Independent leaks: apportioned_leakage ≈ leakage_estimate (no suppression).

    When features A and B are independent, removing B doesn't change A's
    contribution; apportioned and individual should be within tolerance.
    """
    df, contract, _cfg, folds, baseline_auc = independent_base

    summary = run_ablation(df, contract, baseline_auc, folds, model_factory=_fast_factory)

    ablated = [e for e in summary.individual if e.ablated]
    assert len(ablated) == 2, "both forbidden features must be ablated"

    for entry in ablated:
        assert not math.isnan(entry.apportioned_leakage), (
            f"{entry.feature}: apportioned_leakage is NaN"
        )
        diff = abs(entry.apportioned_leakage - entry.leakage_estimate)
        assert diff < 0.10, (
            f"{entry.feature}: apportioned={entry.apportioned_leakage:.4f} differs "
            f"from individual={entry.leakage_estimate:.4f} by {diff:.4f} "
            "(expected < 0.10 for independent features)"
        )


# ── Single-ablated-feature ────────────────────────────────────────────────────

def test_single_ablated_apportioned_equals_individual(single_base):
    """Single ablated feature: apportioned_leakage == leakage_estimate exactly.

    By definition: if there is only one ablated feature, marginal given no
    other leaks present == individual retrain-without.  No extra evaluate_folds
    call is made.
    """
    df, contract, _cfg, folds, baseline_auc = single_base

    summary = run_ablation(df, contract, baseline_auc, folds, model_factory=_fast_factory)

    ablated = [e for e in summary.individual if e.ablated]
    assert len(ablated) == 1

    entry = ablated[0]
    assert entry.apportioned_leakage == entry.leakage_estimate, (
        f"single-feature: apportioned={entry.apportioned_leakage} != "
        f"individual={entry.leakage_estimate}"
    )
    # max_correlation should be NaN (no pair to compute correlation on)
    assert math.isnan(summary.max_correlation_among_ablated)


# ── max_correlation_among_ablated ────────────────────────────────────────────

def test_max_correlation_correct_value(correlated_base):
    """max_correlation_among_ablated ≈ 0.90 for the designed correlated pair."""
    df, contract, _cfg, folds, baseline_auc = correlated_base

    summary = run_ablation(df, contract, baseline_auc, folds, model_factory=_corr_factory)

    assert not math.isnan(summary.max_correlation_among_ablated), (
        "max_correlation_among_ablated must be finite for 2 ablated features"
    )
    assert 0.70 <= summary.max_correlation_among_ablated <= 1.0, (
        f"expected max_corr ≈ 0.90, got {summary.max_correlation_among_ablated:.4f}"
    )


# ── Determinism: serial vs parallel ──────────────────────────────────────────

def test_determinism_apportioned_and_json_byte_identical(correlated_base):
    """n_jobs=1 and n_jobs=2 produce identical apportioned_leakage and byte-identical JSON.

    warn_floor=-1.0 forces ablation regardless of leakage magnitude.
    n_permutations=0 skips the null loop (out of scope).
    """
    from zekan.reports.json_export import verdict_to_dict
    from zekan.severity.audit import run_audit

    df, contract, cfg, _folds, _baseline = correlated_base

    common = dict(model_factory=_corr_factory, n_permutations=0, warn_floor=-1.0)
    report_s = run_audit(df, contract, cfg, **common, n_jobs=1)
    report_p = run_audit(df, contract, cfg, **common, n_jobs=2)

    attr_s = report_s.measured_damage.feature_attribution
    attr_p = report_p.measured_damage.feature_attribution
    assert attr_s is not None and attr_p is not None

    for es, ep in zip(attr_s.individual, attr_p.individual):
        assert es.feature == ep.feature
        assert es.apportioned_leakage == ep.apportioned_leakage, (
            f"{es.feature}: apportioned differs: serial={es.apportioned_leakage} "
            f"parallel={ep.apportioned_leakage}"
        )

    json_s = json.dumps(verdict_to_dict(report_s), sort_keys=True, indent=2)
    json_p = json.dumps(verdict_to_dict(report_p), sort_keys=True, indent=2)
    assert json_s == json_p, (
        "parallel-path JSON differs from serial:\n" + _first_diff(json_s, json_p)
    )


# ── Additive JSON ─────────────────────────────────────────────────────────────

def test_additive_json_keys_and_schema_version(correlated_base):
    """verdict_to_dict: schema_version stays "1"; new keys present; no breakage.

    Checks:
    - schema_version == "1" (not bumped)
    - individual[*] now has "apportioned_leakage"
    - feature_attribution has "max_correlation_among_ablated"
    - existing keys (leakage_estimate, auc_without, etc.) still present
    """
    from zekan.reports.json_export import verdict_to_dict
    from zekan.severity.audit import run_audit

    df, contract, cfg, _folds, _baseline = correlated_base

    report = run_audit(df, contract, cfg,
                       model_factory=_corr_factory, n_permutations=0, warn_floor=-1.0)
    d = verdict_to_dict(report)

    assert d["schema_version"] == "1"

    fa = d["measured_damage"]["feature_attribution"]
    assert fa is not None
    assert "max_correlation_among_ablated" in fa

    for item in fa["individual"]:
        assert "apportioned_leakage" in item, (
            f"apportioned_leakage missing from individual entry: {item}"
        )
        # Existing keys must still be present
        for key in ("feature", "leakage_estimate", "auc_without", "ablated"):
            assert key in item, f"existing key '{key}' missing"


# ── Helper ────────────────────────────────────────────────────────────────────

def _first_diff(a: str, b: str) -> str:
    for i, (la, lb) in enumerate(zip(a.splitlines(), b.splitlines())):
        if la != lb:
            return f"line {i + 1}: serial={la!r}  parallel={lb!r}"
    return f"lengths differ: {len(a)} vs {len(b)}"
