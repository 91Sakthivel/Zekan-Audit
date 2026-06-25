"""Unit tests for the severity engine (Phase 2b)."""

from __future__ import annotations

import math

import pytest
from sklearn.ensemble import RandomForestClassifier

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.config.schema import GotchaConfig, SplitPolicy
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.severity.engine import SeverityResult, run_severity_analysis


def _fast_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


def _make_contract(**kwargs) -> PredictionContract:
    defaults: dict = dict(
        prediction_problem="test churn prediction",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kwargs)
    return PredictionContract(**defaults)


def _make_config(contract: PredictionContract) -> GotchaConfig:
    return GotchaConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
        ),
    )


@pytest.fixture(scope="module")
def clean_df():
    return make_clean_dataset(n_entities=300, snapshots_per_entity=5, seed=42)


@pytest.fixture(scope="module")
def clean_contract():
    return _make_contract()


@pytest.fixture(scope="module")
def clean_config(clean_contract):
    return _make_config(clean_contract)


@pytest.fixture(scope="module")
def clean_result(clean_df, clean_contract, clean_config):
    return run_severity_analysis(clean_df, clean_contract, clean_config, _fast_clf)


# ── Result shape ──────────────────────────────────────────────────────────────

def test_result_fields_present(clean_result: SeverityResult) -> None:
    assert clean_result.status in ("pass", "warn", "fail", "unavailable")
    assert clean_result.metric == "roc_auc"
    assert not math.isnan(clean_result.naive_auc)
    assert not math.isnan(clean_result.estimated_deployable_auc)
    assert not math.isnan(clean_result.total_optimism)
    assert not math.isnan(clean_result.fixable_leakage)
    assert not math.isnan(clean_result.nonfixable_optimism)
    assert isinstance(clean_result.per_fold, list)
    assert len(clean_result.caveat) > 0


def test_result_per_fold_populated(clean_result: SeverityResult) -> None:
    assert len(clean_result.per_fold) >= 1
    for pf in clean_result.per_fold:
        assert 0.0 <= pf.auc_with_forbidden <= 1.0
        assert 0.0 <= pf.auc_without_forbidden <= 1.0


def test_aucs_in_range(clean_result: SeverityResult) -> None:
    assert 0.5 < clean_result.naive_auc < 1.0, (
        f"naive_auc={clean_result.naive_auc:.4f} not in (0.5, 1.0)"
    )
    assert 0.5 < clean_result.estimated_deployable_auc < 1.0, (
        f"estimated_deployable_auc={clean_result.estimated_deployable_auc:.4f}"
    )


# ── Invariant ─────────────────────────────────────────────────────────────────

def test_invariant_total_optimism_approx(clean_result: SeverityResult) -> None:
    """total_optimism ~= fixable_leakage + nonfixable_optimism within 0.05."""
    reconstructed = clean_result.fixable_leakage + clean_result.nonfixable_optimism
    assert abs(clean_result.total_optimism - reconstructed) < 0.05, (
        f"Invariant broken: total={clean_result.total_optimism:.4f}, "
        f"fixable+nonfixable={reconstructed:.4f}"
    )


# ── Unavailable path ──────────────────────────────────────────────────────────

def test_unavailable_when_too_few_rows() -> None:
    """Severity must be UNAVAILABLE when the dataset is below the row-count gate."""
    tiny_df = make_clean_dataset(n_entities=10, snapshots_per_entity=5, seed=1)
    contract = _make_contract()
    config = _make_config(contract)
    result = run_severity_analysis(tiny_df, contract, config, _fast_clf)
    assert result.status == "unavailable"
    assert result.unavailable_reason is not None
    assert math.isnan(result.naive_auc)


# ── Empty forbidden => zero fixable leakage ───────────────────────────────────

def test_empty_forbidden_zero_fixable_leakage(
    clean_df, clean_contract, clean_config
) -> None:
    """With no forbidden features, B == C so fixable_leakage must be ~0 and status pass/warn."""
    result = run_severity_analysis(clean_df, clean_contract, clean_config, _fast_clf)
    assert abs(result.fixable_leakage) <= 0.02, (
        f"Expected fixable_leakage ~0 but got {result.fixable_leakage:.4f}"
    )
    assert result.status in ("pass", "warn"), (
        f"Clean data with no forbidden features should be pass or warn, got {result.status!r}"
    )


# ── top_k enforcement ─────────────────────────────────────────────────────────

def test_top_k_enforcement() -> None:
    """Ablation must respect top_k and mark excess candidates as not_ablated."""
    import pandas as pd
    from gotcha.severity.ablation import run_ablation
    from gotcha.severity.splitters import temporal_expanding_folds

    df = make_clean_dataset(n_entities=200, snapshots_per_entity=5, seed=42)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    contract = _make_contract(forbidden_after_prediction=feature_cols)

    temp_folds = temporal_expanding_folds(
        df,
        time_col="prediction_time",
        entity_col="entity_id",
        target_col="target",
        n_splits=5,
        min_test_rows=50,
        min_pos=10,
        min_neg=10,
    )

    summary = run_ablation(
        df, contract, 0.75, temp_folds, top_k=2, model_factory=_fast_clf
    )

    ablated_count = sum(1 for e in summary.individual if e.ablated)
    not_ablated_count = sum(1 for e in summary.individual if not e.ablated)
    assert ablated_count <= 2
    assert not_ablated_count >= len(feature_cols) - 2
    for e in summary.individual:
        if not e.ablated:
            assert e.not_ablated_reason is not None
