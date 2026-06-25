"""Benchmark gate cases for the A/B/C correctness triangle (Phase 2b).

Severity band calibration (see engine.FIXABLE_LEAKAGE_* constants):
  near-zero / no-leak : fixable_leakage <= 0.02  (empirical noise floor)
  WARN / possible leak : 0.02 < fixable_leakage < 0.04
  clear leak           : fixable_leakage >= 0.04  (2x noise floor)
  strong leak          : fixable_leakage >= 0.10

Cases 1, 2a, 2b establish the noise floor — they MUST hold at <= 0.02 because
that bound is what justifies the 0.04 clear-leak boundary.  If they drift above
0.02, the band calibration is unfounded and must be revisited.

Case 1  - clean data, empty forbidden
          fixable_leakage <= 0.02  (noise floor)
          total_optimism in [-0.020, 0.033)  (both edges measured across 10 seeds)

Case 2a - covariate drift, empty forbidden
          fixable_leakage <= 0.02 (structural zero: B == C)

Case 2b - concept drift, empty forbidden
          fixable_leakage <= 0.02
          total_optimism >= case-1 total_optimism + 0.03

Case 3  - future_feature declared forbidden
          fixable_leakage >= 0.04  (clear-leak band: 2x noise floor)

Case 4  - correlated leaks (non-gating)
          cumulative ablation warning fires when one-at-a-time understates
"""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.benchmark.injectors import (
    inject_concept_drift,
    inject_correlated_leaks,
    inject_covariate_drift,
    inject_future_feature,
)
from gotcha.config.schema import GotchaConfig, SplitPolicy
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.severity.engine import run_severity_analysis


def _fast_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _make_contract(**kwargs) -> PredictionContract:
    defaults: dict = dict(
        prediction_problem="benchmark churn",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kwargs)
    return PredictionContract(**defaults)


def _make_config(contract: PredictionContract, leak_lookahead: int = 1) -> GotchaConfig:
    return GotchaConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
            leak_lookahead=leak_lookahead,
        ),
    )


# Shared base dataset: 500 entities x 10 periods = 5 000 rows
@pytest.fixture(scope="module")
def base_df():
    return make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=42)


# ── Case 1: clean baseline ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def case1_result(base_df):
    contract = _make_contract()
    config = _make_config(contract)
    return run_severity_analysis(base_df, contract, config, _fast_clf)


def test_case1_fixable_leakage_near_zero(case1_result) -> None:
    assert abs(case1_result.fixable_leakage) <= 0.02, (
        f"Case 1 fixable_leakage={case1_result.fixable_leakage:.4f}, expected <=0.02"
    )


def test_case1_total_optimism_banded(case1_result) -> None:
    # Both edges measured across 10 seeds (all pass):
    #   upper = observed_max (0.013) + noise_floor (0.020) = 0.033
    #   lower = -noise_floor = -0.020  (temporal can beat grouped by chance on clean data)
    # Case 2b min across same 10 seeds = 0.048 > 0.033 upper gate — no overlap.
    assert -0.020 <= case1_result.total_optimism < 0.033, (
        f"Case 1 total_optimism={case1_result.total_optimism:.4f} outside [-0.020, 0.033)"
    )


def test_case1_status_not_unavailable(case1_result) -> None:
    assert case1_result.status in ("pass", "warn"), (
        f"Case 1 clean data should be pass or warn, got {case1_result.status!r}"
    )


def test_case1_invariant(case1_result) -> None:
    reconstructed = case1_result.fixable_leakage + case1_result.nonfixable_optimism
    assert abs(case1_result.total_optimism - reconstructed) < 0.05, (
        f"Invariant: total={case1_result.total_optimism:.4f}, "
        f"fixable+nonfixable={reconstructed:.4f}"
    )


# ── Case 2a: covariate drift, empty forbidden ─────────────────────────────────

@pytest.fixture(scope="module")
def case2a_result(base_df):
    df_drift, _ = inject_covariate_drift(base_df)
    contract = _make_contract()
    config = _make_config(contract)
    return run_severity_analysis(df_drift, contract, config, _fast_clf)


def test_case2a_fixable_leakage_near_zero(case2a_result) -> None:
    """Covariate drift with no forbidden features => B == C => fixable_leakage ~0."""
    assert abs(case2a_result.fixable_leakage) <= 0.02, (
        f"Case 2a fixable_leakage={case2a_result.fixable_leakage:.4f}, expected <=0.02"
    )


# ── Case 2b: concept drift, empty forbidden ───────────────────────────────────

@pytest.fixture(scope="module")
def case2b_result(base_df):
    df_drift, _ = inject_concept_drift(base_df)
    contract = _make_contract()
    config = _make_config(contract)
    return run_severity_analysis(df_drift, contract, config, _fast_clf)


def test_case2b_fixable_leakage_near_zero(case2b_result) -> None:
    """Concept drift with no forbidden features => B == C => fixable_leakage ~0."""
    assert abs(case2b_result.fixable_leakage) <= 0.02, (
        f"Case 2b fixable_leakage={case2b_result.fixable_leakage:.4f}, expected <=0.02"
    )


def test_case2b_total_optimism_elevated_vs_clean(case1_result, case2b_result) -> None:
    """Concept drift must raise total_optimism >= clean baseline + 0.03."""
    gap = case2b_result.total_optimism - case1_result.total_optimism
    assert gap >= 0.03, (
        f"Concept drift did not elevate total_optimism enough: "
        f"clean={case1_result.total_optimism:.4f}, "
        f"drift={case2b_result.total_optimism:.4f}, gap={gap:.4f} (need >=0.03)"
    )


# ── Case 3: declared future leakage => clear positive fixable_leakage ─────────

@pytest.fixture(scope="module")
def case3_result(base_df):
    df_leak, record = inject_future_feature(base_df)
    contract = _make_contract(forbidden_after_prediction=record.planted_columns)
    config = _make_config(contract)
    return run_severity_analysis(df_leak, contract, config, _fast_clf)


def test_case3_fixable_leakage_clear_positive(case3_result) -> None:
    """Declared temporal leak must produce fixable_leakage >= 0.04.

    0.04 = 2x the empirical no-leak noise floor (0.02) established by Cases 1/2.
    This is the clear-leak band boundary: a signal at 2x noise floor is unlikely
    to be a measurement artifact.
    """
    assert case3_result.fixable_leakage >= 0.04, (
        f"Case 3 fixable_leakage={case3_result.fixable_leakage:.4f}, expected >=0.04"
    )


def test_case3_status_not_pass(case3_result) -> None:
    assert case3_result.status not in ("pass", "unavailable"), (
        f"Case 3 with real leakage should fail/warn, got status={case3_result.status}"
    )


# ── Case 4: correlated leaks -- non-gating warning check ──────────────────────

def test_case4_correlated_leaks_cumulative_warning(base_df) -> None:
    """Correlated leak pair should fire the one-at-a-time understatement warning."""
    from gotcha.severity.ablation import run_ablation
    from gotcha.severity.metrics import evaluate_folds
    from gotcha.severity.splitters import temporal_expanding_folds

    df_leak, record = inject_correlated_leaks(base_df)
    forbidden = record.planted_columns  # [corr_leak_alpha, corr_leak_beta]

    contract = _make_contract(forbidden_after_prediction=forbidden)

    temp_folds = temporal_expanding_folds(
        df_leak,
        time_col="prediction_time",
        entity_col="entity_id",
        target_col="target",
        n_splits=5,
        min_test_rows=50,
        min_pos=10,
        min_neg=10,
    )

    excluded = {"entity_id", "prediction_time", "target"}
    all_features = [c for c in df_leak.columns if c not in excluded]
    baseline = evaluate_folds(df_leak, all_features, "target", temp_folds, _fast_clf)

    summary = run_ablation(
        df_leak, contract, baseline.mean_auc, temp_folds,
        top_k=10, model_factory=_fast_clf,
    )

    # Both correlated columns must be ablated
    assert len(summary.individual) == 2
    assert all(e.ablated for e in summary.individual), (
        "Expected both corr_leak columns to be ablated"
    )
    # Cumulative ablation must be computed (>= 2 ablated features)
    assert summary.cumulative is not None

    # Understatement flag must fire: cumulative leakage substantially exceeds max
    # individual because dropping either column alone barely changes AUC (the partner
    # compensates), but dropping both removes the full shared z[T+1] source.
    assert summary.one_at_a_time_understates, (
        "Correlated-leak pair: cumulative leakage should substantially exceed max "
        "individual leakage -- dropping either alone barely changes AUC because the "
        "partner compensates; the understatement flag must fire"
    )

    # Print diagnostics regardless (non-gating)
    print(
        f"\nCase 4 -- correlated leaks:"
        f"\n  alpha individual leakage: {summary.individual[0].leakage_estimate:.4f}"
        f"\n  beta  individual leakage: {summary.individual[1].leakage_estimate:.4f}"
        f"\n  cumulative leakage:       {summary.cumulative.cumulative_leakage:.4f}"
        f"\n  understatement warning:   {summary.one_at_a_time_understates}"
    )
