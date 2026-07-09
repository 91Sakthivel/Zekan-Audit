"""Tests for Upgrade H: skipped-fold transparency + min-evaluable gate.

Two moves verified:
  MOVE A — FoldCI exposes folds_evaluated, folds_skipped, skip_reasons.
  MOVE B — build_verdict returns UNCONFIRMED_HIGH_DAMAGE when folds_evaluated
            < min_valid_folds (early-return path).

All tests are deterministic (fixed seeds, n_permutations=0).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.config.schema import SplitPolicy, ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.reports.json_export import verdict_to_dict
from zekan.severity.audit import run_audit
from zekan.severity.verdict import build_verdict


# ── Factories (module-level for loky) ────────────────────────────────────────

def _factory():
    return RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)


# ── Shared contract/config helpers ────────────────────────────────────────────

def _make_contract(forbidden: list[str]) -> PredictionContract:
    return PredictionContract(
        prediction_problem="fold-transparency-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=forbidden,
    )


def _make_config(min_valid_folds: int = 3, n_splits: int = 5) -> ZekanConfig:
    return ZekanConfig(
        contract=_make_contract(["leaky_label_proxy"]),
        split_policy=SplitPolicy(min_valid_folds=min_valid_folds, n_splits=n_splits),
    )


# ── Dataset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def normal_dataset():
    """Standard dataset: 200 entities × 6 snapshots → enough folds to pass gate."""
    df_clean = make_clean_dataset(n_entities=200, snapshots_per_entity=6, seed=42)
    df, _ = inject_label_proxy(df_clean, seed=1)
    return df


# ── MOVE A: transparency tests ────────────────────────────────────────────────

def test_fold_ci_has_three_new_fields(normal_dataset):
    """FoldCI object exposes folds_evaluated, folds_skipped, skip_reasons."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    fci = report.fold_ci

    assert hasattr(fci, "folds_evaluated")
    assert hasattr(fci, "folds_skipped")
    assert hasattr(fci, "skip_reasons")


def test_folds_evaluated_positive_on_normal_data(normal_dataset):
    """Normal-sized dataset: folds_evaluated >= 1 (at least one fold ran)."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    assert report.fold_ci.folds_evaluated >= 1


def test_folds_evaluated_plus_skipped_equals_total(normal_dataset):
    """folds_evaluated + folds_skipped must equal the total fold list length."""
    from zekan.severity.splitters import temporal_expanding_folds

    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    fci = report.fold_ci

    # Cross-check by reproducing the fold list independently
    folds = temporal_expanding_folds(
        df,
        time_col=contract.prediction_time,
        entity_col=contract.entity_id,
        target_col=contract.target,
        n_splits=cfg.split_policy.n_splits,
        min_test_rows=cfg.split_policy.min_test_rows_per_fold,
        min_pos=cfg.split_policy.min_positive_cases_per_fold,
        min_neg=cfg.split_policy.min_negative_cases_per_fold,
    )
    assert fci.folds_evaluated + fci.folds_skipped == len(folds)


def test_skip_reasons_is_list(normal_dataset):
    """skip_reasons is always a list (empty when no skips)."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    assert isinstance(report.fold_ci.skip_reasons, list)


def test_fold_ci_json_contains_new_keys(normal_dataset):
    """verdict_to_dict: fold_ci object in JSON has folds_evaluated/skipped/skip_reasons."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    d = verdict_to_dict(report)

    fci_json = d["fold_ci"]
    assert "folds_evaluated" in fci_json
    assert "folds_skipped" in fci_json
    assert "skip_reasons" in fci_json
    assert isinstance(fci_json["skip_reasons"], list)


# ── MOVE B: min-evaluable gate tests ─────────────────────────────────────────

def test_gate_trips_when_min_valid_folds_exceeds_n_splits(normal_dataset):
    """Gate fires when min_valid_folds > n_splits (total temporal folds created).

    n_splits=3 → at most 3 temporal folds.  Setting min_valid_folds=5 ensures
    folds_evaluated (≤3) < min_valid_folds (5) → UNCONFIRMED_HIGH_DAMAGE.
    Dataset is large enough (200 entities × 6 snapshots) to pass the engine
    hard-minimum, so the result reaches build_verdict.
    """
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config(min_valid_folds=5, n_splits=3)

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE", (
        f"expected UNCONFIRMED_HIGH_DAMAGE from gate, got {report.policy_decision.verdict}; "
        f"folds_evaluated={report.fold_ci.folds_evaluated}"
    )


def test_gate_instability_note_contains_fold_count(normal_dataset):
    """When gate trips, instability_note mentions evaluated vs total fold counts."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config(min_valid_folds=5, n_splits=3)

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    note = report.fold_ci.instability_note
    assert "evaluable folds" in note or "insufficient" in note.lower(), (
        f"instability_note does not mention fold starvation: {note!r}"
    )


def test_gate_does_not_trip_on_normal_data_default_min(normal_dataset):
    """Normal-sized data with default min_valid_folds=3 and n_splits=5 passes gate."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config(min_valid_folds=3, n_splits=5)

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    # Gate must not have fired: folds_evaluated >= min_valid_folds
    assert report.fold_ci.folds_evaluated >= 3, (
        f"expected >= 3 evaluated folds, got {report.fold_ci.folds_evaluated}"
    )


def test_gate_json_is_additive_schema_version_unchanged(normal_dataset):
    """Gate path: schema_version stays '1'; fold_ci keys present; verdict correct."""
    df = normal_dataset
    contract = _make_contract(["leaky_label_proxy"])
    cfg = _make_config(min_valid_folds=5, n_splits=3)

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    d = verdict_to_dict(report)

    assert d["schema_version"] == "1"
    assert "folds_evaluated" in d["fold_ci"]
    assert "folds_skipped" in d["fold_ci"]
    assert "skip_reasons" in d["fold_ci"]
    assert d["policy_decision"]["verdict"] == "UNCONFIRMED_HIGH_DAMAGE"


# ── Runtime fold starvation: gate fires on actual viability-check skips ───────
#
# The tests above (min_valid_folds > n_splits) verify the gate logic against
# a config-constrained fold count.  The tests below verify it fires when folds
# are *skipped at runtime* by _skip_reason (too few positives per test window),
# even with n_splits >= min_valid_folds.

def _make_skip_all_config(min_valid_folds: int = 3) -> ZekanConfig:
    """Config that forces ALL temporal folds to skip via the positives viability check.

    make_clean_dataset(n_entities=100) → n_periods=10, n_rows=1000, base_rate=0.3.
    With n_splits=5 the temporal fold edges are [0,2,3,5,7,8,10], giving test
    windows of 1-2 periods = 100-200 rows × 30% base_rate ≈ 30-60 positives.
    Setting min_positive_cases_per_fold=100 ensures every window has too few
    positives (30-60 < 100) and is skipped by _skip_reason at runtime.
    """
    return ZekanConfig(
        contract=_make_contract([]),
        split_policy=SplitPolicy(
            n_splits=5,
            min_valid_folds=min_valid_folds,
            min_positive_cases_per_fold=100,
        ),
    )


@pytest.fixture(scope="module")
def skip_all_dataset():
    """Clean dataset (no leaky feature) where all temporal folds skip at runtime.

    n_entities=100, n_periods=10 (max(6,10)), n_rows=1000, base_rate=0.3 →
    ~300 positives total. Each test window ≈ 60 positives < min_positive_cases_per_fold=100.
    """
    return make_clean_dataset(n_entities=100, snapshots_per_entity=6, seed=99)


def _make_graduated_leaky_dataset(seed: int = 42) -> pd.DataFrame:
    """Dataset with graduated positives: sparse early periods, dense late periods.

    100 entities × 12 monthly periods = 1200 rows.
    - Periods T0–T7 (early, 8 periods): 2 % positive rate  → ~4 positives per
      2-period test window → fails min_positive_cases_per_fold=20 → SKIP.
    - Periods T8–T11 (late, 4 periods): 60 % positive rate → ~120 positives per
      2-period test window → passes all viability checks → EVALUATE.

    With n_splits=5 and edges [0,2,4,6,8,10,12]:
      Folds 0-2  test=[T2,T3], [T4,T5], [T6,T7]  → SKIP (early periods)
      Folds 3-4  test=[T8,T9], [T10,T11]          → EVALUATE (late periods)
      → folds_evaluated=2, folds_skipped=3; default min_valid_folds=3 → gate fires.

    leaky_proxy ≈ target + N(0, 0.05): near-perfect forbidden feature.
    Declared forbidden → produces fixable_leakage > 0 from the 2 evaluated folds.
    """
    rng = np.random.default_rng(seed)
    n_entities = 100
    n_periods = 12

    period_dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    period_strs = [d.strftime("%Y-%m-%d") for d in period_dates]

    entity_ids = np.repeat([f"entity_{i:05d}" for i in range(n_entities)], n_periods)
    period_ids = np.tile(period_strs, n_entities)
    n_rows = len(entity_ids)

    # Genuine feature: entity-level signal (not correlated with target by design)
    entity_effects = rng.normal(0, 1, n_entities)
    eff = np.repeat(entity_effects, n_periods)
    feature_0 = 0.7 * eff + rng.normal(0, 0.6, n_rows)

    # Target: sparse early (T0-T7), dense late (T8-T11)
    t_rank = np.tile(np.arange(n_periods), n_entities)
    base_prob = np.where(t_rank >= 8, 0.6, 0.02).astype(float)
    target = rng.binomial(1, base_prob)

    # Leaky feature: near-copy of target (strong post-prediction leakage)
    leaky_proxy = target.astype(float) + rng.normal(0, 0.05, n_rows)

    return pd.DataFrame({
        "entity_id": entity_ids,
        "prediction_time": period_ids,
        "feature_0": feature_0,
        "leaky_proxy": leaky_proxy,
        "target": target,
    })


def test_gate_trips_on_runtime_fold_starvation(skip_all_dataset):
    """Gate fires when folds are skipped at runtime via the positives viability check.

    n_splits=5 >= min_valid_folds=3, so the config alone would not trip the gate.
    At runtime, _skip_reason rejects all 5 folds (each test window has ~30-60
    positives < min_positive_cases_per_fold=100), leaving folds_evaluated=0 < 3.
    """
    df = skip_all_dataset
    contract = _make_contract([])
    cfg = _make_skip_all_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)

    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE", (
        f"expected UNCONFIRMED_HIGH_DAMAGE from runtime starvation, "
        f"got {report.policy_decision.verdict}; "
        f"folds_skipped={report.fold_ci.folds_skipped}, "
        f"folds_evaluated={report.fold_ci.folds_evaluated}"
    )
    assert report.fold_ci.folds_skipped > 0, (
        f"expected runtime-skipped folds; got folds_skipped={report.fold_ci.folds_skipped} "
        "(check that min_positive_cases_per_fold=100 actually skips the test windows)"
    )
    note = report.fold_ci.instability_note
    assert "evaluable folds" in note, (
        f"instability_note should cite the fold-count cause; got: {note!r}"
    )


def test_gate_fires_regardless_of_fl_low(skip_all_dataset):
    """Gate fires when fl < warn_floor — verdict would be PASS without the gate.

    With all temporal folds skipped, the engine falls back to fixable_leakage=0.0.
    0.0 < warn_floor=0.10 → policy ladder alone would give PASS (not detected,
    low fl). The gate fires first and returns UNCONFIRMED_HIGH_DAMAGE instead.
    """
    df = skip_all_dataset
    contract = _make_contract([])
    cfg = _make_skip_all_config()

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)
    fl = report.measured_damage.fixable_leakage

    assert fl < 0.10, (
        f"expected fl < warn_floor (0.10) to exercise the low-fl gate path; got fl={fl:.4f}"
    )
    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE", (
        f"gate should override the PASS policy ladder verdict even at low fl={fl:.4f}; "
        f"got {report.policy_decision.verdict}"
    )


def test_gate_fires_regardless_of_fl_high():
    """Gate fires even when 2 evaluated folds measured real leakage (fl > 0).

    The graduated dataset produces 3 runtime-skipped folds (early periods, sparse
    positives) and 2 evaluated folds (late periods, dense positives + leaky_proxy).
    folds_evaluated=2 < min_valid_folds=3 → gate fires despite fl > 0.

    Without the gate, with n_permutations=0 and fl > warn_floor, the policy
    ladder would give UNCONFIRMED_HIGH_DAMAGE for a different reason (not detected
    + high fl). The gate fires for the fold-count reason, making the verdict
    evidence-based rather than magnitude-based.
    """
    df = _make_graduated_leaky_dataset(seed=42)
    contract = PredictionContract(
        prediction_problem="partial-skip-leaky-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_proxy"],
    )
    cfg = ZekanConfig(contract=contract)  # default SplitPolicy: n_splits=5, min_valid_folds=3

    report = run_audit(df, contract, cfg, model_factory=_factory, n_permutations=0)

    assert report.fold_ci.folds_skipped >= 1, (
        f"expected >= 1 runtime-skipped fold from early-period sparse positives; "
        f"got folds_skipped={report.fold_ci.folds_skipped}"
    )
    assert report.fold_ci.folds_evaluated < 3, (
        f"expected folds_evaluated < min_valid_folds=3 to trip gate; "
        f"got folds_evaluated={report.fold_ci.folds_evaluated}"
    )
    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE", (
        f"gate should fire; got {report.policy_decision.verdict}"
    )
    fl = report.measured_damage.fixable_leakage
    assert not math.isnan(fl) and fl > 0.0, (
        f"expected fl > 0 from the 2 evaluated late-period folds with leaky_proxy; "
        f"got fl={fl}"
    )


def test_fold_starved_reason_distinct_from_null_unconfirmed(skip_all_dataset, normal_dataset):
    """Fold-starved and null-unconfirmed both produce UNCONFIRMED_HIGH_DAMAGE
    but are distinguishable by FoldCI.instability_note.

    Fold-starved (Case A):   instability_note names the fold-count cause.
    Null-unconfirmed (Case B): instability_note is empty — verdict came from
                               policy ladder (fl >= warn_floor, not detected).
    """
    # Case A: all temporal folds skip at runtime → gate fires
    report_a = run_audit(
        skip_all_dataset, _make_contract([]), _make_skip_all_config(),
        model_factory=_factory, n_permutations=0,
    )

    # Case B: normal data + leaky feature, n_permutations=0 → detected=False,
    # fl >= warn_floor → policy gives UNCONFIRMED_HIGH_DAMAGE (gate does not fire)
    report_b = run_audit(
        normal_dataset, _make_contract(["leaky_label_proxy"]), _make_config(),
        model_factory=_factory, n_permutations=0,
    )

    note_a = report_a.fold_ci.instability_note
    note_b = report_b.fold_ci.instability_note

    assert report_a.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"
    assert report_b.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"

    assert "evaluable folds" in note_a, (
        f"fold-starved note should cite fold count; got: {note_a!r}"
    )
    assert "evaluable folds" not in note_b, (
        f"null-unconfirmed should not mention fold starvation; got: {note_b!r}"
    )
    assert note_a != note_b, (
        "fold-starved and null-unconfirmed instability_notes must be distinguishable"
    )
