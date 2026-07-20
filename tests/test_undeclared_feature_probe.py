"""Tests for probe_undeclared_feature_screen (Upgrade 1 step 1e).

Covers:
  - the temporal-vs-random wiring invariant (pre-registered in
    UPGRADE1_PREREGISTRATION.md -- catches a rand_folds copy-paste)
  - NEAR_CERTAIN_UNDECLARED_LEAK: fires when every fold clears the floor,
    does not fire when one fold dips below, ties are all flagged
  - the screenability gate (not-screenable reporting, never silent)
  - the ranked informational panel's "screened X of Y" accounting
  - name_pattern_score corroboration (reused from ablation.py verbatim)

All fixtures here are self-contained synthetic panels -- no dependency on
the external Test B CSVs (those live outside the repo; see
UPGRADE1_CALIBRATION.md's Provenance section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import IssueType
from zekan.detectors.undeclared_feature_probe import (
    NEAR_CERTAIN_AUC_FLOOR,
    PANEL_TOP_N,
    WIDE_DATA_CAP,
    UndeclaredFeaturePanel,
    probe_undeclared_feature_screen,
)
from zekan.severity.estimators import DEFAULT_ESTIMATOR_NAME, _build_factory
from zekan.severity.metrics import evaluate_folds
from zekan.severity.splitters import random_grouped_folds, temporal_expanding_folds

_MODEL = _build_factory(DEFAULT_ESTIMATOR_NAME)


def _model_factory():
    return _build_factory(DEFAULT_ESTIMATOR_NAME)()


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="undeclared-screen-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=[],
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _panel_df(n_entities: int = 200, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    """Balanced entity x period panel with a random binary target -- large
    enough (1,600 rows, ~200-400 rows/fold) to clear the screenability
    gate's default floors (_MIN_NONMISSING_COUNT=100,
    _MIN_MINORITY_CLASS_COUNT=20) comfortably."""
    rows = []
    for e in range(n_entities):
        for t in range(n_periods):
            rows.append((f"E{e}", pd.Timestamp("2020-01-01") + pd.DateOffset(months=t)))
    df = pd.DataFrame(rows, columns=["entity_id", "prediction_time"])
    rng = np.random.default_rng(seed)
    df["target"] = rng.integers(0, 2, len(df))
    return df


def _temporal_folds(df: pd.DataFrame) -> list:
    return temporal_expanding_folds(
        df, time_col="prediction_time", entity_col="entity_id", target_col="target",
        n_splits=5, min_test_rows=10, min_pos=2, min_neg=2,
    )


def _random_folds(df: pd.DataFrame) -> list:
    return random_grouped_folds(
        df, entity_col="entity_id", target_col="target",
        n_splits=5, min_test_rows=10, min_pos=2, min_neg=2,
    )


def _run(df, contract=None, folds=None, **kw):
    side = {}
    contract = contract or _contract()
    folds = folds if folds is not None else _temporal_folds(df)
    records = probe_undeclared_feature_screen(
        df, contract, folds=folds, model_factory=_model_factory, n_jobs=1,
        side_channel=side, **kw,
    )
    panel = side.get("undeclared_feature_panel")
    return records, panel


# ── Temporal-vs-random wiring invariant ────────────────────────────────────────

def test_temporal_vs_random_invariant_screen_reports_temporal_score():
    """A feature whose relationship to target REVERSES over time
    (inject_concept_drift's own documented mechanism -- see
    zekan/benchmark/injectors.py) scores materially differently under
    temporal vs random evaluation. When the screen is called with TEMPORAL
    folds (the only kind _run_structural_probes ever supplies it, per the
    pre-registered invariant), its reported score must match the
    directly-computed TEMPORAL figure, not the random one -- this is the
    test that would catch a rand_folds copy-paste.
    """
    from zekan.benchmark.fixtures import make_clean_dataset
    from zekan.benchmark.injectors import inject_concept_drift

    base = make_clean_dataset(n_entities=1000, seed=0)
    df, _ = inject_concept_drift(
        base, strength=4.0, time_col="prediction_time", entity_col="entity_id", seed=8,
    )
    contract = _contract(entity_id="entity_id", prediction_time="prediction_time",
                          target="target")

    temp_folds = temporal_expanding_folds(
        df, time_col="prediction_time", entity_col="entity_id", target_col="target",
        n_splits=5, min_test_rows=10, min_pos=2, min_neg=2,
    )
    rand_folds = random_grouped_folds(
        df, entity_col="entity_id", target_col="target",
        n_splits=5, min_test_rows=10, min_pos=2, min_neg=2,
    )

    temporal_direct = evaluate_folds(df, ["concept_drift_feat"], "target", temp_folds, _MODEL)
    random_direct = evaluate_folds(df, ["concept_drift_feat"], "target", rand_folds, _MODEL)

    # Sanity: the fixture actually produces a material gap (mechanism check,
    # not just a hopeful assumption).
    assert abs(temporal_direct.mean_auc - random_direct.mean_auc) > 0.05

    # Called with TEMPORAL folds (the only kind the real pipeline ever
    # supplies) -- reported score must match the temporal figure.
    _, panel_temporal = _run(df, contract=contract, folds=temp_folds)
    entry = next(e for e in panel_temporal.entries if e.feature == "concept_drift_feat")
    assert entry.univariate_auc == pytest.approx(temporal_direct.mean_auc, abs=1e-9)
    assert abs(entry.univariate_auc - random_direct.mean_auc) > 0.05

    # Called (misuse) with RANDOM folds -- proves the score is genuinely
    # folds-driven, not hardcoded/insensitive to the folds argument.
    _, panel_random = _run(df, contract=contract, folds=rand_folds)
    entry_r = next(e for e in panel_random.entries if e.feature == "concept_drift_feat")
    assert entry_r.univariate_auc == pytest.approx(random_direct.mean_auc, abs=1e-9)


# ── NEAR_CERTAIN_UNDECLARED_LEAK ────────────────────────────────────────────────

def test_near_certain_fires_when_every_fold_clears_floor():
    df = _panel_df()
    df["near_perfect"] = df["target"]  # exact copy -> AUC=1.0 in every fold

    records, panel = _run(df)

    assert len(records) == 1
    rec = records[0]
    assert rec.issue_type == IssueType.NEAR_CERTAIN_UNDECLARED_LEAK
    assert rec.source_layer.value == "flagged_suspicious"
    assert rec.confirmed is False
    detail = rec.evidence.structural_detail
    assert detail.feature == "near_perfect"
    assert detail.univariate_auc == pytest.approx(1.0)
    assert detail.threshold_compared_against == NEAR_CERTAIN_AUC_FLOOR
    assert detail.name_pattern_score == 0.0  # "near_perfect" matches no suspicious pattern

    entry = next(e for e in panel.entries if e.feature == "near_perfect")
    assert entry.univariate_auc == pytest.approx(1.0)


def test_near_certain_does_not_fire_when_one_fold_dips_below_floor():
    """near_perfect is an exact target-copy EXCEPT within fold 0's own test
    window (2020-02 to 2020-03, see this module's own empirical fold-layout
    check), where it's replaced with independent random noise -- degrading
    only that fold's AUC. absolute + tie-robust means EVERY fold must clear
    the floor; one weak fold must suppress the flag entirely, not average out.
    """
    df = _panel_df()
    corrupted_mask = df["prediction_time"].between("2020-02-01", "2020-03-01")
    rng = np.random.default_rng(1)
    near_perfect = df["target"].to_numpy().copy()
    near_perfect[corrupted_mask.to_numpy()] = rng.integers(
        0, 2, int(corrupted_mask.sum())
    )
    df["near_perfect"] = near_perfect

    records, panel = _run(df)

    assert len(records) == 0, [r.evidence.structural_detail.feature for r in records]
    entry = next(e for e in panel.entries if e.feature == "near_perfect")
    # Mean AUC still high (4 of 5 folds perfect) but NOT flagged -- confirms
    # the criterion is per-fold-absolute, not mean-based.
    assert entry.univariate_auc > 0.7
    assert entry.univariate_auc < 1.0


def test_near_certain_tie_robust_both_features_flagged():
    """Two independent near-perfect features -- BOTH must be flagged, not
    just the higher-ranked one (UPGRADE1_PREREGISTRATION.md's explicit
    requirement: a dataset can contain more than one target-adjacent column)."""
    df = _panel_df()
    df["copy_a"] = df["target"]
    df["copy_b"] = df["target"]

    records, panel = _run(df)

    flagged_features = {r.evidence.structural_detail.feature for r in records}
    assert flagged_features == {"copy_a", "copy_b"}
    assert len(records) == 2


def test_near_certain_never_suppressed_by_low_auc_neighbors():
    """A near-perfect feature alongside ordinary honest noise features --
    the honest features must not appear in near_certain, only the real one."""
    df = _panel_df()
    df["near_perfect"] = df["target"]
    rng = np.random.default_rng(2)
    df["honest_noise"] = rng.normal(size=len(df))

    records, _ = _run(df)
    flagged = {r.evidence.structural_detail.feature for r in records}
    assert flagged == {"near_perfect"}


# ── Name-pattern corroboration (reused verbatim from ablation.py) ─────────────

def test_name_pattern_score_corroborates_on_near_certain_when_pattern_matches():
    df = _panel_df()
    df["final_diagnosis"] = df["target"]  # matches _SUSPICIOUS_PATTERNS' final_ prefix

    records, panel = _run(df)
    rec = next(r for r in records if r.evidence.structural_detail.feature == "final_diagnosis")
    assert rec.evidence.structural_detail.name_pattern_score == 1.0

    entry = next(e for e in panel.entries if e.feature == "final_diagnosis")
    assert entry.name_pattern_score == 1.0


def test_name_pattern_score_does_not_gate_near_certain():
    """Corroboration only: a near-perfect feature with an UNsuspicious name
    ('readmitted'-shaped -- verified in UPGRADE1_CALIBRATION.md's read-first
    notes to score 0.0 under this exact machinery) still fires NEAR_CERTAIN."""
    df = _panel_df()
    df["ordinary_column_name"] = df["target"]

    records, _ = _run(df)
    assert len(records) == 1
    assert records[0].evidence.structural_detail.name_pattern_score == 0.0


# ── Screenability gate ──────────────────────────────────────────────────────────

def test_mostly_missing_feature_is_reported_not_screenable():
    df = _panel_df()
    rng = np.random.default_rng(3)
    # 97% missing, matching the pre-registration's own Diabetes-130 'weight'
    # example -- only 3% of rows carry a real value.
    vals = rng.normal(size=len(df))
    missing_mask = rng.random(len(df)) < 0.97
    vals[missing_mask] = np.nan
    df["mostly_missing"] = vals

    records, panel = _run(df)

    reasons = {e.feature: e.reason for e in panel.not_screenable}
    assert "mostly_missing" in reasons
    assert "non-missing" in reasons["mostly_missing"]
    # Never silently skipped or scored on noise: not in the scored panel entries.
    assert not any(e.feature == "mostly_missing" for e in panel.entries)
    assert not any(
        r.evidence.structural_detail.feature == "mostly_missing" for r in records
    )


def test_screened_x_of_y_accounting_excludes_not_screenable():
    df = _panel_df()
    rng = np.random.default_rng(4)
    vals = rng.normal(size=len(df))
    vals[rng.random(len(df)) < 0.97] = np.nan
    df["mostly_missing"] = vals
    df["honest_feature"] = rng.normal(size=len(df))

    _, panel = _run(df)

    assert panel.total_features == 2  # mostly_missing + honest_feature
    assert panel.screened_count == 1  # only honest_feature clears the gate
    assert len(panel.not_screenable) == 1
    assert panel.not_screenable[0].feature == "mostly_missing"


def test_panel_entries_capped_at_top_n_and_sorted_descending():
    df = _panel_df(n_entities=300, n_periods=8)
    rng = np.random.default_rng(5)
    n_features = PANEL_TOP_N + 5
    for i in range(n_features):
        # Graded signal strength so ranking is unambiguous, no ties.
        df[f"f{i}"] = (
            df["target"] * (n_features - i) + rng.normal(size=len(df)) * 3.0
        )

    _, panel = _run(df)

    assert panel.total_features == n_features
    assert panel.screened_count == n_features
    assert len(panel.entries) == PANEL_TOP_N
    aucs = [e.univariate_auc for e in panel.entries]
    assert aucs == sorted(aucs, reverse=True)


def test_no_temporal_folds_returns_nothing_and_does_not_populate_side_channel():
    """TEMPORAL FOLDS ONLY -- when folds is empty/None, the screen must not
    run at all (mirrors every other needs_folds=True probe's existing
    behavior in _run_structural_probes)."""
    df = _panel_df()
    side: dict = {}
    records = probe_undeclared_feature_screen(
        df, _contract(), folds=None, model_factory=_model_factory, side_channel=side,
    )
    assert records == []
    assert "undeclared_feature_panel" not in side


# ── Registered correctly in the probe registry ──────────────────────────────────

def test_registered_in_probe_registry_with_correct_capability_flags():
    from zekan.severity.audit import _build_probe_registry

    specs = {spec.fn.__name__: spec for spec in _build_probe_registry()}
    spec = specs["probe_undeclared_feature_screen"]
    assert spec.needs_folds is True
    assert spec.needs_model is True
    assert spec.needs_matrix is True
    assert spec.needs_budget is True
    assert spec.needs_side_channel is True


def test_suspected_tier_detail_struct_remains_registered_but_unused():
    """UPGRADE1_CALIBRATION.md's step-1d finding: SUSPECTED_UNDECLARED_LEAK's
    BH-FDR design fails its own falsification condition and is DEFERRED, not
    implemented. SuspectedUndeclaredLeakDetail must stay registered in the
    schema (for a future calibrated design) but this probe must never
    construct one."""
    from zekan.detectors.schema import IssueType, _REGISTRY

    assert IssueType.SUSPECTED_UNDECLARED_LEAK in _REGISTRY  # still registered

    df = _panel_df()
    df["near_perfect"] = df["target"]
    df["honest_but_notable"] = df["target"] * 0.3 + np.random.default_rng(6).normal(size=len(df))
    records, _ = _run(df)
    assert all(
        r.issue_type == IssueType.NEAR_CERTAIN_UNDECLARED_LEAK for r in records
    ), "probe must never emit SUSPECTED_UNDECLARED_LEAK (deferred, not implemented)"
