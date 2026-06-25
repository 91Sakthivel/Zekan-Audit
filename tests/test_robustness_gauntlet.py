"""Robustness gauntlet: ~20 adversarially-ugly datasets through the full engine + four probes.

Every case asserts exactly one of three outcomes:
  (A) crash loud with a typed error (pytest.raises)
  (B) return status='unavailable' (graceful degradation)
  (C) return a correct specific verdict (pass / warn / fail / internal_fail)

Never acceptable: untyped crash, silent wrong verdict, false-fail on honest data.

Engine-level cases use run_severity_analysis directly.
Probe-level cases invoke the four probes independently on pathological data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.benchmark.injectors import inject_covariate_drift
from gotcha.config.schema import GotchaConfig, SplitPolicy
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.detectors.duplicate_probe import probe_cross_fold_duplicates, probe_raw_duplicates
from gotcha.detectors.entity_contamination_risk import probe_entity_contamination_risk
from gotcha.detectors.splitter_contract_probe import probe_splitter_contract_violation
from gotcha.severity.engine import run_severity_analysis
from gotcha.severity.splitters import FoldIndices, FoldMeta


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="gauntlet-test",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _make_config(contract: PredictionContract) -> GotchaConfig:
    """Reduced fold minimums so 1000-row datasets produce valid folds."""
    return GotchaConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
        ),
    )


def _fast_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


def _make_fold(
    fold_idx: int,
    train_idx: list[int],
    test_idx: list[int],
    skipped: bool = False,
) -> FoldIndices:
    meta = FoldMeta(
        fold_idx=fold_idx,
        strategy="grouped_cv",
        train_rows=len(train_idx),
        test_rows=len(test_idx),
        train_base_rate=0.5,
        test_base_rate=0.5,
        entity_overlap_count=0,
        entity_overlap_pct=0.0,
        skipped=skipped,
    )
    return FoldIndices(
        train_idx=np.array(train_idx, dtype=int),
        test_idx=np.array(test_idx, dtype=int),
        meta=meta,
    )


def _minimal_engine_df() -> pd.DataFrame:
    """100 entities × 10 periods = 1000 rows.

    row_count_and_folds: 1000 rows → PASS (≥1000), 10 periods → PASS.
    temporal_periods_count: 10 periods ≥ 6 → PASS.
    Both severity-blocker checks are PASS → can_compute_severity=True.
    """
    return make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — Engine → "unavailable" (graceful degradation on bad input)
# Expectation for every case: result.status == "unavailable"
# The engine must never crash; it must explain why via unavailable_reason.
# ═══════════════════════════════════════════════════════════════════════════════

class TestGauntletEngineUnavailable:

    def test_empty_dataframe(self):
        """Completely empty DataFrame — every contract column check fails."""
        df = pd.DataFrame()
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"
        assert result.unavailable_reason is not None

    def test_single_row(self):
        """1 row: row_count < 500 → row_count_and_folds FAIL → blocks severity."""
        df = pd.DataFrame({
            "entity_id": ["e0"],
            "prediction_time": ["2022-01-01"],
            "feature_0": [1.0],
            "target": [0],
        })
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_too_few_rows(self):
        """10 entities × 10 periods = 100 rows — below the 500-row hard gate."""
        df = make_clean_dataset(n_entities=10, snapshots_per_entity=5, seed=0)
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_missing_entity_id_column(self):
        """entity_id column absent — entity_id_exists check FAIL."""
        df = _minimal_engine_df().drop(columns=["entity_id"])
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_missing_prediction_time_column(self):
        """prediction_time column absent — prediction_time_parseable check FAIL."""
        df = _minimal_engine_df().drop(columns=["prediction_time"])
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_missing_declared_forbidden_column(self):
        """Contract declares a forbidden column that does not exist in the df."""
        df = _minimal_engine_df()
        contract = _contract(forbidden_after_prediction=["column_that_does_not_exist"])
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_all_rows_same_timestamp(self):
        """All rows share one prediction_time: 1 distinct period < 3 → FAIL."""
        df = _minimal_engine_df().copy()
        df["prediction_time"] = "2022-01-01"
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"

    def test_multiclass_target(self):
        """Target has 3 classes — target_binary_clean check FAIL."""
        rng = np.random.default_rng(7)
        df = _minimal_engine_df().copy()
        df["target"] = rng.integers(0, 3, len(df))
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status == "unavailable"


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — Engine → correct verdict on honest but ugly input (must NOT crash,
# must NOT false-fail, must NOT return "unavailable")
# ═══════════════════════════════════════════════════════════════════════════════

class TestGauntletEngineCorrectVerdict:

    def test_honest_minimal_viable_not_fail(self):
        """Minimal engine-viable clean dataset must not be flagged as leaking."""
        df = _minimal_engine_df()
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status in {"pass", "warn"}, (
            f"Clean data returned {result.status!r}; expected 'pass' or 'warn'"
        )
        assert result.status != "unavailable"

    def test_drift_no_forbidden_fixable_leakage_near_zero(self):
        """Covariate drift injected but NO forbidden features declared.

        B and C use the same feature set (all features) so fixable_leakage = B - C ≈ 0.
        Covariate drift may widen total_optimism (A-C) but must not inflate fixable_leakage.
        Engine must run (not return 'unavailable') and fixable_leakage must stay near 0.
        """
        df = _minimal_engine_df()
        df_drift, record = inject_covariate_drift(df, seed=42)
        assert not record.is_leak, "covariate drift is not a leak — sanity check"
        contract = _contract()
        result = run_severity_analysis(df_drift, contract, _make_config(contract), _fast_clf)
        assert result.status != "unavailable"
        assert abs(result.fixable_leakage) <= 0.05, (
            f"No forbidden features declared → B=C → fixable_leakage should be ~0, "
            f"got {result.fixable_leakage:.4f}"
        )

    def test_nonmonotonic_row_order_no_crash(self):
        """Rows shuffled into non-chronological order.

        temporal_expanding_folds sorts periods internally; the engine must handle
        arbitrary row ordering without crashing and return a valid verdict.
        """
        df = _minimal_engine_df().sample(frac=1, random_state=99).reset_index(drop=True)
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status in {"pass", "warn", "fail"}, (
            f"Shuffled-row dataset returned unexpected status {result.status!r}"
        )

    def test_constant_feature_column_no_crash(self):
        """One completely constant feature column (zero variance).

        RandomForest ignores constant features; the engine must not crash.
        Clean data + no forbidden features → must not return 'fail'.
        """
        df = _minimal_engine_df().copy()
        df["constant_feat"] = 0.0
        contract = _contract()
        result = run_severity_analysis(df, contract, _make_config(contract), _fast_clf)
        assert result.status in {"pass", "warn"}, (
            f"Clean data with constant feature returned {result.status!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — Probe-level: ugly data must not cause crashes or silent wrong verdicts
# ═══════════════════════════════════════════════════════════════════════════════

class TestGauntletProbes:

    # ── probe_raw_duplicates ────────────────────────────────────────────────

    def test_all_identical_rows_warns_with_correct_count(self):
        """N rows all sharing identical feature+target content → N-1 excess copies.

        entity_id and prediction_time differ per row but are excluded from the hash,
        so the content hash is the same for all rows.
        """
        n = 50
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],
            "prediction_time": ["2022-01-01"] * n,
            "feature_0": [1.0] * n,
            "feature_1": [2.0] * n,
            "target": [0] * n,
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"
        assert result.evidence.structural_detail.duplicate_rows == n - 1

    def test_nan_feature_column_no_crash_and_unique_rows(self):
        """One feature column entirely NaN; remaining features uniquely identify rows.

        NaN sentinel replacement must not equate rows that differ on other features.
        Probe must not crash; result must be 'pass' (no content duplicates).
        """
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=1)
        # Explicitly make feature_1 unique (ascending integers) so there's no
        # accidental collision even if other NaN-sentineled features would match.
        df = df.copy()
        df["feature_0"] = float("nan")
        df["feature_1"] = np.arange(len(df), dtype=float)  # guaranteed unique
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "pass", (
            f"Unique rows with one NaN feature incorrectly flagged: "
            f"duplicate_rows={result.evidence.structural_detail.duplicate_rows}"
        )

    def test_entity_id_excluded_from_content_hash(self):
        """Rows differing ONLY in entity_id are content-duplicates.

        Design invariant: entity_id is excluded from the content hash so that the
        same observation recorded under different IDs is detected as a duplicate.
        """
        n = 10
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(n)],   # all distinct
            "prediction_time": ["2022-01-01"] * n,
            "feature_0": [1.0] * n,                      # all identical
            "feature_1": [2.0] * n,
            "target": [0] * n,
        })
        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn", (
            "Rows differing only in entity_id must be flagged as content-duplicates"
        )
        assert result.evidence.structural_detail.duplicate_rows == n - 1

    def test_duplicate_heavy_planted_detail_counts(self):
        """Planted 3 identical groups of 4 rows each: 3 × (4-1) = 9 excess copies."""
        n_unique = 20
        group_size = 4
        n_groups = 3
        rng = np.random.default_rng(11)
        total = n_unique + n_groups * group_size
        df = pd.DataFrame({
            "entity_id": [f"e{i}" for i in range(total)],
            "prediction_time": ["2022-01-01"] * total,
            "feature_0": rng.normal(0, 1, total),
            "target": rng.integers(0, 2, total).astype(int),
        })
        for g in range(n_groups):
            anchor = n_unique + g * group_size
            for j in range(1, group_size):
                df.loc[anchor + j, "feature_0"] = df.loc[anchor, "feature_0"]
                df.loc[anchor + j, "target"] = df.loc[anchor, "target"]

        result = probe_raw_duplicates(df, _contract())
        assert result.status == "warn"
        assert result.evidence.structural_detail.duplicate_rows == n_groups * (group_size - 1)

    # ── probe_entity_contamination_risk ──────────────────────────────────────

    def test_probe_ecr_missing_entity_id_col_unavailable_not_pass(self):
        """entity_id column absent → 'unavailable', never a silent 'pass'."""
        df = pd.DataFrame({
            "not_entity": ["a", "b", "c"],
            "prediction_time": ["2022-01", "2022-02", "2022-03"],
            "feature_0": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "unavailable"
        assert result.status != "pass", (
            "Missing entity_id must not silently pass — that would be a silent wrong verdict"
        )

    def test_probe_ecr_single_period_is_cross_sectional_pass(self):
        """All rows at the same prediction_time → cross-sectional → must not fire."""
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=3)
        df = df.copy()
        df["prediction_time"] = "2022-01-01"
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "pass"
        assert result.evidence.structural_detail.recurring_entities == 0

    def test_probe_ecr_longitudinal_warns_on_real_dataset(self):
        """make_clean_dataset produces a full panel: all entities recur across periods.

        Must fire 'warn' — this is the primary false-negative guard.
        """
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=4)
        result = probe_entity_contamination_risk(df, _contract())
        assert result.status == "warn", (
            "Longitudinal dataset must fire WARN; got 'pass' — false negative"
        )
        assert result.evidence.structural_detail.recurring_entities > 0

    # ── probe_splitter_contract_violation ─────────────────────────────────────

    def test_probe_scv_empty_folds_unavailable_not_pass(self):
        """No folds provided → 'unavailable', never a silent 'pass'."""
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=5)
        result = probe_splitter_contract_violation([], df, _contract())
        assert result.status == "unavailable"
        assert result.status != "pass"

    def test_probe_scv_all_entities_spanning_internal_fail(self):
        """Fabricated folds where every entity appears in both train and test.

        This represents the worst case of a grouped-splitter bug: random assignment
        that splits all entities' rows across the train/test boundary.
        """
        df = pd.DataFrame({
            "entity_id": ["e0", "e0", "e1", "e1", "e2", "e2"],
            "prediction_time": ["2022-01", "2022-02"] * 3,
            "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        # Each entity's rows are split across train (even indices) and test (odd indices)
        fold = _make_fold(0, train_idx=[0, 2, 4], test_idx=[1, 3, 5])
        result = probe_splitter_contract_violation([fold], df, _contract())
        assert result.status == "internal_fail"
        assert result.evidence.structural_detail.spanning_entities_per_fold == {0: 3}
        assert result.evidence.structural_detail.affected_folds == [0]

    # ── cross-probe ──────────────────────────────────────────────────────────

    def test_longitudinal_with_planted_dupes_both_probes_fire(self):
        """A dataset that is both longitudinal and has planted duplicates.

        Both probe_raw_duplicates and probe_entity_contamination_risk must fire
        their correct verdicts independently on the same dataset.  Neither must
        crash or interfere with the other.
        """
        df = make_clean_dataset(n_entities=40, snapshots_per_entity=5, seed=6)
        # Append 5 rows whose content duplicates the first 5 rows.
        # entity_id and prediction_time are excluded from the hash, so only
        # feature+target content matters — these 5 rows create 5 excess copies.
        dup_rows = df.iloc[:5].copy()
        df_with_dupes = pd.concat([df, dup_rows], ignore_index=True)

        raw_result = probe_raw_duplicates(df_with_dupes, _contract())
        ecr_result = probe_entity_contamination_risk(df_with_dupes, _contract())

        assert raw_result.status == "warn", (
            f"Planted duplicates not detected: status={raw_result.status!r}"
        )
        assert raw_result.evidence.structural_detail.duplicate_rows == 5

        assert ecr_result.status == "warn", (
            f"Longitudinal dataset not flagged by ECR: status={ecr_result.status!r}"
        )
        assert ecr_result.evidence.structural_detail.recurring_entities > 0
