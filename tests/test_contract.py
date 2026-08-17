"""Tests for the prediction contract schema and validation (Phase 1)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from zekan.contract.contract_checks import (
    CheckStatus,
    apply_categorical_mapping,
    build_categorical_mapping,
    validate_contract,
)
from zekan.contract.prediction_contract import PredictionContract


# ── fixture helpers ───────────────────────────────────────────────────────────

def _make_df(
    n_periods: int = 10,
    n_per_period: int = 150,
    n_target_classes: int = 2,
) -> pd.DataFrame:
    """Synthetic churn dataset: n_periods monthly snapshots, n_per_period customers each."""
    rng = np.random.default_rng(42)
    base = date(2023, 1, 1)
    rows = []
    for p in range(n_periods):
        snapshot = (base + timedelta(days=30 * p)).isoformat()
        for i in range(n_per_period):
            rows.append({
                "customer_id": f"c{i:04d}",
                "snapshot_date": snapshot,
                "tenure_months": int(rng.integers(1, 60)),
                "spend_last_30d": round(float(rng.uniform(0, 500)), 2),
                "leaky_col": int(rng.integers(0, 2)),
                "churned": int(rng.random() < 0.2),
            })
    df = pd.DataFrame(rows)

    if n_target_classes != 2:
        # Introduce a third class by overwriting ~10 % of rows with class 2.
        n = len(df)
        replace_idx = rng.choice(n, size=max(1, n // 10), replace=False)
        df = df.copy()
        df.loc[df.index[replace_idx], "churned"] = 2

    return df


def _make_contract(**overrides: object) -> PredictionContract:
    defaults: dict[str, object] = dict(
        prediction_problem="churn",
        entity_id="customer_id",
        prediction_time="snapshot_date",
        target="churned",
        available_features_until="snapshot_date",
        forbidden_after_prediction=[],
        schema_version="1",
        zekan_version="0.1.0",
    )
    defaults.update(overrides)
    return PredictionContract(**defaults)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_valid_contract_passes():
    """10 periods × 150 rows = 1 500 rows — all checks should pass."""
    df = _make_df()
    result = validate_contract(_make_contract(), df)

    failing = [c for c in result.checks if c.status != CheckStatus.PASS]
    assert result.passed, f"Unexpected non-PASS checks: {failing}"
    assert result.can_compute_severity


def test_missing_forbidden_column_fails_check_4():
    """Declaring a forbidden column that is absent in the dataframe must FAIL check 4."""
    df = _make_df()
    contract = _make_contract(forbidden_after_prediction=["nonexistent_column"])
    result = validate_contract(contract, df)

    assert not result.passed
    check = next(c for c in result.checks if c.name == "forbidden_columns_exist")
    assert check.status == CheckStatus.FAIL
    assert "nonexistent_column" in check.message


def test_non_binary_target_fails_check_3():
    """A target with three distinct values must FAIL check 3."""
    df = _make_df(n_target_classes=3)
    result = validate_contract(_make_contract(), df)

    assert not result.passed
    check = next(c for c in result.checks if c.name == "target_binary_clean")
    assert check.status == CheckStatus.FAIL


def test_too_few_time_periods_blocks_severity():
    """4 periods (< 6) warns on temporal_periods_count and sets can_compute_severity=False."""
    df = _make_df(n_periods=4, n_per_period=150)  # 600 rows, 4 periods
    result = validate_contract(_make_contract(), df)

    check = next(c for c in result.checks if c.name == "temporal_periods_count")
    assert check.status in (CheckStatus.WARN, CheckStatus.FAIL)
    assert not result.can_compute_severity


def test_class_balance_message_has_no_numpy_types():
    """Count dict in the balance check message must use plain ints, not np.int64."""
    df = _make_df()
    result = validate_contract(_make_contract(), df)
    check = next(c for c in result.checks if c.name == "target_class_balance")
    assert check.status == CheckStatus.PASS
    assert "np.int64" not in check.message
    assert "int64" not in check.message


def test_partial_unparseable_prediction_time_fails():
    """Some (not all) rows unparseable as datetime must FAIL, not WARN --
    temporal_expanding_folds crashes on ANY parse failure, not just 100%."""
    df = _make_df()
    df = df.copy()
    df.loc[df.index[:5], "snapshot_date"] = "not-a-date"
    result = validate_contract(_make_contract(), df)

    assert not result.passed
    assert not result.can_compute_severity
    check = next(c for c in result.checks if c.name == "prediction_time_parseable")
    assert check.status == CheckStatus.FAIL
    assert "snapshot_date" in check.message
    assert "5" in check.message


def test_all_unparseable_prediction_time_fails_with_teaching_message():
    """A prediction_time column that never parses (e.g. raw large-integer visit
    IDs, like Diabetes-130's encounter_id) must FAIL with a message that names
    the column and says what to do about it -- not just what's wrong."""
    df = _make_df()
    df = df.copy()
    df["snapshot_date"] = [str(197661240 + i) for i in range(len(df))]
    result = validate_contract(_make_contract(), df)

    assert not result.passed
    assert not result.can_compute_severity
    check = next(c for c in result.checks if c.name == "prediction_time_parseable")
    assert check.status == CheckStatus.FAIL
    assert "snapshot_date" in check.message
    assert "time signal" in check.message
    assert "derive" in check.message


def test_non_numeric_feature_column_fails():
    """A feature column containing values that can't be cast to float must
    FAIL -- evaluate_folds does df[feature_cols].to_numpy(dtype=float) on the
    whole feature block for every fold, and a single non-numeric value there
    breaks every fold, not just the row it's in."""
    df = _make_df()
    df = df.copy()
    df["race"] = "Caucasian"
    df.loc[df.index[:3], "race"] = "AfricanAmerican"
    result = validate_contract(_make_contract(), df)

    assert not result.passed
    assert not result.can_compute_severity
    check = next(c for c in result.checks if c.name == "feature_columns_numeric")
    assert check.status == CheckStatus.FAIL
    assert "race" in check.message
    assert "numeric" in check.message.lower()


def test_all_numeric_features_pass():
    """A dataframe whose feature columns are all already numeric must PASS
    feature_columns_numeric explicitly, not just implicitly via the overall
    valid-contract test."""
    df = _make_df()
    result = validate_contract(_make_contract(), df)

    check = next(c for c in result.checks if c.name == "feature_columns_numeric")
    assert check.status == CheckStatus.PASS


def test_available_features_until_after_prediction_time_fails():
    """available_features_until > prediction_time on any row must FAIL the logical check."""
    df = _make_df()
    df = df.copy()
    # Set available_until to 30 days AFTER snapshot_date — every row violates the constraint.
    df["available_until"] = (
        pd.to_datetime(df["snapshot_date"]) + pd.Timedelta(days=30)
    ).dt.strftime("%Y-%m-%d")

    contract = _make_contract(available_features_until="available_until")
    result = validate_contract(contract, df)

    check = next(c for c in result.checks if c.name == "available_features_until_logical")
    assert check.status == CheckStatus.FAIL, (
        f"Expected FAIL when available_until > snapshot_date, got {check.status}: {check.message}"
    )
    assert "available_features_until > prediction_time" in check.message


# ── categorical encoding (CATEGORICAL_SUPPORT_PREREGISTRATION.md) ──────────────

def test_categorical_mapping_nan_sentinel_collision_gets_distinct_codes():
    """A declared categorical column containing BOTH real NaN and the literal
    string "__NaN__" must not collide onto the same code: _pick_nan_sentinel
    must extend the base sentinel until it no longer matches a real value in
    the column, or the two distinct raw values (actual missingness, and a
    real category that happens to spell the sentinel) would collapse onto
    one code -- breaking the bijection section 4 relies on for Theil's U
    invariance."""
    df = pd.DataFrame({"nominal": ["a", "__NaN__", None, "b", "__NaN__"]})
    mapping = build_categorical_mapping(df, ["nominal"])

    col_map = mapping["nominal"]
    codes, sentinel = col_map["codes"], col_map["nan_sentinel"]

    assert sentinel != "__NaN__", "base sentinel collided with a real value and must have been extended"
    assert "__NaN__" in codes, "the literal string is still a real category and must keep its own code"
    assert sentinel in codes, "the (extended) sentinel must stand for actual NaN"
    assert codes["__NaN__"] != codes[sentinel], "real value and actual NaN must never share a code"


def test_apply_categorical_mapping_records_unseen_values():
    """A value absent from a column's codes (e.g. a category seen only in a
    later audit of different data under the same contract) must be counted
    in unseen_counts, not silently folded onto an existing code -- "silence
    is not clearance"."""
    build_df = pd.DataFrame({"nominal": ["a", "b", "a"]})
    mapping = build_categorical_mapping(build_df, ["nominal"])

    apply_df = pd.DataFrame({"nominal": ["a", "b", "c", "c"]})  # "c" never seen at build time
    unseen_counts: dict[str, int] = {}
    result = apply_categorical_mapping(apply_df, mapping, unseen_counts=unseen_counts)

    assert unseen_counts == {"nominal": 2}
    codes = mapping["nominal"]["codes"]
    assert result["nominal"].iloc[0] == codes["a"]
    assert result["nominal"].iloc[1] == codes["b"]
    assert pd.isna(result["nominal"].iloc[2]), "unseen value must map to NaN, not collide onto an existing code"
    assert pd.isna(result["nominal"].iloc[3])


def test_apply_categorical_mapping_no_unseen_values_leaves_dict_untouched():
    """A column with zero unseen values must not appear in unseen_counts at
    all -- distinguishes "checked, found nothing" from "never checked"."""
    df = pd.DataFrame({"nominal": ["a", "b", "a"]})
    mapping = build_categorical_mapping(df, ["nominal"])

    unseen_counts: dict[str, int] = {}
    apply_categorical_mapping(df, mapping, unseen_counts=unseen_counts)

    assert unseen_counts == {}


def test_build_categorical_mapping_is_deterministic():
    """The same data run through build_categorical_mapping twice must
    produce byte-identical mappings -- no target, no randomness, sorted-
    unique only."""
    df = pd.DataFrame({"nominal": ["z", "a", None, "m", "a", "__NaN__"]})
    m1 = build_categorical_mapping(df, ["nominal"])
    m2 = build_categorical_mapping(df, ["nominal"])

    assert m1 == m2
