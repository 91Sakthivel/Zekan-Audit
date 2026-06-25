"""Tests for the prediction contract schema and validation (Phase 1)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gotcha.contract.contract_checks import CheckStatus, validate_contract
from gotcha.contract.prediction_contract import PredictionContract


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
        gotcha_version="0.1.0",
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
