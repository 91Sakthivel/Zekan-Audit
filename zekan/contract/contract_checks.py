"""Contract validation: structural and semantic checks against a real DataFrame.

All checks return CheckResult objects - nothing raises. Callers inspect the
ValidationResult to decide whether to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from zekan.contract.prediction_contract import PredictionContract


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str


# Checks whose non-PASS status blocks severity computation even when the
# contract itself is structurally valid.
_SEVERITY_BLOCKERS = frozenset({"temporal_periods_count", "row_count_and_folds"})


@dataclass
class ValidationResult:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no check carries FAIL status."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def can_compute_severity(self) -> bool:
        """True when passed AND every severity-critical check is PASS (not just non-FAIL)."""
        if not self.passed:
            return False
        return all(
            c.status == CheckStatus.PASS
            for c in self.checks
            if c.name in _SEVERITY_BLOCKERS
        )


# ── individual check functions ────────────────────────────────────────────────

def _check_prediction_time(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    """A prediction_time column must parse as datetime for EVERY row.

    temporal_expanding_folds (splitters.py) calls pd.to_datetime on the whole
    column with no error suppression -- a single unparseable row raises there.
    So ANY parse failure here must FAIL the gate, not just a 100% failure: a
    partial failure is exactly as fatal to the temporal splitter as a total
    one, so this check has no WARN outcome -- only PASS or FAIL.
    """
    col = c.prediction_time
    if col not in df.columns:
        return CheckResult("prediction_time_parseable", CheckStatus.FAIL,
                           f"Column '{col}' not found in dataframe")
    parsed = pd.to_datetime(df[col], errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad == len(df):
        return CheckResult("prediction_time_parseable", CheckStatus.FAIL,
                           f"Column '{col}' cannot be read as a time signal (0 of {len(df)} "
                           f"row(s) parsed as a date/time). Zekan audits leakage over time and "
                           f"needs an ordered, parseable time column -- provide a real date/time "
                           f"column, or derive one (e.g. an ordinal period column) from your data.")
    if n_bad > 0:
        return CheckResult("prediction_time_parseable", CheckStatus.FAIL,
                           f"Column '{col}' cannot be used as a time signal: {n_bad} of {len(df)} "
                           f"row(s) failed to parse as a date/time, and temporal folding requires "
                           f"every row to parse. Provide a column where every value is a valid "
                           f"date/time, or derive one (e.g. an ordinal period column) from your data.")
    return CheckResult("prediction_time_parseable", CheckStatus.PASS,
                       f"Column '{col}' exists and parses as datetime")


def _check_entity_id(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    col = c.entity_id
    if col not in df.columns:
        return CheckResult("entity_id_exists", CheckStatus.FAIL,
                           f"Column '{col}' not found in dataframe")
    miss_pct = df[col].isna().mean() * 100
    if miss_pct > 0:
        return CheckResult("entity_id_exists", CheckStatus.WARN,
                           f"Column '{col}' found but {miss_pct:.1f}% of values are missing")
    return CheckResult("entity_id_exists", CheckStatus.PASS,
                       f"Column '{col}' found, 0.0% missing")


def _check_target_binary(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    col = c.target
    if col not in df.columns:
        return CheckResult("target_binary_clean", CheckStatus.FAIL,
                           f"Target column '{col}' not found in dataframe")
    classes = df[col].dropna().unique()
    n_classes = len(classes)
    if n_classes != 2:
        sample = sorted(str(v) for v in classes[:8])
        return CheckResult("target_binary_clean", CheckStatus.FAIL,
                           f"Target '{col}' has {n_classes} unique value(s) {sample!r}; expected exactly 2")
    reserved = {c.entity_id, c.prediction_time, c.available_features_until}
    if col in reserved:
        return CheckResult("target_binary_clean", CheckStatus.FAIL,
                           f"Target '{col}' is also declared as an ID or time column")
    return CheckResult("target_binary_clean", CheckStatus.PASS,
                       f"Target '{col}' is binary with classes {sorted(str(v) for v in classes)!r}")


def _check_forbidden_columns(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    if not c.forbidden_after_prediction:
        return CheckResult("forbidden_columns_exist", CheckStatus.PASS,
                           "No forbidden columns declared")
    missing = [col for col in c.forbidden_after_prediction if col not in df.columns]
    if missing:
        return CheckResult("forbidden_columns_exist", CheckStatus.FAIL,
                           f"Forbidden columns not found in dataframe: {missing}")
    return CheckResult("forbidden_columns_exist", CheckStatus.PASS,
                       f"All {len(c.forbidden_after_prediction)} forbidden column(s) present in dataframe")


def _check_available_features_logical(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    afu_col, pt_col = c.available_features_until, c.prediction_time
    if afu_col == pt_col:
        return CheckResult("available_features_until_logical", CheckStatus.PASS,
                           "available_features_until is the same column as prediction_time")
    if afu_col not in df.columns or pt_col not in df.columns:
        return CheckResult("available_features_until_logical", CheckStatus.WARN,
                           "Cannot verify temporal ordering: one or both columns are missing")
    afu = pd.to_datetime(df[afu_col], errors="coerce")
    pt = pd.to_datetime(df[pt_col], errors="coerce")
    violations = int((afu > pt).sum())
    if violations > 0:
        return CheckResult("available_features_until_logical", CheckStatus.FAIL,
                           f"{violations} row(s) have available_features_until > prediction_time")
    return CheckResult("available_features_until_logical", CheckStatus.PASS,
                       "available_features_until <= prediction_time for all rows")


def _check_temporal_periods(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    col = c.prediction_time
    if col not in df.columns:
        return CheckResult("temporal_periods_count", CheckStatus.FAIL,
                           "prediction_time column missing - cannot assess temporal periods")
    n = int(df[col].nunique())
    if n < 3:
        return CheckResult("temporal_periods_count", CheckStatus.FAIL,
                           f"Only {n} distinct time period(s); minimum 3 required for any temporal folding")
    if n < 6:
        return CheckResult("temporal_periods_count", CheckStatus.WARN,
                           f"{n} distinct time periods; recommend >= 6 for reliable severity estimation")
    return CheckResult("temporal_periods_count", CheckStatus.PASS,
                       f"{n} distinct time periods - sufficient for temporal splitting")


def _check_target_class_balance(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    col = c.target
    if col not in df.columns:
        return CheckResult("target_class_balance", CheckStatus.FAIL,
                           f"Target column '{col}' missing - cannot check class balance")
    counts = df[col].value_counts()
    if len(counts) < 2:
        return CheckResult("target_class_balance", CheckStatus.FAIL,
                           f"Target '{col}' has only one class present in the data")
    minority_pct = float(counts.min()) / len(df) * 100
    if minority_pct < 1.0:
        return CheckResult("target_class_balance", CheckStatus.WARN,
                           f"Minority class is {minority_pct:.2f}% of data; severity estimates may be unreliable")
    counts_int = {int(k): int(v) for k, v in counts.items()}
    return CheckResult("target_class_balance", CheckStatus.PASS,
                       f"Both classes present; minority is {minority_pct:.1f}% ({counts_int})")


def _check_cost_model(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    if c.cost_model is None:
        return CheckResult("cost_model_ranges_valid", CheckStatus.PASS,
                           "No cost model declared (optional)")
    cm = c.cost_model
    issues: list[str] = []
    for attr in ("discount_cost", "customer_value", "conversion_lift"):
        low, high = getattr(cm, attr)
        if low < 0 or high < 0:
            issues.append(f"{attr}: values must be non-negative (got [{low}, {high}])")
        elif low > high:
            issues.append(f"{attr}: low ({low}) > high ({high})")
    if issues:
        return CheckResult("cost_model_ranges_valid", CheckStatus.FAIL,
                           "Cost model invalid - " + "; ".join(issues))
    return CheckResult("cost_model_ranges_valid", CheckStatus.PASS,
                       "Cost model ranges are valid (all non-negative, low <= high)")


def _check_row_count_and_folds(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    n_rows = len(df)
    n_periods = int(df[c.prediction_time].nunique()) if c.prediction_time in df.columns else 0
    issues: list[str] = []
    hard_fail = False

    if n_rows < 500:
        issues.append(f"{n_rows} rows (hard minimum is 500; got {n_rows})")
        hard_fail = True
    elif n_rows < 1000:
        issues.append(f"{n_rows} rows (recommend >= 1000 for reliable severity estimates)")

    if n_periods < 3:
        issues.append(f"{n_periods} distinct period(s) (minimum 3 required for temporal folding)")
        hard_fail = True

    if not issues:
        return CheckResult("row_count_and_folds", CheckStatus.PASS,
                           f"{n_rows} rows across {n_periods} periods - sufficient for severity computation")
    status = CheckStatus.FAIL if hard_fail else CheckStatus.WARN
    return CheckResult("row_count_and_folds", status, "; ".join(issues))


# ── public API ────────────────────────────────────────────────────────────────

def validate_contract(contract: PredictionContract, df: pd.DataFrame) -> ValidationResult:
    """Run all contract checks against a loaded DataFrame.

    Returns a ValidationResult with per-check outcomes and aggregate booleans.
    Never raises - all outcomes are captured in the result.
    """
    checks = [
        _check_prediction_time(contract, df),
        _check_entity_id(contract, df),
        _check_target_binary(contract, df),
        _check_forbidden_columns(contract, df),
        _check_available_features_logical(contract, df),
        _check_temporal_periods(contract, df),
        _check_target_class_balance(contract, df),
        _check_cost_model(contract, df),
        _check_row_count_and_folds(contract, df),
    ]
    return ValidationResult(checks=checks)
