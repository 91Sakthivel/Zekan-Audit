"""Contract validation: structural and semantic checks against a real DataFrame.

All checks return CheckResult objects - nothing raises. Callers inspect the
ValidationResult to decide whether to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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


def _check_feature_columns_numeric(c: PredictionContract, df: pd.DataFrame) -> CheckResult:
    """Every feature column must be castable to float for EVERY row, UNLESS
    it is declared in contract.categorical_features (CATEGORICAL_SUPPORT_
    PREREGISTRATION.md 3(a)/3(b)): a declared column is ordinal-encoded
    before the cast (see build_categorical_mapping/apply_categorical_mapping
    below), so it is exempted from this coercion requirement -- the encoding
    step is what makes it numeric by the time evaluate_folds runs, not this
    check.

    evaluate_folds (metrics.py) does df[feature_cols].to_numpy(dtype=float) on
    the whole feature block at once for every fold -- a single non-numeric
    value in ANY feature column raises there, and it raises for every fold
    since they all share the same feature_cols. So ANY coercion failure here
    must FAIL the gate, not just a widespread one: this check has no WARN
    outcome -- only PASS or FAIL, the same reasoning as prediction_time_parseable.

    Candidate feature columns mirror engine._feature_cols exactly (all columns
    except entity_id, prediction_time, available_features_until, and target).
    forbidden_after_prediction columns are NOT excluded here: engine.py still
    includes them in model A/B's "all_features" set, so they must be numeric
    (or declared categorical) too, not just the columns that end up in model
    C's "safe_features".
    """
    excluded = {c.entity_id, c.prediction_time, c.available_features_until, c.target}
    feature_cols = [col for col in df.columns if col not in excluded]
    declared_categorical = set(c.categorical_features)

    bad_cols: dict[str, int] = {}
    for col in feature_cols:
        if col in declared_categorical:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((coerced.isna() & ~df[col].isna()).sum())
        if n_bad > 0:
            bad_cols[col] = n_bad

    if bad_cols:
        names = list(bad_cols.keys())
        shown = names[:10]
        detail = ", ".join(f"'{col}' ({bad_cols[col]} value(s))" for col in shown)
        if len(names) > 10:
            detail += f", and {len(names) - 10} more"
        # Actionable per pre-registration 3(b): a copy-pasteable declaration
        # covering EVERY still-failing column (not just the 10 shown above),
        # so declaring them is a paste, not a retype.
        suggestion = "categorical_features:\n" + "\n".join(f"  - {col!r}" for col in names)
        return CheckResult("feature_columns_numeric", CheckStatus.FAIL,
                           f"{len(bad_cols)} feature column(s) contain values that cannot be "
                           f"read as numbers: {detail}. Zekan's models require every feature "
                           f"column to be numeric -- if these are genuinely nominal columns "
                           f"(not broken data), declare them under categorical_features so "
                           f"Zekan can ordinal-encode them before auditing. Copy-paste into "
                           f"your contract:\n{suggestion}")
    return CheckResult("feature_columns_numeric", CheckStatus.PASS,
                       f"All {len(feature_cols)} feature column(s) are numeric "
                       f"({len(declared_categorical & set(feature_cols))} declared categorical)")


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
        _check_feature_columns_numeric(contract, df),
        _check_temporal_periods(contract, df),
        _check_target_class_balance(contract, df),
        _check_cost_model(contract, df),
        _check_row_count_and_folds(contract, df),
    ]
    return ValidationResult(checks=checks)


def candidate_features(contract: PredictionContract, df: pd.DataFrame) -> list[str]:
    """Non-forbidden feature columns under this contract, in `df` column order.

    Excludes the contract's own role columns (entity_id, prediction_time,
    available_features_until, target) and every declared
    forbidden_after_prediction column. This is a CONTRACT concept -- which
    columns count as a screenable feature under a given contract -- shared by
    every structural probe that screens non-forbidden features (Upgrade 1's
    undeclared_feature_probe.py, Upgrade (H)'s near_bijection_probe.py), owned
    by neither. Extracted here (Upgrade H) from what was previously inline,
    private logic in undeclared_feature_probe.py so a second probe needing the
    same candidate set would not have to duplicate it.
    """
    excluded = {
        contract.entity_id,
        contract.prediction_time,
        contract.available_features_until,
        contract.target,
    }
    forbidden = set(contract.forbidden_after_prediction or [])
    return [c for c in df.columns if c not in excluded and c not in forbidden]


# ── Categorical encoding (CATEGORICAL_SUPPORT_PREREGISTRATION.md 3(a)/3(d)) ────
# Ordinal, sorted-unique -> 0..k-1, deterministic, target-free -- matching
# zekan/benchmark/prepare_test_b.py's _build_ordinal_mappings semantics.
# Unlike that script (which infers "every non-numeric, non-role column"),
# this operates ONLY on contract.categorical_features: declared, never
# inferred (3(a)) -- a nominal column and a column of numeral-looking
# strings are different things, and only the user knows which is which.

_CATEGORICAL_NAN_SENTINEL_BASE = "__NaN__"
"""Same idea as near_bijection_probe.py's own _NAN_SENTINEL: NaN is treated
as its own explicit category, not dropped or imputed -- "every value present
gets a code" (pre-registration item 1) includes missingness itself. Plain
`sorted(df[col].unique())` (prepare_test_b.py's literal implementation) would
raise on a column that mixes real NaN with strings; Test B's own raw data
never exercised that path (undeclared_feature_probe.py's docstring notes
this as an unexercised gap), but Freddie Mac's frame tables can carry real
NaN in a declared-categorical column, so this must not crash on it.

This is a BASE value, not the literal sentinel every column uses -- see
_pick_nan_sentinel: if a column's own real values happen to already contain
this literal string, using it unmodified would collide two distinct raw
values (actual NaN, and that real value) onto one code, breaking the
bijection CATEGORICAL_SUPPORT_PREREGISTRATION.md section 4 relies on for
Theil's U invariance. Collision is made impossible, not just unlikely."""


def _pick_nan_sentinel(non_null_object_values: set) -> str:
    """Return a NaN sentinel guaranteed not to collide with any of a
    column's own real (non-null) values.

    Starts from _CATEGORICAL_NAN_SENTINEL_BASE and appends underscores until
    the result is absent from `non_null_object_values` -- deterministic
    (same real values always produce the same sentinel) and exhaustive (a
    finite column has finitely many values, so this always terminates).
    Preserves the section 4 bijection claim: the sentinel this returns is
    never equal to any value already in the column, so NaN and that value
    can never collapse onto the same code.
    """
    sentinel = _CATEGORICAL_NAN_SENTINEL_BASE
    while sentinel in non_null_object_values:
        sentinel += "_"
    return sentinel


def build_categorical_mapping(
    df: pd.DataFrame, categorical_features: list[str]
) -> dict[str, dict]:
    """Sorted-unique -> 0..k-1 ordinal mapping for each declared column.

    Columns in `categorical_features` that are not present in `df` are
    skipped (mirrors prepare_test_b.py's _apply_ordinal_mappings: a contract
    may be reused across data files that don't all carry every declared
    column). Uses ONLY each column's own values -- no target, no randomness
    -- so the mapping is deterministic and reproducible: the same column
    values always produce the same map. Sort key is `str` (not a bare value
    sort) so a column need not be internally comparable/homogeneous to be
    encoded -- this is the one deliberate generalization beyond
    prepare_test_b.py's bare `sorted()`, made necessary by NaN-safety (see
    _CATEGORICAL_NAN_SENTINEL_BASE above).

    Returns {column: {"codes": {raw_value: code}, "nan_sentinel": str}} --
    the sentinel actually used for THIS column (picked by _pick_nan_sentinel,
    per-column since collision depends on that column's own values) travels
    with the mapping so apply_categorical_mapping never has to re-derive or
    guess it, including when applied later to a different dataframe under
    the same contract.
    """
    mappings: dict[str, dict] = {}
    for col in categorical_features:
        if col not in df.columns:
            continue
        non_null = set(df[col].dropna().astype("object").unique())
        sentinel = _pick_nan_sentinel(non_null)
        x = df[col].astype("object").where(df[col].notna(), sentinel)
        uniques = sorted(x.unique(), key=str)
        codes = {v: i for i, v in enumerate(uniques)}
        mappings[col] = {"codes": codes, "nan_sentinel": sentinel}
    return mappings


def apply_categorical_mapping(
    df: pd.DataFrame,
    mapping: dict[str, dict],
    unseen_counts: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Apply a precomputed mapping (from build_categorical_mapping) to `df`.

    Returns a copy -- the input `df` is never mutated. Columns in `mapping`
    that are not present in `df` are skipped, same as build_categorical_mapping.
    A value not present in that column's codes (e.g. a category seen only in
    a later audit of different data under the same contract) maps to NaN via
    pandas' ordinary `.map()` behavior -- not silently coerced to an existing
    code.

    unseen_counts
        Optional out-parameter, same pattern as audit.py's `side_channel`
        (a caller-supplied dict this function writes into, so the primary
        return type -- a DataFrame -- never has to change shape to carry
        secondary information; metrics.py's `df = apply_categorical_mapping(
        df, categorical_map)` call keeps working unmodified). When given, is
        populated with {column: n_unseen} for every column where one or more
        values mapped to NaN because they were absent from that column's
        codes -- "silence is not clearance": an unseen value quietly
        becoming NaN must be countable by the caller, not just discoverable
        later by noticing an unexplained NaN downstream. A column with zero
        unseen values is not added. None (the default) skips counting --
        every caller that doesn't pass this gets identical behavior to
        before this parameter existed.
    """
    if not mapping:
        return df
    df = df.copy()
    for col, col_map in mapping.items():
        if col not in df.columns:
            continue
        codes = col_map["codes"]
        sentinel = col_map["nan_sentinel"]
        x = df[col].astype("object").where(df[col].notna(), sentinel)
        mapped = x.map(codes)
        if unseen_counts is not None:
            n_unseen = int(mapped.isna().sum())
            if n_unseen > 0:
                unseen_counts[col] = n_unseen
        df[col] = mapped
    return df
