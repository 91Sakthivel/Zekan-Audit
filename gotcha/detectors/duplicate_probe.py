"""Phase-3 content-hash duplicate probes.

Two probes, one mechanism. Both hash the feature+target columns of each row
(excluding panel-identity columns) and compare hash sets to detect duplicates.

RAW_DUPLICATE_ROWS  -> IssueType.ROW_DUPLICATION
    Checks the full dataset for rows whose feature+target content is identical.
    Any exact duplicate indicates an ETL copy error.
    Status: 'warn' if excess copies > 0; 'pass' otherwise.

CROSS_FOLD_DUPLICATE  -> IssueType.CROSS_FOLD_DUPLICATE
    For each evaluation fold (must be the same FoldIndices objects the engine
    evaluated on -- do not regenerate), checks whether any test-set row has
    the same feature+target content as a training-set row.
    Status: 'fail' if any contamination exists; 'pass' otherwise.

Hash content decision
---------------------
Columns EXCLUDED from the hash:
    entity_id, prediction_time, available_features_until

These are the panel-identity columns. They legitimately repeat in longitudinal
data: the same entity appears at multiple time points (entity_id repeats across
rows), and the same time period contains multiple entities (prediction_time
repeats across rows). Including them would suppress real duplicates: two
observations from different time periods with identical feature content would not
collide if prediction_time were in the hash, even though the measurement content
is genuinely identical.

Columns INCLUDED in the hash:
    all remaining columns (feature columns + target)

Two rows hash-equal only when their entire feature vector AND label are identical.
This is a genuine duplicate observation, not a legitimate longitudinal repetition.

NaN/None handling: all NA variants (float NaN, None, pd.NA, pd.NaT) are replaced
by a canonical sentinel string before hashing, so NaN == NaN in comparisons.

Status threshold rationale
--------------------------
RAW, any excess copy -> 'warn':
    Exact floating-point equality across all feature columns cannot occur by
    coincidence in real-world observational data. The only source is an ETL copy
    bug. There is no defensible "small enough to ignore" count. The
    duplicate_fraction field lets the user judge operational severity.

CROSS_FOLD, any contaminating row -> 'fail':
    The validity of an evaluation fold is binary: either the model was trained
    on a test answer or it was not. One contaminating row invalidates that fold's
    performance estimate. The contamination fraction is informational, not a gate.
"""

from __future__ import annotations

import pandas as pd

from gotcha.contract.prediction_contract import PredictionContract
from gotcha.detectors.schema import (
    CrossFoldDuplicateDetail,
    Evidence,
    IssueRecord,
    IssueType,
    RowDuplicationDetail,
)
from gotcha.severity.splitters import FoldIndices


# ── Hash machinery ────────────────────────────────────────────────────────────

_NULL_SENTINEL = "__NULL__"


def _hash_columns(df: pd.DataFrame, contract: PredictionContract) -> list[str]:
    """Return columns included in the content hash (all columns except panel keys).

    Raises ValueError when no hashable columns remain so the caller fails loud
    rather than silently hashing an empty tuple per row.
    """
    excluded = {
        contract.entity_id,
        contract.prediction_time,
        contract.available_features_until,
    }
    cols = [c for c in df.columns if c not in excluded]
    if not cols:
        raise ValueError(
            f"No hashable columns remain after excluding identity columns "
            f"{sorted(excluded)}. DataFrame columns: {list(df.columns)}. "
            f"The DataFrame must contain at least one column beyond the "
            f"contract's entity_id, prediction_time, and available_features_until."
        )
    return cols


def _compute_row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """64-bit row hash over selected columns.

    All NA variants (float NaN, None, pd.NA, pd.NaT) are replaced by a canonical
    sentinel string so that NaN == NaN in duplicate comparison. Without this,
    two rows with identical NaN values would never collide.

    Uses pd.util.hash_pandas_object (vectorized, deterministic seed=0). At
    N=1M rows, the 64-bit collision probability is ~3e-8 -- acceptable for
    duplicate detection (not a cryptographic application).
    """
    sub = df[cols].astype(object)           # uniform dtype for all NA variants
    sub = sub.where(sub.notna(), other=_NULL_SENTINEL)
    return pd.util.hash_pandas_object(sub, index=False)


# ── RAW_DUPLICATE_ROWS ────────────────────────────────────────────────────────

def probe_raw_duplicates(
    df: pd.DataFrame,
    contract: PredictionContract,
) -> IssueRecord:
    """Check the full dataset for exact content-duplicate rows.

    Hash columns: all feature columns + target (entity_id, prediction_time,
    and available_features_until excluded -- they legitimately repeat in panels).

    Returns:
        status='warn'  if any excess copies exist (count > 0)
        status='pass'  otherwise
    """
    if df.empty:
        raise ValueError(
            "probe_raw_duplicates: DataFrame is empty. "
            "Pass a non-empty dataset."
        )

    cols = _hash_columns(df, contract)
    hashes = _compute_row_hashes(df, cols)

    hash_counts = hashes.value_counts()
    dup_hash_counts = hash_counts[hash_counts > 1]

    # Excess copies: N identical rows -> N-1 are removable duplicates.
    # This is the count a dedup operation would eliminate.
    excess_copies = int((dup_hash_counts - 1).sum())
    total_rows = len(df)
    duplicate_fraction = excess_copies / total_rows

    detail = RowDuplicationDetail(
        duplicate_rows=excess_copies,
        duplicate_fraction=duplicate_fraction,
    )

    _why = (
        "Duplicate rows inflate effective sample size and can over-weight "
        "certain patterns, biasing model estimates and evaluation metrics."
    )

    if excess_copies == 0:
        return IssueRecord(
            issue_type=IssueType.ROW_DUPLICATION,
            status="pass",
            what="No exact content-duplicate rows found in the dataset.",
            why=_why,
            how_much=f"0 duplicate rows out of {total_rows} total (0.00%).",
            next_fix="No action required.",
            evidence=Evidence(
                measured_value=0.0,
                threshold=0.0,
                metric_name="duplicate_fraction",
                structural_detail=detail,
            ),
        )

    n_patterns = len(dup_hash_counts)
    return IssueRecord(
        issue_type=IssueType.ROW_DUPLICATION,
        status="warn",
        what=(
            f"{excess_copies} excess content-duplicate rows found "
            f"({duplicate_fraction:.2%} of {total_rows} rows); "
            f"{n_patterns} distinct duplicated pattern(s)."
        ),
        why=_why,
        how_much=(
            f"{excess_copies} excess copies out of {total_rows} total rows "
            f"({duplicate_fraction:.2%}); "
            f"{n_patterns} distinct duplicated content pattern(s)."
        ),
        next_fix=(
            "Deduplicate the raw dataset before ingestion. "
            "Identify the ETL step that produced the copies and fix the root cause."
        ),
        evidence=Evidence(
            measured_value=duplicate_fraction,
            threshold=0.0,
            metric_name="duplicate_fraction",
            structural_detail=detail,
        ),
    )


# ── CROSS_FOLD_DUPLICATE ──────────────────────────────────────────────────────

def probe_cross_fold_duplicates(
    df: pd.DataFrame,
    contract: PredictionContract,
    folds: list[FoldIndices],
) -> IssueRecord:
    """Check for content-hash collision between train and test sets within each fold.

    folds must be the same FoldIndices objects the engine evaluated on.
    Regenerating the folds (e.g. with different seeds or params) would check
    the wrong train/test split. Skipped folds are excluded from all counts.

    Returns:
        status='fail'  if any test row's content hash appears in its fold's train set
        status='pass'  otherwise
    """
    if df.empty:
        raise ValueError(
            "probe_cross_fold_duplicates: DataFrame is empty."
        )
    if not folds:
        raise ValueError(
            "probe_cross_fold_duplicates: folds list is empty. "
            "Pass the FoldIndices returned by the splitter (same objects the engine used)."
        )

    cols = _hash_columns(df, contract)
    hashes = _compute_row_hashes(df, cols)
    hash_arr = hashes.values   # numpy array indexed by DataFrame position

    total_test_rows = 0
    total_dup_rows = 0
    affected_folds: list[int] = []
    rows_per_fold: dict[int, int] = {}

    for fold in folds:
        if fold.meta.skipped:
            continue

        train_hash_set = set(hash_arr[fold.train_idx])
        dup_count = int(
            sum(1 for h in hash_arr[fold.test_idx] if h in train_hash_set)
        )
        total_test_rows += len(fold.test_idx)
        total_dup_rows += dup_count

        if dup_count > 0:
            affected_folds.append(fold.meta.fold_idx)
            rows_per_fold[fold.meta.fold_idx] = dup_count

    duplicate_fraction = (
        total_dup_rows / total_test_rows if total_test_rows > 0 else 0.0
    )

    detail = CrossFoldDuplicateDetail(
        duplicate_rows=total_dup_rows,
        duplicate_fraction=duplicate_fraction,
        affected_folds=affected_folds,
        rows_per_fold=rows_per_fold,
    )

    _why = (
        "A test row whose feature+target content also appears in the training "
        "set was literally memorized by the model at training time, invalidating "
        "that fold's performance estimate and undermining the leakage audit verdict."
    )

    n_folds_evaluated = sum(1 for f in folds if not f.meta.skipped)

    if total_dup_rows == 0:
        return IssueRecord(
            issue_type=IssueType.CROSS_FOLD_DUPLICATE,
            status="pass",
            what="No train/test content overlap found across all evaluation folds.",
            why=_why,
            how_much=(
                f"0 contaminating rows across {n_folds_evaluated} evaluated fold(s)."
            ),
            next_fix="No action required.",
            evidence=Evidence(
                measured_value=0.0,
                metric_name="cross_fold_duplicate_fraction",
                structural_detail=detail,
            ),
        )

    fold_summary = ", ".join(
        f"fold {k}: {v} row(s)" for k, v in sorted(rows_per_fold.items())
    )
    return IssueRecord(
        issue_type=IssueType.CROSS_FOLD_DUPLICATE,
        status="fail",
        what=(
            f"{total_dup_rows} test row(s) share exact content with their "
            f"fold's training set across {len(affected_folds)} fold(s) "
            f"({fold_summary})."
        ),
        why=_why,
        how_much=(
            f"{total_dup_rows} of {total_test_rows} test rows contaminated "
            f"({duplicate_fraction:.2%}); affected folds: {affected_folds}."
        ),
        next_fix=(
            "Remove the duplicate rows from the raw dataset before running gotcha. "
            "Trace the ETL pipeline to find where the same observation was "
            "recorded at multiple time points with different timestamps."
        ),
        evidence=Evidence(
            measured_value=duplicate_fraction,
            threshold=0.0,
            metric_name="cross_fold_duplicate_fraction",
            structural_detail=detail,
        ),
    )
