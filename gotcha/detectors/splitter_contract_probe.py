"""SPLITTER_CONTRACT_VIOLATION probe — Gotcha's own grouped splitter self-check.

Internal integrity test: verifies that Gotcha's grouped splitter never placed the
same entity in both train and test within any single fold.  If this probe fires,
the bug is in Gotcha's splitter code, not in the user's data or split choice.

Trigger
-------
Binary structural condition: fire 'internal_fail' if any entity_id value appears
in BOTH the train indices AND the test indices of the same non-skipped fold.
Zero spanning entities across all evaluated folds → 'pass'.
No percentage threshold — the grouped-CV contract is binary: an entity is either
held out or it is not.

Skipped folds (meta.skipped=True) are excluded entirely from the check; they are
excluded from the engine evaluation and should not contribute here either.

Wording contract
----------------
All narrative frames Gotcha as the source of any violation — "Gotcha's grouped
splitter placed the same entity in both train and test of fold N".  No language
implies user fault; this is an internal sanity check, not a user-facing audit
finding.

Robustness
----------
- Empty folds list        → status='unavailable', no crash
- All folds skipped       → status='unavailable' (nothing checkable)
- entity_id col absent    → status='unavailable', no crash
- Single fold             → binary check applies normally
- Degenerate partitions   → correct: if all entities in one side, spanning=0 → pass
"""

from __future__ import annotations

import pandas as pd

from gotcha.contract.prediction_contract import PredictionContract
from gotcha.detectors.schema import (
    Evidence,
    IssueRecord,
    IssueType,
    SplitterContractViolationDetail,
)
from gotcha.severity.splitters import FoldIndices


_WHY = (
    "Gotcha's grouped splitter guarantees that no entity appears in both train "
    "and test within the same fold; a violation means the splitter has a bug and "
    "every evaluation result from this run is unreliable."
)


def probe_splitter_contract_violation(
    folds: list[FoldIndices],
    df: pd.DataFrame,
    contract: PredictionContract,
) -> IssueRecord:
    """Internal self-check: did Gotcha's own grouped splitter honour entity grouping?

    Returns:
        status='pass'          no entity spans train and test in any evaluated fold
        status='internal_fail' at least one entity appears in both partitions of a fold
        status='unavailable'   nothing to check (empty/all-skipped folds, missing column)
    """
    # ── Guard: entity_id column must exist ───────────────────────────────────────
    if contract.entity_id not in df.columns:
        return IssueRecord(
            issue_type=IssueType.SPLITTER_CONTRACT_VIOLATION,
            status="unavailable",
            what=(
                f"Splitter contract check skipped: column {contract.entity_id!r} "
                f"(entity_id) not found in the dataset."
            ),
            why=_WHY,
            how_much=f"Column {contract.entity_id!r} is absent; probe requires it.",
            next_fix=(
                "Ensure the dataset contains an entity_id column matching the "
                "contract, then re-run."
            ),
            evidence=Evidence(),
        )

    # ── Guard: at least one evaluable fold ───────────────────────────────────────
    evaluable = [f for f in folds if not f.meta.skipped]

    if not evaluable:
        reason = (
            "no folds provided" if not folds
            else f"all {len(folds)} fold(s) are marked skipped"
        )
        return IssueRecord(
            issue_type=IssueType.SPLITTER_CONTRACT_VIOLATION,
            status="unavailable",
            what=f"Splitter contract check skipped: {reason}.",
            why=_WHY,
            how_much=f"{len(folds)} fold(s) provided, 0 evaluable.",
            next_fix=(
                "Pass the FoldIndices returned by the splitter; ensure at least "
                "one fold is not skipped."
            ),
            evidence=Evidence(),
        )

    # ── Check each fold for entity spanning ──────────────────────────────────────
    total_entities = int(df[contract.entity_id].nunique())
    entity_col = df[contract.entity_id]

    affected_folds: list[int] = []
    spanning_entities_per_fold: dict[int, int] = {}

    for fold in evaluable:
        train_entities = set(entity_col.iloc[fold.train_idx])
        test_entities = set(entity_col.iloc[fold.test_idx])
        spanning = train_entities & test_entities
        if spanning:
            affected_folds.append(fold.meta.fold_idx)
            spanning_entities_per_fold[fold.meta.fold_idx] = len(spanning)

    total_folds_checked = len(evaluable)
    detail = SplitterContractViolationDetail(
        affected_folds=affected_folds,
        spanning_entities_per_fold=spanning_entities_per_fold,
        total_folds_checked=total_folds_checked,
        total_entities_checked=total_entities,
    )

    # ── Pass: contract upheld ────────────────────────────────────────────────────
    if not affected_folds:
        return IssueRecord(
            issue_type=IssueType.SPLITTER_CONTRACT_VIOLATION,
            status="pass",
            what=(
                "Gotcha's grouped splitter correctly held out entire entities: "
                "no entity appears in both train and test within any fold."
            ),
            why=_WHY,
            how_much=(
                f"0 spanning entities across {total_folds_checked} evaluated fold(s); "
                f"{total_entities} distinct entities checked."
            ),
            next_fix="No action required.",
            evidence=Evidence(
                measured_value=0.0,
                threshold=0.0,
                metric_name="spanning_entity_count",
                structural_detail=detail,
            ),
        )

    # ── Internal fail: contract violated ─────────────────────────────────────────
    total_violations = sum(spanning_entities_per_fold.values())
    fold_summary = ", ".join(
        f"fold {k}: {v} entity/entities"
        for k, v in sorted(spanning_entities_per_fold.items())
    )
    return IssueRecord(
        issue_type=IssueType.SPLITTER_CONTRACT_VIOLATION,
        status="internal_fail",
        what=(
            f"Gotcha's grouped splitter placed the same entity in both train and "
            f"test in {len(affected_folds)} of {total_folds_checked} fold(s) "
            f"— internal contract violation ({fold_summary})."
        ),
        why=_WHY,
        how_much=(
            f"{total_violations} spanning entity occurrence(s) across "
            f"{len(affected_folds)} affected fold(s): {fold_summary}; "
            f"{total_entities} entities total."
        ),
        next_fix=(
            "File a bug against gotcha.severity.splitters — the grouped splitter "
            "must never place the same entity in both train and test.  "
            "Do not interpret any evaluation results from this run."
        ),
        evidence=Evidence(
            measured_value=float(total_violations),
            threshold=0.0,
            metric_name="spanning_entity_count",
            structural_detail=detail,
        ),
    )
