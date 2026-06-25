"""ENTITY_CONTAMINATION_RISK probe — longitudinal data structure risk advisory.

Confirms a structural data property: whether the same entity (identified by
contract.entity_id) appears across more than one distinct prediction_time period.
This is a RISK ADVISORY (confirmed=False in the registry): the probe reports what
the dataset structure looks like, not what split the user applied.

Trigger
-------
Fire WARN when any entity appears in >1 distinct prediction_time value
(recurring_entities >= 1).  This is the conservative advisory choice: once any
entity spans multiple periods, a random row-level split has nonzero probability of
placing that entity in both train and test partitions.  There is no principled lower
threshold because the risk is structural — even a single recurring entity creates
the condition.  The advisory nature (confirmed=False) means conservatism is always
correct here: no false accusation is made, only a conditional warning.

Pass condition: every entity appears in exactly one period (cross-sectional data).
A random split on cross-sectional data cannot produce entity-level train/test overlap.

Wording contract
----------------
The user-facing message is conditional throughout: a random split WOULD leak — not
that a wrong split was used or that leakage occurred.  Gotcha owns the evaluation
loop and cannot observe what split the caller applied outside of Gotcha; the probe
observes dataset structure only.

Messages MUST NOT imply "wrong split", "you leaked", "your split is contaminated",
or "split strategy detected".  The confirmed ENTITY_CONTAMINATION probe (scope
CROSS_FOLD, confirmed=True) handles the case where contamination is directly
observed; this probe handles the structural-risk advisory only.

Robustness
----------
- entity_id column absent   → status='unavailable', no crash
- prediction_time absent    → status='unavailable', no crash
- All rows null in either   → status='unavailable', no crash
- Cross-sectional data      → status='pass' (critical false-positive guard)
- Single entity / many periods  → warn (1 recurring entity is genuine recurrence)
- Many entities / single period → pass (all obs_per_entity == 1)
- Nulls in identity columns → dropped deterministically before all counts
"""

from __future__ import annotations

import pandas as pd

from gotcha.contract.prediction_contract import PredictionContract
from gotcha.detectors.schema import (
    EntityContaminationRiskDetail,
    Evidence,
    IssueRecord,
    IssueType,
)


_WHY = (
    "When the same entity appears in both train and test partitions, the model is "
    "evaluated on observations it was trained on, inflating performance estimates; "
    "Gotcha evaluates under the declared temporal/grouped contract, which avoids this."
)


def probe_entity_contamination_risk(
    df: pd.DataFrame,
    contract: PredictionContract,
) -> IssueRecord:
    """Advisory probe: is the dataset longitudinal in a way that creates random-split risk?

    Returns:
        status='warn'        entities recur across multiple prediction_time periods
        status='pass'        dataset is cross-sectional (each entity at exactly one period)
        status='unavailable' entity_id or prediction_time column absent, or no usable rows
    """
    # ── Guard: required columns must exist ───────────────────────────────────────
    if contract.entity_id not in df.columns:
        return IssueRecord(
            issue_type=IssueType.ENTITY_CONTAMINATION_RISK,
            status="unavailable",
            what=(
                f"Entity contamination risk check skipped: column "
                f"{contract.entity_id!r} (entity_id) not found in the dataset."
            ),
            why=_WHY,
            how_much=f"Column {contract.entity_id!r} is absent; probe requires it.",
            next_fix=(
                "Add an entity_id column to the dataset, or update the contract "
                "to reference the correct column name."
            ),
            evidence=Evidence(),
        )

    if contract.prediction_time not in df.columns:
        return IssueRecord(
            issue_type=IssueType.ENTITY_CONTAMINATION_RISK,
            status="unavailable",
            what=(
                f"Entity contamination risk check skipped: column "
                f"{contract.prediction_time!r} (prediction_time) not found in the dataset."
            ),
            why=_WHY,
            how_much=f"Column {contract.prediction_time!r} is absent; probe requires it.",
            next_fix=(
                "Add a prediction_time column to the dataset, or update the contract "
                "to reference the correct column name."
            ),
            evidence=Evidence(),
        )

    # ── Drop nulls in identity columns ────────────────────────────────────────────
    # Nulls in entity_id or prediction_time make panel membership ambiguous.
    # Drop them deterministically; remaining rows drive all counts.
    valid = df.dropna(subset=[contract.entity_id, contract.prediction_time])

    if valid.empty:
        return IssueRecord(
            issue_type=IssueType.ENTITY_CONTAMINATION_RISK,
            status="unavailable",
            what=(
                "Entity contamination risk check skipped: no rows with non-null "
                "entity_id and prediction_time remain after dropping nulls."
            ),
            why=_WHY,
            how_much=(
                f"0 valid rows of {len(df)} total after dropping nulls in "
                f"{contract.entity_id!r} and {contract.prediction_time!r}."
            ),
            next_fix=(
                "Investigate missing values in entity_id and prediction_time "
                "before running the probe."
            ),
            evidence=Evidence(),
        )

    # ── Panel statistics ──────────────────────────────────────────────────────────
    # Per-entity: how many distinct prediction_time values does each entity appear in?
    periods_per_entity = (
        valid.groupby(contract.entity_id)[contract.prediction_time].nunique()
    )
    total_entities = len(periods_per_entity)
    recurring_entities = int((periods_per_entity > 1).sum())
    entity_recurrence_fraction = recurring_entities / total_entities

    # Row counts per entity for median/max observation stats.
    row_counts = valid.groupby(contract.entity_id).size()
    median_obs = float(row_counts.median())
    max_obs = int(row_counts.max())

    distinct_periods = int(valid[contract.prediction_time].nunique())

    detail = EntityContaminationRiskDetail(
        recurring_entities=recurring_entities,
        total_entities=total_entities,
        entity_recurrence_fraction=entity_recurrence_fraction,
        median_obs_per_entity=median_obs,
        max_obs_per_entity=max_obs,
        distinct_periods=distinct_periods,
    )

    # ── Pass: cross-sectional ─────────────────────────────────────────────────────
    if recurring_entities == 0:
        return IssueRecord(
            issue_type=IssueType.ENTITY_CONTAMINATION_RISK,
            status="pass",
            what=(
                "Dataset is cross-sectional: each entity appears in exactly one "
                "prediction period; a random split cannot produce entity-level "
                "train/test overlap."
            ),
            why=_WHY,
            how_much=(
                f"{total_entities:,} entities across {distinct_periods} period(s); "
                f"0 entities recur across multiple periods."
            ),
            next_fix="No action required.",
            evidence=Evidence(
                measured_value=0.0,
                threshold=0.0,
                metric_name="entity_recurrence_fraction",
                structural_detail=detail,
            ),
        )

    # ── Warn: longitudinal ────────────────────────────────────────────────────────
    return IssueRecord(
        issue_type=IssueType.ENTITY_CONTAMINATION_RISK,
        status="warn",
        what=(
            f"The dataset is longitudinal: {recurring_entities:,} of "
            f"{total_entities:,} entities "
            f"({entity_recurrence_fraction:.1%}) appear across more than one "
            f"prediction period; a random row-level split WOULD place the same "
            f"entity in both train and test."
        ),
        why=_WHY,
        how_much=(
            f"{recurring_entities:,} of {total_entities:,} entities "
            f"({entity_recurrence_fraction:.1%}) recur across {distinct_periods} "
            f"period(s); median {median_obs:.1f} obs/entity, "
            f"max {max_obs} obs/entity."
        ),
        next_fix=(
            "If running your own evaluation outside Gotcha, use a split strategy "
            "that holds out complete entities or complete time periods "
            "(e.g. GroupKFold on entity_id, or TimeSeriesSplit on prediction_time) "
            "rather than sampling individual rows at random."
        ),
        evidence=Evidence(
            measured_value=entity_recurrence_fraction,
            threshold=0.0,
            metric_name="entity_recurrence_fraction",
            structural_detail=detail,
        ),
    )
