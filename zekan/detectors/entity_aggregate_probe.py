"""FORBIDDEN_ENTITY_LEVEL_AGGREGATE probe — constant-within-entity forbidden feature.

Detects when a declared forbidden feature is constant within repeated entities and
varies across entities.  This structural pattern is consistent with an entity-level
aggregate or pre-split artifact (e.g. entity_churn_rate = mean lifetime churn).

Why this matters
----------------
The within-entity permutation null (the default null method in zekan) is a no-op on
constant-within-entity features: shuffling values within an entity that all have the
same value produces an identical column.  All 100 null samples therefore equal the
observed fixable_leakage, giving p_value = 1.0 and leaving detection perpetually
unconfirmed.  The verdict is structurally INCONCLUSIVE, but the leakage is real.

This probe provides a deterministic annotation explaining WHY the verdict is
INCONCLUSIVE: not because the sample size is too small, but because the feature's
structural property makes the null uninformative.

Confirmed vs. statistical
--------------------------
confirmed=True in the registry means the STRUCTURAL FACT (constancy within entity,
variation across entities) is certain.  It does NOT mean the metric impact (fl) is
confirmed — the policy verdict is never changed.  The annotation is explanatory only.

Trigger (all conditions must hold)
------------------------------------
1. contract declares an entity_id column AND it exists in df
2. the feature is in contract.forbidden_after_prediction AND exists in df
3. Restrict to entities with >= 2 observations (the "repeated-entity subset").
   Singletons are excluded: a feature that appears constant only because each entity
   has one row is not structurally meaningful.
4. At least 2 repeated entities must exist in this subset.
5. Within EVERY repeated entity, the feature has exactly ONE unique value (exact
   constancy: std==0 / nunique==1 — NO epsilon tolerance, NO near-constant).
6. Across entities, the feature varies: between_entity_unique_count > 1.
   Dead / globally-constant features do NOT fire.

One IssueRecord is returned per forbidden feature that meets all trigger conditions.
"""

from __future__ import annotations

import pandas as pd

from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import (
    Evidence,
    ForbiddenEntityLevelAggregateDetail,
    IssueRecord,
    IssueType,
)


_WHY = (
    "A forbidden feature that is constant within each entity but varies across "
    "entities encodes entity-level information that is not available at prediction "
    "time for new entities.  The within-entity permutation null cannot detect this "
    "pattern (null is a no-op on constant-within-entity columns), so the standard "
    "detection gate always reports the verdict as unconfirmed.  The structural "
    "evidence is deterministic, however: the feature encodes entity-level signal "
    "that inflates offline AUC but will not be present at deployment."
)

_REPORT_LANGUAGE = (
    "Confirmed structural leakage pattern: feature '{feature}' is constant within "
    "each repeated entity and varies across entities — consistent with an "
    "entity-level aggregate or pre-split artifact. The permutation null cannot "
    "statistically confirm this pattern, but the structural evidence is "
    "deterministic. Treat the measured inflation as suspicious and inspect how "
    "this feature was created."
)


def probe_forbidden_entity_level_aggregate(
    df: pd.DataFrame,
    contract: PredictionContract,
) -> list[IssueRecord]:
    """Structural probe: do any forbidden features form entity-level aggregates?

    Returns one IssueRecord per forbidden feature that meets ALL trigger conditions.
    Returns an empty list when no forbidden features are declared, when the entity_id
    column is absent, or when no forbidden feature meets the trigger.
    """
    # ── Guard: entity_id column must exist ──────────────────────────────────────
    if contract.entity_id not in df.columns:
        return []

    forbidden = [
        f for f in (contract.forbidden_after_prediction or [])
        if f in df.columns
    ]
    if not forbidden:
        return []

    # ── Repeated-entity subset: entities with >= 2 observations ────────────────
    # Drop rows where entity_id is null (panel membership is ambiguous).
    valid = df.dropna(subset=[contract.entity_id])
    entity_counts = valid.groupby(contract.entity_id).size()
    repeated_entities = entity_counts[entity_counts >= 2].index
    n_repeated = len(repeated_entities)

    if n_repeated < 2:
        # Singleton-only or single-repeated-entity dataset: structural gate not met.
        return []

    repeated_df = valid[valid[contract.entity_id].isin(repeated_entities)]

    findings: list[IssueRecord] = []

    for feature in forbidden:
        # ── Condition 5: within EVERY repeated entity, feature has exactly 1 unique value ──
        per_entity_nunique = (
            repeated_df.groupby(contract.entity_id)[feature].nunique()
        )
        if per_entity_nunique.max() != 1:
            # At least one entity has more than one distinct value — not constant within entity.
            continue

        # ── Condition 6: across entities, the feature varies ────────────────────
        # Take one representative value per entity (all values are identical within entity).
        entity_values = repeated_df.groupby(contract.entity_id)[feature].first()
        between_entity_unique_count = int(entity_values.nunique())
        if between_entity_unique_count <= 1:
            # Globally constant (dead column): do not fire.
            continue

        # ── All conditions met: build finding ───────────────────────────────────
        detail = ForbiddenEntityLevelAggregateDetail(
            feature=feature,
            entity_col=contract.entity_id,
            eligible_entities=n_repeated,
            within_entity_constant=True,
            between_entity_unique_count=between_entity_unique_count,
            statistical_confirmation="not_required",
            verdict_effect="annotate_only",
        )

        what = _REPORT_LANGUAGE.format(feature=feature)

        how_much = (
            f"Feature '{feature}' is constant within all {n_repeated} repeated "
            f"entities and takes {between_entity_unique_count} distinct values "
            f"across those entities.  Within-entity permutation null is a no-op "
            f"on this feature (p_value ≡ 1.0 by construction)."
        )

        findings.append(IssueRecord(
            issue_type=IssueType.FORBIDDEN_ENTITY_LEVEL_AGGREGATE,
            status="warn",
            what=what,
            why=_WHY,
            how_much=how_much,
            next_fix=(
                f"Inspect how '{feature}' was constructed. If it aggregates "
                "target or outcome values across ALL time periods for each entity "
                "(including future periods relative to the prediction time), it "
                "encodes future information and must be removed from training. "
                "Replace with a lag-safe version computed only from periods prior "
                "to each prediction time."
            ),
            evidence=Evidence(
                metric_name="within_entity_unique_count",
                measured_value=1.0,
                threshold=1.0,
                structural_detail=detail,
            ),
        ))

    return findings
