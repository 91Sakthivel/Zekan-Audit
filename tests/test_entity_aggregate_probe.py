"""Tests for probe_forbidden_entity_level_aggregate (FORBIDDEN_ENTITY_LEVEL_AGGREGATE).

Trigger conditions (all must hold to fire):
  1. entity_id exists in df
  2. feature is in forbidden_after_prediction AND exists in df
  3. repeated-entity subset (>= 2 obs per entity) has >= 2 members
  4. within EVERY repeated entity, feature is exactly constant (nunique == 1)
  5. across entities, feature varies (between_entity_unique_count > 1)

Negative cases: globally-constant feature, feature varies within entity,
feature not forbidden, singleton-only dataset.
Integration: full run_audit() with entity_churn_rate-style data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zekan.config.schema import ZekanConfig, SplitPolicy
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.entity_aggregate_probe import probe_forbidden_entity_level_aggregate
from zekan.detectors.schema import (
    ForbiddenEntityLevelAggregateDetail,
    IssueType,
    IssueSeverity,
    SourceLayer,
    EvidenceScope,
    ImpactType,
)
from zekan.severity.audit import run_audit


# ── Shared helpers ────────────────────────────────────────────────────────────

def _contract(**kw) -> PredictionContract:
    defaults = dict(
        prediction_problem="agg-probe-test",
        entity_id="entity_id",
        prediction_time="snapshot_date",
        target="churned",
        available_features_until="snapshot_date",
        forbidden_after_prediction=["entity_agg"],
    )
    defaults.update(kw)
    return PredictionContract(**defaults)


def _make_panel(
    n_entities: int = 4,
    n_periods: int = 3,
    agg_fn=None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal longitudinal panel.

    agg_fn(entity_idx) -> scalar for entity_agg column.
    Default: entity index itself (constant within, varies across).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        agg_val = e if agg_fn is None else agg_fn(e)
        for t in range(n_periods):
            rows.append({
                "entity_id": f"e{e}",
                "snapshot_date": f"2022-{t+1:02d}-01",
                "feature_x": float(rng.normal()),
                "entity_agg": float(agg_val),
                "churned": int(rng.random() > 0.7),
            })
    return pd.DataFrame(rows)


# ── POSITIVE: probe fires on correct input ────────────────────────────────────

def test_positive_probe_fires():
    df = _make_panel(n_entities=4, n_periods=3)
    contract = _contract()
    results = probe_forbidden_entity_level_aggregate(df, contract)
    assert len(results) == 1, "Expected exactly one finding for entity_agg"
    rec = results[0]
    assert rec.issue_type == IssueType.FORBIDDEN_ENTITY_LEVEL_AGGREGATE
    assert rec.status == "warn"


def test_positive_confirmed_true():
    df = _make_panel(n_entities=4, n_periods=3)
    rec = probe_forbidden_entity_level_aggregate(df, _contract())[0]
    # confirmed is a @computed_field derived from registry — True for this issue type
    assert rec.confirmed is True


def test_positive_source_layer():
    df = _make_panel(n_entities=4, n_periods=3)
    rec = probe_forbidden_entity_level_aggregate(df, _contract())[0]
    assert rec.source_layer == SourceLayer.DETECTED_STRUCTURAL


def test_positive_severity_high():
    df = _make_panel(n_entities=4, n_periods=3)
    rec = probe_forbidden_entity_level_aggregate(df, _contract())[0]
    assert rec.severity == IssueSeverity.HIGH


def test_positive_detail_fields():
    df = _make_panel(n_entities=4, n_periods=3)
    rec = probe_forbidden_entity_level_aggregate(df, _contract())[0]
    detail = rec.evidence.structural_detail
    assert isinstance(detail, ForbiddenEntityLevelAggregateDetail)
    assert detail.feature == "entity_agg"
    assert detail.entity_col == "entity_id"
    assert detail.within_entity_constant is True
    assert detail.between_entity_unique_count == 4  # e0..e3 → values 0.0..3.0
    assert detail.eligible_entities == 4
    assert detail.statistical_confirmation == "not_required"
    assert detail.verdict_effect == "annotate_only"


def test_positive_report_language_in_what():
    df = _make_panel(n_entities=4, n_periods=3)
    rec = probe_forbidden_entity_level_aggregate(df, _contract())[0]
    assert "entity_agg" in rec.what
    assert "constant within" in rec.what
    assert "entity-level aggregate" in rec.what
    assert "permutation null cannot" in rec.what


# ── NEGATIVE: globally constant (dead column) ─────────────────────────────────

def test_negative_globally_constant_does_not_fire():
    df = _make_panel(n_entities=4, n_periods=3, agg_fn=lambda e: 99.0)
    # entity_agg == 99.0 for ALL rows; between_entity_unique_count == 1 → dead column guard
    results = probe_forbidden_entity_level_aggregate(df, _contract())
    assert results == [], "Dead column must NOT fire"


# ── NEGATIVE: feature varies within entity ────────────────────────────────────

def test_negative_varies_within_entity_does_not_fire():
    # Build panel where entity_agg changes per period for each entity
    rows = []
    for e in range(4):
        for t in range(3):
            rows.append({
                "entity_id": f"e{e}",
                "snapshot_date": f"2022-{t+1:02d}-01",
                "feature_x": 0.0,
                "entity_agg": float(e * 10 + t),  # different per period
                "churned": 0,
            })
    df = pd.DataFrame(rows)
    results = probe_forbidden_entity_level_aggregate(df, _contract())
    assert results == [], "Feature that varies within entity must NOT fire"


# ── NEGATIVE: feature not declared as forbidden ───────────────────────────────

def test_negative_not_forbidden_does_not_fire():
    df = _make_panel(n_entities=4, n_periods=3)
    # override forbidden list to something else entirely
    contract = _contract(forbidden_after_prediction=["some_other_col"])
    results = probe_forbidden_entity_level_aggregate(df, contract)
    assert results == [], "Non-forbidden feature must NOT fire"


def test_negative_no_forbidden_declared_does_not_fire():
    df = _make_panel(n_entities=4, n_periods=3)
    contract = _contract(forbidden_after_prediction=[])
    results = probe_forbidden_entity_level_aggregate(df, contract)
    assert results == []


# ── NEGATIVE: singleton-only dataset (all entities have 1 observation) ────────

def test_negative_singletons_only_does_not_fire():
    # n_periods=1: each entity appears exactly once → no repeated-entity subset
    df = _make_panel(n_entities=6, n_periods=1)
    results = probe_forbidden_entity_level_aggregate(df, _contract())
    assert results == [], "Singleton-only dataset must NOT fire (no repeated-entity subset)"


def test_negative_single_repeated_entity_does_not_fire():
    # Only 1 entity has >= 2 observations; structural gate requires >= 2 repeated entities
    rows = [
        {"entity_id": "e0", "snapshot_date": "2022-01-01", "entity_agg": 0.0, "feature_x": 0.0, "churned": 0},
        {"entity_id": "e0", "snapshot_date": "2022-02-01", "entity_agg": 0.0, "feature_x": 0.0, "churned": 0},
        {"entity_id": "e1", "snapshot_date": "2022-01-01", "entity_agg": 1.0, "feature_x": 0.0, "churned": 1},
        # e1 appears only once (singleton)
    ]
    df = pd.DataFrame(rows)
    results = probe_forbidden_entity_level_aggregate(df, _contract())
    assert results == [], "Only 1 repeated entity: structural gate (>= 2) not met"


# ── NEGATIVE: entity_id column missing ────────────────────────────────────────

def test_negative_missing_entity_id_returns_empty():
    df = _make_panel(n_entities=4, n_periods=3)
    df = df.drop(columns=["entity_id"])
    contract = _contract(entity_id="entity_id")
    results = probe_forbidden_entity_level_aggregate(df, contract)
    assert results == []


# ── MULTI-FEATURE: only matching features fire ────────────────────────────────

def test_multi_feature_only_matching_fires():
    df = _make_panel(n_entities=4, n_periods=3)
    # entity_agg: constant within entity (will fire)
    # feature_x:  varies per row (will NOT fire even if forbidden)
    df["noisy_feat"] = df["feature_x"]  # per-row noise, not constant within entity
    contract = _contract(forbidden_after_prediction=["entity_agg", "noisy_feat"])
    results = probe_forbidden_entity_level_aggregate(df, contract)
    fired = [r.evidence.structural_detail.feature for r in results]
    assert "entity_agg" in fired
    assert "noisy_feat" not in fired


# ── INTEGRATION: full run_audit gives INCONCLUSIVE + annotation ───────────────

def _make_entity_churn_rate_dataset(seed: int = 42) -> pd.DataFrame:
    """Recreate the smoke-test INCONCLUSIVE dataset (entity_churn_rate forbidden feature).

    entity_churn_rate = mean(churned across ALL periods) for each entity.
    Constant within entity → within-entity null is a no-op → p_value = 1.0.
    170 entities × 6 periods = 1020 rows clears the 1000-row severity gate and gives
    fl > warn_floor at n_permutations=0.
    """
    rng = np.random.default_rng(seed)
    N_ENTITIES = 170
    PERIODS = [f"2022-{m:02d}-01" for m in range(1, 7)]

    rows = []
    for eid in range(N_ENTITIES):
        entity_risk = float(rng.normal(0.0, 1.0))
        for t, period in enumerate(PERIODS):
            logit = 0.8 * entity_risk + 0.05 * t + float(rng.normal(0.0, 0.55))
            rows.append({
                "entity_id": f"cust_{eid:04d}",
                "snapshot_date": period,
                "tenure_months": max(1, int(rng.normal(22, 9))),
                "monthly_spend": max(0.0, round(float(rng.normal(70, 28)), 2)),
                "support_tickets": max(0, int(rng.poisson(1.1))),
                "login_count": max(0, round(float(rng.normal(10, 4)), 1)),
                "_logit": logit,
            })

    df = pd.DataFrame(rows)
    thresh = float(np.percentile(df["_logit"].values, 70))
    df["churned"] = (df["_logit"] > thresh).astype(int)
    df = df.drop(columns=["_logit"])
    df["entity_churn_rate"] = df.groupby("entity_id")["churned"].transform("mean")
    return df[["entity_id","snapshot_date","tenure_months","monthly_spend",
               "support_tickets","login_count","entity_churn_rate","churned"]]


def _audit_contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="integration test — entity_churn_rate",
        entity_id="entity_id",
        prediction_time="snapshot_date",
        target="churned",
        available_features_until="snapshot_date",
        forbidden_after_prediction=["entity_churn_rate"],
    )


def _audit_config() -> ZekanConfig:
    return ZekanConfig(
        contract=_audit_contract(),
        split_policy=SplitPolicy(n_splits=5, min_test_rows_per_fold=20,
                                 min_positive_cases_per_fold=5,
                                 min_negative_cases_per_fold=5),
    )


def test_integration_verdict_inconclusive_with_annotation():
    df = _make_entity_churn_rate_dataset()
    contract = _audit_contract()
    config = _audit_config()

    # n_permutations=0: skips null → detection=False → any fl >= warn_floor → INCONCLUSIVE.
    # This keeps the integration test fast (no 100-permutation null).
    report = run_audit(df, contract, config, n_permutations=0)

    # Verdict is unchanged: annotate-only
    assert report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE", (
        f"Expected UNCONFIRMED_HIGH_DAMAGE, got {report.policy_decision.verdict}"
    )

    # Structural annotation is present
    assert len(report.structural_annotations) == 1
    ann = report.structural_annotations[0]
    assert ann.issue_type == IssueType.FORBIDDEN_ENTITY_LEVEL_AGGREGATE
    assert ann.status == "warn"
    detail = ann.evidence.structural_detail
    assert isinstance(detail, ForbiddenEntityLevelAggregateDetail)
    assert detail.feature == "entity_churn_rate"
    assert detail.within_entity_constant is True
    assert detail.between_entity_unique_count > 1


def test_integration_text_render_contains_structural_finding():
    df = _make_entity_churn_rate_dataset()
    report = run_audit(df, _audit_contract(), _audit_config(), n_permutations=0)
    from zekan.reports.text_view import render_verdict
    rendered = render_verdict(report)
    assert "STRUCTURAL FINDING" in rendered
    assert "entity_churn_rate" in rendered
    assert "constant within" in rendered


def test_integration_html_render_contains_structural_finding():
    df = _make_entity_churn_rate_dataset()
    report = run_audit(df, _audit_contract(), _audit_config(), n_permutations=0)
    from zekan.reports.html_view import render_verdict_html
    rendered = render_verdict_html(report)
    assert "STRUCTURAL FINDING" in rendered
    assert "entity_churn_rate" in rendered


def test_integration_clean_dataset_no_annotation():
    """On a dataset with no entity-level-aggregate forbidden features, no annotation fires."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "entity_id": [f"e{i}" for i in range(n)],
        "snapshot_date": ["2022-01-01"] * (n // 2) + ["2022-02-01"] * (n // 2),
        "feature_a": rng.normal(size=n),
        "future_flag": rng.integers(0, 2, size=n).astype(float),  # random, varies within entity
        "churned": rng.integers(0, 2, size=n),
    })
    # Add second period observations for entities so we have a longitudinal panel
    contract = PredictionContract(
        prediction_problem="clean-test",
        entity_id="entity_id",
        prediction_time="snapshot_date",
        target="churned",
        available_features_until="snapshot_date",
        forbidden_after_prediction=["future_flag"],
    )
    results = probe_forbidden_entity_level_aggregate(df, contract)
    assert results == [], "Random per-row feature must not trigger the aggregate probe"
