"""NEAR_BIJECTION_UNDECLARED_LEAK probe -- deterministic value-to-label partition.

Upgrade (H). Locked design per UPGRADE_H_PREREGISTRATION.md /
UPGRADE_H_CALIBRATION.md: for every non-forbidden candidate feature, compute
Theil's U (the uncertainty coefficient) -- (H(Y) - H(Y|X)) / H(Y) -- whole-
frame, by counting a value/label joint frequency table. No estimator, no
folds, no RNG: this is a deterministic counting fact about the data, not a
model score -- the same class of check as FORBIDDEN_ENTITY_LEVEL_AGGREGATE
(DETECTED_STRUCTURAL, DATA_ONLY, confirmed=True), NOT Upgrade 1's
FLAGGED_SUSPICIOUS/confirmed=False screen (undeclared_feature_probe.py).

Candidate set: reuses contract.contract_checks.candidate_features -- a
CONTRACT concept (which columns are non-forbidden features under a given
contract) shared with Upgrade 1's screen, owned by neither probe module.

Support floor and criterion, both derived in UPGRADE_H_CALIBRATION.md, not
re-derived here -- see SUPPORT_FLOOR / CRITERION below for the exact
evidence each constant rests on. No screenability gate and no wide-data cap:
every candidate is scored (screened_count == total_features always) --
the calibration's own cardinality-ceiling investigation found a ratio-based
cap to be n-sensitive (wrongly excludes a real feature like diag_1 at small
n) and the support floor alone already carries the guard's entire measured
protective value against the ID trap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zekan.contract.contract_checks import candidate_features
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import (
    Evidence,
    IssueRecord,
    IssueType,
    NearBijectionUndeclaredLeakDetail,
)

# Support floor: values with count < SUPPORT_FLOOR are pooled into ONE 'rare'
# bucket before scoring. Derived in UPGRADE_H_CALIBRATION.md's "SUPPORT FLOOR
# RE-DERIVATION" addendum: of the six candidate floors tested {2,5,10,20,50,
# 100}, all six tied on both hard requirements (an independent feature's
# worst-case chance-inflation across every tested K stays far below CRITERION,
# and encounter_id collapses to exactly U=0.0 in every frame) -- the lowest,
# 2, was recommended because a higher floor buys no additional protection
# against CRITERION while steadily eroding real diag_1/diag_2/diag_3 signal.
# The "NEAR-ID ANCHOR: patient_nbr" addendum then stress-tested floor=2
# against a real near-ID column that (unlike encounter_id) is NOT fully
# absorbed by pooling -- it stayed at U=0.353, a 0.637 margin below CRITERION.
SUPPORT_FLOOR: int = 2

# Near-bijection criterion: a feature clears this after pooling only when its
# value determines the target almost exactly. Sits deep inside the empty gap
# UPGRADE_H_CALIBRATION.md measured between the honest ceiling (~0.03, raw)
# and the leak anchors (planted_leak=0.6417, readmitted=1.0) -- see that
# document's "Measured margin" section.
CRITERION: float = 0.99

_NAN_SENTINEL = "__NaN__"
_RARE_SENTINEL = "__RARE__"


def _binary_entropy(n1: float, n0: float) -> float:
    n = n1 + n0
    if n <= 0:
        return 0.0
    h = 0.0
    for n_k in (n1, n0):
        if n_k > 0:
            p = n_k / n
            h -= p * np.log2(p)
    return h


def _score_feature(x: pd.Series, y: pd.Series, support_floor: int) -> dict:
    """Whole-frame Theil's U for one feature, after rare-value pooling.

    NaN is treated as its own explicit category, not dropped -- missingness
    itself can correlate with the target; dropping rows would erase exactly
    the signal this check should be able to see (same rationale recorded in
    UPGRADE_H_CALIBRATION.md's Method notes -- not exercised there either,
    since no Test B column carries a real NaN, but the choice is deliberate,
    not an oversight).
    """
    n_total = len(y)
    n1_total = int(y.sum())
    n0_total = n_total - n1_total
    h_y = _binary_entropy(n1_total, n0_total)

    x_cat = x.astype("object").where(x.notna(), _NAN_SENTINEL)
    table = (
        pd.DataFrame({"x": x_cat, "y": y})
        .groupby("x", dropna=False)["y"]
        .agg(n1="sum", count="count")
    )
    table["n0"] = table["count"] - table["n1"]

    n_distinct_raw = len(table)

    rare_mask = table["count"] < support_floor
    n_rows_pooled = int(table.loc[rare_mask, "count"].sum())
    if rare_mask.any():
        kept = table.loc[~rare_mask]
        rare_n1 = table.loc[rare_mask, "n1"].sum()
        rare_n0 = table.loc[rare_mask, "n0"].sum()
        rare_row = pd.DataFrame(
            {"n1": [rare_n1], "count": [n_rows_pooled], "n0": [rare_n0]},
            index=[_RARE_SENTINEL],
        )
        pooled_table = pd.concat([kept, rare_row])
    else:
        pooled_table = table

    n_distinct_after_pooling = len(pooled_table)

    if h_y <= 0:
        theil_u = float("nan")
    else:
        weights = pooled_table["count"].to_numpy() / n_total
        ents = np.array([_binary_entropy(r.n1, r.n0) for r in pooled_table.itertuples()])
        h_yx = float(np.sum(weights * ents))
        theil_u = (h_y - h_yx) / h_y

    return dict(
        theil_u=theil_u,
        n_distinct_raw=n_distinct_raw,
        n_distinct_after_pooling=n_distinct_after_pooling,
        n_rows_pooled=n_rows_pooled,
    )


def probe_near_bijection(
    df: pd.DataFrame,
    contract: PredictionContract,
) -> list[IssueRecord]:
    """Structural probe: do any non-forbidden features nearly perfectly
    partition the target (Theil's U >= CRITERION, after rare-value pooling)?

    Returns one IssueRecord per feature that clears the criterion. Returns an
    empty list when there are no candidate features or none clear it.
    Deterministic and annotate-only -- never changes policy_decision.verdict.
    """
    candidates = candidate_features(contract, df)
    total_features = len(candidates)
    if total_features == 0:
        return []

    y = df[contract.target].astype(int)

    findings: list[IssueRecord] = []
    for feature in candidates:
        scored = _score_feature(df[feature], y, SUPPORT_FLOOR)
        theil_u = scored["theil_u"]
        if not (theil_u >= CRITERION):
            continue

        detail = NearBijectionUndeclaredLeakDetail(
            feature=feature,
            theil_u=theil_u,
            support_floor=SUPPORT_FLOOR,
            threshold_compared_against=CRITERION,
            n_distinct_raw=scored["n_distinct_raw"],
            n_distinct_after_pooling=scored["n_distinct_after_pooling"],
            n_rows_pooled=scored["n_rows_pooled"],
            screened_count=total_features,
            total_features=total_features,
        )

        findings.append(IssueRecord(
            issue_type=IssueType.NEAR_BIJECTION_UNDECLARED_LEAK,
            status="fail",
            what=(
                f"Feature '{feature}' scores Theil's U={theil_u:.4f} after pooling "
                f"(support floor={SUPPORT_FLOOR}) -- its values determine the target "
                "almost exactly. Not a declared forbidden feature."
            ),
            why=(
                "Theil's U measures how much knowing this feature's value reduces "
                "uncertainty about the target, relative to the target's own base rate "
                "-- unlike raw purity, it does not reward a feature for simply "
                "predicting the majority class. A value this close to a deterministic "
                f"partition (U >= {CRITERION:.2f}) is a fact about how the data is "
                "arranged, not a model's inference -- see UPGRADE_H_CALIBRATION.md for "
                "the measured honest-tail ceiling and leak anchors this criterion was "
                "derived against."
            ),
            how_much=(
                f"theil_u={theil_u:.4f} (support_floor={SUPPORT_FLOOR}, "
                f"criterion={CRITERION:.2f}); {scored['n_distinct_raw']} distinct "
                f"value(s) before pooling, {scored['n_distinct_after_pooling']} after; "
                f"{scored['n_rows_pooled']} row(s) pooled into the rare bucket; "
                f"screened {total_features} of {total_features} non-forbidden "
                "features this run."
            ),
            next_fix=(
                f"Inspect '{feature}': if it encodes the target directly or "
                "indirectly (e.g. a near-duplicate column, or derived from "
                "post-outcome data), add it to the contract's "
                "forbidden_after_prediction and re-run."
            ),
            evidence=Evidence(
                measured_value=theil_u,
                threshold=CRITERION,
                metric_name="theil_u",
                structural_detail=detail,
            ),
        ))

    return findings
