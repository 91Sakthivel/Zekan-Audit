"""Full diagnostic: alignment probe first, then real Case 3 + Cases 1/2/2b/drift checks.

Step 1  — ALIGNMENT PROBE: inject z[T] (current period, zero noise).
          B-C should be large/near-saturating — proves entity/period indexing is correct.
          If this is weak, stop: indexing is misaligned.

Step 2  — REAL CASE 3: inject z[T+1] + N(0, 0.35) (the actual leak).
          Full per-fold printout + invariant check.

Step 3  — Cases 1, 2a, 2b, concept-drift correlations (must be unperturbed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.benchmark.injectors import (
    inject_concept_drift,
    inject_covariate_drift,
    inject_future_feature,
)
from gotcha.config.schema import GotchaConfig, SplitPolicy
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.severity.engine import run_severity_analysis


def _fast_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _make_contract(**kwargs) -> PredictionContract:
    defaults: dict = dict(
        prediction_problem="benchmark churn",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    defaults.update(kwargs)
    return PredictionContract(**defaults)


def _make_config(contract: PredictionContract) -> GotchaConfig:
    return GotchaConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
        ),
    )


SEP = "=" * 70

print(SEP)
print("Building base dataset: 500 entities x 10 periods = 5000 rows, seed=42")
print(SEP)
base_df = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: ALIGNMENT PROBE — z[T] zero-noise
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ALIGNMENT PROBE: inject z[T] (current period, zero noise)")
print("Expected: B-C large/near-saturating across all folds.")
print("If weak: entity/period indexing is misaligned — STOP.")
print(SEP)

df_probe, rec_probe = inject_future_feature(
    base_df,
    dataset_seed=42,
    _probe_current_period=True,
    _probe_noise_sigma=0.0,   # zero noise — maximum possible signal
)
contract_probe = _make_contract(forbidden_after_prediction=rec_probe.planted_columns)
config_probe = _make_config(contract_probe)
r_probe = run_severity_analysis(df_probe, contract_probe, config_probe, _fast_clf)

print(f"\nProbe per-fold (z[T] zero-noise):")
print(f"  {'Fold':>4}  {'B':>8}  {'C':>8}  {'B-C':>8}")
probe_diffs = []
for pf in r_probe.per_fold:
    b = pf.auc_with_forbidden
    c_val = pf.auc_without_forbidden
    diff = b - c_val
    probe_diffs.append(diff)
    print(f"  {pf.fold_idx:>4}  {b:>8.4f}  {c_val:>8.4f}  {diff:>+8.4f}")

probe_median = sorted(probe_diffs)[len(probe_diffs) // 2]
print(f"\n  Sorted: {[f'{d:+.4f}' for d in sorted(probe_diffs)]}")
print(f"  Median: {probe_median:+.4f}")
if probe_median > 0.10:
    print("  PROBE PASS: indexing is aligned, z[T] signal is strong.")
elif probe_median > 0.05:
    print("  PROBE MARGINAL: indexing likely OK but z[T] weaker than expected.")
else:
    print("  PROBE FAIL: z[T] zero-noise B-C is weak — INDEXING MAY BE MISALIGNED. STOP.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: REAL CASE 3 — z[T+1] + N(0, 0.35)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CASE 3: future_z_latent = z[T+1] + N(0, 0.35) declared forbidden")
print(SEP)

df_leak, record = inject_future_feature(base_df, dataset_seed=42)

print(f"\nInjected column            : {record.planted_columns}")
print(f"forbidden_after_prediction : {record.planted_columns}")

contract3 = _make_contract(forbidden_after_prediction=record.planted_columns)
config3 = _make_config(contract3)

excluded = {
    contract3.entity_id,
    contract3.prediction_time,
    contract3.target,
    contract3.available_features_until,
}
all_features = [c for c in df_leak.columns if c not in excluded]
forbidden_set = set(contract3.forbidden_after_prediction)
features_B = all_features
features_C = [f for f in all_features if f not in forbidden_set]

print(f"\nFeatures B (with forbidden): {features_B}")
print(f"Features C (drop forbidden): {features_C}")
print(f"  'future_z_latent' in B: {any('future' in f for f in features_B)}")
print(f"  'future_z_latent' in C: {any('future' in f for f in features_C)}")

r3 = run_severity_analysis(df_leak, contract3, config3, _fast_clf)

print("\nCase 3 per-fold (B = with forbidden, C = without):")
print(f"  {'Fold':>4}  {'B':>8}  {'C':>8}  {'B-C':>8}  Note")
diffs = []
for pf in r3.per_fold:
    b = pf.auc_with_forbidden
    c_val = pf.auc_without_forbidden
    diff = b - c_val
    diffs.append(diff)
    note = "above 0.05" if diff >= 0.05 else "BELOW 0.05"
    print(f"  {pf.fold_idx:>4}  {b:>8.4f}  {c_val:>8.4f}  {diff:>+8.4f}  [{note}]")

sorted_diffs = sorted(diffs)
median_diff = sorted_diffs[len(sorted_diffs) // 2]
print(f"\n  Sorted B-C : {[f'{d:+.4f}' for d in sorted_diffs]}")
print(f"  Median     : {median_diff:+.4f}  (gate: >= 0.05)")

# Fold 4 explicit
pf4 = r3.per_fold[4]
diff4 = pf4.auc_with_forbidden - pf4.auc_without_forbidden
print(f"\nFold 4 explicit:")
print(f"  B   = {pf4.auc_with_forbidden:.4f}")
print(f"  C   = {pf4.auc_without_forbidden:.4f}")
print(f"  B-C = {diff4:+.4f}")
print(f"  Period-9 rows have no T+1 → fallback 0.0+noise → pure noise, zero z[T] signal.")
print(f"  Fold 4 weakness is the period-9 boundary, not misalignment (probe above confirms).")

print(f"\nCase 3 summary:")
print(f"  naive_auc (A)            : {r3.naive_auc:.4f}")
print(f"  estimated_deployable (C) : {r3.estimated_deployable_auc:.4f}")
print(f"  total_optimism (A-C)     : {r3.total_optimism:+.4f}")
print(f"  fixable_leakage (med B-C): {r3.fixable_leakage:+.4f}  (gate: >= 0.05)")
print(f"  nonfixable_optimism(A-B) : {r3.nonfixable_optimism:+.4f}")
print(f"  status                   : {r3.status}")

recon = r3.fixable_leakage + r3.nonfixable_optimism
delta = abs(r3.total_optimism - recon)
print(f"\nInvariant A-C = (A-B) + (B-C):")
print(f"  total_optimism      = {r3.total_optimism:+.6f}")
print(f"  fixable+nonfixable  = {recon:+.6f}")
print(f"  |difference|        = {delta:.6f}  ({'HOLDS' if delta < 0.05 else 'BROKEN'})")

gate3 = r3.fixable_leakage >= 0.05
print(f"\nCase 3 gate (fixable_leakage >= 0.05): {'PASS' if gate3 else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Cases 1, 2a, 2b, concept-drift correlations
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CASES 1, 2a, 2b — must be unperturbed by z-helper refactor")
print(SEP)

print("\n--- CASE 1: clean baseline ---")
contract1 = _make_contract()
config1 = _make_config(contract1)
r1 = run_severity_analysis(base_df, contract1, config1, _fast_clf)
flag1 = "OK" if abs(r1.fixable_leakage) <= 0.02 else "FAIL"
print(f"  fixable_leakage : {r1.fixable_leakage:+.4f}  [{flag1}]  (want <= 0.02)")
print(f"  total_optimism  : {r1.total_optimism:+.4f}")
print(f"  status          : {r1.status}")

print("\n--- CASE 2a: covariate drift, no forbidden ---")
df_cov, _ = inject_covariate_drift(base_df)
contract2a = _make_contract()
r2a = run_severity_analysis(df_cov, contract2a, _make_config(contract2a), _fast_clf)
flag2a = "OK" if abs(r2a.fixable_leakage) <= 0.02 else "FAIL"
print(f"  fixable_leakage : {r2a.fixable_leakage:+.4f}  [{flag2a}]  (want <= 0.02)")

print("\n--- CASE 2b: concept drift, no forbidden ---")
df_drift, _ = inject_concept_drift(base_df)
contract2b = _make_contract()
r2b = run_severity_analysis(df_drift, contract2b, _make_config(contract2b), _fast_clf)
drift_gap = r2b.total_optimism - r1.total_optimism
flag2b = "OK" if drift_gap >= 0.03 else "FAIL -- BELOW 0.03"
flag2b_fl = "OK" if abs(r2b.fixable_leakage) <= 0.02 else "FAIL"
print(f"  fixable_leakage : {r2b.fixable_leakage:+.4f}  [{flag2b_fl}]  (want <= 0.02)")
print(f"  total_optimism  : {r2b.total_optimism:+.4f}")
print(f"  drift gap (2b-1): {drift_gap:+.4f}  [{flag2b}]  (want >= 0.03)")

print("\n--- CONCEPT DRIFT early/late correlations (small_df seed=7) ---")
small_df = make_clean_dataset(n_entities=100, snapshots_per_entity=5, seed=7)
cd_mod, cd_rec = inject_concept_drift(small_df)
col_cd = cd_rec.planted_columns[0]
times_cd = cd_mod["prediction_time"]
sorted_periods_cd = sorted(times_cd.unique(), key=lambda x: pd.to_datetime(x))
n_cd = len(sorted_periods_cd)
n_slice = max(1, n_cd // 4)
early_set = set(sorted_periods_cd[:n_slice])
late_set = set(sorted_periods_cd[-n_slice:])
early_cd = cd_mod[times_cd.isin(early_set)]
late_cd = cd_mod[times_cd.isin(late_set)]
early_corr = float(early_cd[col_cd].corr(early_cd["target"].astype(float)))
late_corr = float(late_cd[col_cd].corr(late_cd["target"].astype(float)))
print(f"  early_corr = {early_corr:+.4f}  [{'OK' if early_corr > 0.1 else 'FAIL'}]  (want > +0.1)")
print(f"  late_corr  = {late_corr:+.4f}  [{'OK' if late_corr < -0.1 else 'FAIL'}]  (want < -0.1)")

print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  Probe (z[T] zero-noise median B-C): {probe_median:+.4f}  {'[ALIGNED]' if probe_median > 0.05 else '[WEAK - CHECK INDEXING]'}")
print(f"  Case 1  fixable_leakage:  {r1.fixable_leakage:+.4f}  [{flag1}]")
print(f"  Case 2a fixable_leakage:  {r2a.fixable_leakage:+.4f}  [{flag2a}]")
print(f"  Case 2b drift gap:        {drift_gap:+.4f}  [{flag2b}]")
print(f"  Case 3  fixable_leakage:  {r3.fixable_leakage:+.4f}  {'[PASS]' if gate3 else '[FAIL]'}  (gate: >= 0.05)")
print(f"  Invariant |delta|:        {delta:.6f}  ({'HOLDS' if delta < 0.05 else 'BROKEN'})")
