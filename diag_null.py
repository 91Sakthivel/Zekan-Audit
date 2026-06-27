"""Permutation null characterization — Step 1 of 2.

Runs: method (a) vs (b) comparison, stability check, full 10-seed per-case table.
Does NOT change reported severity bands or status logic.

Sections:
  0. Timing probe (20 permutations on Case 3, seed=42)
  1. Case 2 FIRST: drift feature declared forbidden, method (a) vs (b) comparison
  2. Stability: two independent batches of N=100, method (a), on Case 2a + Case 3
  3. Full 10-seed per-case table (method a)
"""
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import (
    inject_covariate_drift, inject_concept_drift,
    inject_future_feature, inject_correlated_leaks,
)
from zekan.config.schema import ZekanConfig, SplitPolicy
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.engine import run_severity_analysis, FIXABLE_LEAKAGE_CLEAR_LEAK
from zekan.severity.null_baseline import estimate_fixable_leakage_null

SEP = "=" * 76
SEEDS = [1, 2, 3, 7, 13, 17, 42, 99, 123, 999]
N_PERM = 100


def _clf():
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _contract(**kw):
    d = dict(
        prediction_problem="null-diag",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
    )
    d.update(kw)
    return PredictionContract(**d)


def _config(contract):
    return ZekanConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
            leak_lookahead=1,
        ),
    )


def _run(df, contract, config):
    return run_severity_analysis(df, contract, config, _clf)


def _null(df, contract, result, n_perm, seed, method="within_entity"):
    return estimate_fixable_leakage_null(
        df, contract, _config(contract), _clf,
        observed_fixable_leakage=result.fixable_leakage,
        n_permutations=n_perm, seed=seed, method=method,
    )


def _verdict(p):
    if p < 0.01:
        return "OUTSIDE (p<0.01)"
    if p < 0.05:
        return "OUTSIDE (p<0.05)"
    return "inside null"


# ── Case builders (one per case) ──────────────────────────────────────────────

def build_2a(seed):
    """Covariate drift declared forbidden — no leak, inside null."""
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_covariate_drift(b)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    cfg = _config(c)
    return d, c, run_severity_analysis(d, c, cfg, _clf)


def build_2b(seed):
    """Concept drift declared forbidden — no leak, inside null."""
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_concept_drift(b)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    cfg = _config(c)
    return d, c, run_severity_analysis(d, c, cfg, _clf)


def build_3(seed):
    """Future latent feature declared forbidden — genuine leak, outside null."""
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_future_feature(b, dataset_seed=seed)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    cfg = _config(c)
    return d, c, run_severity_analysis(d, c, cfg, _clf)


def build_4(seed):
    """Correlated leak pair declared forbidden — cumulative shared-source leak."""
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_correlated_leaks(b, dataset_seed=seed)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    cfg = _config(c)
    return d, c, run_severity_analysis(d, c, cfg, _clf)


# ── Section 0: timing probe ───────────────────────────────────────────────────
print(SEP)
print("SECTION 0: timing probe — 20 permutations on Case 3, seed=42")
print(SEP)
sys.stdout.flush()

df3_42, c3_42, r3_42 = build_3(42)
t0 = time.perf_counter()
_probe = _null(df3_42, c3_42, r3_42, n_perm=20, seed=0)
probe_elapsed = time.perf_counter() - t0

per_perm = probe_elapsed / 20
total_estimate = per_perm * N_PERM * 4 * len(SEEDS) / 60
print(f"20 permutations: {probe_elapsed:.1f}s  (~{per_perm:.2f}s per permutation)")
print(f"Projected {N_PERM} permutations:   ~{per_perm*N_PERM:.0f}s")
print(
    f"Projected {N_PERM} perm x 4 cases x {len(SEEDS)} seeds:   ~{total_estimate:.0f} minutes"
)
print()
sys.stdout.flush()


# ── Section 1: method (a) vs (b) on Case 2 + Case 3 ─────────────────────────
print(SEP)
print("SECTION 1: method (a) within_entity vs (b) target_within_period  [seed=42]")
print("  Expect: Cases 2a/2b inside null with both methods; Case 3 outside with (a).")
print("  Purpose: confirm (a) is the tighter null; (b) shown for comparison only.")
print(SEP)
sys.stdout.flush()

df2a_42, c2a_42, r2a_42 = build_2a(42)
df2b_42, c2b_42, r2b_42 = build_2b(42)
# Case 3 already built above as df3_42, c3_42, r3_42

for case_label, df_c, contract_c, result_c in [
    ("Case 2a  covariate drift forbidden", df2a_42, c2a_42, r2a_42),
    ("Case 2b  concept drift  forbidden", df2b_42, c2b_42, r2b_42),
    ("Case 3   future_z_latent forbidden", df3_42, c3_42, r3_42),
]:
    print(f"\n--- {case_label} ---")
    print(f"  observed fixable_leakage = {result_c.fixable_leakage:+.4f}")
    for method, label in [("within_entity", "(a)"), ("target_within_period", "(b)")]:
        null = _null(df_c, contract_c, result_c, n_perm=100, seed=7, method=method)
        flag = "** OUTSIDE NULL **" if null.p_value < 0.05 else "inside null"
        print(
            f"  method {label} {method:26s}: "
            f"null_med={null.null_median:+.4f}  "
            f"null_95th={null.null_95th:+.4f}  "
            f"p={null.p_value:.3f}  {flag}  "
            f"({null.elapsed_seconds:.1f}s)"
        )
    sys.stdout.flush()

print()
sys.stdout.flush()


# ── Section 2: stability — two independent batches, method (a) ───────────────
print(SEP)
print("SECTION 2: stability — two independent batches of N=100, method (a)")
print("  Goal: |batch0_95th - batch1_95th| <= ~0.005 (N=100 is stable)")
print(SEP)
sys.stdout.flush()

for case_label, df_c, contract_c, result_c in [
    ("Case 2a  covariate drift forbidden", df2a_42, c2a_42, r2a_42),
    ("Case 3   future_z_latent forbidden", df3_42, c3_42, r3_42),
]:
    print(f"\n--- {case_label} ---")
    print(f"  observed = {result_c.fixable_leakage:+.4f}")
    batch_results = []
    for batch_seed in [0, 99999]:
        null = _null(df_c, contract_c, result_c, n_perm=100, seed=batch_seed)
        batch_results.append(null)
        print(
            f"  seed={batch_seed:>6}: "
            f"med={null.null_median:+.4f}  "
            f"95th={null.null_95th:+.4f}  "
            f"p={null.p_value:.3f}  "
            f"({null.elapsed_seconds:.1f}s)"
        )
        sys.stdout.flush()
    delta = abs(batch_results[0].null_95th - batch_results[1].null_95th)
    verdict_str = "STABLE" if delta <= 0.005 else "UNSTABLE — increase N"
    print(f"  |batch0_95th - batch1_95th| = {delta:.4f}  {verdict_str}")

print()
sys.stdout.flush()


# ── Section 3: full 10-seed per-case table (method a) ────────────────────────
print(SEP)
print(f"SECTION 3: full 10-seed table  method=(a) within_entity  N={N_PERM}")
print("  Seeds:", SEEDS)
print("  Cases: 2a (cov-drift), 2b (concept-drift), 3 (future-feat), 4 (corr-pair)")
print(SEP)
sys.stdout.flush()

header = (
    f"{'seed':>5}  {'observed':>9}  {'null_med':>9}  {'null_95th':>9}  "
    f"{'p_value':>8}  result"
)

CASES = [
    ("Case 2a  covariate drift declared forbidden", build_2a),
    ("Case 2b  concept drift declared forbidden",   build_2b),
    ("Case 3   future_z_latent forbidden",          build_3),
    ("Case 4   corr-leak pair forbidden",           build_4),
]

for case_label, builder in CASES:
    print(f"\n--- {case_label} ---")
    print(header)
    sys.stdout.flush()
    for seed in SEEDS:
        df_c, contract_c, result_c = builder(seed)
        null = _null(df_c, contract_c, result_c, n_perm=N_PERM, seed=seed)
        p = null.p_value
        print(
            f"{seed:>5}  {result_c.fixable_leakage:>+9.4f}  "
            f"{null.null_median:>+9.4f}  {null.null_95th:>+9.4f}  "
            f"{p:>8.3f}  {_verdict(p)}"
        )
        sys.stdout.flush()

print()
print(SEP)
print("SECTION 3 complete.")
print("  Expected: Cases 2a/2b all inside null (p>=0.05) across 10 seeds")
print("  Expected: Case 3 all outside null (p<0.05) across 10 seeds")
print("  Expected: Case 4 outside null — understatement pattern")
print(SEP)
