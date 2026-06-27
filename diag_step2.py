"""Step-2 proof benchmark — null-standardized verdict on 4 cases × 10 seeds.

Sections
--------
0. Timing probe (N=20 on Case 3 seed=42)
1. Verdict table: 2a, 2b, 3, 4 × 10 seeds
   Columns: seed | observed | null_med | null_99th | null_IQR | p | NSL_IQR | NSL_med | old_band | status
2. NSL stability for Cases 3 and 4 (IQR vs median-based denominator)
3. Case-4 before/after: null_99th > 0.04 seeds prove the fixed band is wrong
4. Summary: drift must PASS; leak cases graded; NSL cutoff check
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
from zekan.severity.engine import (
    _NSL_NOTE_THRESHOLD, _NSL_WARN_THRESHOLD, _NULL_ALPHA, _NSL_EPS,
    FIXABLE_LEAKAGE_CLEAR_LEAK,
    leakage_issue_record, run_severity_analysis,
)

SEP = "=" * 84
SEEDS = [1, 2, 3, 7, 13, 17, 42, 99, 123, 999]
N_PERM = 100


def _clf():
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _contract(**kw):
    d = dict(
        prediction_problem="step2-diag",
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


def build_2a(seed):
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_covariate_drift(b)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    return d, c, _config(c)


def build_2b(seed):
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_concept_drift(b)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    return d, c, _config(c)


def build_3(seed):
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_future_feature(b, dataset_seed=seed)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    return d, c, _config(c)


def build_4(seed):
    b = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    d, r = inject_correlated_leaks(b, dataset_seed=seed)
    c = _contract(forbidden_after_prediction=r.planted_columns)
    return d, c, _config(c)


def _nsl_med(result):
    """NSL with (null_99th - null_median) denominator — computed from stored fields."""
    if result.null_99th is None or result.null_median is None:
        return float("nan")
    denom = max(result.null_99th - result.null_median, _NSL_EPS)
    return (result.fixable_leakage - result.null_99th) / denom


# ── Section 0: timing probe ───────────────────────────────────────────────────
print(SEP)
print("SECTION 0: timing probe — N=20 on Case 3, seed=42")
print(SEP)
sys.stdout.flush()

_df3, _c3, _cfg3 = build_3(42)
t0 = time.perf_counter()
_r_probe = run_severity_analysis(_df3, _c3, _cfg3, _clf, n_permutations=20, null_seed=0)
probe_s = time.perf_counter() - t0
per_perm = probe_s / 20
print(f"  20 perms: {probe_s:.1f}s  (~{per_perm:.2f}s/perm)")
print(f"  Projected {N_PERM} perm × 4 cases × {len(SEEDS)} seeds: ~{per_perm*N_PERM*4*len(SEEDS)/60:.0f} min")
print()
sys.stdout.flush()


# ── Section 1: Verdict table ──────────────────────────────────────────────────
HEADER = (
    f"{'seed':>5}  {'observed':>9}  {'null_med':>9}  {'null_99th':>9}  "
    f"{'null_IQR':>9}  {'p':>7}  {'NSL_IQR':>8}  {'NSL_med':>8}  "
    f"{'old_band':>8}  status"
)

print(SEP)
print("SECTION 1: Verdict table — null-standardized (IQR denominator, alpha=0.01)")
print(f"  ladder: p>={_NULL_ALPHA}->PASS | p<{_NULL_ALPHA}: NSL<{_NSL_NOTE_THRESHOLD}->NOTE "
      f"| NSL<{_NSL_WARN_THRESHOLD}->WARN | else->FAIL")
print(f"  old_band = FAIL if observed >= {FIXABLE_LEAKAGE_CLEAR_LEAK}")
print(SEP)

CASES = [
    ("Case 2a  covariate drift",  build_2a, "2a"),
    ("Case 2b  concept drift",    build_2b, "2b"),
    ("Case 3   future_z_latent",  build_3,  "3"),
    ("Case 4   corr-leak pair",   build_4,  "4"),
]

all_rows: dict[str, list[dict]] = {}

for case_label, builder, tag in CASES:
    print(f"\n--- {case_label} ---")
    print(HEADER)
    sys.stdout.flush()
    rows = []
    for seed in SEEDS:
        df, c, cfg = builder(seed)
        result = run_severity_analysis(df, c, cfg, _clf, n_permutations=N_PERM, null_seed=seed)
        status = leakage_issue_record(result).status
        old_band = "FAIL" if result.fixable_leakage >= FIXABLE_LEAKAGE_CLEAR_LEAK else "pass"
        nsl_iqr = result.nsl
        nsl_m = _nsl_med(result)
        row = dict(
            seed=seed, obs=result.fixable_leakage,
            null_med=result.null_median, null_99th=result.null_99th,
            null_iqr=result.null_iqr, p=result.p_value,
            nsl_iqr=nsl_iqr, nsl_med=nsl_m, status=status, old_band=old_band,
        )
        rows.append(row)
        nsl_iqr_s = f"{nsl_iqr:>8.2f}" if nsl_iqr is not None else "     n/a"
        nsl_med_s = f"{nsl_m:>8.2f}" if not np.isnan(nsl_m) else "     n/a"
        print(
            f"{seed:>5}  {result.fixable_leakage:>+9.4f}  "
            f"{result.null_median:>+9.4f}  {result.null_99th:>+9.4f}  "
            f"{result.null_iqr:>9.4f}  {result.p_value:>7.4f}  "
            f"{nsl_iqr_s}  {nsl_med_s}  {old_band:>8}  {status}"
        )
        sys.stdout.flush()
    all_rows[tag] = rows

print()
sys.stdout.flush()


# ── Section 2: NSL stability ──────────────────────────────────────────────────
print(SEP)
print("SECTION 2: NSL stability — Cases 3 and 4 (IQR vs median-based denominator)")
print("  Goal: STATUS column stable across 10 seeds; exact NSL value may vary.")
print(SEP)

for tag, label in [("3", "Case 3  future_z_latent"), ("4", "Case 4  corr-leak pair")]:
    rows = all_rows[tag]
    nsl_iqr_v = [r["nsl_iqr"] for r in rows if r["nsl_iqr"] is not None]
    nsl_med_v = [r["nsl_med"] for r in rows if not np.isnan(r["nsl_med"])]
    statuses = [r["status"] for r in rows]
    ps = [r["p"] for r in rows]

    print(f"\n  {label}")
    print(f"    p-values:   min={min(ps):.4f}  max={max(ps):.4f}  "
          f"n_outside_null(p<{_NULL_ALPHA})={sum(1 for p in ps if p < _NULL_ALPHA)}/10")
    if nsl_iqr_v:
        print(f"    NSL_IQR:  min={min(nsl_iqr_v):+.2f}  max={max(nsl_iqr_v):+.2f}  "
              f"std={float(np.std(nsl_iqr_v)):.3f}"
              f"  [all values: {[f'{v:.2f}' for v in nsl_iqr_v]}]")
    if nsl_med_v:
        print(f"    NSL_med:  min={min(nsl_med_v):+.2f}  max={max(nsl_med_v):+.2f}  "
              f"std={float(np.std(nsl_med_v)):.3f}"
              f"  [all values: {[f'{v:.2f}' for v in nsl_med_v]}]")
    counts = dict(zip(*np.unique(statuses, return_counts=True)))
    print(f"    STATUS:   {counts}")
    stable = len(set(statuses)) == 1
    print(f"    STABLE:   {'YES' if stable else 'NO — status oscillates across seeds'}")
    # Check if NSL_med has higher std than NSL_IQR (justifies IQR choice)
    if nsl_iqr_v and nsl_med_v:
        if np.std(nsl_med_v) > np.std(nsl_iqr_v) * 1.5:
            print(f"    NOTE: NSL_med std ({np.std(nsl_med_v):.3f}) > 1.5× NSL_IQR std "
                  f"({np.std(nsl_iqr_v):.3f}) — IQR denominator is more stable.")

print()
sys.stdout.flush()


# ── Section 3: Case-4 before/after ────────────────────────────────────────────
print(SEP)
print("SECTION 3: Case-4 before/after — proof that fixed 0.04 band is wrong")
print(f"  The per-dataset null_99th for Case 4 may exceed {FIXABLE_LEAKAGE_CLEAR_LEAK}.")
print(f"  When null_99th > {FIXABLE_LEAKAGE_CLEAR_LEAK}: the null itself produces values above")
print(f"  the old threshold, so any observed value in [0.04, null_99th] is inside")
print(f"  the null distribution — old band FAILs; null-standardized PASSes.")
print(SEP)

rows4 = all_rows["4"]
print(f"\n  Case 4: null_99th distribution across 10 seeds:")
print(f"  {'seed':>5}  {'observed':>9}  {'null_99th':>9}  {'p':>7}  {'old_band':>8}  {'new_status'}")
wide = []
for r in rows4:
    n99 = r["null_99th"]
    old = r["old_band"]
    new = r["status"]
    flag = ""
    if n99 is not None and n99 > FIXABLE_LEAKAGE_CLEAR_LEAK:
        flag = "  <- null_99th > 0.04"
        wide.append(r)
    print(f"  {r['seed']:>5}  {r['obs']:>+9.4f}  {n99:>+9.4f}  "
          f"{r['p']:>7.4f}  {old:>8}  {new}{flag}")

print()
if wide:
    print(f"  {len(wide)} seeds have null_99th > {FIXABLE_LEAKAGE_CLEAR_LEAK}.")
    print(f"  These seeds demonstrate: the per-dataset noise floor for Case 4 EXCEEDS")
    print(f"  the old global 0.04 threshold.  A fixed band cannot distinguish 'inside")
    print(f"  Case-4 null' from 'genuine leakage' on these seeds; the per-dataset null can.")
    changed_verdict = [r for r in wide if r["old_band"] == "FAIL" and r["status"] != "fail"]
    if changed_verdict:
        print(f"\n  Seeds where old_band=FAIL and null verdict differs:")
        for r in changed_verdict:
            print(f"    seed={r['seed']}  observed={r['obs']:+.4f}  null_99th={r['null_99th']:+.4f}  "
                  f"p={r['p']:.4f}  old=FAIL  new={r['status']}")
    else:
        print(f"\n  On these seeds the observed value is also above null_99th (p<0.01),")
        print(f"  so both old and new give a 'fail'/'warn'/'note' verdict.  The structural")
        print(f"  point remains: null_99th > 0.04 proves the 0.04 floor is dataset-specific,")
        print(f"  not universal.  A hypothetical observed value of 0.041 with null_99th=0.06")
        print(f"  would be: old_band=FAIL, null=PASS (p>0.01).")
else:
    print(f"  All Case-4 null_99th values are <= {FIXABLE_LEAKAGE_CLEAR_LEAK} on these seeds.")
    print(f"  This means the IQR-based null is narrow for Case 4 — the upgrade still matters")
    print(f"  because the null shape varies per dataset; a universal 0.04 cannot adapt.")

print()
sys.stdout.flush()


# ── Section 4: Summary ────────────────────────────────────────────────────────
print(SEP)
print("SECTION 4: Summary")
print(SEP)

for tag, label, must_pass in [("2a", "Case 2a  covariate drift", True),
                                ("2b", "Case 2b  concept drift",   True)]:
    rows = all_rows[tag]
    statuses = [r["status"] for r in rows]
    all_pass = all(s == "pass" for s in statuses)
    neg_obs = [r for r in rows if r["obs"] < 0]
    print(f"\n  {label}")
    print(f"    Statuses: {dict(zip(*np.unique(statuses, return_counts=True)))}")
    print(f"    All PASS: {'YES' if all_pass else 'NO -- REGRESSION'}")
    if neg_obs:
        neg_statuses = [r["status"] for r in neg_obs]
        print(f"    {len(neg_obs)} seeds have negative fixable_leakage -> statuses: {neg_statuses} (must all be pass)")

for tag, label in [("3", "Case 3  future_z_latent"), ("4", "Case 4  corr-leak pair")]:
    rows = all_rows[tag]
    statuses = [r["status"] for r in rows]
    print(f"\n  {label}")
    print(f"    Statuses: {dict(zip(*np.unique(statuses, return_counts=True)))}")
    n_pass = sum(1 for s in statuses if s == "pass")
    if n_pass:
        print(f"    WARNING: {n_pass} seeds PASS — check whether these are genuine false-negatives")
        print(f"    or whether the NSL cutoffs need adjustment")
    # Check if NSL cutoffs (1.0, 2.0) actually separate the cases
    outside_null = [r for r in rows if r["p"] is not None and r["p"] < _NULL_ALPHA]
    if outside_null:
        nsl_v = [r["nsl_iqr"] for r in outside_null if r["nsl_iqr"] is not None]
        print(f"    NSL_IQR for outside-null seeds: {[f'{v:.2f}' for v in nsl_v]}")
        below_note = sum(1 for v in nsl_v if v < _NSL_NOTE_THRESHOLD)
        below_warn = sum(1 for v in nsl_v if _NSL_NOTE_THRESHOLD <= v < _NSL_WARN_THRESHOLD)
        above_fail = sum(1 for v in nsl_v if v >= _NSL_WARN_THRESHOLD)
        print(f"    NOTE(NSL<{_NSL_NOTE_THRESHOLD})={below_note}  WARN({_NSL_NOTE_THRESHOLD}<=NSL<{_NSL_WARN_THRESHOLD})={below_warn}  FAIL(NSL>={_NSL_WARN_THRESHOLD})={above_fail}")
        if below_note > 0:
            print(f"    Cutoff suggestion: if {label} should be WARN minimum,")
            print(f"      lower _NSL_NOTE_THRESHOLD below min observed NSL = {min(nsl_v):.2f}")

print()
print(SEP)
print("diag_step2 complete.")
print(SEP)
