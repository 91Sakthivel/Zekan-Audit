#!/usr/bin/env python3
"""NSL boundary calibration sweep -- does NSL=2.0 mark a natural WARN/FAIL line?

WHAT THIS DOES
--------------
Injects a graded future leak (forbidden = alpha * z[T+1] + noise) across 5
signal-strength levels with 5 noise-seed replications each.  Runs the full
permutation null (n_permutations=100) at every (alpha, seed) point to obtain
the actual NSL value the production verdict ladder would assign.

n_entities=2000 (20k rows) matches the default make_clean_dataset scale.
SplitPolicy (min_test_rows=50, min_pos/neg=10) and n_estimators=30 match
the benchmark scripts (diag_case3.py, diag_step2.py) so results are directly
comparable to the known n=500 baseline (min NSL ~1.48 at full-strength leak).

PRIMARY DIAGNOSTIC -- null tightening at scale
-----------------------------------------------
fl saturates with the AR(1) rho=0.80 channel.  The lever at larger n is the
null: null_99th moves toward null_median, null_iqr shrinks, so
(fl - null_99th) / null_iqr can rise even with fl flat.  The per-run
null-geometry columns (null_median, q99-med, fl-q99, iqr) track this directly.

DISCIPLINE
----------
* Alpha levels and seeds are fixed before any results are observed.
* The verdict is read off the gradient, never tuned onto it.
* fix-the-leak-not-the-gate applies.
* NSL >= 1.0 is the earned engine boundary.
* NSL >= 2.0 stays a policy default -- do not move it until the data says so.

INVOCATION
----------
    python -m gotcha.benchmark.nsl_boundary_sweep [--output PATH]
or
    python gotcha/benchmark/nsl_boundary_sweep.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from gotcha.benchmark.fixtures import make_clean_dataset
from gotcha.benchmark.injectors import inject_graded_future_leak
from gotcha.config.schema import GotchaConfig, SplitPolicy
from gotcha.contract.prediction_contract import PredictionContract
from gotcha.severity.engine import leakage_issue_record, run_severity_analysis


# ---- Sweep parameters (fixed before any results are observed) ----------------

# 5 key alpha levels spanning the expected NSL 0->2+ gradient.
# SNR = alpha * 0.7 / 0.35 = 2 * alpha.
ALPHA_LEVELS: list[float] = [0.0, 0.60, 1.10, 1.60, 2.50]

# 5 injector seeds per level -> 5 noise realizations.
INJECTOR_SEEDS: list[int] = [0, 1, 2, 3, 4]

# Full permutation null (100 draws) for stable q99 and IQR estimates.
N_PERMUTATIONS: int = 100

# Dataset: 2000 entities x 10 periods = 20 000 rows.
# row_count_and_folds: 20000 >> 1000 -> PASS.
# temporal_periods_count: 10 >= 6 -> PASS.
# SplitPolicy matches benchmark (diag_case3.py): min_test_rows=50, min_pos/neg=10.
N_ENTITIES: int = 2000
DATASET_SEED: int = 0

# Noise floor: 0.35 matches inject_future_feature(window_size=1) exactly.
# SNR = alpha * sigma_z / noise_sigma = alpha * 0.7 / 0.35 = 2 * alpha.
NOISE_SIGMA: float = 0.35

# AR(1) rho locked in make_clean_dataset / _make_z_mat -- reported as constant column.
RHO: float = 0.80


# ---- Helpers -----------------------------------------------------------------

def _clf() -> RandomForestClassifier:
    # n_estimators=30 matches benchmark scripts (diag_case3.py, diag_step2.py).
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="nsl-boundary-sweep",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["graded_future_leak"],
    )


def _config(contract: PredictionContract) -> GotchaConfig:
    # Benchmark-aligned split policy.
    return GotchaConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
        ),
    )


# ---- Result record -----------------------------------------------------------

@dataclass
class SweepRow:
    alpha: float
    seed: int
    snr: float
    fixable_leakage: float
    null_median: float
    null_99th: float
    null_iqr: float
    p_value: float
    nsl: float
    verdict: str   # pass / note / warn / fail


# ---- Sweep ------------------------------------------------------------------

def run_sweep(verbose: bool = True) -> list[SweepRow]:
    contract = _contract()
    cfg = _config(contract)

    if verbose:
        print(f"Loading dataset: n_entities={N_ENTITIES}, seed={DATASET_SEED}")
    df_clean = make_clean_dataset(n_entities=N_ENTITIES, seed=DATASET_SEED)
    n_rows = len(df_clean)
    n_periods = int(df_clean["prediction_time"].nunique())
    if verbose:
        print(f"  {n_rows} rows | {n_periods} periods | base_rate={df_clean['target'].mean():.3f} | rho={RHO}")
        total_runs = len(ALPHA_LEVELS) * len(INJECTOR_SEEDS)
        print(
            f"\nSweep: {len(ALPHA_LEVELS)} alpha levels x {len(INJECTOR_SEEDS)} seeds"
            f" = {total_runs} runs | n_permutations={N_PERMUTATIONS}\n"
        )
        # Column header for per-run verbose output
        print(
            f"  {'#':>5} | {'a':>5} {'SNR':>4} {'s':>1} |"
            f" {'fl':>7} {'nmed':>7} {'q99':>7} {'iqr':>7} {'q99-med':>7} {'fl-q99':>7} |"
            f" {'NSL':>6} {'p':>7} | {'verdict'}"
        )
        print("  " + "-" * 105)

    rows: list[SweepRow] = []
    total = len(ALPHA_LEVELS) * len(INJECTOR_SEEDS)
    i = 0
    t_sweep_start = time.perf_counter()

    for alpha in ALPHA_LEVELS:
        snr = alpha * 0.7 / NOISE_SIGMA   # = 2 * alpha with default sigma

        for seed in INJECTOR_SEEDS:
            i += 1
            t0 = time.perf_counter()

            df_leaked, _ = inject_graded_future_leak(
                df_clean,
                alpha=alpha,
                noise_sigma=NOISE_SIGMA,
                seed=seed,
                dataset_seed=DATASET_SEED,
            )

            result = run_severity_analysis(
                df_leaked,
                contract,
                cfg,
                model_factory=_clf,
                n_permutations=N_PERMUTATIONS,
                null_seed=seed,
            )

            if result.status == "unavailable":
                print(
                    f"  [{i}/{total}] alpha={alpha:.2f} seed={seed}"
                    " -> UNAVAILABLE -- aborting sweep"
                )
                sys.exit(1)

            issue = leakage_issue_record(result)
            verdict = issue.status   # pass / note / warn / fail (production verdict)

            nsl_val = result.nsl if result.nsl is not None else float("nan")
            null_med = result.null_median if result.null_median is not None else float("nan")
            null_q99 = result.null_99th if result.null_99th is not None else float("nan")
            null_iqr = result.null_iqr if result.null_iqr is not None else float("nan")
            pval = result.p_value if result.p_value is not None else float("nan")
            fl = result.fixable_leakage

            q99_minus_med = null_q99 - null_med
            fl_minus_q99 = fl - null_q99

            elapsed = time.perf_counter() - t0

            row = SweepRow(
                alpha=alpha,
                seed=seed,
                snr=snr,
                fixable_leakage=fl,
                null_median=null_med,
                null_99th=null_q99,
                null_iqr=null_iqr,
                p_value=pval,
                nsl=nsl_val,
                verdict=verdict,
            )
            rows.append(row)

            if verbose:
                print(
                    f"  [{i:>2}/{total}] | {alpha:>5.2f} {snr:>4.1f} {seed:>1} |"
                    f" {fl:>+7.4f} {null_med:>+7.4f} {null_q99:>+7.4f}"
                    f" {null_iqr:>7.4f} {q99_minus_med:>+7.4f} {fl_minus_q99:>+7.4f} |"
                    f" {nsl_val:>+6.2f} {pval:>7.4f} | {verdict:<5}  ({elapsed:.1f}s)"
                )
                sys.stdout.flush()

    t_total = time.perf_counter() - t_sweep_start
    if verbose:
        print(f"\nSweep complete in {t_total:.0f}s ({t_total/total:.1f}s/run average)")

    return rows


# ---- Report -----------------------------------------------------------------

def report(rows: list[SweepRow]) -> None:
    by_alpha: dict[float, list[SweepRow]] = defaultdict(list)
    for r in rows:
        by_alpha[r.alpha].append(r)

    # ---- Per-run table -------------------------------------------------------
    W = 115
    print("\n" + "=" * W)
    print("NSL SCALE SWEEP n=2000 -- per-run null geometry")
    print(
        f"  {'alpha':>5} {'SNR':>4} {'rho':>4} {'seed':>4} |"
        f" {'fl':>7} {'nmed':>7} {'q99':>7} {'iqr':>7} {'q99-med':>7} {'fl-q99':>7} |"
        f" {'NSL':>6} {'p':>7} | verdict"
    )
    print("  " + "-" * (W - 2))

    for alpha in sorted(by_alpha):
        snr = by_alpha[alpha][0].snr
        for r in sorted(by_alpha[alpha], key=lambda x: x.seed):
            q99_minus_med = r.null_99th - r.null_median
            fl_minus_q99 = r.fixable_leakage - r.null_99th
            print(
                f"  {alpha:>5.2f} {snr:>4.1f} {RHO:>4.2f} {r.seed:>4d} |"
                f" {r.fixable_leakage:>+7.4f} {r.null_median:>+7.4f} {r.null_99th:>+7.4f}"
                f" {r.null_iqr:>7.4f} {q99_minus_med:>+7.4f} {fl_minus_q99:>+7.4f} |"
                f" {r.nsl:>+6.2f} {r.p_value:>7.4f} | {r.verdict}"
            )

    print("  " + "-" * (W - 2))

    # ---- Per-level summary ---------------------------------------------------
    W2 = 95
    print("\n" + "=" * W2)
    print("NSL BOUNDARY SWEEP -- per-level distribution")
    print(
        f"{'alpha':>5} | {'SNR':>4} | {'NSL_min':>7} | {'NSL_med':>7} | {'NSL_max':>7} |"
        f" {'fl_med':>7} | {'q99_med':>7} | {'iqr_med':>7} | verdicts"
    )
    print("-" * W2)

    nsl_1_alpha: float | None = None
    nsl_2_alpha: float | None = None
    prev_med: float | None = None

    for alpha in sorted(by_alpha):
        lvl = by_alpha[alpha]
        snr = lvl[0].snr
        nsls = [r.nsl for r in lvl]
        fls = [r.fixable_leakage for r in lvl]
        q99s = [r.null_99th for r in lvl]
        iqrs = [r.null_iqr for r in lvl]

        nsl_min = min(nsls)
        nsl_med = float(np.median(nsls))
        nsl_max = max(nsls)
        fl_med = float(np.median(fls))
        q99_med = float(np.median(q99s))
        iqr_med = float(np.median(iqrs))

        vc: dict[str, int] = {}
        for r in lvl:
            vc[r.verdict] = vc.get(r.verdict, 0) + 1
        vstr = "  ".join(
            f"{cnt}x{v}"
            for v, cnt in sorted(
                vc.items(),
                key=lambda x: {"fail": 0, "warn": 1, "note": 2, "pass": 3}[x[0]],
            )
        )

        if nsl_1_alpha is None and nsl_med >= 1.0:
            nsl_1_alpha = alpha
        if nsl_2_alpha is None and nsl_med >= 2.0:
            nsl_2_alpha = alpha

        marker = ""
        if prev_med is not None:
            if prev_med < 1.0 <= nsl_med:
                marker = " <-- median NSL crosses 1.0"
            if prev_med < 2.0 <= nsl_med:
                marker += " <-- median NSL crosses 2.0"

        print(
            f"{alpha:>5.2f} | {snr:>4.1f} | {nsl_min:>+7.2f} | {nsl_med:>+7.2f} |"
            f" {nsl_max:>+7.2f} | {fl_med:>+7.4f} | {q99_med:>+7.4f} | {iqr_med:>7.4f} | {vstr}{marker}"
        )
        prev_med = nsl_med

    print("-" * W2)

    # Zone population
    outside_null = [r for r in rows if r.p_value < 0.01]
    z_note = [r for r in outside_null if r.nsl < 1.0]
    z_warn = [r for r in outside_null if 1.0 <= r.nsl < 2.0]
    z_fail = [r for r in outside_null if r.nsl >= 2.0]
    z_pass = [r for r in rows if r.p_value >= 0.01]

    print(f"\nZone population across all {len(rows)} runs:")
    print(f"  pass  (p >= 0.01)           : {len(z_pass):>3} runs")
    print(f"  note  (p < 0.01, NSL < 1)  : {len(z_note):>3} runs")
    print(f"  warn  (p < 0.01, 1<=NSL<2) : {len(z_warn):>3} runs")
    print(f"  fail  (p < 0.01, NSL >= 2) : {len(z_fail):>3} runs")

    print(f"\nFirst alpha where median NSL >= 1.0 : {nsl_1_alpha}")
    print(f"First alpha where median NSL >= 2.0 : {nsl_2_alpha}")

    if z_warn:
        w_nsls = sorted(r.nsl for r in z_warn)
        print(
            f"\nWARN-zone NSLs (1.0-2.0):"
            f"  min={w_nsls[0]:>+.2f}  max={w_nsls[-1]:>+.2f}  n={len(w_nsls)}"
        )
    if z_fail:
        f_nsls = sorted(r.nsl for r in z_fail)
        print(
            f"FAIL-zone NSLs (>=2.0):  "
            f"  min={f_nsls[0]:>+.2f}  max={f_nsls[-1]:>+.2f}  n={len(f_nsls)}"
        )

    # Null tightening diagnostic
    print("\nNull-geometry by level (median across 5 seeds):")
    print(f"  {'alpha':>5} | {'fl_med':>7} | {'nmed_med':>8} | {'q99_med':>7} | {'iqr_med':>7} | {'q99-med':>7} | {'fl-q99':>7}")
    print("  " + "-" * 65)
    for alpha in sorted(by_alpha):
        lvl = by_alpha[alpha]
        fl_med = float(np.median([r.fixable_leakage for r in lvl]))
        nmed_med = float(np.median([r.null_median for r in lvl]))
        q99_med = float(np.median([r.null_99th for r in lvl]))
        iqr_med = float(np.median([r.null_iqr for r in lvl]))
        q99_minus_med = q99_med - nmed_med
        fl_minus_q99 = fl_med - q99_med
        print(
            f"  {alpha:>5.2f} | {fl_med:>+7.4f} | {nmed_med:>+8.4f} | {q99_med:>+7.4f}"
            f" | {iqr_med:>7.4f} | {q99_minus_med:>+7.4f} | {fl_minus_q99:>+7.4f}"
        )

    # Gradient monotonicity check
    print("\nPer-level median fixable_leakage (monotonicity check):")
    prev_fl = None
    for alpha in sorted(by_alpha):
        fls = [r.fixable_leakage for r in by_alpha[alpha]]
        fl_med = float(np.median(fls))
        arrow = ""
        if prev_fl is not None:
            arrow = " ^" if fl_med > prev_fl else " v" if fl_med < prev_fl else " ="
        print(f"  alpha={alpha:.2f}  fl_med={fl_med:+.5f}{arrow}")
        prev_fl = fl_med

    print()


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    rows = run_sweep(verbose=True)
    report(rows)
