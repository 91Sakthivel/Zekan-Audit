#!/usr/bin/env python3
"""NSL boundary calibration sweep -- does NSL=2.0 mark a natural WARN/FAIL line?

WHAT THIS DOES
--------------
Injects a graded future leak (forbidden = alpha * z[T+1] + noise) across 5
signal-strength levels with 5 noise-seed replications each.  Runs the full
permutation null (n_permutations=100 by default) at every (alpha, seed) point
to obtain the actual NSL value the production verdict ladder would assign.

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

F2b EXTENSION -- across-entity channel + spawn_v2 re-validation
-----------------------------------------------------------------
Since spec 1, run_severity_analysis already runs BOTH the within-entity and
across-entity permutation nulls transparently on every call this script
makes -- this script previously just never read the across fields.  F2b adds:
  - nsl_across / p_value_across / null_iqr_across / null_99th_across /
    n_permutations_across, read straight off the SAME SeverityResult.
  - detection_channel (within_entity / across_entity / both / "") via
    build_verdict(result) -- the OR-combined production detection gate.
  - An explicit scheme confirmation: since F2a, the null uses "spawn_v2"
    (SeedSequence-based) seeding, not the retired "serial_v1" shared-stream
    scheme.  The confirmed scheme is echoed in the header, printed in every
    table, and written as a column in the CSV -- a calibration table must
    never leave the seeding scheme ambiguous.
  - --jobs, passed straight through to run_severity_analysis (both nulls
    parallelize -- this is what makes the full sweep affordable).
  - --n-permutations and --quick, so the harness can be smoke-tested cheaply
    (a handful of permutations, 1 alpha x 1 seed) before committing to the
    real n_permutations=100 x 25-cell run.
  - CSV output (default scratch/f2b_calibration.csv) in addition to the
    stdout table, in fixed (alpha, seed) sweep order.
  - An explicit, loud flag if the across channel comes back None/NaN on
    EVERY cell -- inject_graded_future_leak's forbidden column varies by
    BOTH entity and period (not constant within entity), so it is not the
    same shape as the across-entity blind-spot case (spec 1's
    entity_churn_rate).  If across degenerates here, that is itself a
    finding: this fixture cannot earn the across boundary and a different
    (entity-constant-aggregate) fixture is needed.

DISCIPLINE
----------
* Alpha levels and seeds are fixed before any results are observed.
* The verdict is read off the gradient, never tuned onto it.
* fix-the-leak-not-the-gate applies.
* NSL >= 1.0 is the earned engine boundary for WITHIN-entity -- re-validating
  that boundary under spawn_v2 is Q1 below.  The across-entity boundary has
  NEVER been earned; this script's across columns exist to start earning it
  (Q2) -- 1.0 is NOT assumed to be meaningful for across until the data says so.
* NSL >= 2.0 stays a policy default -- do not move it until the data says so.

INVOCATION
----------
    python -m zekan.benchmark.nsl_boundary_sweep [--output PATH] [--jobs N]
        [--n-permutations N] [--quick]
or
    python zekan/benchmark/nsl_boundary_sweep.py [same flags]

--quick runs a single (alpha, seed) cell at n_permutations=5 (overridable via
--n-permutations) to prove the harness executes end-to-end before committing
to the full calibration run.  A bare invocation with no flags is the real
calibration: all 25 cells, n_permutations=100, n_jobs=1.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_graded_future_leak
from zekan.config.schema import ZekanConfig, SplitPolicy
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.engine import leakage_issue_record, run_severity_analysis
from zekan.severity.verdict import build_verdict


# ---- Sweep parameters (fixed before any results are observed) ----------------

# 5 key alpha levels spanning the expected NSL 0->2+ gradient.
# SNR = alpha * 0.7 / 0.35 = 2 * alpha.
ALPHA_LEVELS: list[float] = [0.0, 0.60, 1.10, 1.60, 2.50]

# 5 injector seeds per level -> 5 noise realizations.
INJECTOR_SEEDS: list[int] = [0, 1, 2, 3, 4]

# Full permutation null (100 draws) for stable q99 and IQR estimates.
# This is the DEFAULT for a bare invocation; --n-permutations / --quick override.
N_PERMUTATIONS: int = 100

# --quick smoke mode: a single mid-strength alpha (near the expected NSL~1
# boundary), a single seed, and a tiny permutation count -- just enough to
# prove the harness executes and populates every column.
QUICK_ALPHA_LEVELS: list[float] = [1.10]
QUICK_INJECTOR_SEEDS: list[int] = [0]
QUICK_N_PERMUTATIONS: int = 5

DEFAULT_OUTPUT_PATH: str = "scratch/f2b_calibration.csv"

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


def _config(contract: PredictionContract) -> ZekanConfig:
    # Benchmark-aligned split policy.
    return ZekanConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5,
            min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10,
            min_negative_cases_per_fold=10,
        ),
    )


def _confirmed_scheme() -> str:
    """Read the null-seeding scheme this checkout actually uses.

    NullResult.scheme's class-level default is exactly what
    estimate_fixable_leakage_null stamps on every within_entity/across_entity
    result (see null_baseline.py) -- reading it here means the calibration
    table's scheme label is never a hand-typed assumption that can drift out
    of sync with the code that produced the numbers.
    """
    from zekan.severity.null_baseline import NullResult

    for f in dataclasses.fields(NullResult):
        if f.name == "scheme":
            return f.default
    return "unknown"


# ---- Result record -----------------------------------------------------------

@dataclasses.dataclass
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
    verdict: str   # pass / note / warn / fail -- WITHIN-entity only (leakage_issue_record)

    # F2b additions -- across-entity channel (additive; spec 1 + F2a).
    # NaN / 0 when the across null did not run (no-op guard: <2 entities;
    # not expected to fire on this fixture's n_entities=2000).
    nsl_across: float
    p_value_across: float
    null_iqr_across: float
    null_99th_across: float
    n_permutations_across: int
    detection_channel: str   # within_entity / across_entity / both / "" (OR-combined, build_verdict)


# ---- Sweep ------------------------------------------------------------------

def run_sweep(
    n_permutations: int = N_PERMUTATIONS,
    n_jobs: int = 1,
    alpha_levels: list[float] | None = None,
    injector_seeds: list[int] | None = None,
    verbose: bool = True,
) -> tuple[list[SweepRow], str]:
    """Run the sweep; returns (rows, confirmed_scheme).

    alpha_levels / injector_seeds default to the real-calibration ALPHA_LEVELS /
    INJECTOR_SEEDS; pass QUICK_ALPHA_LEVELS / QUICK_INJECTOR_SEEDS for a smoke run.
    """
    alpha_levels = ALPHA_LEVELS if alpha_levels is None else alpha_levels
    injector_seeds = INJECTOR_SEEDS if injector_seeds is None else injector_seeds

    scheme = _confirmed_scheme()

    contract = _contract()
    cfg = _config(contract)

    if verbose:
        print(f"Confirmed null seeding scheme: {scheme}")
        if scheme != "spawn_v2":
            print(
                "  *** WARNING: expected 'spawn_v2' (F2a). This checkout is measuring "
                f"under '{scheme}' -- results below are NOT the spawn_v2 calibration. ***"
            )
        print(f"Loading dataset: n_entities={N_ENTITIES}, seed={DATASET_SEED}")

    df_clean = make_clean_dataset(n_entities=N_ENTITIES, seed=DATASET_SEED)
    n_rows = len(df_clean)
    n_periods = int(df_clean["prediction_time"].nunique())
    if verbose:
        print(f"  {n_rows} rows | {n_periods} periods | base_rate={df_clean['target'].mean():.3f} | rho={RHO}")
        total_runs = len(alpha_levels) * len(injector_seeds)
        print(
            f"\nSweep: {len(alpha_levels)} alpha levels x {len(injector_seeds)} seeds"
            f" = {total_runs} runs | n_permutations={n_permutations} | n_jobs={n_jobs} | scheme={scheme}\n"
        )
        # Column header for per-run verbose output
        print(
            f"  {'#':>5} | {'a':>5} {'SNR':>4} {'s':>1} |"
            f" {'fl':>7} {'NSL':>6} {'p':>7} {'verd':>5} |"
            f" {'NSLx':>6} {'px':>7} {'channel':>13} |"
        )
        print("  " + "-" * 90)

    rows: list[SweepRow] = []
    total = len(alpha_levels) * len(injector_seeds)
    i = 0
    t_sweep_start = time.perf_counter()

    for alpha in alpha_levels:
        snr = alpha * 0.7 / NOISE_SIGMA   # = 2 * alpha with default sigma

        for seed in injector_seeds:
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
                n_permutations=n_permutations,
                null_seed=seed,
                n_jobs=n_jobs,
            )

            if result.status == "unavailable":
                print(
                    f"  [{i}/{total}] alpha={alpha:.2f} seed={seed}"
                    " -> UNAVAILABLE -- aborting sweep"
                )
                sys.exit(1)

            issue = leakage_issue_record(result)
            verdict = issue.status   # pass / note / warn / fail (within-entity production verdict)

            detection_channel = build_verdict(result).engine_detection.detection_channel

            nsl_val = result.nsl if result.nsl is not None else float("nan")
            null_med = result.null_median if result.null_median is not None else float("nan")
            null_q99 = result.null_99th if result.null_99th is not None else float("nan")
            null_iqr = result.null_iqr if result.null_iqr is not None else float("nan")
            pval = result.p_value if result.p_value is not None else float("nan")
            fl = result.fixable_leakage

            nsl_across_val = result.nsl_across if result.nsl_across is not None else float("nan")
            pval_across = result.p_value_across if result.p_value_across is not None else float("nan")
            null_iqr_across = result.null_iqr_across if result.null_iqr_across is not None else float("nan")
            null_q99_across = result.null_99th_across if result.null_99th_across is not None else float("nan")
            n_perm_across = result.n_permutations_across

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
                nsl_across=nsl_across_val,
                p_value_across=pval_across,
                null_iqr_across=null_iqr_across,
                null_99th_across=null_q99_across,
                n_permutations_across=n_perm_across,
                detection_channel=detection_channel,
            )
            rows.append(row)

            if verbose:
                channel_str = detection_channel or "(none)"
                print(
                    f"  [{i:>2}/{total}] | {alpha:>5.2f} {snr:>4.1f} {seed:>1} |"
                    f" {fl:>+7.4f} {nsl_val:>+6.2f} {pval:>7.4f} {verdict:>5} |"
                    f" {nsl_across_val:>+6.2f} {pval_across:>7.4f} {channel_str:>13} |"
                    f"  ({elapsed:.1f}s)"
                )
                sys.stdout.flush()

    t_total = time.perf_counter() - t_sweep_start
    if verbose:
        print(f"\nSweep complete in {t_total:.0f}s ({t_total/total:.1f}s/run average)")

    return rows, scheme


# ---- CSV output ---------------------------------------------------------------

def write_csv(rows: list[SweepRow], path: str, scheme: str) -> None:
    """Write rows to a CSV in fixed sweep order (deterministic), with the
    confirmed scheme as an explicit column on every row (never ambient/implicit)."""
    import csv

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scheme"] + [f.name for f in dataclasses.fields(SweepRow)]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:  # already in fixed (alpha, seed) sweep order -- do not re-sort
            d = dataclasses.asdict(r)
            d["scheme"] = scheme
            writer.writerow(d)


# ---- Report -----------------------------------------------------------------

def report(rows: list[SweepRow], scheme: str, quick: bool = False) -> None:
    by_alpha: dict[float, list[SweepRow]] = defaultdict(list)
    for r in rows:
        by_alpha[r.alpha].append(r)

    if quick:
        print("\n" + "#" * 70)
        print("# SMOKE MODE (--quick): tiny sample, NOT a calibration result. #")
        print("#" * 70)

    print(f"\nSCHEME = {scheme}" + ("  (expected spawn_v2)" if scheme == "spawn_v2" else "  *** UNEXPECTED ***"))

    # ---- Per-run table -------------------------------------------------------
    W = 130
    print("\n" + "=" * W)
    print(f"NSL SCALE SWEEP n={N_ENTITIES} scheme={scheme} -- per-run null geometry (within + across)")
    print(
        f"  {'alpha':>5} {'SNR':>4} {'rho':>4} {'seed':>4} |"
        f" {'fl':>7} {'nmed':>7} {'q99':>7} {'iqr':>7} |"
        f" {'NSL':>6} {'p':>7} | verdict |"
        f" {'NSLx':>6} {'px':>7} {'iqrx':>7} {'q99x':>7} {'nperm_x':>7} | channel"
    )
    print("  " + "-" * (W - 2))

    for alpha in sorted(by_alpha):
        snr = by_alpha[alpha][0].snr
        for r in sorted(by_alpha[alpha], key=lambda x: x.seed):
            print(
                f"  {alpha:>5.2f} {snr:>4.1f} {RHO:>4.2f} {r.seed:>4d} |"
                f" {r.fixable_leakage:>+7.4f} {r.null_median:>+7.4f} {r.null_99th:>+7.4f}"
                f" {r.null_iqr:>7.4f} |"
                f" {r.nsl:>+6.2f} {r.p_value:>7.4f} | {r.verdict:>7} |"
                f" {r.nsl_across:>+6.2f} {r.p_value_across:>7.4f} {r.null_iqr_across:>7.4f}"
                f" {r.null_99th_across:>+7.4f} {r.n_permutations_across:>7d} | {r.detection_channel or '(none)'}"
            )

    print("  " + "-" * (W - 2))

    # ---- Degenerate-across guard (loud, not silent NaNs) ----------------------
    across_present = [
        r for r in rows
        if r.n_permutations_across > 0 and not np.isnan(r.nsl_across)
    ]
    if not across_present:
        print("\n" + "!" * W)
        print("!!! ACROSS-CHANNEL DEGENERATE ON THIS FIXTURE !!!")
        print(
            "nsl_across is None/NaN (or n_permutations_across == 0) for EVERY "
            f"(alpha, seed) cell in this {len(rows)}-run sweep."
        )
        print(
            "inject_graded_future_leak's forbidden column varies by BOTH entity "
            "and period (alpha * z[entity, T+1] + noise) -- it is not an "
            "entity-constant aggregate, so this is not the shape the across-entity "
            "channel was built to catch (spec 1's blind spot). This fixture cannot "
            "be used to earn the across-entity NSL boundary on its own."
        )
        print(
            "A DIFFERENT fixture (entity-constant aggregate, e.g. "
            "tests/test_entity_aggregate_probe.py's _make_entity_churn_rate_dataset) "
            "is needed to calibrate the across channel."
        )
        print("!" * W)
    elif len(across_present) < len(rows):
        print(
            f"\nNOTE: across channel populated on {len(across_present)}/{len(rows)} cells "
            "-- some cells did not run the across null (check n_permutations_across==0 rows above)."
        )

    # ---- Per-level summary: within AND across side by side --------------------
    W2 = 110
    print("\n" + "=" * W2)
    print("NSL BOUNDARY SWEEP -- per-level distribution (within vs across)")
    print(
        f"{'alpha':>5} | {'SNR':>4} |"
        f" {'W_min':>6} {'W_med':>6} {'W_max':>6} |"
        f" {'X_min':>6} {'X_med':>6} {'X_max':>6} |"
        f" {'fl_med':>7} | verdicts"
    )
    print("-" * W2)

    nsl_1_alpha: float | None = None
    nsl_2_alpha: float | None = None
    nsl_1_alpha_across: float | None = None
    prev_med: float | None = None
    prev_med_across: float | None = None

    for alpha in sorted(by_alpha):
        lvl = by_alpha[alpha]
        snr = lvl[0].snr
        nsls = [r.nsl for r in lvl]
        nsls_across_all = [r.nsl_across for r in lvl]
        nsls_across = [v for v in nsls_across_all if not np.isnan(v)]
        fls = [r.fixable_leakage for r in lvl]

        nsl_min, nsl_med, nsl_max = min(nsls), float(np.median(nsls)), max(nsls)
        if nsls_across:
            nsl_min_x = min(nsls_across)
            nsl_med_x = float(np.median(nsls_across))
            nsl_max_x = max(nsls_across)
        else:
            nsl_min_x = nsl_med_x = nsl_max_x = float("nan")

        fl_med = float(np.median(fls))

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
        if nsl_1_alpha_across is None and nsls_across and nsl_med_x >= 1.0:
            nsl_1_alpha_across = alpha

        marker = ""
        if prev_med is not None:
            if prev_med < 1.0 <= nsl_med:
                marker += " <-- W crosses 1.0"
            if prev_med < 2.0 <= nsl_med:
                marker += " <-- W crosses 2.0"
        if prev_med_across is not None and nsls_across and not np.isnan(nsl_med_x):
            if prev_med_across < 1.0 <= nsl_med_x:
                marker += " <-- X crosses 1.0"

        print(
            f"{alpha:>5.2f} | {snr:>4.1f} |"
            f" {nsl_min:>+6.2f} {nsl_med:>+6.2f} {nsl_max:>+6.2f} |"
            f" {nsl_min_x:>+6.2f} {nsl_med_x:>+6.2f} {nsl_max_x:>+6.2f} |"
            f" {fl_med:>+7.4f} | {vstr}{marker}"
        )
        prev_med = nsl_med
        if nsls_across:
            prev_med_across = nsl_med_x

    print("-" * W2)

    # ---- Q1: within-entity re-validation under spawn_v2 ------------------------
    outside_null = [r for r in rows if r.p_value < 0.01]
    z_note = [r for r in outside_null if r.nsl < 1.0]
    z_warn = [r for r in outside_null if 1.0 <= r.nsl < 2.0]
    z_fail = [r for r in outside_null if r.nsl >= 2.0]
    z_pass = [r for r in rows if r.p_value >= 0.01]

    print(f"\nQ1 (within, re-validate under {scheme}) -- zone population across all {len(rows)} runs:")
    print(f"  pass  (p >= 0.01)           : {len(z_pass):>3} runs")
    print(f"  note  (p < 0.01, NSL < 1)  : {len(z_note):>3} runs")
    print(f"  warn  (p < 0.01, 1<=NSL<2) : {len(z_warn):>3} runs")
    print(f"  fail  (p < 0.01, NSL >= 2) : {len(z_fail):>3} runs")
    print(f"  First alpha where median NSL >= 1.0 : {nsl_1_alpha}")
    print(f"  First alpha where median NSL >= 2.0 : {nsl_2_alpha}")

    # ---- Q2: across-entity -- does it have its own meaningful boundary? -------
    if across_present:
        outside_null_x = [r for r in rows if not np.isnan(r.p_value_across) and r.p_value_across < 0.01]
        z_note_x = [r for r in outside_null_x if r.nsl_across < 1.0]
        z_warn_x = [r for r in outside_null_x if 1.0 <= r.nsl_across < 2.0]
        z_fail_x = [r for r in outside_null_x if r.nsl_across >= 2.0]
        z_pass_x = [r for r in rows if not np.isnan(r.p_value_across) and r.p_value_across >= 0.01]

        print(f"\nQ2 (across, earn for the first time) -- zone population across all {len(rows)} runs:")
        print(f"  pass  (p >= 0.01)           : {len(z_pass_x):>3} runs")
        print(f"  note  (p < 0.01, NSL < 1)  : {len(z_note_x):>3} runs")
        print(f"  warn  (p < 0.01, 1<=NSL<2) : {len(z_warn_x):>3} runs")
        print(f"  fail  (p < 0.01, NSL >= 2) : {len(z_fail_x):>3} runs")
        print(f"  First alpha where median NSL_across >= 1.0 : {nsl_1_alpha_across}")
    else:
        print("\nQ2 (across) -- SKIPPED: across channel degenerate on this fixture (see warning above).")

    print()


# ---- Entry point --------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="NSL boundary calibration sweep (F2b: spawn_v2 re-validation + across-entity channel).",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel workers passed to run_severity_analysis (both nulls; loky backend). Default 1 = serial.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=None,
        help="Permutation draws per null. Default: 100 (real calibration), or 5 under --quick.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Tiny smoke mode: 1 alpha x 1 seed, n_permutations=5 (unless --n-permutations given) "
             "-- proves the harness runs before committing to the full sweep.",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    if args.quick:
        alpha_levels = QUICK_ALPHA_LEVELS
        injector_seeds = QUICK_INJECTOR_SEEDS
        n_permutations = args.n_permutations if args.n_permutations is not None else QUICK_N_PERMUTATIONS
    else:
        alpha_levels = ALPHA_LEVELS
        injector_seeds = INJECTOR_SEEDS
        n_permutations = args.n_permutations if args.n_permutations is not None else N_PERMUTATIONS

    rows, scheme = run_sweep(
        n_permutations=n_permutations,
        n_jobs=args.jobs,
        alpha_levels=alpha_levels,
        injector_seeds=injector_seeds,
        verbose=True,
    )
    report(rows, scheme, quick=args.quick)
    write_csv(rows, args.output, scheme)
    print(f"CSV written: {args.output}")


if __name__ == "__main__":
    main()
