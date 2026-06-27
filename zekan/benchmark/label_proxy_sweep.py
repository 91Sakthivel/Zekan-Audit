#!/usr/bin/env python3
"""Label-proxy sweep: does target-copy leakage separate from temporal leakage by fl magnitude?

Sweeps inject_label_proxy across flip_rate levels at n_entities=200 to measure
fixable_leakage (fl) against the temporal n=200 baseline (fl_med ~0.03-0.10).

PRIMARY QUESTION
----------------
Does direct target-copy fl land qualitatively above the rho-capped temporal
saturation ceiling (~0.096 at n=200)?  A clear magnitude gap = empirical basis
for where an fl-based severity floor could sit.

SECONDARY
---------
Verify detection gate fires for every real proxy run:
  p < 0.01  AND  NSL >= 1.0

NSL is engine-identical (result.nsl, IQR denominator).
No floor/boundary is written to the engine.

TEMPORAL n=200 REFERENCE (from bkbfbwb35 / bxpjle0za runs)
-----------------------------------------------------------
  clean  (alpha=0.00):  fl_med = +0.028
  alpha=0.60 SNR=1.2:   fl_med = +0.073
  alpha=0.90 SNR=1.8:   fl_med = +0.078
  alpha=1.10 SNR=2.2:   fl_med = +0.087  (first alpha where median NSL >= 1.0)
  alpha=1.60 SNR=3.2:   fl_med = +0.096  (saturation ceiling)
  alpha=2.50 SNR=5.0:   fl_med = +0.094
  alpha=3.00 SNR=6.0:   fl_med = +0.090

INVOCATION
----------
    python -m zekan.benchmark.label_proxy_sweep
or
    python zekan/benchmark/label_proxy_sweep.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.engine import leakage_issue_record, run_severity_analysis


# ---- Sweep parameters --------------------------------------------------------

# flip_rate = fraction of labels randomly flipped.  Lower = stronger proxy leak.
# 0.05: 5% noise  (~Corr(proxy,y)=0.90) -- near-perfect label copy
# 0.20: 20% noise (~Corr(proxy,y)=0.60) -- moderate degradation
# 0.40: 40% noise (~Corr(proxy,y)=0.20) -- near-random (heavy degradation)
FLIP_LEVELS: list[float] = [0.05, 0.20, 0.40]

INJECTOR_SEEDS: list[int] = [0, 1, 2, 3, 4]
N_PERMUTATIONS: int = 100
N_ENTITIES: int = 200
DATASET_SEED: int = 0

# Temporal n=200 saturation ceiling for comparison in report output.
TEMPORAL_CEILING_FL: float = 0.096   # fl_med at alpha=1.60 (saturation)
TEMPORAL_CLEAN_FL: float = 0.028     # fl_med at alpha=0.00 (clean baseline)


# ---- Helpers -----------------------------------------------------------------

def _clf() -> RandomForestClassifier:
    # n_estimators=20 matches the original n=200 temporal sweep for comparability.
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


def _contract() -> PredictionContract:
    return PredictionContract(
        prediction_problem="label-proxy-sweep",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )


# ---- Result record -----------------------------------------------------------

@dataclass
class ProxyRow:
    flip_rate: float
    seed: int
    fixable_leakage: float
    null_median: float
    null_99th: float
    null_iqr: float
    p_value: float
    nsl: float
    verdict: str   # pass / note / warn / fail
    gate_fires: bool   # p < 0.01 AND NSL >= 1.0


# ---- Sweep ------------------------------------------------------------------

def run_sweep(verbose: bool = True) -> list[ProxyRow]:
    contract = _contract()
    config = ZekanConfig(contract=contract)   # default SplitPolicy -- matches n=200 temporal baseline

    if verbose:
        print(f"Loading dataset: n_entities={N_ENTITIES}, seed={DATASET_SEED}")
    df_clean = make_clean_dataset(n_entities=N_ENTITIES, seed=DATASET_SEED)
    n_rows = len(df_clean)
    n_periods = int(df_clean["prediction_time"].nunique())
    if verbose:
        print(f"  {n_rows} rows | {n_periods} periods | base_rate={df_clean['target'].mean():.3f}")
        print(f"\nTemporal n=200 reference: clean fl={TEMPORAL_CLEAN_FL:.3f}  ceiling fl={TEMPORAL_CEILING_FL:.3f}")
        total_runs = len(FLIP_LEVELS) * len(INJECTOR_SEEDS)
        print(
            f"\nSweep: {len(FLIP_LEVELS)} flip levels x {len(INJECTOR_SEEDS)} seeds"
            f" = {total_runs} runs | n_permutations={N_PERMUTATIONS}\n"
        )
        print(
            f"  {'#':>5} | {'flip':>5} {'s':>1} |"
            f" {'fl':>7} {'nmed':>7} {'q99':>7} {'iqr':>7} {'q99-med':>7} {'fl-q99':>7} |"
            f" {'NSL':>6} {'p':>7} | {'gate':>5} {'verdict'}"
        )
        print("  " + "-" * 108)

    rows: list[ProxyRow] = []
    total = len(FLIP_LEVELS) * len(INJECTOR_SEEDS)
    i = 0
    t_sweep_start = time.perf_counter()

    for flip_rate in FLIP_LEVELS:
        for seed in INJECTOR_SEEDS:
            i += 1
            t0 = time.perf_counter()

            df_leaked, _ = inject_label_proxy(
                df_clean,
                flip_rate=flip_rate,
                seed=seed,
            )

            result = run_severity_analysis(
                df_leaked,
                contract,
                config,
                model_factory=_clf,
                n_permutations=N_PERMUTATIONS,
                null_seed=seed,
            )

            if result.status == "unavailable":
                print(f"  [{i}/{total}] flip={flip_rate:.2f} seed={seed} -> UNAVAILABLE -- aborting")
                sys.exit(1)

            issue = leakage_issue_record(result)
            verdict = issue.status

            nsl_val = result.nsl if result.nsl is not None else float("nan")
            null_med = result.null_median if result.null_median is not None else float("nan")
            null_q99 = result.null_99th if result.null_99th is not None else float("nan")
            null_iqr = result.null_iqr if result.null_iqr is not None else float("nan")
            pval = result.p_value if result.p_value is not None else float("nan")
            fl = result.fixable_leakage

            q99_minus_med = null_q99 - null_med
            fl_minus_q99 = fl - null_q99
            gate_fires = (pval < 0.01) and (nsl_val >= 1.0)

            elapsed = time.perf_counter() - t0

            row = ProxyRow(
                flip_rate=flip_rate,
                seed=seed,
                fixable_leakage=fl,
                null_median=null_med,
                null_99th=null_q99,
                null_iqr=null_iqr,
                p_value=pval,
                nsl=nsl_val,
                verdict=verdict,
                gate_fires=gate_fires,
            )
            rows.append(row)

            if verbose:
                print(
                    f"  [{i:>2}/{total}] | {flip_rate:>5.2f} {seed:>1} |"
                    f" {fl:>+7.4f} {null_med:>+7.4f} {null_q99:>+7.4f}"
                    f" {null_iqr:>7.4f} {q99_minus_med:>+7.4f} {fl_minus_q99:>+7.4f} |"
                    f" {nsl_val:>+6.2f} {pval:>7.4f} | {'YES':>5}  {verdict:<5}  ({elapsed:.1f}s)"
                    if gate_fires else
                    f"  [{i:>2}/{total}] | {flip_rate:>5.2f} {seed:>1} |"
                    f" {fl:>+7.4f} {null_med:>+7.4f} {null_q99:>+7.4f}"
                    f" {null_iqr:>7.4f} {q99_minus_med:>+7.4f} {fl_minus_q99:>+7.4f} |"
                    f" {nsl_val:>+6.2f} {pval:>7.4f} | {'NO':>5}  {verdict:<5}  ({elapsed:.1f}s)"
                )
                sys.stdout.flush()

    t_total = time.perf_counter() - t_sweep_start
    if verbose:
        print(f"\nSweep complete in {t_total:.0f}s ({t_total/total:.1f}s/run average)")

    return rows


# ---- Report -----------------------------------------------------------------

def report(rows: list[ProxyRow]) -> None:
    by_flip: dict[float, list[ProxyRow]] = defaultdict(list)
    for r in rows:
        by_flip[r.flip_rate].append(r)

    # ---- Per-level summary --------------------------------------------------
    W = 100
    print("\n" + "=" * W)
    print("LABEL-PROXY SWEEP n=200 -- fl vs temporal baseline")
    print(
        f"{'flip':>5} | {'fl_min':>7} | {'fl_med':>7} | {'fl_max':>7} |"
        f" {'NSL_med':>7} | {'p_med':>7} | {'gate':>5} | verdicts"
    )
    print("-" * W)

    for flip in sorted(by_flip, reverse=False):
        lvl = by_flip[flip]
        fls = [r.fixable_leakage for r in lvl]
        nsls = [r.nsl for r in lvl]
        pvals = [r.p_value for r in lvl]

        fl_min = min(fls)
        fl_med = float(np.median(fls))
        fl_max = max(fls)
        nsl_med = float(np.median(nsls))
        p_med = float(np.median(pvals))
        gates_fire = sum(1 for r in lvl if r.gate_fires)
        gate_str = f"{gates_fire}/5"

        vc: dict[str, int] = {}
        for r in lvl:
            vc[r.verdict] = vc.get(r.verdict, 0) + 1
        vstr = "  ".join(
            f"{cnt}x{v}"
            for v, cnt in sorted(
                vc.items(),
                key=lambda x: {"fail": 0, "warn": 1, "note": 2, "pass": 3}.get(x[0], 9),
            )
        )
        print(
            f"{flip:>5.2f} | {fl_min:>+7.4f} | {fl_med:>+7.4f} | {fl_max:>+7.4f} |"
            f" {nsl_med:>+7.2f} | {p_med:>7.4f} | {gate_str:>5} | {vstr}"
        )

    print("-" * W)

    # ---- Temporal reference -------------------------------------------------
    print(f"\nTemporal n=200 reference (same n_entities=200, default SplitPolicy, n_estimators=20):")
    ref = [
        ("clean  alpha=0.00 SNR=0.0", 0.028),
        ("leak   alpha=0.60 SNR=1.2", 0.073),
        ("leak   alpha=1.10 SNR=2.2", 0.087),
        ("leak   alpha=1.60 SNR=3.2", 0.096),
        ("leak   alpha=2.50 SNR=5.0", 0.094),
    ]
    for label, fl_ref in ref:
        print(f"  {label}: fl_med = {fl_ref:+.3f}")

    print(f"\n  Temporal rho-capped ceiling: fl_med ~ {TEMPORAL_CEILING_FL:.3f}")

    # ---- Null geometry by level ---------------------------------------------
    print("\nNull geometry by level (median across 5 seeds):")
    print(f"  {'flip':>5} | {'fl_med':>7} | {'nmed':>7} | {'q99':>7} | {'iqr':>7} | {'q99-med':>7} | {'fl-q99':>7}")
    print("  " + "-" * 65)
    for flip in sorted(by_flip):
        lvl = by_flip[flip]
        fl_med = float(np.median([r.fixable_leakage for r in lvl]))
        nmed = float(np.median([r.null_median for r in lvl]))
        q99 = float(np.median([r.null_99th for r in lvl]))
        iqr = float(np.median([r.null_iqr for r in lvl]))
        print(
            f"  {flip:>5.2f} | {fl_med:>+7.4f} | {nmed:>+7.4f} | {q99:>+7.4f}"
            f" | {iqr:>7.4f} | {q99-nmed:>+7.4f} | {fl_med-q99:>+7.4f}"
        )

    # ---- fl-gap assessment --------------------------------------------------
    proxy_fl_min = min(
        float(np.median([r.fixable_leakage for r in lvl]))
        for lvl in by_flip.values()
    )
    print(f"\nfl gap: lowest proxy fl_med = {proxy_fl_min:+.4f}  vs  temporal ceiling = {TEMPORAL_CEILING_FL:.4f}")
    gap = proxy_fl_min - TEMPORAL_CEILING_FL
    if gap > 0.05:
        print(f"  -> CLEAR GAP of {gap:+.4f} above temporal ceiling.")
    elif gap > 0:
        print(f"  -> MARGINAL separation of {gap:+.4f} above temporal ceiling.")
    else:
        print(f"  -> NO separation ({gap:+.4f}): proxy fl overlaps with temporal fl.")

    print()


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    rows = run_sweep(verbose=True)
    report(rows)
