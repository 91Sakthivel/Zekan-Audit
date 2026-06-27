#!/usr/bin/env python3
"""Phase 5 calibration: measure per-fold ci_ratio across 5 key scenarios.

Throwaway script -- does NOT touch the live verdict path.
Runs run_severity_analysis(n_permutations=0) to extract per_fold data,
then computes fold-level t-interval stats and prints the calibration table.

Goal: earn the confidence tier breakpoints from measured fold variability,
not from assumed values.

SCENARIOS
---------
  1. clean,        n=200   (noise forbidden column, zero signal)
  2. clean,        n=2000  (noise forbidden column, zero signal)
  3. temporal-leak n=200,  alpha=1.60
  4. temporal-leak n=2000, alpha=1.60
  5. label-proxy   n=200,  flip_rate=0.05

5 seeds [0,1,2,3,4] per scenario = 25 total runs.
n_permutations=0 -- null not needed, only per_fold fold_leakage values.

CI: t-interval centered on fl_mean, df = K-1 (K = interior fold count).
    Guard: if K < 2, emit ci_ratio=None with note.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import t as scipy_t
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_graded_future_leak, inject_label_proxy
from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.engine import run_severity_analysis


# ── Constants ──────────────────────────────────────────────────────────────────

SEEDS: list[int] = [0, 1, 2, 3, 4]
DATASET_SEED: int = 0   # matches all existing sweeps
NOISE_SIGMA: float = 0.35

# floor for max(abs(fl_mean), _CI_EPS) -- prevents divide-by-zero on near-zero fl_mean
_CI_EPS: float = 0.001


# ── Scenario descriptors ───────────────────────────────────────────────────────

SCENARIOS: list[dict] = [
    {"label": "clean    n=200",  "n": 200,  "leak": None},
    {"label": "clean    n=2000", "n": 2000, "leak": None},
    {"label": "temporal n=200",  "n": 200,  "leak": "graded", "alpha": 1.60},
    {"label": "temporal n=2000", "n": 2000, "leak": "graded", "alpha": 1.60},
    {"label": "proxy    n=200",  "n": 200,  "leak": "proxy",  "flip_rate": 0.05},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clf() -> RandomForestClassifier:
    # n_estimators=20 matches the n=200 sweep default for comparability.
    return RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1)


def _build_contract(forbidden_col: str, n_entities: int) -> tuple[PredictionContract, ZekanConfig]:
    contract = PredictionContract(
        prediction_problem="phase5-calibration",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=[forbidden_col],
    )
    return contract, ZekanConfig(contract=contract)


# ── Per-run result ─────────────────────────────────────────────────────────────

@dataclass
class CalibRow:
    scenario: str
    seed: int
    n_entities: int
    n_folds: int               # interior (non-terminal) fold count
    fl_mean: float             # mean of interior fold leakages
    fl_std: float              # std, ddof=1
    fl_se: float               # fl_std / sqrt(n_folds)
    t_crit: Optional[float]    # scipy_t.ppf(0.975, df=n_folds-1)
    ci_half: Optional[float]   # t_crit * fl_se
    ci_low: Optional[float]    # fl_mean - ci_half
    ci_high: Optional[float]   # fl_mean + ci_half
    ci_ratio: Optional[float]  # ci_half / max(abs(fl_mean), _CI_EPS)
    pooled: float              # result.fixable_leakage (pooled OOF)
    pooled_vs_fold_gap: float  # pooled - fl_mean
    note: str


# ── Per-run computation ────────────────────────────────────────────────────────

def _run_one(scenario: dict, seed: int) -> CalibRow:
    n = scenario["n"]
    leak = scenario.get("leak")

    df = make_clean_dataset(n_entities=n, seed=DATASET_SEED)

    if leak == "graded":
        alpha = scenario["alpha"]
        df, _ = inject_graded_future_leak(
            df, alpha=alpha, noise_sigma=NOISE_SIGMA,
            seed=seed, dataset_seed=DATASET_SEED,
        )
        forbidden_col = "graded_future_leak"

    elif leak == "proxy":
        df, _ = inject_label_proxy(df, flip_rate=scenario["flip_rate"], seed=seed)
        forbidden_col = "leaky_label_proxy"

    else:
        # Clean: add a pure-noise forbidden column (zero signal, nonzero fold noise)
        # so the CI reflects real fold variability around fl≈0.
        df = df.copy()
        rng = np.random.default_rng(seed)
        df["noise_forbidden"] = rng.standard_normal(size=len(df))
        forbidden_col = "noise_forbidden"

    contract, config = _build_contract(forbidden_col, n)

    result = run_severity_analysis(
        df, contract, config,
        model_factory=_clf,
        n_permutations=0,
    )

    if result.status == "unavailable":
        nan = float("nan")
        return CalibRow(
            scenario=scenario["label"], seed=seed, n_entities=n,
            n_folds=0, fl_mean=nan, fl_std=nan, fl_se=nan,
            t_crit=None, ci_half=None, ci_low=None, ci_high=None,
            ci_ratio=None, pooled=nan, pooled_vs_fold_gap=nan,
            note=f"UNAVAILABLE: {result.unavailable_reason}",
        )

    interior = [pf.fold_leakage for pf in result.per_fold if not pf.is_terminal]
    k = len(interior)
    pooled = result.fixable_leakage

    if k < 2:
        fl_mean = float(np.mean(interior)) if k == 1 else float("nan")
        return CalibRow(
            scenario=scenario["label"], seed=seed, n_entities=n,
            n_folds=k, fl_mean=fl_mean, fl_std=float("nan"), fl_se=float("nan"),
            t_crit=None, ci_half=None, ci_low=None, ci_high=None,
            ci_ratio=None, pooled=pooled,
            pooled_vs_fold_gap=(pooled - fl_mean) if k == 1 else float("nan"),
            note="insufficient folds for interval (df<=0)",
        )

    arr = np.array(interior, dtype=float)
    fl_mean = float(arr.mean())
    fl_std = float(arr.std(ddof=1))
    fl_se = fl_std / float(np.sqrt(k))
    t_crit = float(scipy_t.ppf(0.975, df=k - 1))
    ci_half = t_crit * fl_se
    ci_low = fl_mean - ci_half
    ci_high = fl_mean + ci_half
    ci_ratio = ci_half / max(abs(fl_mean), _CI_EPS)
    gap = pooled - fl_mean

    return CalibRow(
        scenario=scenario["label"], seed=seed, n_entities=n,
        n_folds=k, fl_mean=fl_mean, fl_std=fl_std, fl_se=fl_se,
        t_crit=t_crit, ci_half=ci_half, ci_low=ci_low, ci_high=ci_high,
        ci_ratio=ci_ratio, pooled=pooled, pooled_vs_fold_gap=gap,
        note="",
    )


# ── Print helpers ──────────────────────────────────────────────────────────────

def _f(v: Optional[float], fmt: str = "+.4f", width: int = 8) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # None or NaN
        return f"{'None':>{width}}"
    return f"{v:{fmt}}"


def _print_table(rows: list[CalibRow]) -> None:
    W = 135
    print("=" * W)
    print("PHASE 5 CALIBRATION -- per-fold CI (t-interval, df=K-1)")
    print(f"  eps floor = {_CI_EPS}  |  CI: [fl_mean - t*se, fl_mean + t*se]")
    print()
    hdr = (
        f"  {'scenario':<18} {'sd':>2} {'K':>2}  "
        f"{'fl_mean':>8} {'fl_std':>7} {'fl_se':>7} {'t_crit':>6}  "
        f"{'ci_half':>7} {'ci_ratio':>8}  "
        f"{'pooled':>8} {'gap':>7}  note"
    )
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in rows:
        note_str = f"  [{r.note}]" if r.note else ""
        ci_half_s = _f(r.ci_half, ".4f", 7) if r.ci_half is not None else f"{'None':>7}"
        ci_ratio_s = _f(r.ci_ratio, "+.3f", 8) if r.ci_ratio is not None else f"{'None':>8}"
        t_s = _f(r.t_crit, ".2f", 6) if r.t_crit is not None else f"{'None':>6}"
        print(
            f"  {r.scenario:<18} {r.seed:>2} {r.n_folds:>2}  "
            f"{_f(r.fl_mean, '+.4f')} {_f(r.fl_std, '.4f', 7)} {_f(r.fl_se, '.4f', 7)} {t_s}  "
            f"{ci_half_s} {ci_ratio_s}  "
            f"{_f(r.pooled, '+.4f')} {_f(r.pooled_vs_fold_gap, '+.4f', 7)}"
            f"{note_str}"
        )
    print("=" * W)


def _print_summary(rows: list[CalibRow]) -> None:
    by_sc: dict[str, list[CalibRow]] = defaultdict(list)
    for r in rows:
        by_sc[r.scenario].append(r)

    print("\nPER-SCENARIO ci_ratio summary (t-interval, min/med/max across 5 seeds):")
    print(f"  {'scenario':<18}  {'K_typ':>5}  {'t_crit_typ':>10}  {'ci_ratio_min':>12}  {'ci_ratio_med':>12}  {'ci_ratio_max':>12}")
    print("  " + "-" * 80)
    for sc in [s["label"] for s in SCENARIOS]:
        rs = by_sc[sc]
        valid_ratios = [r.ci_ratio for r in rs if r.ci_ratio is not None]
        k_vals = [r.n_folds for r in rs if r.n_folds >= 2]
        t_vals = [r.t_crit for r in rs if r.t_crit is not None]
        k_med = int(np.median(k_vals)) if k_vals else 0
        t_med = float(np.median(t_vals)) if t_vals else float("nan")
        if valid_ratios:
            print(
                f"  {sc:<18}  {k_med:>5}  {t_med:>10.3f}  "
                f"{min(valid_ratios):>12.3f}  {float(np.median(valid_ratios)):>12.3f}  {max(valid_ratios):>12.3f}"
            )
        else:
            print(f"  {sc:<18}  {k_med:>5}  {'None':>10}  {'None':>12}  {'None':>12}  {'None':>12}")


def _print_regime_distribution(rows: list[CalibRow]) -> None:
    regimes = [
        ("clean    small-n (n=200)",   lambda r: r.scenario == "clean    n=200"),
        ("clean    large-n (n=2000)",  lambda r: r.scenario == "clean    n=2000"),
        ("temporal small-n (n=200)",   lambda r: r.scenario == "temporal n=200"),
        ("temporal large-n (n=2000)",  lambda r: r.scenario == "temporal n=2000"),
        ("proxy    small-n (n=200)",   lambda r: r.scenario == "proxy    n=200"),
    ]

    print("\nCI_RATIO DISTRIBUTION BY REGIME (t-interval):")
    print(f"  {'regime':<28}  {'n':>4}  {'min':>7}  {'p25':>7}  {'med':>7}  {'p75':>7}  {'max':>7}")
    print("  " + "-" * 78)
    for label, pred in regimes:
        vals = [r.ci_ratio for r in rows if pred(r) and r.ci_ratio is not None]
        if vals:
            arr = np.array(vals)
            print(
                f"  {label:<28}  {len(vals):>4}  "
                f"{arr.min():>7.3f}  {float(np.percentile(arr, 25)):>7.3f}  "
                f"{float(np.median(arr)):>7.3f}  {float(np.percentile(arr, 75)):>7.3f}  "
                f"{arr.max():>7.3f}"
            )
        else:
            print(f"  {label:<28}  {'n/a':>4}")


def _print_breakpoint_assessment(rows: list[CalibRow]) -> None:
    valid = [r for r in rows if r.ci_ratio is not None]
    if not valid:
        print("\nNo valid ci_ratio values to assess.")
        return

    all_ratios = np.array([r.ci_ratio for r in valid])

    below_033 = int(np.sum(all_ratios < 0.33))
    below_067 = int(np.sum(all_ratios < 0.67))
    n = len(all_ratios)

    print(f"\nCANDIDATE BREAKPOINT ASSESSMENT (high<0.33, medium<0.67, low>=0.67):")
    print(f"  total valid runs : {n}")
    print(f"  ci_ratio < 0.33  (would be 'high')  : {below_033}/{n} = {100*below_033/n:.0f}%")
    print(f"  0.33-0.67        (would be 'medium') : {below_067-below_033}/{n} = {100*(below_067-below_033)/n:.0f}%")
    print(f"  >= 0.67          (would be 'low')    : {n-below_067}/{n} = {100*(n-below_067)/n:.0f}%")

    print()
    print("  Per-scenario placement under candidate cuts:")
    by_sc: dict[str, list[CalibRow]] = defaultdict(list)
    for r in valid:
        by_sc[r.scenario].append(r)

    for sc in [s["label"] for s in SCENARIOS]:
        rs = by_sc.get(sc, [])
        if not rs:
            continue
        ratios = sorted(r.ci_ratio for r in rs)
        tags = []
        for v in ratios:
            if v < 0.33:
                tags.append("high")
            elif v < 0.67:
                tags.append("medium")
            else:
                tags.append("low")
        placement = ", ".join(f"{v:.3f}({t})" for v, t in zip(ratios, tags))
        print(f"    {sc:<18}: {placement}")

    print()
    print("  NOTE: breakpoints NOT locked. Report to user before editing verdict.py.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    total = len(SCENARIOS) * len(SEEDS)
    print(f"Phase 5 calibration: {len(SCENARIOS)} scenarios x {len(SEEDS)} seeds = {total} runs")
    print(f"n_permutations=0 | t-interval df=K-1 | ci_ratio = ci_half / max(|fl_mean|, {_CI_EPS})")
    print()

    all_rows: list[CalibRow] = []
    run_idx = 0
    t_start = time.perf_counter()

    for scenario in SCENARIOS:
        print(f"  --- {scenario['label']} ---")
        for seed in SEEDS:
            run_idx += 1
            t0 = time.perf_counter()
            row = _run_one(scenario, seed)
            elapsed = time.perf_counter() - t0
            all_rows.append(row)

            if row.ci_ratio is not None:
                ci_str = f"{row.ci_ratio:+.3f}"
            else:
                ci_str = " None"
            if row.ci_half is not None:
                ci_half_str = f"{row.ci_half:.4f}"
            else:
                ci_half_str = "  None"
            note_str = f"  [{row.note}]" if row.note else ""

            print(
                f"    [{run_idx:>2}/{total}] seed={seed}"
                f"  K={row.n_folds}"
                f"  fl_mean={row.fl_mean:+.4f}"
                f"  ci_half={ci_half_str}"
                f"  ci_ratio={ci_str}"
                f"  ({elapsed:.1f}s)"
                + note_str,
                flush=True,
            )

    t_total = time.perf_counter() - t_start
    print(f"\nAll {total} runs complete in {t_total:.0f}s ({t_total/total:.1f}s/run avg)\n")

    _print_table(all_rows)
    _print_summary(all_rows)
    _print_regime_distribution(all_rows)
    _print_breakpoint_assessment(all_rows)


if __name__ == "__main__":
    main()
