"""Deterministic data prep for Test B (Diabetes-130 real-data experiments).

Produces three audit-input CSVs from the raw Diabetes-130 file, one per
pre-registered test (see zekan/benchmark/results/TEST_B_PREREGISTRATION.md
and its ADDENDUM_1). Deriving the binary target and planting the B-2 leak
are DEFINING the experiment, not cleaning data: every other column is
carried through byte-for-byte as read from the raw file (missing-value
'?' sentinels, the 97%-missing 'weight' column, everything).

Same raw file in -> same three CSVs out, every time. The only randomness
is the B-2 label-noise flip, which is seeded (SEED) for reproducibility.

Usage
-----
    python -m zekan.benchmark.prepare_test_b \\
        --raw "C:\\path\\to\\diabetic_data.csv" \\
        --outdir "C:\\path\\to\\outdir"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed for the B-2 planted-leak label noise. Named constant so the
# plant is reproducible and the seed choice is visible, not buried.
SEED: int = 20260716

# Fraction of rows whose planted_leak label is flipped relative to the true
# target — makes planted_leak a strong-but-imperfect proxy, not a literal
# duplicate of the target column.
LEAK_FLIP_RATE: float = 0.05

TARGET_SOURCE_COL = "readmitted"
TARGET_COL = "readmitted_lt30"
PLANTED_LEAK_COL = "planted_leak"
CONTROL_COL = "weight"  # untouched-column check: must be byte-identical to raw


def _derive_target(df: pd.DataFrame) -> pd.Series:
    return (df[TARGET_SOURCE_COL] == "<30").astype(int).astype(str)


def _plant_leak(target: pd.Series, seed: int, flip_rate: float) -> tuple[pd.Series, int]:
    """Copy `target`, flip `flip_rate` fraction of labels at a fixed seed."""
    rng = np.random.default_rng(seed)
    values = target.astype(int).to_numpy(copy=True)
    n = len(values)
    n_flip = round(flip_rate * n)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    values[flip_idx] = 1 - values[flip_idx]
    return pd.Series(values.astype(str), index=target.index), n_flip


def _question_mark_count(df: pd.DataFrame, col: str) -> int:
    return int((df[col] == "?").sum())


def build_outputs(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target = _derive_target(raw_df)

    # ── B-1: specificity — raw readmitted dropped, nothing planted ────────────
    b1 = raw_df.drop(columns=[TARGET_SOURCE_COL]).copy()
    b1[TARGET_COL] = target

    # ── B-2: sensitivity — raw readmitted dropped, one declared leak planted ──
    b2 = b1.copy()
    planted, n_flip = _plant_leak(target, SEED, LEAK_FLIP_RATE)
    b2[PLANTED_LEAK_COL] = planted

    # ── B-3: honest unknown — raw readmitted retained, undeclared ─────────────
    b3 = raw_df.copy()
    b3[TARGET_COL] = target

    return {
        "testB1_specificity.csv": b1,
        "testB2_sensitivity.csv": b2,
        "testB3_honest_unknown.csv": b3,
    }, target, planted, n_flip


def verify(raw_df: pd.DataFrame, outputs: dict[str, pd.DataFrame],
           target: pd.Series, planted: pd.Series, n_flip: int) -> None:
    n_raw = len(raw_df)
    print("=" * 90)
    print("VERIFICATION REPORT")
    print("=" * 90)

    pos_rate_raw = target.astype(int).mean()
    print(f"\nTarget positive rate ({TARGET_COL} == 1): {pos_rate_raw:.4%} "
          f"({int(target.astype(int).sum())} of {n_raw})")
    assert abs(pos_rate_raw - 0.1116) < 0.001, f"positive rate drifted: {pos_rate_raw}"

    raw_q = _question_mark_count(raw_df, CONTROL_COL)
    print(f"\nControl column '{CONTROL_COL}' '?' count in RAW: {raw_q}")

    for name, df in outputs.items():
        print(f"\n--- {name} ---")
        assert len(df) == n_raw, f"{name}: row count {len(df)} != raw {n_raw}"
        print(f"  rows: {len(df)} (matches raw: {len(df) == n_raw})")

        pos_rate = df[TARGET_COL].astype(int).mean()
        print(f"  {TARGET_COL} positive rate: {pos_rate:.4%}")
        assert abs(pos_rate - pos_rate_raw) < 1e-9, f"{name}: positive rate mismatch"

        has_readmitted = TARGET_SOURCE_COL in df.columns
        print(f"  contains raw '{TARGET_SOURCE_COL}': {has_readmitted}")

        q = _question_mark_count(df, CONTROL_COL)
        print(f"  '{CONTROL_COL}' '?' count: {q} (matches raw: {q == raw_q})")
        assert q == raw_q, f"{name}: '{CONTROL_COL}' '?' count {q} != raw {raw_q}"

        print(f"  columns ({len(df.columns)}): {list(df.columns)}")

    # B-1 / B-2 must NOT carry raw readmitted; B-3 must.
    b1, b2, b3 = (outputs["testB1_specificity.csv"],
                  outputs["testB2_sensitivity.csv"],
                  outputs["testB3_honest_unknown.csv"])
    assert TARGET_SOURCE_COL not in b1.columns, "B-1 must not contain raw readmitted"
    assert TARGET_SOURCE_COL not in b2.columns, "B-2 must not contain raw readmitted"
    assert TARGET_SOURCE_COL in b3.columns, "B-3 must contain raw readmitted"

    # B-2 planted_leak: high but imperfect correlation with target.
    corr = float(np.corrcoef(planted.astype(int), target.astype(int))[0, 1])
    print(f"\nB-2 planted_leak: {n_flip} labels flipped "
          f"({n_flip / n_raw:.2%} of {n_raw} rows)")
    print(f"B-2 planted_leak correlation with target: {corr:.4f}")
    assert n_flip > 0, "no flips applied — planted_leak would be a literal duplicate"
    assert corr < 1.0, "planted_leak is a literal duplicate of the target"
    assert corr > 0.5, f"planted_leak correlation too low to be a meaningful leak: {corr}"

    print("\nAll assertions passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Path to diabetic_data.csv")
    parser.add_argument("--outdir", default=None,
                         help="Output directory for the three CSVs "
                              "(default: same directory as --raw)")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    outdir = Path(args.outdir) if args.outdir else raw_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path, keep_default_na=False, dtype=str)

    outputs, target, planted, n_flip = build_outputs(raw_df)

    verify(raw_df, outputs, target, planted, n_flip)

    print("\n" + "=" * 90)
    print("WRITING OUTPUTS")
    print("=" * 90)
    for name, df in outputs.items():
        out_path = outdir / name
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  wrote {out_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
