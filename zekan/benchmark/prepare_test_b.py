"""Deterministic data prep for Test B (Diabetes-130 real-data experiments).

Produces three audit-input CSVs from the raw Diabetes-130 file, one per
pre-registered test (see zekan/benchmark/results/TEST_B_PREREGISTRATION.md
and its ADDENDUM_1, ADDENDUM_2, and ADDENDUM_3). Deriving the binary target,
deriving the ordinal-period time column, ordinal-encoding categorical
features, and planting the B-2 leak are DEFINING the experiment, not
cleaning data: every other column is carried through byte-for-byte as read
from the raw file (missing-value '?' sentinels, the 97%-missing 'weight'
column, everything -- '?' becomes an ordinary encoded category, not a
cleaned-away value).

Same raw file in -> same three CSVs out, every time. The only randomness
is the B-2 label-noise flip, which is seeded (SEED) for reproducibility.
The period_ordinal assignment and the ordinal encoding are both pure
functions of the data's own values -- no randomness, no target information.

Usage
-----
    python -m zekan.benchmark.prepare_test_b \\
        --raw "C:\\path\\to\\diabetic_data.csv" \\
        --outdir "C:\\path\\to\\outdir"
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
CONTROL_COL = "weight"  # mess-preservation check: encoded, but category count must be invariant
ENTITY_ORDER_COL = "encounter_id"  # real-order proxy: globally unique, increases with time
ENTITY_ID_COL = "patient_nbr"  # contract's entity_id -- excluded from encoding

# ── Ordinal-period time column ──────────────────────────────────────────────
# Addendum 2 (TEST_B_PREREGISTRATION_ADDENDUM_2.md): raw encounter_id is
# unparseable as a date AND degenerate as a period (every row is its own
# period). Zekan's severity core is inherently temporal and needs a
# datetime-PARSEABLE prediction_time (see the ready-then-crash fix,
# commit 2832600). period_ordinal encodes the REAL encounter order into N
# equal-count sequential buckets, each labeled with a valid but synthetic
# ISO date -- an honest ordinal encoding, not a fabricated calendar.
#
# N=24 is chosen from the actual fold math (see _print_period_rationale),
# not a round number picked for its own sake:
#   - contract_checks._check_temporal_periods requires >=3 periods (FAIL
#     below that) and recommends >=6 for PASS. 24 clears the PASS floor
#     with a real 4x margin, not just barely.
#   - With n_splits=5, temporal_expanding_folds divides periods into
#     n_splits+1=6 roughly-equal blocks (np.linspace(0, N, n_splits+2)).
#     Each fold's TEST block therefore spans ~N/6 periods regardless of N,
#     and with equal-count period buckets, each test block holds
#     ~total_rows/(n_splits+1) rows -- an invariant of N, not something N
#     controls. For this dataset that's ~101766/6 ~= 16961 rows/block,
#     clearing min_test_rows_per_fold=100 and min_positive_cases_per_fold=20
#     (at ~11.16% positive, ~1893 positives/block) by roughly two orders of
#     magnitude for virtually any N in a broad valid range (~6 to a few
#     hundred). The fold-size minimums do not pin down a unique N.
#   - What N=24 actually buys, within that broad valid range: real
#     multi-period granularity well past the bare PASS floor (so the
#     expanding-window audit has genuine train/test boundaries to choose
#     among, not one coarse block), while each individual period is still
#     a large, well-powered chunk on its own (~4240 rows, ~473 positives
#     per period -- clears the per-fold minimums even at single-period
#     granularity, before any block-aggregation margin is counted).
N_PERIODS: int = 24
PERIOD_COL = "period_ordinal"
PERIOD_BASE_DATE = pd.Timestamp("2000-01-01")  # arbitrary anchor; ordinal, not calendar

# ── Ordinal encoding of categorical features ────────────────────────────────
# Addendum 3 (TEST_B_PREREGISTRATION_ADDENDUM_3.md): the first real B-1 run
# crashed on evaluate_folds's df[feature_cols].to_numpy(dtype=float) cast,
# because Diabetes-130 is mostly raw categorical text (race, gender, age
# bucket, diag_1/2/3, ~20 drug-dosage columns). Addendum 3 pre-registered the
# fix: ordinal/label encoding (sorted-unique -> 0..k-1, same semantics as
# sklearn's OrdinalEncoder), not one-hot (diag_1/2/3 alone would explode into
# 700+ columns each) and not target encoding (would inject target information
# into the features, manufacturing leakage). The mapping uses ONLY each
# column's own values -- no target, no randomness -- and every value present,
# including the '?' sentinel, gets an ordinary code: nothing is cleaned or
# imputed away.
#
# Role columns are excluded from encoding entirely (they aren't features):
# entity_id, the derived period_ordinal (prediction_time AND
# available_features_until), and the derived binary target. Everything else
# -- including the raw 3-way 'readmitted' column that B-3 deliberately
# leaves in as the undeclared blind-spot column -- is a feature and gets
# encoded if it isn't already numeric.
ROLE_COLS_EXCLUDED_FROM_ENCODING = {ENTITY_ID_COL, PERIOD_COL, TARGET_COL}
ENCODING_MAP_JSON = Path(__file__).resolve().parent / "results" / "TEST_B_ENCODING_MAP.json"
ENCODING_MAP_MD = Path(__file__).resolve().parent / "results" / "TEST_B_ENCODING_MAP.md"
_INLINE_MAPPING_MAX_CATEGORIES = 15  # small enough to show in full in the .md summary


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


def _print_period_rationale(n_rows: int, pos_rate: float, n_splits: int,
                             min_test_rows: int, min_pos: int, n_periods: int) -> None:
    block_periods = n_periods / (n_splits + 1)
    period_rows = n_rows / n_periods
    block_rows = n_rows / (n_splits + 1)  # invariant of n_periods for balanced buckets
    block_pos = block_rows * pos_rate
    period_pos = period_rows * pos_rate
    print("=" * 90)
    print("STEP 1a — PERIOD COUNT RATIONALE")
    print("=" * 90)
    print(f"n_rows={n_rows}  positive_rate={pos_rate:.4%}  n_splits={n_splits}  "
          f"min_test_rows_per_fold={min_test_rows}  min_positive_cases_per_fold={min_pos}")
    print(f"temporal_expanding_folds divides periods into n_splits+1={n_splits + 1} "
          f"roughly-equal blocks -> each test block spans ~{block_periods:.2f} periods, "
          f"independent of n_periods for equal-count buckets.")
    print(f"block rows ~= n_rows/(n_splits+1) = {n_rows}/{n_splits + 1} = {block_rows:.1f}")
    print(f"block positives ~= {block_rows:.1f} * {pos_rate:.4f} = {block_pos:.1f} "
          f"(clears min_positive_cases_per_fold={min_pos} by {block_pos / min_pos:.1f}x)")
    print(f"Chosen N_PERIODS={n_periods}: period rows ~= {n_rows}/{n_periods} = {period_rows:.1f}, "
          f"period positives ~= {period_pos:.1f} (clears {min_pos} by {period_pos / min_pos:.1f}x "
          f"even at single-period granularity)")
    print(f"N={n_periods} clears contract_checks' >=6-period PASS recommendation "
          f"by {n_periods / 6:.1f}x")
    print()


def _assign_period_ordinal(df: pd.DataFrame, n_periods: int, id_col: str) -> pd.Series:
    """Rank all rows by `id_col` ascending (real order) and bucket into
    `n_periods` equal-count sequential buckets, each labeled with a valid,
    monotonically-increasing ISO date. Pure function of id_col -- no RNG.
    """
    ids = df[id_col].astype(np.int64).to_numpy()
    order = np.argsort(ids, kind="stable")  # row positions in ascending id order
    n = len(ids)
    bucket_by_rank = np.floor(np.arange(n) * n_periods / n).astype(int)
    bucket_by_rank = np.minimum(bucket_by_rank, n_periods - 1)

    bucket = np.empty(n, dtype=int)
    bucket[order] = bucket_by_rank  # scatter rank-ordered buckets back to original row positions

    period_dates = [
        (PERIOD_BASE_DATE + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
        for i in range(n_periods)
    ]
    return pd.Series([period_dates[b] for b in bucket], index=df.index, name=PERIOD_COL)


def _is_non_numeric(series: pd.Series) -> bool:
    """True if any value in `series` fails to coerce to a number.

    Mirrors contract_checks._check_feature_columns_numeric's own definition
    of "non-numeric" exactly, so a column this function says is fine is a
    column that check will also say is fine.
    """
    coerced = pd.to_numeric(series, errors="coerce")
    n_bad = int((coerced.isna() & ~series.isna()).sum())
    return n_bad > 0


def _build_ordinal_mappings(df: pd.DataFrame, exclude: set[str]) -> dict[str, dict[str, int]]:
    """Compute a sorted-unique -> 0..k-1 mapping for every non-role,
    non-numeric column in `df`. Computed once from the full column set and
    reused everywhere, so the same category always gets the same code across
    all three outputs. Uses only each column's own values -- no target, no
    randomness -- so the mapping is deterministic and reproducible.
    """
    mappings: dict[str, dict[str, int]] = {}
    for col in df.columns:
        if col in exclude:
            continue
        if not _is_non_numeric(df[col]):
            continue
        uniques = sorted(df[col].unique())
        mappings[col] = {v: i for i, v in enumerate(uniques)}
    return mappings


def _apply_ordinal_mappings(df: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Apply precomputed mappings to every column of `df` that has one.
    Columns not present in `df` (e.g. 'readmitted', absent from B-1/B-2) are
    skipped. Encoded columns are stored as digit strings, matching this
    script's all-string DataFrame convention.
    """
    df = df.copy()
    for col, mapping in mappings.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(mapping).astype(str)
    return df


def _mapping_hash(mapping: dict[str, int]) -> str:
    """Short, stable hash of a column's mapping for cross-run verification."""
    canonical = json.dumps(mapping, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_encoding_map(mappings: dict[str, dict[str, int]],
                         json_path: Path, md_path: Path) -> None:
    """Write the complete, reproducible encoding record (Addendum 3's promise:
    'the exact columns encoded and their code mappings will be written down').
    JSON carries every mapping in full; the .md gives a fast human-readable
    summary plus the full mapping inline for small columns.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": "ordinal (sorted-unique -> 0..k-1, sklearn OrdinalEncoder semantics)",
        "target_free": True,
        "role_columns_excluded": sorted(ROLE_COLS_EXCLUDED_FROM_ENCODING),
        "columns": {
            col: {
                "n_categories": len(mapping),
                "hash": _mapping_hash(mapping),
                "mapping": mapping,
            }
            for col, mapping in sorted(mappings.items())
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                          encoding="utf-8")

    lines = [
        "# Test B ordinal-encoding map",
        "",
        "Generated by `zekan/benchmark/prepare_test_b.py` -- see "
        "`TEST_B_PREREGISTRATION_ADDENDUM_3.md` for why (the crash it fixes) "
        "and why ordinal encoding specifically (not one-hot, not target encoding).",
        "",
        "Method: sorted-unique -> 0..k-1 per column, using only that column's own "
        "values. No target information, no randomness. `'?'` and every other raw "
        "sentinel value is an ordinary category -- nothing is cleaned or imputed.",
        "",
        f"Role columns excluded from encoding: {', '.join(sorted(ROLE_COLS_EXCLUDED_FROM_ENCODING))}.",
        "",
        f"Full machine-readable mapping (every column, in full): `{json_path.name}`.",
        "",
        "## Summary",
        "",
        "| column | n_categories | hash |",
        "|---|---|---|",
    ]
    for col, mapping in sorted(mappings.items()):
        lines.append(f"| `{col}` | {len(mapping)} | `{_mapping_hash(mapping)}` |")

    small = {c: m for c, m in sorted(mappings.items()) if len(m) <= _INLINE_MAPPING_MAX_CATEGORIES}
    if small:
        lines += ["", "## Full mapping for small columns (<= "
                       f"{_INLINE_MAPPING_MAX_CATEGORIES} categories)", ""]
        for col, mapping in small.items():
            lines.append(f"**`{col}`** ({len(mapping)} categories):")
            lines.append("")
            for value, code in sorted(mapping.items(), key=lambda kv: kv[1]):
                lines.append(f"- `{value!r}` -> `{code}`")
            lines.append("")

    large = [c for c, m in mappings.items() if len(m) > _INLINE_MAPPING_MAX_CATEGORIES]
    if large:
        lines += ["## Large columns (see JSON for full mapping)", ""]
        for col in sorted(large):
            lines.append(f"- `{col}`: {len(mappings[col])} categories, "
                         f"hash `{_mapping_hash(mappings[col])}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target = _derive_target(raw_df)
    period_ordinal = _assign_period_ordinal(raw_df, N_PERIODS, ENTITY_ORDER_COL)

    # ── B-1: specificity — raw readmitted dropped, nothing planted ────────────
    b1 = raw_df.drop(columns=[TARGET_SOURCE_COL]).copy()
    b1[TARGET_COL] = target
    b1[PERIOD_COL] = period_ordinal

    # ── B-2: sensitivity — raw readmitted dropped, one declared leak planted ──
    b2 = b1.copy()
    planted, n_flip = _plant_leak(target, SEED, LEAK_FLIP_RATE)
    b2[PLANTED_LEAK_COL] = planted

    # ── B-3: honest unknown — raw readmitted retained, undeclared ─────────────
    b3 = raw_df.copy()
    b3[TARGET_COL] = target
    b3[PERIOD_COL] = period_ordinal

    # ── Ordinal-encode categorical features -- LAST transform before write ────
    # Mapping is built once from raw_df (the superset of all real columns,
    # including 'readmitted', before any output is column-subset/superset'd)
    # so every output that shares a column shares the exact same code for the
    # exact same category. Role columns (entity_id, period_ordinal, target)
    # are excluded; planted_leak and encounter_id are already numeric-as-string
    # and are left untouched by _is_non_numeric's own definition of "fine".
    mappings = _build_ordinal_mappings(raw_df, ROLE_COLS_EXCLUDED_FROM_ENCODING)
    b1 = _apply_ordinal_mappings(b1, mappings)
    b2 = _apply_ordinal_mappings(b2, mappings)
    b3 = _apply_ordinal_mappings(b3, mappings)

    return {
        "testB1_specificity.csv": b1,
        "testB2_sensitivity.csv": b2,
        "testB3_honest_unknown.csv": b3,
    }, target, planted, n_flip, period_ordinal, mappings


def verify(raw_df: pd.DataFrame, outputs: dict[str, pd.DataFrame],
           target: pd.Series, planted: pd.Series, n_flip: int,
           period_ordinal: pd.Series, mappings: dict[str, dict[str, int]]) -> None:
    n_raw = len(raw_df)
    print("=" * 90)
    print("VERIFICATION REPORT")
    print("=" * 90)

    # ── period_ordinal checks ───────────────────────────────────────────────
    print(f"\n--- {PERIOD_COL} ---")
    parsed = pd.to_datetime(period_ordinal, errors="coerce")
    n_bad = int(parsed.isna().sum())
    n_distinct = int(period_ordinal.nunique())
    print(f"  distinct values: {n_distinct} (expected {N_PERIODS})")
    print(f"  parse failures (n_bad): {n_bad} (must be 0 -- this is exactly the "
          f"condition contract_checks._check_prediction_time now requires)")
    assert n_distinct == N_PERIODS, f"expected {N_PERIODS} distinct periods, got {n_distinct}"
    assert n_bad == 0, f"{n_bad} period_ordinal values failed to parse as datetime"

    counts = period_ordinal.value_counts().sort_index()
    print(f"  bucket size: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f} (balance check)")
    assert counts.max() - counts.min() <= 1, (
        f"buckets not balanced: min={counts.min()}, max={counts.max()}"
    )

    pos_by_period = target.astype(int).groupby(period_ordinal).sum()
    print(f"  positives per period: min={int(pos_by_period.min())}, "
          f"max={int(pos_by_period.max())} (must clear min_positive_cases_per_fold=20)")
    assert pos_by_period.min() >= 20, (
        f"a period has only {pos_by_period.min()} positives, below the per-fold minimum"
    )

    ids = raw_df[ENTITY_ORDER_COL].astype(np.int64).to_numpy()
    order = np.argsort(ids, kind="stable")
    period_int = parsed.to_numpy()[order]  # period_ordinal in ascending encounter_id order
    monotonic = bool(np.all(period_int[1:] >= period_int[:-1]))
    print(f"  monotonic non-decreasing in encounter_id order: {monotonic}")
    assert monotonic, "period_ordinal is not monotonic non-decreasing in encounter_id order"

    pos_rate_raw = target.astype(int).mean()
    print(f"\nTarget positive rate ({TARGET_COL} == 1): {pos_rate_raw:.4%} "
          f"({int(target.astype(int).sum())} of {n_raw})")
    assert abs(pos_rate_raw - 0.1116) < 0.001, f"positive rate drifted: {pos_rate_raw}"

    raw_q = _question_mark_count(raw_df, CONTROL_COL)
    print(f"\nControl column '{CONTROL_COL}' raw '?' count: {raw_q}")

    # ── ordinal encoding checks ─────────────────────────────────────────────
    print(f"\n--- ordinal encoding ---")
    print(f"  columns encoded: {len(mappings)}")
    print(f"  encoded column names: {sorted(mappings.keys())}")

    raw_weight_nunique = int(raw_df[CONTROL_COL].nunique())
    print(f"  '{CONTROL_COL}' raw distinct value count (incl. '?'): {raw_weight_nunique}")
    assert CONTROL_COL in mappings, f"'{CONTROL_COL}' was not encoded -- unexpected"
    assert "?" in mappings[CONTROL_COL], f"'?' sentinel missing from '{CONTROL_COL}' mapping"
    assert len(mappings[CONTROL_COL]) == raw_weight_nunique, (
        f"'{CONTROL_COL}' encoded category count {len(mappings[CONTROL_COL])} "
        f"!= raw distinct count {raw_weight_nunique}"
    )
    print(f"  '{CONTROL_COL}' encoded category count: {len(mappings[CONTROL_COL])} "
          f"(matches raw: {len(mappings[CONTROL_COL]) == raw_weight_nunique}); "
          f"'?' present as an ordinary category: True")

    for name, df in outputs.items():
        print(f"\n--- {name} ---")
        assert len(df) == n_raw, f"{name}: row count {len(df)} != raw {n_raw}"
        print(f"  rows: {len(df)} (matches raw: {len(df) == n_raw})")

        pos_rate = df[TARGET_COL].astype(int).mean()
        print(f"  {TARGET_COL} positive rate: {pos_rate:.4%}")
        assert abs(pos_rate - pos_rate_raw) < 1e-9, f"{name}: positive rate mismatch"

        has_readmitted = TARGET_SOURCE_COL in df.columns
        print(f"  contains raw '{TARGET_SOURCE_COL}': {has_readmitted}")

        out_weight_nunique = int(df[CONTROL_COL].nunique())
        print(f"  '{CONTROL_COL}' encoded distinct code count: {out_weight_nunique} "
              f"(matches raw distinct count: {out_weight_nunique == raw_weight_nunique})")
        assert out_weight_nunique == raw_weight_nunique, (
            f"{name}: '{CONTROL_COL}' encoded distinct count {out_weight_nunique} "
            f"!= raw distinct count {raw_weight_nunique}"
        )

        assert PERIOD_COL in df.columns, f"{name}: missing '{PERIOD_COL}'"
        out_n_bad = int(pd.to_datetime(df[PERIOD_COL], errors="coerce").isna().sum())
        print(f"  '{PERIOD_COL}' present, parse failures: {out_n_bad} (must be 0)")
        assert out_n_bad == 0, f"{name}: '{PERIOD_COL}' has {out_n_bad} parse failures"

        non_numeric_cols = [
            col for col in df.columns
            if col not in ROLE_COLS_EXCLUDED_FROM_ENCODING and _is_non_numeric(df[col])
        ]
        print(f"  non-numeric feature columns remaining: {non_numeric_cols} (must be empty)")
        assert not non_numeric_cols, f"{name}: still non-numeric after encoding: {non_numeric_cols}"

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

    target_preview = _derive_target(raw_df)
    _print_period_rationale(
        n_rows=len(raw_df),
        pos_rate=target_preview.astype(int).mean(),
        n_splits=5,
        min_test_rows=100,
        min_pos=20,
        n_periods=N_PERIODS,
    )

    outputs, target, planted, n_flip, period_ordinal, mappings = build_outputs(raw_df)

    verify(raw_df, outputs, target, planted, n_flip, period_ordinal, mappings)

    print("\n" + "=" * 90)
    print("WRITING ENCODING MAP")
    print("=" * 90)
    _write_encoding_map(mappings, ENCODING_MAP_JSON, ENCODING_MAP_MD)
    print(f"  wrote {ENCODING_MAP_JSON} ({len(mappings)} column(s) encoded)")
    print(f"  wrote {ENCODING_MAP_MD}")

    print("\n" + "=" * 90)
    print("WRITING OUTPUTS")
    print("=" * 90)
    for name, df in outputs.items():
        out_path = outdir / name
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  wrote {out_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
