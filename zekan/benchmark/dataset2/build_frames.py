"""Build the shared frame table for the Freddie Mac SFLLD dataset-2 study.

Follows, exactly, without improvising any parameter:
  DATASET2_FREDDIEMAC_PREREGISTRATION.md            -- panel structure, frames
  DATASET2_ADDENDUM_01_FRAME_DECISIONS.md  section 2 -- delinquency-status
                                                          parsing rules
  DATASET2_ADDENDUM_01_FRAME_DECISIONS.md  section 3 -- 12-month forward
                                                          horizon target
  DATASET2_ADDENDUM_04_SAMPLING_DESIGN.md            -- the case-control
                                                          sample this reads

This script builds ONE joined table with BOTH targets attached. It does not
choose a feature set for any frame -- per pre-registration section 5, that is
a Zekan contract concern, applied later, not here. It does not construct a
Zekan contract and does not run any audit.

Output is written OUTSIDE the repo (default:
C:\\Users\\Hp\\Desktop\\freddiemac\\2018_frames\\) -- raw data and derived
frames are never committed, per the standing project constraint recorded in
DATASET2_FREDDIEMAC_PREREGISTRATION.md section 3.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

LOAN_ID_FIELD = "LOAN IDENTIFIER"
PERIOD_FIELD = "PERIOD"
DLQ_FIELD = "CURRENT LOAN DELINQUENCY STATUS"
ZBC_FIELD = "ZERO BALANCE CODE"

HORIZON_MONTHS = 12          # Addendum 01 section 3: 12-month forward horizon
SERIOUS_DELINQUENCY_CUTOFF = 3  # Addendum 01 section 2: status >= 03 (90+ DPD)
CREDIT_EVENT_CODES = {"02", "03", "09"}  # pre-registration section 5, Frame P


# ── Period arithmetic ───────────────────────────────────────────────────────

def period_to_ordinal(period: str) -> int:
    """YYYYMM -> a month ordinal (year*12 + month), so period arithmetic
    handles gaps in a loan's period sequence correctly (never assumes rows
    are contiguous)."""
    p = period.strip()
    year = int(p[:4])
    month = int(p[4:6])
    return year * 12 + month


# ── Delinquency status parsing (Addendum 01 section 2) ─────────────────────
# Explicit three-way classification. A failed parse is NEVER silently
# defaulted to 0 and NEVER silently dropped -- it is counted and reported.

def parse_status(raw: str) -> tuple[str, object]:
    """Return (kind, value).

    kind == "numeric":      value is an int.
    kind == "RA":            value is the raw string "RA" (REO Acquisition;
                              a post-outcome disposition state, not a
                              delinquency-severity reading -- excluded from
                              the >=3 condition per Addendum 01 section 2).
    kind == "unparseable":   value is the raw (stripped) string -- covers
                              blank, "XX", and anything else undocumented.
    """
    s = raw.strip()
    if s == "RA":
        return "RA", s
    try:
        return "numeric", int(s)
    except ValueError:
        return "unparseable", s


# ── Frame builder ────────────────────────────────────────────────────────────

def build_frame_table(perf: pd.DataFrame, orig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (output_df, stats) where stats carries every count this
    script is required to print."""

    stats: dict = {}

    # ── Delinquency status parsing, counted ──────────────────────────────────
    parsed = perf[DLQ_FIELD].map(parse_status)
    kinds = parsed.map(lambda kv: kv[0])
    n_numeric = int((kinds == "numeric").sum())
    n_ra = int((kinds == "RA").sum())
    n_unparseable = int((kinds == "unparseable").sum())
    unparseable_raw_values = sorted(
        {v for k, v in parsed if k == "unparseable"}
    )
    unexpected_unparseable = sorted(
        v for v in unparseable_raw_values if v not in ("", "XX")
    )
    stats["n_status_numeric"] = n_numeric
    stats["n_status_ra"] = n_ra
    stats["n_status_unparseable"] = n_unparseable
    stats["unparseable_distinct_values"] = unparseable_raw_values
    stats["unexpected_unparseable_values"] = unexpected_unparseable

    # ── Per-loan period -> (status_kind, status_value, zbc) lookup ──────────
    ordinals = perf[PERIOD_FIELD].map(period_to_ordinal)
    zbc_stripped = perf[ZBC_FIELD].str.strip()

    per_loan: dict[str, dict[int, tuple[str, object, str]]] = {}
    per_loan_max_ordinal: dict[str, int] = {}
    for loan_id, ordv, (kind, value), zbc in zip(
        perf[LOAN_ID_FIELD], ordinals, parsed, zbc_stripped
    ):
        d = per_loan.setdefault(loan_id, {})
        d[ordv] = (kind, value, zbc)
        if loan_id not in per_loan_max_ordinal or ordv > per_loan_max_ordinal[loan_id]:
            per_loan_max_ordinal[loan_id] = ordv

    # ── Row-by-row labelling ────────────────────────────────────────────────
    n_dropped_incomplete_window = 0
    n_ra_in_label_windows = 0
    keep_mask: list[bool] = []
    target_delinquency: list[int] = []
    target_creditevent: list[int] = []

    for loan_id, ordv in zip(perf[LOAN_ID_FIELD], ordinals):
        max_ord = per_loan_max_ordinal[loan_id]
        window_end = ordv + HORIZON_MONTHS
        if window_end > max_ord:
            keep_mask.append(False)
            target_delinquency.append(None)
            target_creditevent.append(None)
            n_dropped_incomplete_window += 1
            continue

        loan_periods = per_loan[loan_id]
        dlq_hit = 0
        ce_hit = 0
        ra_in_window = 0
        for w in range(ordv + 1, ordv + HORIZON_MONTHS + 1):
            entry = loan_periods.get(w)
            if entry is None:
                continue
            kind, value, zbc = entry
            if kind == "numeric" and value >= SERIOUS_DELINQUENCY_CUTOFF:
                dlq_hit = 1
            elif kind == "RA":
                ra_in_window += 1
            if zbc in CREDIT_EVENT_CODES:
                ce_hit = 1
        keep_mask.append(True)
        target_delinquency.append(dlq_hit)
        target_creditevent.append(ce_hit)
        n_ra_in_label_windows += ra_in_window

    stats["n_dropped_incomplete_window"] = n_dropped_incomplete_window
    stats["n_ra_in_label_windows"] = n_ra_in_label_windows

    labelled = perf.copy()
    labelled["target_delinquency"] = target_delinquency
    labelled["target_creditevent"] = target_creditevent
    labelled = labelled.loc[keep_mask].copy()
    labelled["target_delinquency"] = labelled["target_delinquency"].astype(int)
    labelled["target_creditevent"] = labelled["target_creditevent"].astype(int)

    # ── Join origination attributes (every performance row) ─────────────────
    orig_no_dupe_id = orig.drop(columns=[LOAN_ID_FIELD])
    orig_for_join = orig[[LOAN_ID_FIELD] + list(orig_no_dupe_id.columns)]
    merged = labelled.merge(
        orig_for_join, on=LOAN_ID_FIELD, how="left", indicator=True,
    )
    n_unjoined = int((merged["_merge"] == "left_only").sum())
    stats["n_unjoined_performance_rows"] = n_unjoined
    merged = merged.drop(columns=["_merge"])

    # ── Deterministic order ──────────────────────────────────────────────────
    merged["_ordinal_sort_key"] = merged[PERIOD_FIELD].map(period_to_ordinal)
    merged = merged.sort_values(
        by=[LOAN_ID_FIELD, "_ordinal_sort_key"], kind="mergesort"
    ).drop(columns=["_ordinal_sort_key"]).reset_index(drop=True)

    return merged, stats


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the shared frame table for the Freddie Mac dataset-2 study."
    )
    p.add_argument(
        "--orig-data",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_sample_cc\sample_cc_orig_2018Q1.txt",
    )
    p.add_argument(
        "--perf-data",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_sample_cc\sample_cc_perf_2018Q1.txt",
    )
    p.add_argument(
        "--output-dir",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_frames",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Both inputs now have a header row (written by build_sample.py).
    perf = pd.read_csv(
        args.perf_data, sep="|", header=0, dtype=str,
        keep_default_na=False, na_values=[],
    )
    orig = pd.read_csv(
        args.orig_data, sep="|", header=0, dtype=str,
        keep_default_na=False, na_values=[],
    )

    output_df, stats = build_frame_table(perf, orig)

    out_path = os.path.join(args.output_dir, "frames_2018Q1_cc.txt")
    output_df.to_csv(out_path, sep="|", header=True, index=False)

    # ── Reporting ─────────────────────────────────────────────────────────────
    print("=== Delinquency status parsing ===")
    print(f"numeric rows: {stats['n_status_numeric']}")
    print(f"RA rows: {stats['n_status_ra']}")
    print(f"unparseable rows: {stats['n_status_unparseable']}")
    print(f"unparseable distinct raw values: {stats['unparseable_distinct_values']}")
    if stats["unexpected_unparseable_values"]:
        print(
            "!!! UNEXPECTED unparseable value(s) outside {blank, XX}: "
            f"{stats['unexpected_unparseable_values']}"
        )

    print("\n=== Target windowing ===")
    print(f"rows dropped (12-month window extends beyond loan's last observed period): "
          f"{stats['n_dropped_incomplete_window']}")
    print(f"RA rows encountered inside label windows (labelled anchors only): "
          f"{stats['n_ra_in_label_windows']}")

    print("\n=== Join check ===")
    print(f"performance rows that failed to join an origination row: "
          f"{stats['n_unjoined_performance_rows']}")

    print("\n=== Output table ===")
    n_rows = len(output_df)
    n_loans = output_df[LOAN_ID_FIELD].nunique()
    print(f"final row count: {n_rows}")
    print(f"final loan count: {n_loans}")
    print(f"period range: {output_df[PERIOD_FIELD].min()} to {output_df[PERIOD_FIELD].max()}")

    for target_col in ("target_delinquency", "target_creditevent"):
        row_rate = output_df[target_col].mean()
        loan_positive = output_df.groupby(LOAN_ID_FIELD)[target_col].max()
        loan_rate = loan_positive.mean()
        print(
            f"{target_col}: row-level positive rate = {row_rate:.4%} "
            f"({int(output_df[target_col].sum())}/{n_rows}); "
            f"loan-level positive rate (ever positive) = {loan_rate:.4%} "
            f"({int(loan_positive.sum())}/{n_loans})"
        )

    print(f"\nsha256 {os.path.basename(out_path)}: {sha256_of(out_path)}")


if __name__ == "__main__":
    main()
