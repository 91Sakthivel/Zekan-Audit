"""Build the shared case-control sample for the Freddie Mac SFLLD dataset-2 study.

Follows DATASET2_ADDENDUM_04_SAMPLING_DESIGN.md (commit 026a378) exactly:
loan-level case-control sampling, all case loans retained, non-case loans
sampled 1:4 without replacement, one shared sample used across every frame.

This script builds the sample ONLY. It does not construct any frame, define
any target, or run any audit -- see the addendum's own scope note (section 8:
frame/target/audit work is separate and not yet done).

Output is written OUTSIDE the repo (default: C:\\Users\\Hp\\Desktop\\freddiemac\\
2018_sample_cc\\) -- raw data and derived samples are never committed, per the
standing project constraint recorded in DATASET2_FREDDIEMAC_PREREGISTRATION.md
section 3.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# Fixed before the sample was built -- see DATASET2_ADDENDUM_04_SAMPLING_DESIGN.md
# section 2 ("case-control sampling at the loan level, seeded"). Never changed
# after the fact to alter which loans land in the sample.
SEED = 20180101

CONTROLS_PER_CASE = 4  # 1:4 ratio, per addendum 04 section 2
CHUNKSIZE = 200_000
CASE_ZBC_CODES = {"02", "03", "09"}

LOAN_ID_FIELD = "LOAN IDENTIFIER"
ZBC_FIELD = "ZERO BALANCE CODE"
PERIOD_FIELD = "PERIOD"


def read_header(path: str) -> list[str]:
    """Column names, in order, from a pipe-delimited header .txt file."""
    with open(path, encoding="utf-8") as f:
        return f.readline().rstrip("\n\r").split("|")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def find_case_loan_ids(perf_data_path: str, perf_cols: list[str]) -> set[str]:
    """First pass over the performance file: which loans are cases.

    A loan is a case if ANY of its rows has Zero Balance Code in
    {02, 03, 09}, compared as stripped strings.
    """
    case_ids: set[str] = set()
    reader = pd.read_csv(
        perf_data_path, sep="|", header=None, names=perf_cols, dtype=str,
        keep_default_na=False, na_values=[], chunksize=CHUNKSIZE,
    )
    for chunk in reader:
        zbc_stripped = chunk[ZBC_FIELD].str.strip()
        mask = zbc_stripped.isin(CASE_ZBC_CODES)
        case_ids.update(chunk.loc[mask, LOAN_ID_FIELD].unique().tolist())
    return case_ids


def sample_control_ids(all_loan_ids: list[str], case_ids: set[str]) -> list[str]:
    """Sample non-case loans 1:4 without replacement, using numpy default_rng(SEED).

    Deterministic: both the candidate pool and the case set are sorted before
    sampling, so no set-iteration order affects the result.
    """
    non_case_ids = sorted(set(all_loan_ids) - case_ids)
    n_controls = min(CONTROLS_PER_CASE * len(case_ids), len(non_case_ids))
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(np.array(non_case_ids), size=n_controls, replace=False)
    return sorted(sampled.tolist())


def write_filtered_origination(
    orig_data_path: str, orig_cols: list[str], selected_ids: set[str], out_path: str
) -> int:
    orig = pd.read_csv(
        orig_data_path, sep="|", header=None, names=orig_cols, dtype=str,
        keep_default_na=False, na_values=[],
    )
    mask = orig[LOAN_ID_FIELD].isin(selected_ids)
    filtered = orig.loc[mask]
    filtered.to_csv(out_path, sep="|", header=True, index=False)
    return len(filtered)


def write_filtered_performance(
    perf_data_path: str, perf_cols: list[str], selected_ids: set[str], out_path: str
) -> tuple[int, str, str]:
    """Second pass over the performance file: write only the sampled loans' rows.

    Returns (row_count, min_period, max_period) over the WRITTEN rows.
    """
    n_rows = 0
    period_min: str | None = None
    period_max: str | None = None
    reader = pd.read_csv(
        perf_data_path, sep="|", header=None, names=perf_cols, dtype=str,
        keep_default_na=False, na_values=[], chunksize=CHUNKSIZE,
    )
    first_chunk = True
    for chunk in reader:
        mask = chunk[LOAN_ID_FIELD].isin(selected_ids)
        kept = chunk.loc[mask]
        if kept.empty:
            continue
        kept.to_csv(
            out_path, sep="|", header=first_chunk, index=False,
            mode="w" if first_chunk else "a",
        )
        first_chunk = False
        n_rows += len(kept)
        cmin = kept[PERIOD_FIELD].min()
        cmax = kept[PERIOD_FIELD].max()
        if period_min is None or cmin < period_min:
            period_min = cmin
        if period_max is None or cmax > period_max:
            period_max = cmax
    if first_chunk:
        # No rows matched at all -- still produce a header-only file.
        pd.DataFrame(columns=perf_cols).to_csv(out_path, sep="|", header=True, index=False)
    return n_rows, period_min, period_max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the shared case-control sample for the Freddie Mac dataset-2 study."
    )
    p.add_argument(
        "--orig-header",
        default=r"C:\Users\Hp\Desktop\freddiemac\origination_data_file_header.txt",
    )
    p.add_argument(
        "--perf-header",
        default=r"C:\Users\Hp\Desktop\freddiemac\performance_data_file_header.txt",
    )
    p.add_argument(
        "--orig-data",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_full\Q1\orig_2018Q1.txt",
    )
    p.add_argument(
        "--perf-data",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_full\Q1\perf_2018Q1.txt",
    )
    p.add_argument(
        "--output-dir",
        default=r"C:\Users\Hp\Desktop\freddiemac\2018_sample_cc",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    orig_cols = read_header(args.orig_header)
    perf_cols = read_header(args.perf_header)

    orig_all_ids = pd.read_csv(
        args.orig_data, sep="|", header=None, names=orig_cols, dtype=str,
        keep_default_na=False, na_values=[], usecols=[LOAN_ID_FIELD],
    )[LOAN_ID_FIELD].tolist()

    case_ids = find_case_loan_ids(args.perf_data, perf_cols)
    control_ids = sample_control_ids(orig_all_ids, case_ids)
    selected_ids = set(case_ids) | set(control_ids)

    out_orig_path = os.path.join(args.output_dir, "sample_cc_orig_2018Q1.txt")
    out_perf_path = os.path.join(args.output_dir, "sample_cc_perf_2018Q1.txt")

    n_orig_rows = write_filtered_origination(args.orig_data, orig_cols, selected_ids, out_orig_path)
    n_perf_rows, period_min, period_max = write_filtered_performance(
        args.perf_data, perf_cols, selected_ids, out_perf_path
    )

    print(f"n cases: {len(case_ids)}")
    print(f"n controls: {len(control_ids)}")
    print(f"n loans total: {len(selected_ids)}")
    print(f"n performance rows: {n_perf_rows}")
    print(f"n origination rows: {n_orig_rows}")
    print(f"min period: {period_min}")
    print(f"max period: {period_max}")
    print(f"sha256 {os.path.basename(out_orig_path)}: {sha256_of(out_orig_path)}")
    print(f"sha256 {os.path.basename(out_perf_path)}: {sha256_of(out_perf_path)}")


if __name__ == "__main__":
    main()
