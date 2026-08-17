# Addendum 02 — Measured Base Rates

Addendum 02 to `DATASET2_FREDDIEMAC_PREREGISTRATION.md` (commit `b582e2f`)
and Addendum 01 (commit `0b66849`). Dated 2026-08-16.

**Status: MEASURED. NO FRAME BUILT, NO AUDIT RUN, NO DESIGN RESPONSE
DECIDED.**

Measurements below were taken directly from the raw sample files, before any
frame construction. Two problems surfaced by measurement are recorded in
§3 and §4. The design responses to those two problems are **deliberately
not decided in this document** — they are left as open candidates, to be
resolved (if at all) in a later, separately dated addendum.

## 1. Data provenance (manifest)

Source: Freddie Mac SFLLD official 50,000-loan sample for vintage 2018,
downloaded from Clarity Data Intelligence. Files `sample_orig_2018.txt` and
`sample_perf_2018.txt`, held outside the repo (`C:\Users\Hp\Desktop\
freddiemac\2018_sample\`).

SHA256 (manifest):
```
sample_orig_2018.txt: 72f625b605d66c5342196d9d203a0ff061e7086cdccb94e9007e7f6a5386da93
sample_perf_2018.txt: b8ed3781a2148498f6e7f59cfd02c07fa58a15ff45eb9091af665e8187d5626e
```

**Correction to Addendum 01 / pre-registration §1**: the earlier 12-loan
local files bundled with the schema documents were **not** the guide's
official sample product. This 50,000-loan file **is** that product —
it matches both the guide's stated definition (a random sample of loans
per vintage year) and its `sample_orig_YYYY.txt` / `sample_perf_YYYY.txt`
naming convention. This is recorded here as a correction; the earlier
documents are not edited.

## 2. Structural measurements

- Origination: 50,000 rows, 50,000 distinct Loan Identifiers.
- Performance: 2,059,564 rows, 50,000 distinct Loan Identifiers.
- Period range: 201801 to 202603.
- Periods per loan: median 30, min 1, max 99.
- Join: 50,000 of 50,000 both directions — zero unmatched either way.

This confirms the entity/period panel mapping recorded in
pre-registration §4 (`entity_id = Loan Identifier`, `period = Period`).

## 3. Finding 1 — Frame P is underpowered

`Zero Balance Code` value counts (raw strings), full table:

```
blank: 2,019,487  |  01: 39,830  |  96: 117  |  16: 59
02: 29            |  15: 18      |  09: 17   |  03: 7
```

- Distinct loans with ZBC in {02, 03, 09}: **53** (0.1060% of 50,000).
- Distinct loans with ZBC in {02, 03, 09, 15}: **71** (0.1420%).
- Corroborating: `Actual Loss` is non-null on only **65 of 2,059,564 rows**
  (0.0032%).

Recorded plainly: this fires the risk explicitly accepted in Addendum 01
§1. 53 positive loans may not survive temporal fold-splitting, which would
make Frame P unusable as a known-answer anchor.

**The response is not decided here.** Three candidate responses are
recorded without choosing among them:

- (a) Accept and report the limitation as-is.
- (b) Reframe Frame P at loan level rather than loan-period level.
- (c) Obtain the full 2018 vintage for more loans.

(c) is increasing `n` for statistical power, which is legitimate and
distinct from threshold tuning — but must itself be recorded as a design
change if taken, not applied silently.

## 4. Finding 2 — DDLPI null rate undermines Frame D's stated mechanism

`Due Date of Last Paid Installment (DDLPI)` is null/blank on **2,021,286 of
2,059,564 rows = 98.1415%**.

Recorded precisely: Frame D exists to test whether Upgrade H's Theil's
U≥0.99 near-bijection criterion fires on a definitional near-duplicate. A
feature present on under 2% of rows **cannot form a near-bijection with
the target**. If Upgrade H does not fire under these conditions, the
result would be **uninterpretable** — indistinguishable between
non-transfer of the threshold (the finding this study is designed to
detect) and the feature simply being absent from almost every row.

This was **not anticipated** in the pre-registration and is a design flaw
in Frame D found by measurement, not a data-quality defect in Freddie
Mac's files.

**The response is not decided here.**

## 5. Frame D target base rate (healthy)

Distinct loans ever reaching numeric status >= 3: **2,708 (5.4160% of
50,000)**.

Recorded: the delinquency **target** is well populated even though the
DDLPI **feature** is not (§4). The two problems are independent — a
healthy target base rate does not resolve the feature-sparsity problem.

## 6. Delinquency status code table — source gap now empirically closed

Addendum 01 §2 recorded that the guide's code table is truncated after 03
("etc."). Measurement shows **74 distinct values**: a continuous numeric
run from 00 upward (00 through 72 observed, tapering to 1–2 occurrences at
the top), plus `RA` (206 rows). `XX` does not occur in this file, and no
blank/empty status occurs.

This is consistent with the guide's statement that the value corresponds
to days delinquent and is capped at 99, and confirms the numeric-comparison
implementation specified in Addendum 01 §2. `RA` remains the only
non-numeric value requiring explicit handling.

## 7. Other null rates measured

```
ELTV:                          0.0000% null
Borrower Assistance Plan:     98.4974% null
Delinquency Due to Disaster:  98.5776% null
Actual Loss:                  99.9968% null
```

## 8. Predictions unchanged

The directional predictions (pre-registration §6) and falsification
conditions (§7) are **unchanged** by this addendum. This document records
measurements only. Any revision to Frame P or Frame D arising from
Findings 1 and 2 will be recorded as a **separate** dated addendum, so
that the original predictions and the measurements that challenged them
remain separately auditable.
