# Dataset 2 Pre-Registration — Freddie Mac Single-Family Loan-Level Dataset

## 1. Date and status

Dated 2026-08-16.

**Status: PRE-REGISTERED, NOT YET RUN.**

No Freddie Mac production data (the full `historical_data_YYYYQn` vintage
files described in the guide's own naming convention — see §3) has been
downloaded. A small local folder, `C:\Users\Hp\Desktop\freddiemac\Sample
Files\` (`origination_sample_file.txt`, `performance_sample_file.txt`), was
inspected in a prior session step — **schema/structure/value-counts only**:
raw line counts (1,000 origination rows / 1,011 performance rows), delimiter,
field-count-per-line cross-check against the layout, distinct-loan count (12),
period range, and `Current Loan Delinquency Status` value counts. No frame was
constructed, no target was defined, and no audit was run against it.

One factual note carried over from that inspection: this local sample does
**not** match the guide's own description of its official sample product.
The guide states (§3, "Single-Family Loan-Level Dataset Sample"): *"The
sample dataset is a simple random sample of 50,000 loans selected from each
full vintage year..."*, and its own naming convention table names sample
files `sample_orig_YYYY.txt` / `sample_perf_YYYY.txt`. The local files are
named `origination_sample_file.txt` / `performance_sample_file.txt` and
contain only 12 distinct loans — neither the naming nor the scale matches
the guide's official sample definition. This is recorded as a fact, not
resolved here; the vintage/subsample actually used for this study is an
open item (§8).

Predictions in §6 are made from schema, documentation, and the structural
inspection above only — never from an audit result, since none has been run.

## 2. Purpose

Second-dataset validation. Every threshold Zekan currently ships —
`warn_floor`, `fail_floor`, the NSL≥1.0 detection gate, Upgrade H's Theil's
U≥0.99 near-bijection criterion, and `SUPPORT_FLOOR` — was calibrated on a
single dataset (Diabetes-130), as this repo's own calibration documents
(`TEST_B_RESULTS.md`, `UPGRADE_H_CALIBRATION.md`, `UPGRADE1_CALIBRATION.md`)
record throughout. This study tests whether those thresholds **transfer** to
a dataset they were never calibrated against — a different domain (mortgage
credit risk vs. hospital readmission), a different panel structure, and an
independently authored schema.

Stated explicitly, before any run: **a negative result here is a publishable
finding, not a failure to be fixed by retuning.** If a threshold calibrated
on Diabetes-130 does not transfer to Freddie Mac data, the correct response
is to record that as a scope limitation of the current calibration — the
same falsification-integrity discipline this repo already applies elsewhere
(see `BOOTSTRAP_CI_PREREGISTRATION.md` §7: "Thresholds and gates are never
tuned to produce a desired outcome"). That discipline applies here
identically.

## 3. Dataset and licensing

**Dataset**: Freddie Mac Single-Family Loan-Level Dataset, per the guide's
own title page: *"Single-Family Loan-Level Dataset General User Guide,
Release 47, July 2026."*

Quoted, on the dataset's stated purpose:

> "Freddie Mac is making the Dataset available as part of a larger effort to
> increase transparency and help investors build more accurate credit
> performance models in support of ongoing credit risk-sharing initiatives."

Quoted, on data quality and usage agreement:

> "The Dataset is a 'living' dataset, and as such may periodically be
> corrected or updated over time. Freddie Mac does not guarantee that the
> information in Dataset is complete or error-free. By utilizing the
> Dataset, you agree to the Dataset Terms and Conditions and are subject to
> the Freddie Mac Web Site Terms and Conditions."

**On non-commercial/academic/research-use wording specifically**: the guide
PDF, as inspected, does **not** contain that wording. "Dataset Terms and
Conditions" and "Freddie Mac Web Site Terms and Conditions" appear only as
**hyperlinks** to external pages; their target text is not embedded in this
document and was not fetched (this task's scope was local-file inspection
only — no web access was performed or authorized). Recording this as a gap
rather than fabricating a quote: **no verbatim non-commercial/academic-use
clause can be quoted from the source actually inspected.** Before any real
download or publication of results derived from this dataset, the linked
Terms and Conditions must be read directly — that is deferred, not assumed
here.

The non-commercial/academic-use wording is published on the dataset's own
web page (https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset),
not in the guide PDF, and states that Freddie Mac requires a licensing
agreement for commercial redistribution of the data in its Single-Family
Loan-Level Dataset, and that use of the dataset continues to be free for
non-commercial, academic/research and for limited use, subject to the
applicable terms and conditions. Sourced from the web page, quoted here as a
secondary reference; the authoritative Terms and Conditions
(https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf)
remain unread and must be read directly before any real download or before
publishing results derived from this dataset.

**Standing project constraint** (this repo's own rule, not Freddie Mac's):
raw data and derived frames are **never** committed to the repo or
redistributed; only audit artifacts (JSON, thresholds, contracts, results
tables) are published. Reproducibility is via a **manifest** (vintage
identifier + file hashes), not data — mirroring the provenance-hash pattern
already used for Test B (`provenance.contract_sha256` /
`provenance.data_sha256` fields observed in `scratch/testB2_10k_histgb.json`
during the earlier inspection).

## 4. Panel structure (guide's own wording)

**Loan Identifier** — identical definition in both the Origination and
Monthly Performance glossaries:

> "The unique identifier assigned to each loan."

Format: `PYYQnXXXXXXX`, with the guide's own breakdown: *"Product F = FRM
and A = ARM; YYQn = origination year and quarter; XXXXXXX = randomly
assigned digits."*

**Period**:

> "The as-of month for loan information contained in the record."

Format: `YYYYMM`.

**Performance-file coverage per loan**, from "Interpreting the Data":

> "One Performance Data file for all of the loans originated during the
> quarter. All performance periods associated with a loan will be contained
> within the same Performance Data file."

> "The monthly performance data file contains monthly loan-level credit
> performance and actual loss data for each loan, starting from the time of
> loan acquisition by Freddie Mac until the earlier of a termination event,
> which is the last period of performance data available for any loan in
> the Dataset."

**Entity_id / period mapping this implies**: `entity_id = Loan Identifier`
(the one field present, identically defined, in both files); `period =
Period` (`YYYYMM`), one row per loan per month. The panel is **unbalanced
by construction** — the guide's own wording ("until the earlier of a
termination event") means each loan's row count varies with how long it
survived before termination, not a fixed window. This is consistent with
what the earlier structural inspection measured directly on the local
sample: periods-per-loan ranged 14–250 (median 56.5) across the 12 distinct
loans present.

## 5. Frames — three, defined now

### Frame C (clean / negative control)

- **Target**: serious delinquency, derived from `Current Loan Delinquency
  Status` (exact cutoff deferred — §8).
- **Features included**: origination-file attributes, plus contemporaneous
  performance fields that are not outcome-adjacent — `Current Actual UPB`,
  `Loan Age`, `Remaining Months to Legal Maturity`, `Current Interest Rate`,
  `Current Non-Interest Bearing UPB`, `Current Interest Bearing UPB`,
  `Estimated Loan-to-Value (ELTV)`, `Servicer Name`.
- **Features excluded**: every field logically or temporally downstream of a
  credit event or disposition outcome — `Zero Balance Code`, `Zero Balance
  Effective Date`, `Zero Balance Removal UPB`, `Delinquent Accrued Interest`,
  `MI Recoveries`, `Net Sales Proceeds`, `Non-MI Recoveries`, `Total
  Expenses` and its components (`Legal Costs`, `Maintenance and Preservation
  Costs`, `Taxes and Insurance`, `Miscellaneous Expenses`), `Actual Loss`,
  `Cumulative Modification Costs`, `Current Period Modification Costs`,
  `Bankruptcy Cramdown Costs`, `Due Date of Last Paid Installment (DDLPI)`,
  `Underwriting Defect and Major Servicing Defect Settlement Date`,
  `Modification Flag`, `Payment Deferral Flag`, `Interest Rate Step
  Indicator`, `Borrower Assistance Plan`, `Delinquency Due to Disaster`,
  `Mortgage Insurance Cancellation Indicator`.
- **Mechanism tested**: false-positive rate. Does the engine correctly
  report TRUSTED/PASS on an honestly constructed feature set with no
  outcome leakage?

### Frame P (positive control)

- **Target**: credit event derived from `Zero Balance Code`. Quoted, the
  guide's stated meanings for the three codes defining the event:

  > "02 = Third Party Sale"
  > "03 = Short Sale or Charge Off"
  > "09 = REO Disposition"

- **Features included**: post-disposition loss fields, included
  **deliberately** — `Actual Loss`, `Net Sales Proceeds`, `MI Recoveries`,
  `Non-MI Recoveries`, `Total Expenses` and its components, `Delinquent
  Accrued Interest`, `Zero Balance Removal UPB`.
- **Features excluded**: none beyond what Frame C excludes for other
  reasons; this frame's entire point is that the included fields ARE the
  leak.
- Quoted, on why this frame is a known leak by the source's own definition:

  > "Actual Loss is calculated for loans with Zero Balance Codes of 02, 03,
  > 09, and 15."

  > "Actual Loss data components of Net Sale Proceeds, Expenses (Legal
  > Costs, Maintenance and Preservation Costs, Taxes and Insurance, &
  > Miscellaneous Expenses), MI Recoveries, Non-MI Recoveries, Zero Balance
  > Removal UPB (disclosed for all Zero Balances, not just dispositions),
  > and Delinquent Accrued Interest will be disclosed at property
  > disposition."

  The guide's own `Actual Loss` calculation formula makes the relationship
  arithmetic, not merely correlational: *"Calculation: (Zero Balance Removal
  UPB + Net Sale Proceeds) + Delinquent Accrued Interest + Expenses + MI
  Recoveries + Non-MI Recoveries."*

- **Mechanism tested**: this frame is stated plainly to be **near-tautological
  by design** — the included fields are the arithmetic components the target
  event triggers, per the guide's own calculation formula above. It exists
  as a **known-answer anchor**: if Zekan does not flag this, nothing else in
  this study can be trusted.

### Frame D (headline / harder test)

- **Target**: serious delinquency derived from `Current Loan Delinquency
  Status` (same target family as Frame C — exact cutoff deferred, §8).
- **Features included**: `Due Date of Last Paid Installment (DDLPI)`,
  otherwise the same feature set as Frame C.
- Quoted, the guide's own definition showing delinquency status is derived
  from DDLPI:

  > "A value corresponding to the number of days the borrower is
  > delinquent, based on the Due Date of Last Paid Installment ('DDLPI')
  > reported by servicers to Freddie Mac."

  And DDLPI's own definition:

  > "The due date that the loan's scheduled principal and interest is paid
  > through, regardless of when the installment payment was actually made."

- **Mechanism tested**: whether Upgrade H's Theil's U≥0.99 near-bijection
  threshold — calibrated exclusively on Diabetes-130 (`UPGRADE_H_VALIDATION.md`
  §29-39, prior inspection) — fires on an **uncalibrated definitional
  near-duplicate**. DDLPI is not identical to delinquency status, but is the
  field the guide's own definition says delinquency status is *computed
  from*, making the relationship a near-deterministic function rather than
  an arbitrary correlation — structurally analogous to, but not assumed
  identical in strength to, whatever near-bijection pattern the U≥0.99
  threshold was originally calibrated against.

## 6. Predictions — directional, committed before any run

No numeric point predictions are made. No real audit has been run against
this dataset (§1); the only inspection performed was schema/structure/
value-counts on a 12-loan local sample, which is not an audit result and
does not inform numeric predictions here.

Committed directional predictions:

- **Frame C**: PASS expected.
- **Frame P**: FAIL expected, `fl` large.
- **Frame D**: DDLPI expected to be flagged. Flagging could come from
  Upgrade H (Theil's U≥0.99) or from Upgrade 1 (`NEAR_CERTAIN`, univariate
  AUC screen), or both. Corroboration by **both** independent checks is
  committed as the strongest possible outcome — mirroring this repo's own
  stated standard for B-3 (`UPGRADE_H_VALIDATION.md`: *"two independent
  checks corroborating the same finding"*), not a new standard invented for
  this study.
- **Expected `fl` ordering**: P > D > C.

## 7. Falsification conditions

- **If Frame C FAILs** → false-positive on unseen data; `warn_floor`/
  `fail_floor` do not transfer and must be re-derived **across both
  datasets**, never patched to fit Freddie Mac alone.
- **If Frame P PASSes** → the engine misses a documented, definitional
  leak. This is the most severe possible failure mode this study can
  surface.
- **If Frame D's DDLPI is not flagged by either screen** → the structural
  checks (Upgrade H, Upgrade 1) are Diabetes-130-specific, not general.
- **If the `fl` ordering (P > D > C) is violated** → `fl`'s severity scale
  is not portable across datasets — a direct hit on the portable-severity
  thesis this whole study exists to test.

Any of these firing is **recorded as a finding**, not silently absorbed.
Thresholds are never tuned to produce the predicted outcome — identical
discipline to `BOOTSTRAP_CI_PREREGISTRATION.md` §7.

## 8. Open items deferred to implementation

- **Exact vintage/quarter** to be used, and target subsample size
  (~100,000 rows intended, for direct comparability with Diabetes-130's
  101,766-row full B-2 run recorded in `TEST_B_RESULTS.md`). Loan-level
  stratified sampling, seeded. Not decided in this document.
- **Exact delinquency-status cutoff** defining "serious." Recorded, not
  guessed: the guide's own `Current Loan Delinquency Status` code table is
  itself truncated — it gives explicit meanings for `00`, `01`, `02`, `03`
  only, then literally writes *"etc."* before jumping to `RA` and `XX`
  (confirmed directly from the guide text during the prior inspection turn).
  This gap in the source is recorded here as a fact affecting cutoff
  selection; it is not filled in by inference in this document.
- **Horizon definition** for the delinquency target (e.g., ever-90+-days-
  delinquent within N months of origination, vs. a point-in-time status
  read) — not decided here.

## 9. Integrity clause

Findings contradicting any prediction recorded in §6 are recorded as dated
addenda. Original predictions in this document are never edited.
