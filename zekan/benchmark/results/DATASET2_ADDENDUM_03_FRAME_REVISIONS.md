# Addendum 03 — Frame Revisions

Addendum 03 to `DATASET2_FREDDIEMAC_PREREGISTRATION.md` (`b582e2f`),
Addendum 01 (`0b66849`), Addendum 02 (`9892522`). Dated 2026-08-16.

**Status: FRAME REVISIONS DECIDED. NO FRAME BUILT, NO AUDIT RUN.**

## 1. Frame P — decision

**Decision: obtain the full 2018 vintage (`historical_data_2018.zip`)
rather than the 50,000-loan official sample.**

Rationale recorded: Addendum 02 Finding 1 measured only 53 loans with Zero
Balance Code in `{02, 03, 09}` in the sample, which may not survive
temporal fold-splitting. The full vintage contains substantially more
loans at the **same event rate**.

Stated explicitly, what is and is not being changed:
- `n` increases.
- The class imbalance is **preserved exactly** — no rebalancing, no
  oversampling.
- No change of vintage (still 2018, per Addendum 01 §1).
- No change to the target definition (still `Zero Balance Code` in
  `{02, 03, 09}`, per pre-registration §5).

Increasing `n` for statistical power is distinct from threshold tuning —
the same distinction Addendum 02 §3 recorded when listing candidate
response (c) without choosing it.

This is recorded as a **design change** from the sample used in Addendum
02. The sample-based measurements in Addendum 02 remain on record
**unchanged** — they are not retracted, only superseded for the purpose of
frame construction going forward.

Recorded as open:
- Whether periods will be subsampled for tractability.
- The full-vintage file hashes (`historical_data_2018.zip` contents), to
  be added to the manifest when downloaded.

## 2. Frame D — DDLPI eliminated as the near-bijection probe

Measurements that eliminate it, all under the null test
`raw_string.strip() == ''` (Addendum 02's null test, unchanged):

- Overall DDLPI null rate: **98.1415%** (2,021,286 of 2,059,564 rows).
- Conditional null rates by delinquency status (row level):
  - status 0: **98.0934%** null (2,001,945 rows)
  - status 1: **99.8931%** null (22,445 rows)
  - status 2: **99.7448%** null (7,444 rows)
  - status >= 3: **99.8292%** null (27,524 rows)
  - status >= 6: **99.7750%** null (16,891 rows)
  - status RA: **91.2621%** null (206 rows)
- Cross-tab: of 27,524 rows with status >= 3, DDLPI populated on **47
  (0.1708%)**.
- Loan level, across the 2,708 loans ever reaching status >= 3: median
  fraction of rows with DDLPI populated = **0.0125** (min 0.0000, max
  0.2500).
- Of the 38,278 rows where DDLPI **is** populated, **38,170** are status
  `"00"`.

Recorded plainly: DDLPI is sparse precisely where the target fires, and
its population is **concentrated on current loans** rather than
delinquent ones. The guide's definitional statement that delinquency
status is computed from DDLPI is a servicing-process fact that does not
manifest as a usable feature relationship in this file.

**Both candidate uses are ruled out by these measurements**:

- **(a) Near-bijection probe** — no coverage where the target lives (0.17%
  of status>=3 rows have DDLPI populated).
- **(b) Sparsity-as-leak probe** — missingness is **anti-correlated** with
  the outcome (DDLPI is populated almost exclusively on `status == 00`
  rows), so there is no leak signal in the missingness pattern itself.

## 3. The near-bijection question remains open

The structural-threshold question is **unchanged** and remains the
study's central question: Upgrade H's Theil's U >= 0.99 criterion was
calibrated exclusively on Diabetes-130, and whether it transfers to any
other dataset is still untested. Eliminating DDLPI removes a **probe**,
not the **question**.

The same limitation applies to `warn_floor`, `fail_floor`, NSL >= 1.0, and
`SUPPORT_FLOOR`: all were earned on a single dataset and are applied as
though general — recorded already in the original pre-registration §2,
restated here as still standing.

## 4. Next step, pre-committed before searching

Recorded, before any search is run: the replacement structural candidate
will be identified by **running Zekan's own Upgrade 1 screen** over the
candidate feature set, not by hand-picking a field from the schema.

Rationale: hand-picking after seeing the data risks selecting a feature
because it produces a desired result; letting the tool surface candidates
is both more honest and closer to real use.

Stated plainly, the pre-commitment: whatever the screen surfaces
(**including nothing**) is recorded as the outcome. If no near-bijection
candidate exists in this dataset, that is recorded as a finding and Frame
D is dropped, with the structural threshold reported as still untested
off Diabetes-130.

## 5. Predictions

Pre-registration §6 predictions for Frame C and Frame P are **unchanged**.

The Frame D prediction (DDLPI flagged by Upgrade H and/or Upgrade 1) is
**voided by measurement, not revised** — recorded that it was falsified
by the data before any run, and that no replacement prediction is made
until a replacement probe is identified per §4. Original text in the
pre-registration is not edited.
