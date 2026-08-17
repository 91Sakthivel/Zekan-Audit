# Addendum 05 — Censoring Rule

Addendum 05 to `DATASET2_FREDDIEMAC_PREREGISTRATION.md` (`b582e2f`),
Addenda 01 (`0b66849`), 02 (`9892522`), 03 (`4ae3fe1`), 04 (`026a378`), and
the sampling script (`10991fc`). Dated 2026-08-17.

**Status: DEFECT RECORDED, FIX DECIDED, NOT YET IMPLEMENTED.**

## 1. What was built and measured

The shared case-control sample was built deterministically
(`build_sample.py`, seed `20180101`): **381 cases**, **1,524 controls**,
**1,905 loans**, **95,775 performance rows**, verified byte-identical
across three independent runs.

Frame table measurements from `build_frames.py`, as first run:

- Delinquency status parsing: **94,143** numeric, **1,632** RA, **0**
  unparseable.
- Rows dropped by the incomplete-window rule: **22,521**.
- RA rows inside label windows: **10,655**.
- Unjoined performance rows: **0**.
- Final: **73,254** rows, **1,828** loans, periods **201801–202503**.
- `target_delinquency`: **15.8026%** row level (11,576/73,254), **25.4376%**
  loan level.
- `target_creditevent`: **0.5174%** row level (379/73,254), **20.7330%**
  loan level.
- Output SHA256:
  `f1b657c1f471ecaf7b64e211a1bb8dd10c86ac156cc2ff75d8ebcd5ae185f5ed`

Recorded: the delinquency-code hazard flagged in Addendum 01 §2 did **not**
materialize — zero unparseable values were encountered.

## 2. Defect — censoring conflation

Stated precisely: `Zero Balance Code` appears only in a loan's **final**
performance record. Under the 12-month forward horizon, only the row at
`T-12` (where `T` is the disposition period) has the event inside its
window; every row after `T-12` is removed by the incomplete-window drop
rule. The result is **exactly one positive row per case loan**: 379
positive rows across 379 positive loans — a 1:1 ratio.

Root cause: the drop rule treats two materially different situations
identically.

- **(a) A loan whose last observed period is the data cutoff (202603) is
  genuinely censored** — the next 12 months are unobserved, so the row
  cannot be labelled. Dropping is correct.
- **(b) A loan that terminated by prepayment or maturity (Zero Balance
  Code 01) is not censored.** It left the risk pool; no credit event can
  subsequently occur. Its rows are legitimate observed **negatives** and
  are currently being discarded.

Consequence: Frame P's designed loan-level enrichment (20.73%) collapses
to **0.5174%** at row level — the level the model actually trains at. The
case-control design from Addendum 04 is therefore **not achieving at row
level what it was built to achieve.**

## 3. Second, related concern — recorded, not yet resolved

With exactly one positive row per case loan located at a fixed offset from
that loan's final period, any feature encoding proximity-to-termination
becomes predictive of the label. This is recorded as an **artifact of the
framing**, not leakage in the source data, and it may confound Frame P's
interpretation.

Stated plainly: this concern is recorded now and is **not** resolved by
the fix in §4.

## 4. Fix decided

**Decision: distinguish censoring from termination in the label window
rule.**

- A row is **dropped** only if the 12-month window extends beyond the
  loan's last observed period **and** that last period is not a
  termination.
- A loan is **terminated** if its final record carries any non-blank Zero
  Balance Code. For terminated loans, periods after termination are
  treated as **observed with no further event** — rows whose windows
  extend past termination are **labelled** (negative for credit event,
  unless the event is the termination itself) rather than dropped.
- Loans whose final observed period is at or near the data cutoff with
  **no** Zero Balance Code remain genuinely censored and continue to be
  dropped.

Recorded, what this does and does not change: it changes which rows are
**labellable**. It does **not** change the target definitions, the
12-month horizon, the `>= 03` cutoff, the sample, or the seed.

## 5. Predictions for the fix, committed before implementing

Stated plainly, before the change is written:

- `target_creditevent` row-level positive rate is expected to **rise
  materially** above 0.5174%, because case loans will contribute more
  than one positive row each and prepaid loans will contribute observed
  negatives instead of drops.
- The dropped-row count is expected to **fall materially** below 22,521.
- `target_delinquency` rates may also shift, since the same rule governs
  both targets. Recorded that this is **expected and is not a defect**.

Recorded: if the credit-event row rate does **not** rise materially, the
diagnosis in §2 is wrong, and that must be recorded as a finding rather
than pursued with further changes.

## 6. Status of falsification conditions

**Unchanged.** The three binding conditions (Frame C false positive, Frame
P miss, Frame D structural non-detection) still stand. The `fl`-ordering
condition remains downgraded to diagnostic per Addendum 04 §7.

## 7. Still open

- Frame D has no probe; the Upgrade 1 search pre-committed in Addendum 03
  §4 has still not been run.
- The proximity-to-termination artifact in §3.
