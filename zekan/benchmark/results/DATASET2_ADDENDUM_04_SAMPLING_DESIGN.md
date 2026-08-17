# Addendum 04 — Sampling Design

Addendum 04 to `DATASET2_FREDDIEMAC_PREREGISTRATION.md` (`b582e2f`),
Addendum 01 (`0b66849`), Addendum 02 (`9892522`), Addendum 03 (`4ae3fe1`).
Dated 2026-08-16.

**Status: SAMPLING DESIGN DECIDED. NO FRAME BUILT, NO AUDIT RUN.**

## 1. Data scope

**Decision: full 2018 Q1 vintage only. Q2–Q4 downloaded but not used.**

Rationale: Q1 alone yields 381 loans with Zero Balance Code in
`{02, 03, 09}` (0.1284% of 296,816 loans), a 7.2x improvement over the
sample's 53, at an essentially unchanged event rate (sample: 0.1060%).
Additional quarters add compute without changing what the study can
conclude.

Manifest hashes:
```
orig_2018Q1.txt: b1c0379c80f3eeff80474aadc477eeebf746dc368425ea7edd5e8705231b86a7
perf_2018Q1.txt: 430dac486d3aa7a60187c2c697e38f7873dc82a457bdacd91722e58a63a320bf
```

Measured structure: **296,816 loans**, **14,285,575 performance rows**,
periods **201801–202603**, median **36** periods per loan, join
**296,816/296,816** both directions with **zero unmatched**. Delinquency
status has **77 distinct values** with `RA` (1,722 rows) the only
non-numeric value.

## 2. Sampling design

**Decision: case-control sampling at the loan level, seeded, shared
across all three frames.**

- Unit is the **loan**, not the row: all periods of a sampled loan are
  retained. Row-level sampling would destroy the panel structure Zekan
  requires.
- All loans with Zero Balance Code in `{02, 03, 09}` are retained (**381**).
- Non-event loans are randomly sampled at a **1:4 ratio** (approx **1,524**
  controls), giving approx **1,905** loans and approx **69,000** rows at
  ~36 periods per loan.
- **One shared sample** is used for Frames C, P, and D. Only the target
  and feature set differ between frames, so the frames remain comparable.
- Ratio rationale: power gain from additional controls flattens sharply
  beyond approximately 4:1 in case-control designs. A 1:20 ratio was
  considered and **rejected** as costing roughly 5x the compute for
  negligible additional power.

## 3. What this changes, stated plainly

Frame P's positive rate rises from **0.1284%** (population) to
**approximately 20%** (sample). This is **deliberate enrichment**, not
preserved prevalence.

This **revises** the claim in Addendum 03 §1 that "class imbalance is
preserved exactly" — that statement was true of moving to the full
vintage, and is **not** true of the sampling design decided here. Recorded
as a correction; Addendum 03 is not edited.

Frame C/D's positive rate rises from **5.51%** (loans ever reaching status
>= 3, population) to an estimated **~24%** at loan level, because event
loans are also overwhelmingly delinquent loans.

## 4. Justification, and its limits

Case-control sampling is standard practice for rare-event credit risk
modeling. The specific technical justification here: `fl` is a
**difference of AUCs**, and AUC is rank-based and largely
prevalence-invariant, so `fl` is far more robust to case-control sampling
than a precision- or PPV-based metric would be.

**The limit, stated explicitly**: this invariance is **partial, not
exact**. The models are **trained** on the reweighted distribution, so
learned decision functions differ from what they would be at population
prevalence. `fl` measured under case-control sampling is therefore **not
guaranteed to equal** `fl` at population prevalence. This is recorded as a
stated limitation of the study, not a solved problem.

## 5. Temporal fold viability — measured, not assumed

Concern raised before measurement: 2018-originated loans reaching
disposition might do so only in 2021+, leaving early expanding-window
folds with zero positives and undefined AUC.

**Measured and resolved**: `{02, 03, 09}` events occur in every year from
2018 onward:
```
2018: 1     2022: 41
2019: 40    2023: 68
2020: 58    2024: 57
2021: 56    2025: 44
            2026: 16
```
Earliest event period: **201811**. **41** events occur before 202001.

Conclusion recorded: every expanding-window fold after the earliest will
contain positives. Only a fold terminating within 2018 would starve.

Temporal expanding-window folds are **retained**. Reason recorded: a
practitioner trains on past periods and predicts forward; random k-fold on
panel data would leak future into past — which is the failure mode this
tool exists to detect.

## 6. COVID regime shift — recorded as a study limitation

Measured: **11,555 of 16,358 loans (71%)** first reach status >= 3 during
calendar 2020, against **1,153** in 2019. Zero Balance Code 01
(prepaid/matured) peaks at **86,602** in 2020 against **39,879** in 2019.

Recorded plainly: Addendum 01 selected the 2018 vintage to avoid COVID
confounding at **origination**. It does not avoid COVID in the
**performance window**. A substantial regime shift sits inside the fold
structure. This is representative of what a practitioner modeling 2018
originations actually faces, and is recorded as a limitation rather than
engineered around.

Also recorded: performance rows decline steeply by year (2018:
**2,810,104** to 2026: **204,366**) as loans prepay out, so
expanding-window folds will be markedly uneven in size.

## 7. Falsification condition revision — fl ordering downgraded

The pre-registration §7 condition *"if the fl ordering P > D > C is
violated, fl's severity scale is not portable"* is **downgraded** from a
falsification condition to a **diagnostic observation**.

Reasoning recorded: the three frames have **different targets**, so their
`fl` values were never strictly comparable — this weakness predates the
sampling design and was not caught when the pre-registration was written.
Case-control enrichment compounds it. An ordering violation would
therefore not cleanly falsify portability.

Recorded as a **weakening** of a pre-registered condition, disclosed as
such. The other three falsification conditions (Frame C false positive,
Frame P miss, Frame D structural non-detection) are **unchanged** and
remain binding.

## 8. Still open

- Frame D has no probe. The Upgrade 1 search pre-committed in Addendum 03
  §4 has **not** been run.
- Exact seed value and the sampling script, to be recorded when written.
- Whether Frame C requires any further feature exclusions beyond those
  listed in pre-registration §5.
