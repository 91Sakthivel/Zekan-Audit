# Addendum 01 — Frame Decisions

Addendum 01 to `DATASET2_FREDDIEMAC_PREREGISTRATION.md` (commit `b582e2f`),
dated 2026-08-16, resolving the three open items recorded in that document's
§8 ("Open items deferred to implementation").

**Status: DECIDED BEFORE ANY DATA DOWNLOAD OR RUN.**

No Freddie Mac production data has been downloaded at the time of writing.
None of the decisions below is informed by observed results — they follow
only from the guide's own documented field behavior and standard industry
practice, cited explicitly where used.

## 1. Vintage

**Decision: 2018 origination vintage.**

Rationale recorded:
- Modern performance fields are populated for this vintage. The guide's own
  notes state `Estimated Loan-to-Value (ELTV)` is *"Only populated for April
  2017 and following periods"*, and `Borrower Assistance Plan` /
  `Delinquency Due to Disaster` are each *"Only populated for January 2014
  and following periods."* A 2018 origination falls after all three cutoffs.
- Origination predates COVID-19, so the delinquency target is not confounded
  by pandemic-era forbearance programs.
- Roughly eight years of seasoning to the March 2026 performance cutoff
  allows credit events to accumulate.

**Accepted risk, recorded explicitly**: recent-vintage loans have low
credit-event rates, so Frame P's positive class may be small. Low event
rates are representative of real-world mortgage-servicing practice and are
**accepted deliberately** — not engineered around by selecting a
crisis-era vintage to inflate the positive-class count.

## 2. Serious-delinquency cutoff

**Decision: `Current Loan Delinquency Status` >= 03 (90+ days delinquent).**

Rationale: 90+ days past due (DPD) is the standard industry definition of
mortgage default and aligns with the Basel one-year default framework and
GSE serious-delinquency reporting.

Quoted, the guide's own enumerations grounding the numeric reading:

> "00 = Current, or less than 30 days delinquent"
> "01 = 30-59 days delinquent"
> "02 = 60 – 89 days delinquent"
> "03 = 90 – 119 days delinquent etc."
> "RA = REO Acquisition"
> "XX = Not Available"

And the guide's statement on capping: *"This value for any given month will
be capped at 99."*

**Implementation hazard, recorded**: the guide's code table is **truncated
after 03** ("etc." — recorded as a source gap in the original
pre-registration §8, not resolved here, only worked around). Codes 04–98
are not individually documented. The cutoff must therefore be implemented
as a **numeric comparison** over codes that parse as integers, with `"RA"`
and `"XX"` handled **explicitly** as non-numeric values rather than
silently coerced (e.g., `int("RA")` failing loudly, or an explicit
type/membership check before comparison) — silent coercion here (for
example, a failed numeric parse defaulting to 0 or being dropped without
logging) would silently produce a **wrong target**, not just a missing
value.

**RA (REO Acquisition) is excluded** from the delinquency target: it is a
post-outcome disposition state, not a delinquency-severity reading, and
mixing it into the `>= 03` comparison would blur the distinction this study
depends on between Frame D (delinquency, a performance signal) and Frame P
(disposition, an outcome/loss signal). Exactly how RA rows are handled
during frame construction — excluded entirely from the labeled panel, or
retained and treated as a terminal/absorbing state — is **not decided
here**; it is carried forward as an open implementation detail (§5).

## 3. Target time framing

**Decision: 12-month forward horizon.** At period `t`, the label is whether
the loan reaches 90+ days delinquent within the following 12 periods.
Features are drawn from period `t` and earlier only.

Rationale: this is the industry-standard formulation for probability-of-
default modeling and matches the Basel one-year PD window.

**Why point-in-time was rejected, recorded explicitly**: a same-period label
is description rather than prediction — reading the current period's own
delinquency status off the current period's own fields is not a forecasting
task. It would also make Frame D's DDLPI relationship
**arithmetically trivial** rather than a genuine test (DDLPI directly
determines the same-period delinquency status per the guide's own
definition, already quoted in the original pre-registration §5), while
simultaneously risking **spurious failures in Frame C** from other
contemporaneous fields that correlate with same-period status without
being genuine leaks under a forward-looking framing.

**Consequence, recorded**: the final 12 periods of each loan cannot be
labeled under a 12-month forward horizon and will be **dropped**, shrinking
the usable panel. This must be **reported in the results** when the study
is actually run — not applied silently as an invisible row-count reduction.

## 4. Predictions unchanged

The directional predictions (§6) and falsification conditions (§7) in
`DATASET2_FREDDIEMAC_PREREGISTRATION.md` are **unchanged** by this
addendum. This document resolves open implementation parameters only — it
does not revise, soften, or add to the committed predictions or
falsification conditions.

## 5. Still open

- **Subsample size and sampling scheme** (~100,000 rows intended, loan-level
  stratified, seeded) — not decided here.
- **Frame P positive-class rate** — to be **measured** as the first step
  after download, and recorded as a further addendum **before any frame is
  built**, not assumed or estimated in advance.
- **RA row handling in construction** — excluded vs. terminal/absorbing
  state, per §2 above — not decided here.
