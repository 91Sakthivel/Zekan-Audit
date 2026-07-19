# Test B results — Diabetes-130, real hospital data

## What this document is

This is the consolidated result of Test B: `TEST_B_PREREGISTRATION.md` plus
its five addenda, run against actual real-world data rather than a
synthetic benchmark fixture. Everything predicted in advance is checked
here against what actually happened, plainly, including where Zekan itself
broke and where it has a real, demonstrated blind spot. Nothing here is
softened after the fact. Where a result doesn't match what was predicted,
that is written down as exactly that — not explained away.

## Dataset and setup

Diabetes-130 (UCI dataset #296, CC BY 4.0): 101,766 hospital-stay rows drawn
from 130 US hospitals, 1999-2008. Confirmed by reading the file directly:
71,518 distinct patients, of whom 16,773 (about a quarter) have two or more
visits, and those repeat visits account for 46% of all rows — real,
usable entity structure, not a synthetic construction. `patient_nbr` is the
entity column throughout.

There is no real calendar timestamp anywhere in the raw data (Addendum 2).
A derived `period_ordinal` column — visit records ranked into 24 sequential
buckets by their true relative order — stands in for time. It is labeled
here, and in every artifact, as **ordinal, not calendar**: real order,
translated into a form Zekan can read, not an invented clock.

Categorical columns (race, gender, age bucket, three diagnosis codes, ~20
drug-dosage columns, and others) are ordinal-encoded, mapped from each
column's own sorted-unique values (Addendum 3) — not one-hot, not target
encoded, and not cleaned. The `'?'` sentinel and every other messy raw value
is encoded like any other category. `weight` (97% missing), `medical_specialty`
(~49% missing), and `payer_code` (~40% missing) are all left exactly as
they are. **The mess is deliberately preserved** — a tool that only copes
with a scrubbed CSV isn't demonstrating anything real.

Target: `readmitted_lt30`, derived from the original three-way `readmitted`
column, binary. At full scale (101,766 rows): 11,357 positive / 90,409
negative = **11.16% positive**.

## The three results

### B-1 — Specificity: MET

**Predicted:** Zekan should not flag `number_inpatient` — a strong, honest,
legitimately-known-in-advance predictor — as leakage.

**Result:** verdict `PASS`, `fixable_leakage = 0.0`. `naive_auc =
deployable_auc = 0.6311` — B and C are identical because B-1 declares no
forbidden columns at all (Addendum 1's design), so there is nothing for the
ablation to attribute and nothing gets flagged, `number_inpatient` included.
Stable across 5 seeds (`stability_seeds_checked=5`, no instability note).

**Condition MET.** No false alarm on an honest, powerful feature.

**Honest scope note:** with `forbidden=[]`, this is a pure specificity test
of the *declared* path — B-1 never exercises the permutation null or the
ablation/attribution machinery at all (both are gated behind having at
least one forbidden column). It answers "does Zekan stay quiet when nothing
is flagged," which is the right question for B-1, but it says nothing about
how Zekan behaves when something legitimate *is* declared forbidden by
mistake, and nothing about the (unbuilt) undeclared-feature screen B-3
motivates below.

### B-2 — Sensitivity: MET

**Predicted:** Zekan detects a planted, declared, target-copy leak at high
severity.

**Result, full 101,766 rows:** verdict **FAIL**. Detected on **both**
channels (within-entity and across-entity), `p = 0.0099` (`p_is_upper_bound
= true` on both — zero permutation draws out of 100 ever reached the
observed leakage; see the p-floor finding below), NSL **35.56** within-entity
/ **100.26** across-entity, `planted_leak` ranked **#1** by the ablation.
`fixable_leakage = 0.3095` (`naive_auc = 0.9639` -> `deployable_auc =
0.6550`). Wall-clock: **557s**.

**Condition MET.** The most blatant form of leakage there is, caught cleanly
at full scale, on both detection channels.

**Scale comparison — the fl-vs-n drift:**

| n | fl | naive | deployable | NSL (within) | estimator |
|---|---|---|---|---|---|
| 10,008 (stratified) | 0.4008 | 0.9664 | 0.5894 | 5.45 | histgb |
| 40,008 (stratified) | 0.3437 | 0.9638 | 0.6423 | 14.33 | histgb |
| 101,766 (full) | 0.3095 | 0.9639 | 0.6550 | 35.56 | histgb |

fl drifts down as n grows: **0.40 -> 0.34 -> 0.31**. The mechanism is
visible directly in the table: `naive_auc` (A) barely moves across scale —
it's already near its ceiling at 10k — but `deployable_auc` (C, the
safe-features-only model) **rises** with n, from 0.589 to 0.655, as model C
gets enough data to actually learn the real, honest signal in the safe
features. `fixable_leakage = B - C`, so as C climbs toward B, the gap
shrinks. This is C improving, not the leak weakening — the leak itself
(`naive_auc`) is essentially flat across scale. NSL rises sharply in the
same direction (5.45 -> 14.33 -> 35.56) precisely because the permutation
null tightens with more data even as fl itself shrinks — the same
null-tightening mechanism `nsl_boundary_sweep.py` documents.

Stated plainly: **fl is n-sensitive through the quality of the safe model**,
not through the leak getting smaller. It does not threaten the floors here —
0.31 sits nowhere near `fail_floor=0.15`, comfortably FAIL at every scale
measured — but it is real calibration knowledge worth recording: fl at one
sample size is not a fixed constant that transfers unchanged to another.

*Provenance note on the 10k row above:* the permanent artifact
`scratch/testB2_strat.json` (predating the histgb default flip, estimator
recorded as the pre-Tier-3-Phase-C literal `"default"`) shows fl=0.390,
NSL=6.13 for the same 10k data under the *old* default (rf) rather than
histgb. The histgb-estimator figures in the table above (fl=0.4008,
NSL=5.4516) are now backed by a persistent artifact,
`scratch/testB2_10k_histgb.json`, which reproduces the earlier same-data,
same-contract Tier 2b-final validation run identically (fl, NSL, verdict,
top_feature, `naive_auc`, `deployable_auc` all match to full precision).
histgb is used throughout the 10k/40k/100k comparison so the estimator is
held constant across scale, matching Tier 3 Phase B's own finding that rf
and histgb agree qualitatively but not to the decimal on NSL magnitude.

### B-3 — The honest unknown: Addendum 1's prediction CONFIRMED

**Predicted (Addendum 1, before this run):** with the raw `readmitted`
column left in, undeclared, Zekan will report a falsely-clean result —
missing an obvious leak because it only ever measures *declared* leakage.

**Result:** `naive_auc = temporal_all_auc = deployable_auc = 1.0` exactly.
`fixable_leakage = 0.0`. `p_value = 1.0`, `NSL = 0.0`. Verdict **PASS**, and
the reported headline reads: *"Zekan found no evidence of leakage within
the declared audit scope."*

**Prediction CONFIRMED**, exactly as written down in advance.

**The mechanism, stated plainly:** the undeclared leaky column sits in both
model A/B (all features) and model C (safe features only) equally — nothing
distinguishes it from any other feature since it was never named forbidden
— so it inflates both sides of the B-C comparison identically and cancels
out completely. The permutation null only ever shuffles *declared* forbidden
columns, so this column is never touched by it either. The ablation/feature-
attribution step is itself gated behind `fixable_leakage > 0`, so with fl
pinned at exactly 0.0, attribution never runs at all — there was never a
chance for it to point at the real cause even in principle.

**The signature to hold onto:** a perfect **1.0 AUC**, reported side by side
with **"no evidence of leakage."** That combination, on its own, in output
from a leakage-detection tool, is the tell — not a hedge, not a low-
confidence caveat, a clean, confident PASS sitting directly next to the
single most extreme AUC value possible.

Pre-flight's own notes on this exact run: it caught `encounter_id` as
ID-like ("should likely be excluded from features") and `examide` /
`citoglipton` as constant, no-signal columns. It had **nothing to say**
about the one column, still sitting undeclared in the feature set, that
predicts the target perfectly. Pre-flight checks for structural properties
of a column in isolation (is it unique, is it constant); it has no
mechanism at all for "does this correlate suspiciously well with the
answer" — a fundamentally different, harder check. That gap is the
concrete, real-data motivation for **Upgrade 1**, an undeclared-feature
screen, as recorded in Addendum 1 before this run and now demonstrated, not
merely argued for, on genuine hospital data.

## Defects Test B found in Zekan

These are findings Test B exists to produce, not footnotes to the three
results above.

**1. `prediction_time` ready-then-crash (Addendum 2, fixed `2832600`).**
Pre-flight only *warned* about an unparseable time column (because a tiny
fraction of visit-ID values happened to accidentally parse as nonsense
dates), so the run proceeded past a green "READY" and crashed mid-audit
trying to sort by date. The one failure mode a trust tool must never have:
saying "fine" and then falling over. Fixed to fail loudly and cleanly at
the gate instead.

**2. Non-numeric feature ready-then-crash (Addendum 3, fixed `8a68c18`).**
Same shape of defect, a different unchecked assumption: pre-flight had no
check that feature columns were actually numeric, so B-1's first real run
sailed past "READY: severity computable" and crashed on `race`'s literal
value `'Caucasian'` inside the model-fitting step. Fixed the same way as
`2832600` — tied to the real thing that breaks (the float conversion),
failing at the gate instead of mid-run.

**3. Superlinear scaling and a slow baseline (Addendum 4) -> Tier 1
(`381d0a2`), Tier 2/2b sequential stopping (`38a06d0`), Tier 3 histgb
default (`b06cebc`).** Measured directly, not estimated: 10,008 rows
completed in 225s; 40,008 rows exceeded 1,800s and was killed at ~44
minutes without finishing; the full 101,766-row B-2 (single seed) didn't
complete in 40+ minutes, and with the 5-seed stability check didn't
complete in 2+ hours. Four times the rows cost more than eight times the
time — worse than linear, something compounding. Before the fixes, **RF
never completed a full-100k B-2 run at all.** After Tier 1's optimizations,
Tier 2/2b-final's adaptive permutation stopping, and Tier 3's histgb
default, the same full-100k B-2 audit **completes in 557s** — the exact
result reported above, not a projection.

**4. The p-floor architectural finding (Addendum 5).** `n_permutations=100`
had been reverse-engineered, whether anyone noticed or not, so that a
zero-exceedance leak's Laplace-corrected p-value (`1/(n+1)`) just barely
cleared `alpha=0.01` — exactly the class of number-tuned-to-pass-a-gate this
project forbids elsewhere, hidden because the fixed design always drew all
100 regardless of the evidence, so the gate always happened to pass by the
time anyone looked. Tier 2b's adaptive stopping exposed it directly: an
unmistakable leak (NSL=6.67) stopped at 36 draws and downgraded from FAIL
to `UNCONFIRMED_HIGH_DAMAGE`, purely because p hadn't run long enough to be
allowed an opinion — not because the leak was small. Resolved by the
asymmetric design (Tier 2b-final): the NOT-DETECTED direction still stops
early with no floor, but a DETECTED conclusion now honestly waits for the
same ~100-draw floor before it's allowed to fire, with `p_is_upper_bound`
surfaced in the JSON so the p-value's own nature — floor value or real count
— is never hidden. `n_drawn=100/100` and `p_is_upper_bound=true` on both
channels in the full-100k B-2 result above confirm this fix is live in the
result this report is built on.

## What remains unvalidated

- **B-1 and B-3 never exercised the null-permutation or ablation path.**
  B-1 declares no forbidden columns (nothing to test the null against); B-3's
  `fixable_leakage` is pinned at exactly 0.0 by the blind spot itself, which
  gates the ablation off entirely. Only B-2 has ever run Zekan's actual
  leakage-measurement machinery on this real dataset.
- **No `--stability` (multi-seed) run at full 100k scale.** Addendum 4
  recorded that a 5-seed stability check on the full 101,766 rows did not
  complete in 2+ hours, even before the performance fixes; that run has not
  been repeated post-fix. Every full-100k result in this report (B-1's
  seed-stability aside, which ran on B-1's much-cheaper `fl=0` path) is a
  single-seed measurement.
- **The across-entity boundary is "validated-safe but not fully earned"** —
  unchanged by this report. B-2 fired `detection_channel="both"` at every
  scale measured, which is consistent with a genuine entity-level structure
  in the planted leak, but this report does not independently re-derive or
  re-calibrate the across-entity NSL boundary against real data; it inherits
  the same caveat `F2b_CALIBRATION.md` already recorded on synthetic data.
- **`sequential_v1` was not used for the closing full-100k runs.** B-2 and
  B-3's headline results (this document's primary evidence) both used
  `null_stopping=fixed_v1`. The asymmetric sequential design (item 4 above)
  is validated on its own 25-cell synthetic sweep and the single 10k real-
  data comparison in `TIER2B_CALIBRATION.md`, but has not been run as the
  closing measurement for the real full-scale B-2/B-3 results reported here.
- **Wall-clock figures throughout (225s, 557s, and the failed 40k/100k
  timings) are single runs**, not repeated or averaged — real measurements,
  not statistically hardened ones.
- **Upgrade 1 (the undeclared-feature screen) is unbuilt.** B-3 demonstrates
  the blind spot exists and is real on genuine data; it does not close it.
  The blind spot remains open in the shipped tool today.

## What Test B earns Zekan the right to claim, and what it does not

Zekan can now claim, with real, messy, 101,766-row hospital data behind the
claim rather than only a synthetic fixture: it stays quiet on a strong,
honest predictor when nothing is declared forbidden (B-1); it catches an
extreme, declared, target-copy leak cleanly, on both detection channels, at
full scale, in under ten minutes (B-2); and its one documented structural
blind spot — leakage in a column nobody thought to suspect — behaves exactly
as its own code predicted it would, not as a surprise (B-3). Two
ready-then-crash defects and one serious performance ceiling were found and
fixed *by this same process*, on this same data, before anyone could rely on
the tool for something that mattered; a fourth, subtler defect (the p-floor)
was caught by Zekan's own validation discipline catching itself mid-fix,
which is arguably the strongest evidence in this whole document that the
discipline works.

It cannot yet claim that this behavior holds under multi-seed stability at
full scale, that the across-entity boundary is calibrated (rather than
merely non-false-alarming) on real data, that the newer adaptive-stopping
design has been exercised as the closing measurement for a real audit, or —
most importantly — that the demonstrated blind spot in B-3 is closed. Test
B's job was to learn the truth about how Zekan behaves on real data, not to
certify it as finished. On that measure, it did its job.
