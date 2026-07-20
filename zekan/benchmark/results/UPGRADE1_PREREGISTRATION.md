# Upgrade 1 pre-registration — undeclared-feature screen

## What this document is

This is written before any calibration run, before any implementation code, before the screen exists in any runnable form. It records the method, the thresholds-to-be-derived, the conditions we agree in advance will count as falsification, and the limitations we already know we're accepting — all before we've seen a single calibration number. Same discipline as `TEST_B_PREREGISTRATION.md` and its addenda: write down what we predict and what would count as failure, then go find out, not the other way around. If calibration produces a result that doesn't match what's written here, we record that plainly; we don't quietly redefine the threshold until it looks right.

## Motivation

`TEST_B_RESULTS.md` recorded B-3's result: `naive_auc = temporal_all_auc = deployable_auc = 1.0` exactly, `fixable_leakage = 0.0`, verdict **PASS**, headline *"Zekan found no evidence of leakage within the declared audit scope."* A perfect, maximal AUC, reported beside a clean bill of health — on real hospital data, because the leaking column was never declared forbidden. Pre-flight caught `encounter_id` (ID-like) and two constant columns; it had nothing to say about the one column that perfectly predicted the target.

This screen exists so that signature — a 1.0 (or near-1.0) AUC sitting next to "no evidence of leakage" — can never again print without a warning standing directly beside it.

## Method, locked before calibration

### Signal: univariate AUC, temporal folds only

For every non-forbidden feature, fit the audit's own estimator on that one column alone, evaluated on the **same temporal expanding folds** `engine.py` already builds for the B/C decomposition (`temporal_expanding_folds`, never `random_grouped_folds`, never an in-sample fit) — reusing the hoisted float32 feature matrix (this session's matrix-hoist work) by column position, not rebuilt per feature.

**The invariant, and why it's locked this way:** an in-sample screen (fit and score on the same rows) would flag almost any moderately-correlated legitimate feature as suspicious under a flexible enough model — it isn't leakage-specific, it's just what flexible models do on training data. A random-fold screen doesn't test whether a feature's information would actually have been available at prediction time; it can't distinguish "genuinely predictive" from "encodes something from later in time," because it never respects a real before/after boundary. This is the exact reason `engine.py`'s own A/B/C decomposition uses temporal folds for B and C and only uses random folds for A (the optimistic upper bound) — the screen is bound by the identical logic, not a separate judgment call.

This gets a locked wiring test before anything else, not just a design note: construct a feature whose univariate AUC materially differs between temporal and random partitioning (a future-leak fixture, in the shape of `inject_graded_future_leak` or equivalent), score it both ways, and assert the screen's reported score equals the **temporal** figure. A test that only checks "the screen returns a number" cannot catch a copy-paste that wires in `rand_folds` by mistake; this one can.

### Multiple-testing control: Benjamini-Hochberg FDR

Screening every non-forbidden feature is running one hypothesis test per feature — 48 simultaneous tests on the Diabetes-130 frame, more on wider data. A single fixed raw AUC cutoff doesn't account for this: as column count grows, the chance that *some* honest feature clears a fixed bar by pure noise grows with it, silently, with no visible change in the tool's behavior. Benjamini-Hochberg FDR control is applied across all screened features' scores for the `SUSPECTED` tier; the FDR level itself is **not fixed here** — it is set from calibration evidence (see Calibration Plan) and recorded, not assumed in advance.

### Two tiers, annotate-only, never verdict-changing

- **`SUSPECTED_UNDECLARED_LEAK`** — clears the FDR-controlled threshold. A statistical flag: "this feature's univariate signal is higher than we'd expect from chance across this many simultaneous tests," nothing stronger.
- **`NEAR_CERTAIN_UNDECLARED_LEAK`** — a **separate, absolute** criterion, not a stricter FDR level, for the regime where a feature is functionally a copy of the target. FDR/percentile logic degenerates exactly at the boundary that matters most here: at AUC ≈ 1.0, a rank-based or percentile-based rule can't meaningfully separate "the worst near-perfect feature" from "the second-worst near-perfect feature" — they're all near the top by construction. The absolute criterion must therefore flag **every** feature tied at or above it, not just the single highest-ranked one — a dataset can (and B-3's did) contain more than one target-adjacent column.

Both tiers are **annotate-only**: they attach to `report.structural_annotations` exactly like `FORBIDDEN_ENTITY_LEVEL_AGGREGATE` and `CORRELATED_LEAK_PAIR` do today, and never alter `policy_decision.verdict`. Registry classification mirrors `CORRELATED_LEAK_PAIR`'s existing row in `zekan/detectors/schema.py`: `source_layer=FLAGGED_SUSPICIOUS`, `confirmed=False` — a univariate score is suggestive, not a confirmed statistical gate the way the permutation null is. `NEAR_CERTAIN` carries a higher intrinsic `severity` than `SUSPECTED` in the registry (mirroring how `ENTITY_CONTAMINATION` sits above `ENTITY_CONTAMINATION_RISK`), but `confirmed=False` applies to both — neither tier claims to have confirmed a genuine leak the way the permutation-null-backed `TEMPORAL_LEAKAGE` issue type does.

### Screenability gate

A feature is only scored if it clears a minimum-information floor: a minimum non-missing row count, and a minimum minority-class representation actually present within the temporal test folds it would be scored on. A column that's 97% missing (`weight`, in Diabetes-130 — deliberately preserved per Test B's own setup) can produce a freak AUC on an effectively tiny sample. A feature that fails this floor is reported **`not screenable`**, with the reason, rather than silently scored on noise or silently skipped without a trace. Honest non-coverage beats noisy coverage, and it beats silent coverage even more — both are visible in the annotation record.

### Wide-data behavior

When the feature count is large enough that fully temporal-scoring every column is impractical, a fast pre-rank (a cheap correlation-style pass, not a full temporal-fold model fit) selects the top-K candidates for full scoring. The cap value and the pre-rank method are both written to provenance. The JSON always reports **"screened X of Y features"** explicitly — silence about a feature is never allowed to be mistaken for that feature having been cleared. An un-screened feature is a stated fact, not an absence.

### Contract allowlist: `known_strong_features`

A new `PredictionContract` field, `known_strong_features`, lets a domain expert name features that are legitimately strong and known in advance (the pre-registration's own `number_inpatient` case, or a domain example like a prior-fraud flag sitting at AUC 0.95). Naming a feature here suppresses the `SUSPECTED` tier for it specifically.

**It never suppresses `NEAR_CERTAIN`.** Nothing legitimate scores ~1.0 against a binary target — that's the entire point of the two-tier split: `SUSPECTED` is a judgment call a domain expert can reasonably override with domain knowledge; `NEAR_CERTAIN` is not a judgment call. Any suppression that does fire is recorded explicitly in the JSON (which feature, that it was contract-declared, not silently dropped) so an auditor reading the report can see a flag was waived, and by what, rather than wondering why an obviously-strong feature never appeared.

### Provenance + diff

A new provenance field, `undeclared_screen` (e.g. `"univariate_v1"`), added the same additive way `null_scheme` and `null_stopping` were — a version string, not a boolean. `zekan diff` treats a screen-version mismatch between two artifacts the same way it already treats a `null_stopping` mismatch: a notice, not silence, since scores from two screen versions aren't directly comparable. Beyond the version-mismatch notice, `diff` also surfaces **new and resolved annotations** between two audits (an annotation present in the new artifact but not the old one, or vice versa) as its own explicit diff output — an undeclared-leak flag appearing or disappearing between two runs of the same pipeline is exactly the kind of change a diff-based workflow exists to catch.

## Calibration plan (what will be measured, stated before measuring)

- **The honest-feature score distribution.** Univariate AUC of all 47 legitimate Diabetes-130 features from the B-2 frame (48 features minus the 1 declared-forbidden `planted_leak`), plus the F2b synthetic fixtures for graded coverage across a wider range of honest-signal strengths than one real dataset alone provides.
- **Named anchors, pre-committed as falsification conditions, not as targets to hit:**
  1. `number_inpatient` — the strongest honest predictor identified in Test B — **must not** be flagged at whatever threshold calibration selects. If the calibrated threshold *would* flag it, that is a **design failure to report**, not a threshold to quietly nudge until it stops firing. We are pre-committing to report that outcome exactly that way if it happens.
  2. Raw `readmitted` (B-3's demonstrated leak, univariate AUC ≈ 1.0) **must** be flagged at the `NEAR_CERTAIN` tier. If it isn't, the absolute criterion is wrong and that is reported as a failure of this design, not adjusted after the fact to make this specific case pass.
- **The threshold is set against the honest distribution's tail, and the margin is the evidence.** The gap between the strongest honest feature's score and the leak's score is recorded as part of the calibration record, explicitly, not just the final threshold number. **If the honest distribution and the leak overlap** — no clean separating gap — **that is a finding to record and STOP on**, not a boundary to split arbitrarily in the middle of an ambiguous region. A screen calibrated by splitting an overlap in half is a screen whose false-positive and false-negative rates were never actually measured, only assumed.
- **Calibration context, stated explicitly, not left implicit:** Diabetes-130 (real data) plus the F2b synthetic fixtures, scored under the **histgb** default estimator (Tier 3 Phase C). The resulting threshold is **estimator-coupled** — histgb and rf have already been shown, in this project's own calibration history, to produce different NSL magnitudes on identical data; there is no reason to assume univariate AUC thresholds would be estimator-invariant either. If the default estimator changes again in the future, this calibration is invalidated and must be re-run, not assumed to still hold. Every annotation this screen ever emits displays **both** the feature's score **and** the threshold it was compared against, so a user auditing a different domain, under a different estimator, can weigh the flag against their own domain knowledge rather than trusting a bare "flagged" label.

## Validation conditions (pre-registered)

- **B-1 re-run:** verdict unchanged (TRUSTED/PASS). `number_inpatient` **not** annotated at either tier.
- **B-3 re-run:** verdict unchanged (**PASS** — the screen never changes verdicts, by design, regardless of what it finds). `NEAR_CERTAIN_UNDECLARED_LEAK` annotation **present**, naming `readmitted`, rendered **prominently** beside the verdict in both the text and HTML views (not appended after "WHAT TO FIX FIRST" the way today's structural annotations are — a `NEAR_CERTAIN` finding earns different visual weight than a routine structural note), with the univariate AUC number shown directly in the rendered text, not only in the JSON.
- **B-2 re-run:** `planted_leak` is declared forbidden there, so it is outside this screen's scope (non-forbidden features only) — the screen must **not** duplicate-flag it. No honest B-2 feature is newly flagged either.
- **The temporal-vs-random wiring test passes** (see Method, above) — a known temporally-leaky feature scores differently under each fold type, and the screen's reported score matches the temporal figure specifically.
- **Resilience, pre-registered as a requirement, not an afterthought:** a single probe's exception can never take down the whole audit. Each probe (including this screen) runs isolated; a caught exception surfaces as its own annotation (a `PROBE_FAILED`-shaped record — exact `IssueType` naming to be finalized at implementation, not pinned here) rather than propagating. The screen also respects a soft time budget and reports honest partial coverage (`"screened 31/48"`, using the same "screened X of Y" surface the wide-data cap already requires) rather than hanging the audit trying to finish every column.

## Scope honesty (accepted limitations, documented now, not built around later)

- **Univariate only.** This screen cannot see multivariate or combination leakage — two individually-innocent columns that jointly encode the target will pass clean. A per-feature permutation-null approach (spec 2) is the designed v2 upgrade path for this gap; it is not part of this build.
- **Threat model is accidental leakage.** This screen assumes a leak nobody noticed, not a leak somebody is actively hiding. An adversary who deliberately launders a leak through noise, binning, or a nonlinear transform specifically to defeat a univariate AUC check can do so. This is a screen for the honest-mistake class of leakage B-3 demonstrated, not an adversarial-robustness guarantee.
- **Binary targets only**, matching Zekan's current overall scope. No provision is made here for multiclass or regression targets.
- **Time-varying leakiness gets averaged, not localized.** A feature that is honest early in the observed period and leaky only late (or vice versa) receives one averaged score across all temporal folds, which can under- or over-state either phase. A temporal-stability probe (Upgrade J in the existing roadmap) is the designed answer to exactly this shape of gap — not something this screen attempts to solve on its own.
- **Frame-only visibility.** The screen only ever sees columns that are already present in the DataFrame handed to Zekan. Leakage introduced upstream — in a join, a feature-store pipeline, or an ETL step before the data ever reaches the audit — is invisible to it and remains Upgrade 2 territory, exactly as the existing scope footer on Zekan's decomposition already states for the declared-leakage measurement itself.

## What still counts as success

Unchanged from the standard this project has already held itself to: success is learning the true behavior of this design against real and synthetic evidence, written down honestly, whichever way it goes — not a calibration run massaged until every named anchor happens to land where we hoped. If `number_inpatient` gets flagged, if the honest and leak distributions overlap, if the temporal wiring test fails on first attempt — every one of those is a valid, complete result for this pre-registration, to be reported exactly as it happens, not explained away.
