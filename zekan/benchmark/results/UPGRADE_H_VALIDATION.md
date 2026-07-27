# Upgrade (H) validation — H4 real-audit results + probe cost

This is the validation evidence for the NEAR_BIJECTION_UNDECLARED_LEAK guard
(`zekan/detectors/near_bijection_probe.py`), pre-registered in
`UPGRADE_H_PREREGISTRATION.md` and calibrated against the real B-1/B-2/B-3
frames in `UPGRADE_H_CALIBRATION.md`. That document measured Theil's U
directly, before any probe code existed. This document records H4: running
the actual registered probe through a real, full `run_audit` call on all
three frames at full scale (`--json`, histgb default, no `--stability`), plus
a controlled, isolated measurement of the probe's own cost. Same discipline
as the two documents above — numbers as measured, not rounded or invented
beyond what was recorded.

**Headline: H4 passes on all three frames, and the annotate-only invariant
holds everywhere.** B-1 stays PASS with zero annotations and `encounter_id`
correctly not flagged. B-3 stays PASS but now carries two independent
annotations on `readmitted` — this probe (Theil's U = 1.0000) and the
pre-existing near-certain-undeclared-leak screen (univariate AUC = 1.0000) —
closing the pre-registered B-3 blind spot exactly as designed. B-2 stays FAIL
on `planted_leak` unchanged, with zero structural annotations from this probe
(`planted_leak` is declared forbidden, so it is not a screen candidate; its
calibrated `U = 0.6417` sits below the `0.99` criterion regardless). No
verdict changed anywhere.

## H4 — full real-audit correctness, all three frames, full 101,766 rows

Command shape: `--json`, histgb (default estimator), no `--stability`.

| frame | verdict | fixable_leakage | structural annotations | `encounter_id` flagged? | screened | wall |
|---|---|---|---|---|---|---|
| B-1 (`testB1_specificity.csv`) | PASS (unchanged) | 0.0 | 0 | No | 48/48 | ~79.7s (cold start) |
| B-3 (`testB3_honest_unknown.csv`) | PASS (unchanged) | 0.0 | 2 (on `readmitted`) | No | 48/48 | ~72.5s |
| B-2 (`testB2_sensitivity.csv`) | FAIL (unchanged) | 0.3094871962253354 | 0 | No | 48/48 | — |

Per-frame detail:

- **B-1**: verdict PASS, unchanged from pre-Upgrade-(H) baseline. `fixable_leakage = 0.0`. Zero structural annotations. `encounter_id` is NOT flagged (the support-floor guard neutralizes it as calibrated). Screened 48/48 candidates. ~79.7s wall (cold start).
- **B-3**: verdict PASS, unchanged. `fixable_leakage = 0.0`. **Two** annotations, both on `readmitted`: `near_bijection_undeclared_leak` (Theil's U = 1.0000) and `near_certain_undeclared_leak` (univariate AUC = 1.0000) — two independent checks corroborating the same finding. `encounter_id` NOT flagged. Screened 48/48. ~72.5s wall. This is the pre-registered B-3 result: the blind spot is closed, and closed by two independent checks agreeing, not one.
- **B-2**: verdict FAIL, unchanged. `fixable_leakage = 0.3094871962253354`, top feature `planted_leak`. Zero structural annotations from this probe — `planted_leak` is declared `forbidden_after_prediction`, so `candidate_features` excludes it from the screen set entirely; its calibrated `U = 0.6417` is below the `0.99` criterion in any case, so it would not have cleared the guard even as a candidate. `encounter_id` NOT flagged. Screened 48/48.

**Annotate-only invariant holds in all three frames**: registering and running this probe changed zero verdicts. B-1 PASS→PASS, B-2 FAIL→FAIL, B-3 PASS→PASS.

## Isolated probe cost (measured, controlled, project venv)

Measured directly by timing `probe_near_bijection(df, contract)` in isolation
(not inferred from audit wall-clock), against the full 101,766-row B-2 frame,
in the project's own `.venv`:

- Full 48-feature pass: **mean 0.786s over 3 repeats** (min 0.777s, max
  0.801s), 0 findings on B-2 (expected — `planted_leak` is excluded as
  forbidden, and no honest feature clears `U >= 0.99`).
- **Invoked exactly once per `run_audit` call** — confirmed with a counting
  wrapper around a real `run_audit` run, not inferred. Structural reason:
  `probe_near_bijection` is registered with `needs_folds=False`
  (`severity/audit.py`'s `_build_probe_registry`), and
  `_run_structural_probes` — its only call site — runs once in `run_audit`,
  after `severity_result` is already computed, never inside the
  permutation-null loop.
- Cost is spread evenly across the ~48 per-feature `groupby` calls, roughly
  15–75ms each — no pathological feature. `encounter_id` (101,766 distinct
  raw values, pooled to 1 after the support floor) is the single slowest
  feature at ~74ms, still a small fraction of the total.

## Cost finding, stated honestly

An earlier, uncontrolled A/B comparison (single run each, not repeated,
not isolated from other machine activity) showed a 187s wall-clock delta
on the full B-2 frame: 1021.6s with the probe registered vs. 834.8s without.
Direct, isolated, controlled measurement of the probe itself (above) shows
it costs **0.79s** — **237x smaller** than that 187s delta. The 187s
therefore cannot be the probe's own cost; it is run-to-run variance in an
uncontrolled, single-sample comparison, not a real regression this probe
introduces.

This is the **third** instance in this project of an obvious-looking timing
culprit that controlled measurement did not support, following the same
pattern as the 1458s B-2 figure (`UPGRADE1_CALIBRATION.md`'s 2026-07-21
correction — a single uncontrolled, contended-machine mtime-delta run,
later shown by a clean re-run to be ~30% *faster* than baseline, not 2.6x
slower) and the pool-churn hypothesis. **Lesson, restated once more**:
wall-clock deltas from single, uncontrolled runs are not evidence on their
own — isolate the component before attributing cost to it.

## Residual limits (carried from calibration, unchanged)

- Single dataset (Diabetes-130).
- Single base rate (11.16%).
- Screen untested under `--stability`.
