# Tier 2b-final — asymmetric sequential stopping, alpha-floor fix

This is the fix for the verdict-flip defect Addendum 5 recorded, built on top
of FIX A (distance-aware NSL interval) and FIX B (persistent worker pool),
both already in the working tree from Tier 2b and unchanged by this phase.
Nothing here touches h, N_min, N_max, alpha, warn_floor, or fail_floor.

## The bug this fixes (recap, see Addendum 5 for the full record)

Tier 2b's FIX A made the decision-stability check unconditional (checked
every batch past `N_min=30`, no longer gated behind "running p already looks
significant") and distance-aware in both directions. That correctly fixed
the measured "median n_drawn=108" defect for clean data, but for a genuine,
strong leak it let the loop stop as early as ~36 draws purely because NSL
looked stable -- while `engine.py`'s reality gate still requires
`p_value < alpha` before NSL is even consulted, and the Laplace-corrected
`p_value = (count_gte+1)/(n+1)` has a hard floor that cannot cross alpha
before ~100 draws when `count_gte==0`, no matter how large NSL is. Measured
directly on `scratch/testB2_strat.csv`: FIX-A-only stopped at 36 draws,
NSL=6.67, and the verdict downgraded from FAIL to UNCONFIRMED_HIGH_DAMAGE --
a real verdict flip on an unmistakable leak.

## Part 1: the alpha floor, derived and corrected

Addendum 5 recorded the floor as `ceil(1/alpha) - 1 = 99`. That figure was
verified against the actual gate before writing any code, and **it was off
by one**:

- `engine.py:497`: `if p_value >= _NULL_ALPHA: <not detected>` -- detection
  needs **p strictly less than alpha**, not `<=`.
- At `n=99`: zero-exceedance `p = 1/(99+1) = 1/100 = 0.01` exactly. `0.01` is
  **not** `< 0.01`. Fails the gate.
- At `n=100`: `p = 1/(100+1) = 1/101 ~= 0.0099`. `0.0099 < 0.01`. Passes --
  and this exactly reproduces the old fixed-N=100 behavior, which is the
  intended backstop.

`_ALPHA_FLOOR_DRAWS` is computed in code (`_derive_alpha_floor_draws` in
`null_baseline.py`) as the smallest `n` satisfying `1/(n+1) < _SEQ_ALPHA` by
direct search over the actual formula -- not hardcoded, not assumed
algebraically. It evaluates to **100**. Addendum 5 (append-only, per its own
discipline) is not edited; this document records the correction instead.

## Part 2: the asymmetric design

Per Besag-Clifford semantics, the two directions are treated differently:

- **NOT-DETECTED** (NSL interval provably below 1.0): stop as soon as
  decision-stable past `N_min=30`. No floor -- "this isn't a leak" is a claim
  the data can support at low n regardless of exceedance count. Unchanged
  from FIX A's behavior in this direction.
- **DETECTED** (NSL interval provably above 1.0): NSL alone is not enough to
  stop. The loop additionally requires `n_valid >= _ALPHA_FLOOR_DRAWS` **and**
  the actual running `p_value = (count_gte+1)/(n+1) < alpha` at the current
  exceedance count before stopping. For the common zero-exceedance case this
  means stopping at (or, due to batch granularity, shortly after) exactly
  100 draws. If `count_gte` is nonzero but below `h=10`, the honest floor for
  *that* count is higher than 100 (the p-value formula requires more draws
  as `count_gte` rises) -- the loop keeps drawing rather than stopping early
  on an NSL claim the p-value gate can't yet back up. Besag-Clifford's
  `h=10` exact rule and the `N_max=500` backstop are both unchanged and still
  apply on top of this.
- RNG: children are still pre-spawned for `N_max` up front, so draw `i`'s
  value is unaffected by any of this -- verified explicitly (see Part 4d).

## Part 3: honest p surface

`p_is_upper_bound` (new field, additive, threaded `NullResult` ->
`SeverityResult` -> `EngineDetection` -> JSON, mirroring the existing
`n_drawn`/`stopped_early` pattern so no existing construction site breaks):
`True` when `p_value` is the Laplace floor (`count_gte==0`), `False` when a
real count backs it, or when the Besag-Clifford exact `h/n` formula applies.

## Part 4: tests

**4a -- 3 mechanical signature fixes.** `_nsl_decision_stable` now takes
`null_99th_lo`/`null_99th_hi` instead of one `null_99th` point (FIX A,
already in the tree). `test_nsl_decision_stable_when_both_ends_agree`,
`test_nsl_decision_unstable_when_bound_straddles_boundary`,
`test_nsl_decision_stable_when_both_ends_below_one`: updated to pass the same
point value for both bounds, reproducing the original degenerate case
exactly. No behavior change.

**4b -- 2 approved behavior re-baselines.**

- `test_sequential_stops_exactly_on_h_exceedance_besag_clifford` (old:
  expected stop at draw 35; a prior turn's investigation found it now stops
  at 30). With this fixture, the 9 early exceedances (all `==1.0`) pin
  `null_99th`'s conservative bound at exactly `[1.0, 1.0]`, making NSL
  provably negative against `observed=0.5` -- a correct, not weakened,
  distance-aware conclusion for this specific data. Rather than hand-tune
  floating-point values to coincidentally dodge that (fragile against future
  changes to the CI method), decision-stability is now mocked out
  (`return_value=False`), isolating Besag-Clifford's counting mechanic
  directly -- the same pattern `test_sequential_runs_to_n_max_when_neither_rule_fires`
  already used.
- `test_sequential_decision_stability_can_defer_stop_past_n_min` (old
  assertion `n_permutations >= 100`, an artifact of the removed pre-Tier-2b
  gate, not a real invariant). Replaced with the actual invariant: a
  zero-exceedance DETECTED case must not stop before `_ALPHA_FLOOR_DRAWS` and
  must not overshoot it materially. Empirically: under FIX-A-only (no
  floor), this exact seed/observed pair stabilizes at n=41; under Tier
  2b-final it correctly waits and stops at n=101 (stability holds from 41
  through 99, is momentarily marginal at exactly 100, clears again the next
  draw).

A **third, unapproved** test break surfaced during this work and is reported
here rather than silently absorbed into the two approved re-baselines: an
internal refactor (extracting `_nsl_interval_bound` so the loop could read
off which side of 1.0 a stable interval landed on) initially made the loop
call that function directly instead of `_nsl_decision_stable`, silently
breaking `test_sequential_runs_to_n_max_when_neither_rule_fires`'s mock of
`_nsl_decision_stable` (it stopped at 100 instead of 500). Fixed by keeping
the loop's call to `_nsl_decision_stable` itself (preserving the mockable
seam) and calling `_nsl_interval_bound` a second time, only when stable, just
to read the direction -- restoring the test to its original, unmodified
behavior without widening the approved re-baseline scope.

**4c -- new tests (6 added, 18 -> 24 in this file):**
`test_sequential_decision_stability_fires_early_not_detected_direction`
(decision-stability alone, no Besag-Clifford help, stops at exactly N_min in
the NOT-DETECTED direction -- also satisfies "ADD a separate test" from 4b);
`test_alpha_floor_draws_is_100_not_99`;
`test_alpha_floor_boundary_n99_fails_n100_passes` (direct arithmetic check of
the `n=99` fails / `n=100` passes boundary);
`test_sequential_detected_direction_waits_for_alpha_floor` (zero-exceedance
DETECTED case, stops at exactly 100, `p_is_upper_bound=True`);
`test_sequential_draw_values_identical_across_n_jobs_even_when_stop_point_differs`
(see 4d); and the existing `test_sequential_determinism_same_seed_same_n_jobs`
parametrization widened from `[1, 2]` to `[1, 2, 12]`.

**4d -- determinism across n_jobs {1, 2, 12} and batch boundaries.**
Repeat-run determinism (same seed, same `n_jobs`, run twice) holds exactly
for all three values. Cross-`n_jobs` comparison is more nuanced and is
reported honestly rather than glossed: `n_jobs=1` vs `n_jobs=2` land on the
**same** stop point (n=30) for the real dataset used here, but this is
because `N_min=30` happens to be divisible by both batch sizes -- it is not
a general guarantee. `n_jobs=12` (batch_size=12) does **not** land on the
same checkpoint (stops at n=36 instead), because the stopping decision is
only evaluated at batch boundaries and 12 doesn't divide 30 evenly. This is
inherent to any batch-checked group-sequential design, present already in
FIX A, and not something this phase introduced or could remove without
checking after every single draw regardless of `n_jobs` (which would defeat
the point of batching). The actual guaranteed invariant -- draw `i`'s value
is byte-identical regardless of `n_jobs` -- was verified directly: the
`n_jobs=1` run's full 30-value array exactly equals the first 30 values of
the `n_jobs=12` run's 36-value array. `fixed_v1` mode (unaffected by any of
this) remains fully `n_jobs`-invariant, confirmed empirically (`n_jobs=1` vs
`n_jobs=12`, identical `n_permutations`, `p_value`, and `null_samples`).

**4e -- real audit, `scratch/testB2_strat.csv --sequential-null --json --jobs 12`
(histgb default, not overridden):**

| | FIX-A-only (buggy) | Tier 2b-final (this fix) | fixed_v1 baseline |
|---|---|---|---|
| n_drawn (within / across) | 36 / 36 | 108 / 108 | 100 / 100 |
| p_value | 0.02703 | 0.009174 | 0.009901 |
| p_is_upper_bound | (field didn't exist) | True / True | True |
| nsl | 6.669 | 5.134 | 5.452 |
| detected | False | **True** | True |
| verdict | UNCONFIRMED_HIGH_DAMAGE | **FAIL** | FAIL |
| detection_channel | `""` | **`"both"`** | `"both"` |
| wall-clock | 56.7s | 101.3s | 150.5s |

**Verdict FAIL is restored**, matching the fixed-N baseline exactly, at
roughly 33% less wall-clock (101s vs 150s) than the fixed-N run. n_drawn
lands at 108 rather than exactly 100 because `--jobs 12` means batch_size=12
and the first checkpoint at or past the 100-draw floor is 108 (12, 24, ...,
96, 108) -- the same batch-granularity effect documented in 4d, and the same
108 the original (pre-any-fix) Tier 2 measurement independently observed.

**4f -- full suite: 750 passed, 0 failed** (up from 744 pre-existing, +6 for
the new tests above; consistent with `test_tier2_sequential_null.py` going
from 18 to 24 collected tests).

## Part 5: re-validation against the 25-cell histgb baseline (hard gates)

`python -m zekan.benchmark.nsl_boundary_sweep --sequential --estimator histgb
--jobs 12 --output zekan/benchmark/results/f2b_calibration_sequential_v2.csv`,
compared programmatically against `f2b_calibration_histgb.csv` (the
fixed_v1/histgb baseline), matched by `(alpha, seed)`.

**(a) Zero verdict/detection_channel flips: PASS.** All 25 cells match the
baseline exactly:

| alpha | seed(s) | verdict / channel (both baseline and this run) |
|---|---|---|
| 0.00 | 0-4 | pass / (none) -- 5/5 |
| 0.60 | 0-4 | fail / both -- 5/5 |
| 1.10 | 0-4 | fail / both -- 5/5 |
| 1.60 | 0-4 | fail / both -- 5/5 |
| 2.50 | 0-4 | fail / both -- 5/5 |

**(b) Clean-cell (snr=0.0) median n_drawn materially below 100: PASS.** All
5 clean cells stopped at `n_permutations=36` (median 36) -- consistent with
the real-audit smoke test's clean-data behavior and well below the 100-draw
floor, which correctly does not apply in the NOT-DETECTED direction.

**(c) Sweep wall-clock faster than 1275s: PASS.** Measured two ways: wall-
clock timestamps around the sweep process (1119s) and the sum of the sweep
script's own per-cell timers (1121.5s over 25 cells). Both agree closely;
reporting **~1120s**, about **12% faster** than the 1275s fixed_v1/histgb
baseline.

Per-cell detail: all 20 leaked cells (`alpha >= 0.60`) landed at
`n_permutations=108` (the same batch-granularity floor-landing as the 4e
real-run measurement, `p_value=0.009174` = `1/109` exactly on every one, all
`p_is_upper_bound`-eligible since `count_gte==0` throughout). NSL ranges:
within-entity `[5.85, 11.36]` on leaked cells, `[-2.15, -0.28]` on clean
cells; across-entity `[16.29, 40.07]` on leaked cells, `[-2.44, -0.56]` on
clean cells -- both channels stay cleanly separated from the ladder boundary
in the same direction as the fixed-N baseline, consistent with gate (a)'s
zero-flip result.

**All three hard gates pass.** Sequential stopping (with the alpha-floor
fix) is not a dead end on this fixture: it delivers the same verdicts as
fixed-N, correctly and without shortcuts, while running faster than fixed-N
overall and roughly 3x faster than fixed-N specifically on clean data.

## What remains unvalidated (stated plainly, not glossed over)

- **One benchmark DGP and one real-data smoke test.** The 25-cell sweep uses
  the same synthetic `make_clean_dataset`/`inject_graded_future_leak`
  fixture as F2b/Tier 3; the only real-data confirmation is the single
  `scratch/testB2_strat.csv` run in 4e. Neither exercises every leak shape
  or scale this tool will see in practice.
- **The "moderate, persistent exceedance count" regime is untested.** Every
  fixture used here (real data included) landed on either `count_gte==0`
  throughout (the common strong-leak or clean case) or the exact `h=10`
  Besag-Clifford path. A case where `count_gte` sits at, say, 3-9 for a long
  stretch without ever reaching 10 -- which per Part 2's design would need
  *more* than 100 draws to honestly cross alpha, and could in principle run
  all the way to `N_max=500` -- was not specifically constructed and
  verified. The logic is believed correct by construction (the honest
  floor for a given `count_gte` is derived directly from the same p-value
  formula, not a separate rule), but it has not been empirically exercised.
- **The across-entity channel's floor behavior is shared code, not
  independently stress-tested.** Within-entity got dedicated fixtures for
  the floor-wait and not-detected-early paths; across-entity uses the exact
  same `estimate_fixable_leakage_null` logic and is covered by the real-data
  smoke test (4e) and the 25-cell sweep (Part 5), but has no isolated,
  mocked unit test of its own for the floor-wait behavior specifically.
- **n_jobs=12's overshoot-past-the-floor amount (108, not 100) is
  dataset/seed dependent**, not a guaranteed constant -- a different batch
  size or a different natural stop point could land further past the floor.
  Only the `N_max=500` backstop bounds this in the worst case.
- **Wall-clock numbers are single runs**, not repeated/averaged, for both
  the 4e real-audit comparison and the Part 5 sweep -- real but not
  statistically hardened measurements, consistent with the same caveat in
  `TIER3_CALIBRATION.md`.
- **`p_is_upper_bound` is not surfaced in `nsl_boundary_sweep.py`'s CSV
  schema.** The field exists end-to-end in the engine/verdict/JSON pipeline
  (verified via the 4e real-audit JSON) but the benchmark sweep script
  itself was not modified to read or write it as a CSV column -- out of
  scope for this phase.
- This document validates that the alpha-floor fix restores verdict parity
  with fixed-N and passes the three pre-registered hard gates on this
  fixture. It does not certify sequential stopping as a permanent
  replacement for fixed-N in production without further real-data
  validation, matching `TIER3_CALIBRATION.md`'s own framing of "evidence for
  a decision, not the decision itself."

## Provenance

Raw 25-cell evidence: `f2b_calibration_sequential_v2.csv` (this directory).
Baseline for comparison: `f2b_calibration_histgb.csv` (this directory, Tier 3
Phase B). Regenerate via:

```
python -m zekan.benchmark.nsl_boundary_sweep --sequential --estimator histgb --jobs N --output PATH
```

Real-audit comparison (4e) used `scratch/testB2_strat.csv` with
`zekan/benchmark/test_b_contracts/testB2_sensitivity.yml`; regenerate via
`zekan audit --data scratch/testB2_strat.csv --config
zekan/benchmark/test_b_contracts/testB2_sensitivity.yml --sequential-null
--jobs 12 --json` (compare against the same command without
`--sequential-null` for the fixed_v1 baseline row).
