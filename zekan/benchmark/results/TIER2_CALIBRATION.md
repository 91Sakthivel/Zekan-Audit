# Tier 2 — sequential/adaptive permutation stopping re-validation

## What was run

`nsl_boundary_sweep --sequential` — the same 25-cell F2b fixture (5 alpha
levels 0.00/0.60/1.10/1.60/2.50 x 5 injector seeds), `n_entities=2000`,
`n_jobs=12`, `scheme=spawn_v2`, but with `null_stopping="sequential_v1"`
(Besag-Clifford exceedance-count rule + decision-stability early stop,
h=10, N_min=30, N_max=500) instead of the fixed `n_permutations=100`.
25 cells, ~66 min wall (3946s). Raw evidence:
`f2b_calibration_sequential_v1.csv` (this directory); baseline for
comparison: `f2b_calibration_spawn_v2.csv`.

## THE GATE (the deliverable): all 25 verdicts match, zero flips

Every one of the 25 cells produces the **identical verdict and
detection_channel** as the fixed-N=100 baseline:

| alpha | seeds | baseline verdict | sequential verdict | match |
|---|---|---|---|---|
| 0.00 | 0-4 | pass (all 5) | pass (all 5) | yes |
| 0.60 | 0-4 | fail / both (all 5) | fail / both (all 5) | yes |
| 1.10 | 0-4 | fail / both (all 5) | fail / both (all 5) | yes |
| 1.60 | 0-4 | fail / both (all 5) | fail / both (all 5) | yes |
| 2.50 | 0-4 | fail / both (all 5) | fail / both (all 5) | yes |

**25/25 verdicts match. 25/25 detection_channels match. Zero mismatches,
zero verdict flips.** Per the pre-registered discipline for this
re-validation, this is the hard requirement, and it passes cleanly —
nothing here required (or received) any tuning.

## n_drawn: what the adaptive rule actually did

Every one of the 25 cells stopped before reaching the locked ceiling
(`N_max=500`) — 0/25 ran to the ceiling, on either channel.

| alpha | seed | n_drawn (within) | n_drawn (across) |
|---|---|---|---|
| 0.00 | 0 | 156 | 396 |
| 0.00 | 1 | 36 | 36 |
| 0.00 | 2 | 36 | 36 |
| 0.00 | 3 | 48 | 48 |
| 0.00 | 4 | 36 | 36 |
| 0.60 - 2.50 | 0-4 (all 20 leaked cells) | 108 | 108 |

Two clear, honest patterns:

**Clean cells (alpha=0.00) mostly stop very early — 36-48 draws** (a
fraction of the fixed default's 100), because with no real signal, permuted
draws exceed the small observed `fixable_leakage` often enough that the
Besag-Clifford h=10 exceedance count is reached quickly. One clean seed
(seed=0) is a genuine outlier at 156/396 draws — its `p_value=0.0641`
matches `10/156` exactly, confirming it stopped via the same h-exceedance
rule, just needing more draws to accumulate 10 exceedances than the other
four clean seeds did. This is expected sampling variance in a Bernoulli-style
count, not a bug.

**Every leaked cell (all 20 cells, alpha >= 0.60) stops at exactly 108
draws, on both channels, regardless of leak strength.** This is a striking
and fully reproducible pattern in this specific benchmark fixture: the
decision-stability check (checked every 12 draws, since `batch_size =
n_jobs = 12` here) happens to resolve at the same checkpoint every time for
this dataset/model/leak-shape combination. We are reporting this
factually, not claiming to know it will hold on every dataset — it is a
property of this fixture, not a proven general constant.

## Honest finding: this run was SLOWER than the fixed-N baseline, not faster

Total wall time: **~66 minutes (3946s)** for the sequential sweep, versus
**~37 minutes** for the original fixed-N=100 baseline — **about 1.8x
slower overall**, despite most cells needing a comparable or only modestly
higher number of draws (108 vs 100 for the 20 leaked cells; far fewer for
4 of 5 clean cells).

The reason is batch-dispatch overhead, not draw count. The fixed-N=100
baseline draws all 100 permutations in **one** `Parallel(n_jobs=12)` call
per null channel — one process-pool dispatch, 100 tasks submitted at once.
Sequential mode checks the stopping criteria every `batch_size=n_jobs=12`
draws, so a cell that resolves at 108 draws requires **9 separate**
`Parallel()` dispatches per channel (18 per cell, both channels) instead of
1. This is the same class of overhead this project already profiled and
partially addressed in the Tier-1 performance work (repeated pool
construction/dispatch cost) — Tier 2's batch size was set to `n_jobs` per
the pre-registered spec for parallel efficiency within a batch, but that
does not eliminate the *per-batch* dispatch overhead when many batches are
needed. This is flagged here as a real, unresolved efficiency tradeoff,
not glossed over: the statistical method is validated and correct; its
current implementation is not yet faster in wall-clock terms on this
benchmark, and would need a different batching strategy (e.g. larger or
adaptively-growing batch sizes) to realize a net speed win. That is future
work, out of scope for this pre-registered spec (h/N_min/N_max are locked;
batch size was specified, not something to retune here to make a number
look better).

## Conclusion

The sequential/adaptive stopping rule (Besag-Clifford + decision-stability,
as implemented) **changes how many permutations are drawn and never the
verdict** — confirmed empirically across the full pre-existing 25-cell
calibration fixture, with zero exceptions. It is not, in its current form,
a performance win on this benchmark; that is a separate, honestly-reported
finding, not a reason to doubt the correctness result.

## Provenance

Raw 25-cell evidence: `f2b_calibration_sequential_v1.csv` (this directory).
Baseline for comparison: `f2b_calibration_spawn_v2.csv` (this directory).
Regenerate via:

```
python -m zekan.benchmark.nsl_boundary_sweep --sequential --jobs N --output PATH
```
