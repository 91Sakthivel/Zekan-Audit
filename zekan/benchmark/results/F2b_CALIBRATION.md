# F2b — NSL gate re-validation under spawn_v2

## What was run

`nsl_boundary_sweep` — 5 alpha levels (0.00 / 0.60 / 1.10 / 1.60 / 2.50) x 5
injector seeds, `n_permutations=100`, `n_entities=2000`, `n_jobs=12`,
`scheme=spawn_v2`. 25 cells, ~37 min wall.

## WITHIN result (the deliverable)

Under spawn_v2, clean runs (alpha=0.00) sit at NSL in **[-2.65, -0.17]**, all
`pass`, `p_value` in [0.0495, 0.90] — comfortably clearing the `p >= 0.01`
PASS gate. Real leaks (alpha >= 0.60) sit at NSL in **[+3.84, +9.21]**, all
`fail`, `p_value = 0.0099` (the Laplace-corrected floor at n=100). The
**[1.0, 2.0) band is empty** — no cell, clean or leaked, lands near the 1.0
line.

**Conclusion: the within-entity NSL >= 1.0 gate is re-validated under
spawn_v2.** Separation between clean and leaked is preserved; no threshold
change is indicated.

Honest caveat: 1.0 is not *pinned* by this data — nothing sits close enough
to it to adjudicate the exact cut — but it sits safely inside a wide empty
gap between the clean and leak clusters. This is the same epistemic status
as the original serial_v1 calibration (see `nsl_boundary_sweep.py`'s
docstring), now reproduced under spawn_v2 rather than newly established.
`null_iqr` across all 25 cells sits in **[0.0018, 0.0034]** — consistent
with the serial-stream finding that null width is set by dataset size, not
leak strength.

## ACROSS result (partial, correctly scoped)

The across-entity channel is even more separated on this fixture: clean NSL
in **[-2.54, -0.16]**, leak NSL in **[+11.50, +28.21]**. Every leaked cell
fired `detection_channel="both"`; every clean cell fired neither channel
(`detection_channel=""`) — 5/5 clean seeds correctly PASS on both channels,
zero false positives.

This shows the across null is well-behaved and **safe** (it does not
false-alarm on clean data). It does **not** mean the across NSL >= 1.0
boundary is *earned*: `inject_graded_future_leak` is a row-level graded leak
(`alpha * z[entity, T+1] + noise`, varying by entity **and** period), not
the entity-level-aggregate shape spec 1's across-entity null specifically
targets (a forbidden column constant within each entity). The across
boundary is therefore **validated-safe but not fully earned** here.
Calibrating it against a genuine entity-aggregate structure is deferred to
real-data Test B.

## Provenance

Raw 25-cell evidence: `f2b_calibration_spawn_v2.csv` (this directory).
Regenerate via:

```
python -m zekan.benchmark.nsl_boundary_sweep --jobs N
```
