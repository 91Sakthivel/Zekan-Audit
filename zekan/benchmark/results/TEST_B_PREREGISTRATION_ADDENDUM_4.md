# Test B pre-registration — Addendum 4

## What this addendum is

This is a fourth dated amendment, written before any profiling or performance code changes. It records a real slowness problem found while running Test B, and the decisions we're committing to before carrying them out. Same discipline as addenda 1 through 3: the record has to show discovery came before the fix, not the other way around.

## What we found (a performance problem, not a correctness one)

B-1 passed. B-2's sensitivity question — does Zekan catch a planted, declared leak? — was answered on a 10,000-row stratified subsample of the real data: the leak was detected, the verdict was FAIL, and `planted_leak` came out ranked first. But B-2 at full scale, on all 101,766 rows, never produced a verdict at all, because the audit does not finish in a usable amount of time.

Here is the measured timing curve, as actually observed, not estimated:

- **10,008 rows** (stratified, all 24 periods): 225 seconds. Completed. The full severity core ran to a result.
- **40,008 rows** (stratified, all 24 periods): exceeded 1,800 seconds and was stopped at roughly 44 minutes, without completing.
- **101,766 rows** (full B-2, single seed, no stability check): did not complete in 40-plus minutes. Stopped.
- **101,766 rows** (full B-2, with the 5-seed stability check): did not complete in 2-plus hours. Stopped.

Four times the rows cost more than eight times the time. That is not what you'd expect if the work simply scaled with the amount of data — it means the cost grows faster than linearly as the row count goes up, and something is compounding.

We want to state a second thing plainly, separate from the scaling question: 225 seconds for a single audit of 10,000 rows is already slow on its own terms, before any question of how it scales. A tool whose stated goals include being fast and practical to run shouldn't take nearly four minutes to audit ten thousand rows once. Both things are real problems here — the baseline cost and the way it scales — and we're recording both, not just the more dramatic scaling one.

## What we observed about worker processes (this is observation, not diagnosis)

Zekan uses a pool of worker processes to parallelize parts of the audit. During the 40,000-row run, starting from a verified-empty process list, the worker processes did not all appear at once as a single group. Twelve workers appeared together with one start time. About three seconds later, two more workers appeared, with a noticeably different amount of accumulated CPU time from the first twelve. The same multi-group pattern was seen in the earlier 100,000-row runs as well.

Because the 40,000-row run started from a confirmed-clean process list, these later-arriving workers are not leftovers from some earlier run still lingering in the background — they appeared fresh, during this one audit.

So the observation, stated carefully and no further than the evidence supports: a single audit appears to create its worker pool more than once. We want to be precise about what we do and don't know yet. We do not yet know which specific piece of code creates these pools, and we do not yet know for certain that recreating the pool is what's actually causing the slowness, as opposed to being a symptom of something else, or a separate issue entirely. This is a documented observation, not a diagnosis, and we're not going to treat it as more than that until it's actually confirmed.

## What this is not

This is not a mistake in how we set up Test B, and it is not evidence that Test B itself is broken. This is Test B doing exactly what it exists to do: surfacing a real problem in Zekan before anyone relies on it for something that matters. It's a different kind of problem than the two we found before — those were correctness defects (a promise of "ready" followed by a crash); this one is a performance defect (a promise of eventually finishing that, in practice, doesn't hold at realistic scale) — but it belongs to the same family of things Test B is designed to catch: gaps between what Zekan claims about itself and what actually happens when you run it on real data.

## What we're going to do about it (recorded before doing it)

Two decisions, stated now, before either is carried out:

**1. Profile first.** We will profile the severity core as a bounded, read-only investigation — to find out where the time is actually going, and to locate every place in the code where a worker pool gets created, by file and line. No performance-related code will be changed until profiling has actually identified the cause. We are not going to guess at a fix, and we are not going to tune anything just to make a number look better without understanding why it was bad in the first place.

**2. Fix only after that.** Once profiling tells us what's actually happening, we'll fix it — and only that, based on what was found, not on the worker-process observation alone, since that observation on its own doesn't yet tell us the mechanism.

One more thing worth recording plainly: this performance work is legitimate engine hardening that Test B surfaced, in exactly the same way it surfaced the two earlier ready-then-crash defects. It is not a new feature we've decided to add and not a change of scope for this project — it's the same kind of finding, just a different failure mode.

## What this changes for the remaining tests

B-2's sensitivity result stands — the leak was planted, declared, and caught, which answers the sensitivity question Test B-2 was designed to ask. But it is explicitly scoped, from here forward, as a result measured on a 10,000-row stratified subsample, not on the full dataset. It will be re-run at full scale once the performance problem is fixed, so we have a real full-scale result to stand alongside the subsample one.

B-3 has not been run yet, and it waits for the same fix — there is no point starting a run we already know may not finish in a usable time.

The consolidated Test B results report will record all of this plainly, including this performance defect itself, as one of the findings from running Test B — not as a footnote or an excuse, but as part of what the test was for.
