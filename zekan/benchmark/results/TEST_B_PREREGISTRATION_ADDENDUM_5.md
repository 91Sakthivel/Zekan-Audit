# Test B pre-registration — Addendum 5

## What this addendum is

This is a fifth dated amendment, written before any fix is implemented. It records an architectural finding surfaced while validating the Tier 2b sequential-stopping fixes, and the decisions we're committing to before carrying them out. Same discipline as addenda 1 through 4: the record has to show discovery came before the fix, not the other way around. Addenda 1 through 4 stay exactly as written — we don't go back and edit them.

## What prompted this

While validating the Tier 2b sequential-stopping fixes on the real B-2 subsample, the audit stopped early — 36 draws, NSL=6.67, an unmistakable leak by that measure — but the verdict came out `UNCONFIRMED_HIGH_DAMAGE` instead of `FAIL`. The downgrade had nothing to do with the size of the leak. It happened purely because the p-value could not cross alpha in 36 draws.

## The finding, stated plainly

Detection requires `p < 0.01` before NSL is even consulted. p is Laplace-corrected: `(exceedances + 1) / (draws + 1)`. With zero exceedances — which is exactly the signature of the *strongest* leaks, the ones where no permuted draw ever comes close to the real signal — the smallest p can possibly be is `1 / (draws + 1)`. That number cannot cross 0.01 until roughly 100 draws have been made, no matter how overwhelming the evidence sitting in front of it.

The old fixed `n_permutations=100` was chosen so that a zero-exceedance leak just barely clears this floor — this is documented in `null_baseline.py`'s own comment. The permutation budget was reverse-engineered to let the gate pass. That is exactly the class of number-tuned-to-pass-a-gate this project forbids elsewhere, and it sat unnoticed in the architecture for one simple reason: the fixed design always drew all 100 regardless of how the evidence looked, so the gate always happened to pass by the time anyone checked it. Nothing ever stopped early enough to expose that the 100 was propping the gate open.

The consequence, stated as plainly as we can: under any early-stopping scheme built on top of this p-value gate, the stronger the leak, the *more* draws it would need to get confirmed. That is backwards. A weak, ambiguous case would resolve fine; an obvious one would get penalized for being obvious.

Credit where it's due: this was not a defect Tier 2b introduced. It was surfaced by the Tier 2b validation gate doing exactly its job — a verdict flip was caught and reported, not tuned away, on real data, before it shipped. The flaw was already sitting in the architecture; Tier 2b's fix to the previous defect (the decision-stability rule demanding IQR precision regardless of distance from the boundary) is what finally let a run stop early enough to walk into it.

## The statistics, honestly stated

The roughly 99-draw floor for certifying `p < 0.01` is not an artifact of bad code. It is the genuine information-theoretic minimum: you cannot certify significance at alpha=0.01 from 36 permutations, full stop, regardless of how the stopping rule is written. No amount of cleverness in the decision-stability check changes that arithmetic.

What was actually wrong is narrower than "the floor exists." What was wrong is that the floor was *implicit and accidental* rather than *explicit and designed*, and that early stopping was allowed to report a downgraded verdict on evidence that simply had not run long enough to be entitled to an opinion either way. The floor itself is correct statistics. Treating a 36-draw sample as if it had already answered the significance question was not.

## The decision (recorded before implementing)

We are committing to an asymmetric sequential design, in keeping with Besag-Clifford semantics rather than against them:

1. **NOT-DETECTED stops early.** When the NSL interval is provably below 1.0 past `N_min=30`, stop — this is the common clean-data case, and it gets the full speedup. There is no floor to respect here: "we can already tell this isn't a leak" is a claim the data can support at low n.

2. **DETECTED conclusions require `n >= ceil(1/alpha) - 1 = 99` draws** — the honest minimum for alpha=0.01 — and never draw meaningfully past that once NSL is stable and exceedances remain at zero. No overshoot: the floor is a requirement, not an excuse to fall back to the old fixed budget's full run.

3. **p is reported as an explicit upper bound** when stopping with zero exceedances (`p <= 1 / (n + 1)`), carried in the JSON as `p_is_upper_bound`. The surface tells the truth about what kind of number p is, instead of quietly presenting a floor-bound estimate as if it were an exact one.

Expected effect, recorded now so it can be checked against later rather than rationalized afterward: clean audits roughly 3x faster; leak audits equal to or slightly faster than the fixed-N design; verdict flips impossible by construction, because a case that would be `DETECTED` always draws the same ~100 permutations the fixed design already drew. Re-validation against the 25-cell histgb baseline remains the hard gate before any of this ships.

## What this is not

This is not a mistake in how Tier 2b was validated, and it is not evidence that the Tier 2b validation gate is broken. It is the validation gate doing exactly what it exists to do: catching a verdict flip before it reached anyone who'd rely on it, and forcing the actual cause into the open instead of letting a parameter tweak paper over it. It belongs to the same family as the findings in the earlier addenda — a gap between what the architecture assumed and what actually happens once a real run is allowed to behave differently than the fixed design always forced it to.

## What this changes for the remaining work

No code changes yet. This addendum exists specifically so that when FIX A/FIX B for the p-floor problem are implemented next, there is a dated record showing the asymmetric design above was decided *before* that implementation, not reverse-engineered afterward to match whatever the code happened to do. The 25-cell histgb re-validation sweep stays blocked until the asymmetric design is implemented and passes it.
