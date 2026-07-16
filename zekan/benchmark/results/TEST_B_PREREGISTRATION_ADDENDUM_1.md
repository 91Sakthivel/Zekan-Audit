# Test B pre-registration — Addendum 1

## What this addendum is and why it's separate

The original pre-registration document (`TEST_B_PREREGISTRATION.md`) was committed before any run, and it stays exactly as written — we don't go back and edit it, on principle. This addendum is a dated amendment, also written and committed *before* Zekan runs on the data.

Here's why that separation matters. Pre-registration — writing down your predictions before you see the result — is only trustworthy if someone can later see exactly what was predicted and when. If we learn something new partway through and just fold it into the original document, we've quietly erased the line between "what we predicted going in" and "what we figured out along the way." That erasure is exactly the kind of self-deception pre-registration exists to prevent. So when we learn something new before the run, the honest move is to file a dated addendum, not rewrite history. The seams — the fact that you can see document 1, then document 2, then eventually the actual result — are the whole integrity story.

## What we learned (by inspecting Zekan's own code, before running it)

Before running anything, we read exactly how Zekan decides what counts as a "feature" and how it measures leakage.

The short version of how it measures leakage: Zekan trains two models on the same data — one model is allowed to use a suspect column, the other is not. The drop in accuracy between the two is Zekan's estimate of how much that column was leaking (letting the model cheat).

The key thing we found: Zekan only measures leakage for columns you explicitly tell it to suspect — its "forbidden" list. A leaky column you don't flag gets used by both models equally. Since both models get to use it, the drop between them is close to zero — the column looks clean, even though it's leaking just as badly as if you had flagged it.

We want to be fair about this: it is not a hidden bug. Zekan's own code documents this limitation in plain words, in the engine's own comments — the decomposition measures only *declared* leakage, and undeclared leaky features appear in every comparison equally and cancel out, staying invisible to the check. We are reporting a known, documented boundary of the current tool, confirmed by actually reading the code — not accusing it of a defect it's hiding from us.

One more thing worth recording plainly: all three of Zekan's detection methods share this same boundary, because all three key off that same "forbidden" list. The two-model accuracy-drop check only compares columns you flagged. The per-feature importance check (which ranks how much each feature matters) only runs at all when the accuracy-drop check has already found something, so it inherits the same blind spot. And the shuffle test (which scrambles a column's values to see if that breaks the leak) only shuffles columns you've flagged as forbidden — it never touches a column you didn't suspect. Three different mechanisms, one shared blind spot.

## Why this matters for our data specifically

Our dataset has a column called `readmitted` — the original three-way answer (`NO` / `>30` / `<30`) — from which we derive our actual yes/no target. If we leave that original column sitting in the data without flagging it, it is a near-perfect copy of the answer. That's the textbook leak: a column that basically *is* the target wearing a different hat.

Per what we just learned, Zekan will not catch this on its own unless we flag it. It will simply sit in both models, both models will use it, and the comparison between them will look clean.

## The new prediction we are committing to, before running

This is the whole point of pre-registration: we call the shot before we look.

We predict that in Test B-3 (the honest-unknown run), if we leave `readmitted` in the data and do not flag it, Zekan will report a falsely-clean result on it. It will not flag this obvious leak.

We want to be clear about what kind of result this is. If it happens, it is expected, not a surprise failure. It is a demonstration, on real data, of a limitation the code already told us about in its own comments. Running it turns a line in a docstring into a shown fact, observed on a genuine hospital dataset rather than assumed from reading source.

And we want to record plainly what this implies: this is the concrete, real-data case for building something Zekan doesn't have yet — an "undeclared-feature screen," a check that could catch a leak you didn't already know to suspect. We're writing this motivation down now, before the run, precisely so it's honest — so it's a prediction we made in advance, not a justification we invented after seeing the result.

## How this changes the three test setups (design decisions, recorded now)

- **B-1 (specificity):** we will remove the raw `readmitted` column from B-1's data entirely, so the clean specificity question — does Zekan stay quiet on the honest, strong predictor `number_inpatient`? — isn't muddied by an obvious leak sitting in the background. B-1 flags nothing as forbidden.
- **B-2 (sensitivity):** we plant a leak and flag it as forbidden — confirming Zekan correctly measures a leak it's told to check for, on real, messy data.
- **B-3 (honest unknown):** we deliberately leave `readmitted` in and do not flag it — this is the blind-spot demonstration described above. We also still watch the "Expired"-at-discharge quirk noted in the original pre-registration.

## What still counts as success

This is unchanged from the original pre-registration: success is learning the truth about how Zekan behaves on real data. Demonstrating a documented limitation honestly is a success, not a failure — as long as we predicted it in advance (which we now have, in writing, above) and we record what actually happens plainly, whichever way it goes.
