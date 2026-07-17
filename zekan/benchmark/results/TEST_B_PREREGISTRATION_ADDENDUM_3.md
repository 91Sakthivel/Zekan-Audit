# Test B pre-registration — Addendum 3

## What this addendum is

This is a third dated amendment, written after B-1's first real run crashed, but before we touch any code or encode anything. It records what happened, why, and the two decisions we're committing to before carrying either of them out. Same discipline as addenda 1 and 2: the record has to show discovery came before the fix, not the other way around.

## What we found (a second real problem in Zekan, caught by actually running it)

B-1's first real run got further than any prior check. Pre-flight passed cleanly — every check PASS, "READY: severity computable" — and the pre-flight data scan printed its notes about every column without incident. Then the run crashed on the very first real computation.

Here's exactly where: inside the step that trains and tests a model on each fold, there's a line that converts all of the feature columns into a single array of plain numbers, so the model can be fit. That conversion hit the `race` column's actual value, `'Caucasian'`, and failed outright: `ValueError: could not convert string to float: 'Caucasian'`.

The reason is straightforward once you see it. That conversion step assumes every feature column is already numeric, and does not check that assumption first — it just tries the conversion and lets it fail. And pre-flight has no check for this either; nothing in the readiness gate looks at whether the features are actually numbers. Diabetes-130 is mostly categorical data: race, gender, the age bucket, the three diagnosis codes, and roughly twenty drug-dosage columns (each recorded as words like "No," "Steady," "Up," "Down," never a number) are all raw text. Every one of them ends up in the feature set, because nothing in the contract or the engine tells Zekan to leave them out — and we deliberately kept all of this raw and unencoded when we built the three CSVs, exactly as we've done with every other messy column in this dataset.

## What this is not

This is not a mistake in how we set up Test B. We didn't do anything wrong preparing B-1's data or its contract. This is Test B doing precisely what it exists to do: surfacing a real problem in Zekan before anyone relies on it for something that matters. And it's the same species of problem we already found and fixed once, at commit 2832600 — pre-flight says everything is fine, and then the engine crashes on an assumption pre-flight never actually checked. Last time the unchecked assumption was that the time column could be parsed. This time it's that the feature columns are already numeric. Same shape of defect, same fix pattern, a different column doing the breaking.

## What we're going to do about it (recorded before doing it)

Two decisions, stated now, before either is carried out:

**1. Fix Zekan's fail-safe hole first.** Before we encode anything, we will make pre-flight detect non-numeric feature columns and fail there, cleanly, with a plain message naming the columns and saying what to do about them — instead of letting the engine discover the problem partway through and fall over. The condition for failing will be tied to the real thing that breaks (the float conversion the model-fitting step actually performs), not an invented rule, following the exact precedent set by the prediction_time fix. This stands on its own regardless of how Test B turns out afterward — it makes Zekan safer for anyone auditing categorical data, not just for this experiment.

**2. Decide, in advance, how the categorical columns will be turned into numbers.** We will use ordinal encoding: each category is replaced by a whole number, assigned in the column's own sorted order — the same approach scikit-learn's `OrdinalEncoder` uses. We are recording this choice before doing it, the same way we recorded the target derivation and the period-column derivation earlier, because how you encode a column is a real methodology decision that can change the result, not a neutral technical detail to skip past.

We are ruling out two alternatives, now, before encoding anything. One-hot encoding would explode into hundreds of new columns just for the three diagnosis-code columns (700+ distinct categories each), making the audit unwieldy for no real benefit here. Target encoding — replacing a category with a number derived from the target itself — would inject target information directly into the features, manufacturing exactly the kind of leakage this whole test exists to check for. Ordinal encoding uses none of the target; it only looks at a column's own values.

The mapping will be built from each column's own sorted-unique values, so it's deterministic and reproducible, and it will be applied identically across all three CSVs. The `'?'` sentinel and every other messy raw value will be encoded like any other category, not cleaned away first — the mess stays in the data. The exact columns encoded and their resulting code mappings will be written down when the encoding actually happens, for the same reason every other data-shaping decision in this project has been documented.

## Why this is still a fair choice

Ordinal encoding imposes an arbitrary numeric order on categories that don't actually have one — `'Caucasian'` becoming, say, 2 and `'Hispanic'` becoming 4 doesn't mean one is worth more than the other. Ordinarily that's a real risk: a model that treats numbers as meaningful magnitudes could be misled by an arbitrary ordering. But Zekan's default model is a random forest, which is tree-based — it only ever asks whether a value falls above or below some threshold, and never treats the size of the gap between two codes as meaningful. An arbitrary but consistent ordering doesn't distort what a tree-based model learns from it. That's a property of the specific model Zekan uses by default, not a property of ordinal encoding in general, and we're recording it here as the actual reason this choice is safe in this context — not as a universal defense of ordinal encoding.

## What this changes for the run

The order of operations, on the record: first, this addendum, written before touching any code. Second, we fix the non-numeric-feature fail-safe hole in Zekan, the same way and for the same reason as the prediction_time fix. Third, we ordinal-encode the categorical columns in `prepare_test_b.py`, using each column's own sorted values, applied identically to all three CSVs. Fourth, only after both of those are done, B-1 runs again — against the fixed Zekan, on the encoded data. The three tests' feature scoping, as recorded in Addendum 1, is unchanged by any of this.
