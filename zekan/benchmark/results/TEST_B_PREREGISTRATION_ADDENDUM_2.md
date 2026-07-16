# Test B pre-registration — Addendum 2

## What this addendum is

This is a second dated amendment, written before we change any code or run any audit. It records something we discovered while preparing to run Test B, and the decisions that follow from it. As with the first addendum, we append rather than edit the earlier documents — the record has to show that discovery came before the fix, not the other way around.

## What we found (a real problem in Zekan, caught before running)

To check for leakage, Zekan needs to understand the order things happened in. It works by comparing how a model performs when it's allowed to peek across time against how it performs when it isn't — that comparison is the core of what the tool does. So it needs a usable "time" column to build that comparison on.

Our dataset has no real calendar date anywhere in it. We planned to use the visit ID as a stand-in for time, since it increases as visits happen. But the visit IDs are large numbers, and when Zekan tries to read them as dates, almost all of them fail to parse.

Here is the actual defect. Zekan's pre-flight check — the step meant to catch problems before doing any real work — only *warns* about this failure, because a tiny fraction of the visit-ID numbers happen to accidentally parse as some nonsense date, so the failure isn't a clean 100%. A warning doesn't stop the run. So Zekan reports "READY — good to go," proceeds, and then crashes partway through with an unhandled error when it tries to sort the visits by date.

We want to state the significance plainly and fairly. This is the one failure mode a trust tool must never have: saying "everything's fine" and then falling over. It is not that Zekan requires a time column — that's correct and intentional. The problem is that when the time column turns out to be unusable, Zekan should say so clearly and stop, not promise success and then crash. We found this by reading the relevant code and testing it directly against the real data, before running anything for real.

## What this is not

To be fair to the tool: requiring a real time signal is not a flaw. It's the essence of what Zekan does. Zekan is built for accumulated, historical data — the data you already have, that you're about to build a model on — and reasoning correctly about time order is its whole job. The problem we found is only the dishonest failure mode (claiming readiness, then crashing), not the underlying requirement that a usable time signal exist.

## What we're going to do about it (recorded before doing it)

Two decisions, stated now, before either is carried out:

**1. Fix Zekan first.** Before running Test B, we will fix Zekan so that an unusable time column fails cleanly and loudly at the pre-flight gate, with a plain message telling the user what's wrong and what to do about it — instead of promising success and then crashing partway through. This makes Zekan better for everyone who uses it, not just for this test. We're recording here that we are fixing a real defect we found, before we run the experiment against the fixed version. The crash itself is already fully documented by direct code tracing in an earlier inspection, so writing it down here loses nothing — we're not reconstructing anything from memory.

**2. Give this dataset a real time signal, honestly.** The data does contain real order: a patient's later visit genuinely happened after their earlier one, and the visit IDs increase as time passes. We will turn that real order into a usable time column by ranking all visits in their true order and grouping them into a moderate number of sequential periods — roughly a few dozen buckets, not one period per individual visit. We want to say this plainly: we are encoding the real order the data already contains into a form Zekan can read. We are not inventing a fake calendar and we are not making up dates. These periods will be labeled as ordinal periods — "1st period," "2nd period," and so on — not real calendar dates, and that label will be stated wherever results are reported.

## Why this is still honest

Nobody is cleaning the data or inventing signal that isn't there. The visit order is real; it reflects genuine sequence in genuine hospital visits. We're translating that real order into a format the tool can use, at a sensible granularity, and we're saying exactly what we did and why. The alternative — forcing a fake calendar of specific, invented dates onto the data — would be less honest than labeling the periods as what they are: an ordering, not a clock. So we're deliberately not doing that.

## What this changes for the run

Three things follow from this addendum: first, we fix the ready-then-crash defect in Zekan before running anything. Second, all three Test B datasets receive the same derived ordinal-period time column. Third, B-1 runs against the fixed version of Zekan, once both of those are done. The three tests' feature scoping, as recorded in Addendum 1, is unchanged by any of this.
