# Test B pre-registration — Diabetes-130 real-data run

## What this document is

We are writing this down *before* Zekan ever touches the real hospital data. That order matters. Once you've seen a result, it's very easy to talk yourself into believing you predicted it all along — "oh, that flag makes sense" or "well, it missed that one because the data is weird." Writing predictions and failure conditions down first closes that door. If the result doesn't match what's written here, we don't get to quietly redefine success; we call it what it is.

There's a plain irony worth naming: Zekan exists to stop people from fooling themselves about their data and their models. The least we can do is not fool ourselves about Zekan.

## The data, in plain terms

The dataset is called Diabetes-130. It's about 101,766 hospital stays, drawn from 130 US hospitals between 1999 and 2008. Each row is one hospital visit. The question the data is meant to answer: **will this patient be readmitted to hospital within 30 days of leaving?**

A few real facts we confirmed by actually reading the file (not from memory or documentation):

- There are 71,518 distinct patients, but only 101,766 rows — meaning a lot of patients show up more than once. Specifically, 16,773 patients (about a quarter of all patients) have two or more visits, and those repeat visits account for 46% of all rows in the dataset. This matters because it's real, usable structure: a patient (what we'll call an **entity** — "the thing that repeats," here a patient) has a history, and Zekan can compare a patient's own repeated rows against each other, which is a much stronger check than just looking at rows in isolation.

- The data is genuinely messy, and we're leaving it that way on purpose. The `weight` column is missing 97% of the time — for almost every patient we simply don't know their weight. `medical_specialty` is missing about 49% of the time, and `payer_code` about 40%. We are **not** cleaning any of this up before running Zekan. Real hospital data looks like this. A tool that only works on a scrubbed, pre-cleaned CSV isn't actually useful, so part of this test is simply: does Zekan coldly cope with a mess, and does it say so honestly, instead of pretending nothing's wrong?

- The honest limitation: there is no real timestamp anywhere in this data. No calendar date, no "visit happened on this day." All we have is an ever-increasing visit ID, which tells us the *order* visits happened in but not *when*. We're not going to pretend otherwise. Where Zekan needs a sense of time, we'll use the visit ID as a stand-in for time — and we'll label it clearly as a stand-in, not a real clock.

## Define the key terms once, plainly

- **Leakage** — the model accidentally gets to see information it wouldn't really have at the moment it makes its prediction. Like peeking at the answer key before the test.
- **Specificity** — not crying wolf. Staying quiet when a feature is strong but honest, instead of flagging everything that looks powerful.
- **Sensitivity** — actually catching a real problem when one is there.
- **Entity** — the thing that repeats. Here, a patient, since the same patient can appear in multiple rows.
- **Verdict** — Zekan's final call on a dataset: trustworthy, risky, or somewhere in between.

## The three tests

Each test below states what we do, what we predict, and — most importantly — the specific result we agree in advance to call a failure, with no talking our way out of it.

### Test B-1 — Specificity (the most important one)

**What we do:** point Zekan at the data and ask it to evaluate `number_inpatient` — a count of how many times this patient has been admitted to hospital before. This genuinely predicts readmission (patients with a history of frequent admissions tend to come back), and it is completely honest: it's known before the patient ever leaves this visit. It is not a form of cheating.

**Prediction:** Zekan should **not** flag `number_inpatient` as leakage. A predictor can be strong and still be legitimate, and Zekan needs to tell the difference.

**Failure condition, committed in advance:** if Zekan flags `number_inpatient` as leakage, that is a real defect — a false alarm on an honest feature. We record it as a **FAILURE**, plainly, and it becomes the next thing to fix. We do not soften this or explain it away after the fact.

**Honest caveat:** the raw file has no data dictionary, so we can't be perfectly certain what time window `number_inpatient` actually covers (prior year? all prior visits?). We are *declaring* it legitimate for the purposes of this test. That bit of uncertainty is itself realistic — a real analyst often isn't 100% sure of a feature's history either, which is exactly the kind of situation an auditor like Zekan needs to be useful in.

### Test B-2 — Sensitivity

**What we do:** deliberately plant a cheating feature into this real, messy data — one that's secretly copied from the answer (whether the patient was actually readmitted). Then we check whether Zekan catches it.

**Prediction:** Zekan detects it, and flags it at high severity.

**Failure condition:** if Zekan does **not** detect a feature that is a direct copy of the answer, it is missing the most blatant form of leakage there is. That is a **FAILURE**, recorded plainly.

### Test B-3 — The honest unknown

**What we do:** run Zekan on the real features as they actually are — nothing planted, nothing removed. Nobody has inserted a leak or a legitimate anchor here. This is genuine discovery: if there's a real problem in this data, this is where it would actually live.

There's no pass/fail prediction for this one — we don't know what we'll find, and that's the point. But we're pre-committing to one thing: **we will hand-investigate anything Zekan flags here.** In particular, we already noticed something during inspection worth watching closely: some patients are recorded with `discharge_disposition_id` = "Expired," meaning the patient died at discharge. A patient who has died obviously cannot be readmitted, so for those rows the answer is close to fixed in advance. We commit to checking whether Zekan reacts to this pattern, and to writing down honestly whether that reaction is a genuine insight (real structural quirk worth knowing about) or a false alarm (Zekan flagging something that isn't actually a modeling problem).

## How we'll run each test

We'll run each audit two ways: once grouping rows by patient (the solid version, using the real repeat-visit structure), and once using visit-order as a stand-in for time (the proxy version, since there's no real clock in this data). We'll also run with the stability check turned on — Zekan reruns itself across several random seeds, and if its final verdict flips depending on which seed it happened to use, that instability is itself a warning sign worth reporting.

Running it two ways and across several seeds, rather than once, is simply more honest: a single run could get lucky or unlucky, and we'd have no way to tell the difference between "Zekan is right" and "Zekan happened to land on the right answer this one time."

## What counts as success for Test B overall

Success here is **not** "Zekan passes everything." Success is "we learn the truth about how Zekan actually behaves on real, messy data." If Zekan fails B-1, that's a valuable and fully valid outcome — the test did exactly its job by catching a real weakness before anyone relied on this tool for something that matters. The only real way to fail Test B as a whole would be to run it carelessly, or to see a bad result and explain it away instead of writing it down.
