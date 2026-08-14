# Upgrade (H) calibration — the NEAR-BIJECTION structural check (Theil's U)

This is evidence for calibrating the guard pre-registered in
`UPGRADE_H_PREREGISTRATION.md`. No probe code exists yet — this measures
Theil's U (the uncertainty coefficient) directly against the real B-1/B-2/B-3
frames, exactly as pre-registered, and reports what the pre-registration's
own falsification conditions say about the design. Same discipline as
`UPGRADE1_CALIBRATION.md`: this is evidence for a decision, written down
honestly whichever way it goes, not a run massaged until the named anchors
land where hoped. **Report, not tune** — no threshold below was searched for
to make an anchor pass; every number is read off the measured distribution.

**Headline: no pre-registered assumption failed on this data.** `encounter_id`
achieves `U = 1.0` raw, exactly as the ID-trap section predicted it would —
and is completely neutralized (`U = 0.0` exactly) by the pre-registered
minimum-support guard at every floor tested (5/10/20/50). Both leak anchors
(`readmitted`, `planted_leak`) clear the honest tail by a wide, clean,
non-overlapping margin at every pooling level. `number_inpatient` never
comes close to triggering. All three named falsification conditions hold.

## Addition 1 — `encounter_id` presence check (done before computing anything)

| frame | `encounter_id` present? |
|---|---|
| B-1 (`testB1_specificity.csv`) | **Yes** |
| B-2 (`testB2_sensitivity.csv`) | **Yes** |
| B-3 (`testB3_honest_unknown.csv`) | **Yes** |

Present in all three — the ID-trap falsification anchor is testable.
Calibration proceeded.

## Frames and columns

| frame | path | rows | cols | forbidden | extra column vs. B-1 |
|---|---|---|---|---|---|
| B-1 | `...\testB1_specificity.csv` | 101,766 | 51 | none | — |
| B-2 | `...\testB2_sensitivity.csv` | 101,766 | 52 | `planted_leak` | `planted_leak` |
| B-3 | `...\testB3_honest_unknown.csv` | 101,766 | 52 | none | `readmitted` (raw, undeclared) |

Column-set diff confirmed directly (`set(B2.columns) - set(B1.columns) == {planted_leak}`,
`set(B3.columns) - set(B1.columns) == {readmitted}`, symmetric differences
empty otherwise) — B-1/B-2/B-3 share the same 48 underlying honest features
with identical values; B-2 adds one planted leak column, B-3 leaves one real
leak (`readmitted`) undeclared. This is why the honest-feature `U` values
below are identical across all three frames — expected, not a bug, and it is
exactly why the three frames are reported separately rather than merged: the
anchors that differentiate them (`planted_leak`, `readmitted`) don't co-occur.

Screened candidates per frame: **B-1: 48** (no forbidden columns, no extra
leak column present). **B-2: 48 screen candidates + 1 reference**
(`planted_leak`, scored but excluded from the screen-candidate set because
it's declared forbidden — Addition 3). **B-3: 49** (49 candidates, including
raw `readmitted` since it's present and undeclared).

## Method notes, as actually run

- **NaN handling**: treated as its own explicit category (`__NaN__` sentinel), per the pre-registration's own justification (missingness itself can correlate with the target). **Not actually exercised on this data** — verified directly: zero columns in any of the three frames contain any NaN value. The sentinel path exists in the script but was never taken; this is recorded honestly rather than silently assumed safe.
- **Entropy base**: log2 (bits) — irrelevant to `U` itself, which is a ratio and base-invariant; used only for readability if `H(Y)` were reported directly (not shown below, since only `U` matters for the guard).
- **Values used exactly as stored** — no binning, no normalization, per the pre-registration.

## Full U distribution, per frame (NOT merged — Addition 2)

<details><summary>Full tables (B-1, B-2, B-3), sorted descending by U (raw)</summary>

### B1 full U distribution (48 features)

| feature | role | U (raw) | n_distinct | distinct/rows | max count | min count |
|---|---|---|---|---|---|---|
| encounter_id | screen_candidate | 1.0000 | 101766 | 1.0000 | 1 | 1 |
| number_inpatient | screen_candidate | 0.031552 | 21 | 0.000206 | 67630 | 1 |
| diag_1 | screen_candidate | 0.024362 | 717 | 0.007046 | 6862 | 1 |
| diag_3 | screen_candidate | 0.021779 | 790 | 0.007763 | 11555 | 1 |
| discharge_disposition_id | screen_candidate | 0.021616 | 26 | 0.000255 | 60234 | 2 |
| diag_2 | screen_candidate | 0.020309 | 749 | 0.007360 | 6752 | 1 |
| number_emergency | screen_candidate | 0.006913 | 33 | 0.000324 | 90383 | 1 |
| medical_specialty | screen_candidate | 0.005260 | 73 | 0.000717 | 49949 | 1 |
| num_medications | screen_candidate | 0.004523 | 75 | 0.000737 | 6086 | 1 |
| time_in_hospital | screen_candidate | 0.004300 | 14 | 0.000138 | 17756 | 1042 |
| number_diagnoses | screen_candidate | 0.003840 | 16 | 0.000157 | 49474 | 7 |
| num_lab_procedures | screen_candidate | 0.002676 | 118 | 0.001160 | 3208 | 1 |
| insulin | screen_candidate | 0.002600 | 4 | 0.000039 | 47383 | 11316 |
| number_outpatient | screen_candidate | 0.002249 | 39 | 0.000383 | 85027 | 1 |
| age | screen_candidate | 0.001786 | 10 | 0.000098 | 26068 | 161 |
| payer_code | screen_candidate | 0.001466 | 18 | 0.000177 | 40256 | 1 |
| diabetesMed | screen_candidate | 0.001084 | 2 | 0.000020 | 78363 | 23403 |
| metformin | screen_candidate | 0.000857 | 4 | 0.000039 | 81778 | 575 |
| admission_source_id | screen_candidate | 0.000768 | 17 | 0.000167 | 57494 | 1 |
| num_procedures | screen_candidate | 0.000669 | 7 | 0.000069 | 46652 | 3078 |
| change | screen_candidate | 0.000543 | 2 | 0.000020 | 54755 | 47011 |
| A1Cresult | screen_candidate | 0.000518 | 4 | 0.000039 | 84748 | 3812 |
| admission_type_id | screen_candidate | 0.000413 | 8 | 0.000079 | 53990 | 10 |
| race | screen_candidate | 0.000390 | 6 | 0.000059 | 76099 | 641 |
| max_glu_serum | screen_candidate | 0.000208 | 4 | 0.000039 | 96420 | 1264 |
| repaglinide | screen_candidate | 0.000156 | 4 | 0.000039 | 100227 | 45 |
| glipizide | screen_candidate | 0.000149 | 4 | 0.000039 | 89080 | 560 |
| glimepiride | screen_candidate | 0.000100 | 4 | 0.000039 | 96575 | 194 |
| pioglitazone | screen_candidate | 0.000088 | 4 | 0.000039 | 94438 | 118 |
| rosiglitazone | screen_candidate | 0.000084 | 4 | 0.000039 | 95401 | 87 |
| weight | screen_candidate | 0.000071 | 10 | 0.000098 | 98569 | 3 |
| glyburide | screen_candidate | 0.000067 | 4 | 0.000039 | 91116 | 564 |
| miglitol | screen_candidate | 0.000057 | 4 | 0.000039 | 101728 | 2 |
| acarbose | screen_candidate | 0.000056 | 4 | 0.000039 | 101458 | 3 |
| chlorpropamide | screen_candidate | 0.000054 | 4 | 0.000039 | 101680 | 1 |
| glyburide-metformin | screen_candidate | 0.000029 | 4 | 0.000039 | 101060 | 6 |
| nateglinide | screen_candidate | 0.000024 | 4 | 0.000039 | 101063 | 11 |
| gender | screen_candidate | 0.000022 | 3 | 0.000029 | 54708 | 3 |
| tolbutamide | screen_candidate | 0.000019 | 2 | 0.000020 | 101743 | 23 |
| troglitazone | screen_candidate | 0.000010 | 2 | 0.000020 | 101763 | 3 |
| tolazamide | screen_candidate | 0.000010 | 3 | 0.000029 | 101727 | 1 |
| metformin-rosiglitazone | screen_candidate | 0.000007 | 2 | 0.000020 | 101764 | 2 |
| acetohexamide | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glimepiride-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| metformin-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glipizide-metformin | screen_candidate | 0.000002 | 2 | 0.000020 | 101753 | 13 |
| citoglipton | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |
| examide | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |

### B2 full U distribution (49 features: 48 screen candidates + 1 known-leak reference)

| feature | role | U (raw) | n_distinct | distinct/rows | max count | min count |
|---|---|---|---|---|---|---|
| encounter_id | screen_candidate | 1.0000 | 101766 | 1.0000 | 1 | 1 |
| planted_leak | known_leak_reference | 0.641748 | 2 | 0.000020 | 86483 | 15283 |
| number_inpatient | screen_candidate | 0.031552 | 21 | 0.000206 | 67630 | 1 |
| diag_1 | screen_candidate | 0.024362 | 717 | 0.007046 | 6862 | 1 |
| diag_3 | screen_candidate | 0.021779 | 790 | 0.007763 | 11555 | 1 |
| discharge_disposition_id | screen_candidate | 0.021616 | 26 | 0.000255 | 60234 | 2 |
| diag_2 | screen_candidate | 0.020309 | 749 | 0.007360 | 6752 | 1 |
| number_emergency | screen_candidate | 0.006913 | 33 | 0.000324 | 90383 | 1 |
| medical_specialty | screen_candidate | 0.005260 | 73 | 0.000717 | 49949 | 1 |
| num_medications | screen_candidate | 0.004523 | 75 | 0.000737 | 6086 | 1 |
| time_in_hospital | screen_candidate | 0.004300 | 14 | 0.000138 | 17756 | 1042 |
| number_diagnoses | screen_candidate | 0.003840 | 16 | 0.000157 | 49474 | 7 |
| num_lab_procedures | screen_candidate | 0.002676 | 118 | 0.001160 | 3208 | 1 |
| insulin | screen_candidate | 0.002600 | 4 | 0.000039 | 47383 | 11316 |
| number_outpatient | screen_candidate | 0.002249 | 39 | 0.000383 | 85027 | 1 |
| age | screen_candidate | 0.001786 | 10 | 0.000098 | 26068 | 161 |
| payer_code | screen_candidate | 0.001466 | 18 | 0.000177 | 40256 | 1 |
| diabetesMed | screen_candidate | 0.001084 | 2 | 0.000020 | 78363 | 23403 |
| metformin | screen_candidate | 0.000857 | 4 | 0.000039 | 81778 | 575 |
| admission_source_id | screen_candidate | 0.000768 | 17 | 0.000167 | 57494 | 1 |
| num_procedures | screen_candidate | 0.000669 | 7 | 0.000069 | 46652 | 3078 |
| change | screen_candidate | 0.000543 | 2 | 0.000020 | 54755 | 47011 |
| A1Cresult | screen_candidate | 0.000518 | 4 | 0.000039 | 84748 | 3812 |
| admission_type_id | screen_candidate | 0.000413 | 8 | 0.000079 | 53990 | 10 |
| race | screen_candidate | 0.000390 | 6 | 0.000059 | 76099 | 641 |
| max_glu_serum | screen_candidate | 0.000208 | 4 | 0.000039 | 96420 | 1264 |
| repaglinide | screen_candidate | 0.000156 | 4 | 0.000039 | 100227 | 45 |
| glipizide | screen_candidate | 0.000149 | 4 | 0.000039 | 89080 | 560 |
| glimepiride | screen_candidate | 0.000100 | 4 | 0.000039 | 96575 | 194 |
| pioglitazone | screen_candidate | 0.000088 | 4 | 0.000039 | 94438 | 118 |
| rosiglitazone | screen_candidate | 0.000084 | 4 | 0.000039 | 95401 | 87 |
| weight | screen_candidate | 0.000071 | 10 | 0.000098 | 98569 | 3 |
| glyburide | screen_candidate | 0.000067 | 4 | 0.000039 | 91116 | 564 |
| miglitol | screen_candidate | 0.000057 | 4 | 0.000039 | 101728 | 2 |
| acarbose | screen_candidate | 0.000056 | 4 | 0.000039 | 101458 | 3 |
| chlorpropamide | screen_candidate | 0.000054 | 4 | 0.000039 | 101680 | 1 |
| glyburide-metformin | screen_candidate | 0.000029 | 4 | 0.000039 | 101060 | 6 |
| nateglinide | screen_candidate | 0.000024 | 4 | 0.000039 | 101063 | 11 |
| gender | screen_candidate | 0.000022 | 3 | 0.000029 | 54708 | 3 |
| tolbutamide | screen_candidate | 0.000019 | 2 | 0.000020 | 101743 | 23 |
| troglitazone | screen_candidate | 0.000010 | 2 | 0.000020 | 101763 | 3 |
| tolazamide | screen_candidate | 0.000010 | 3 | 0.000029 | 101727 | 1 |
| metformin-rosiglitazone | screen_candidate | 0.000007 | 2 | 0.000020 | 101764 | 2 |
| acetohexamide | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glimepiride-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| metformin-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glipizide-metformin | screen_candidate | 0.000002 | 2 | 0.000020 | 101753 | 13 |
| citoglipton | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |
| examide | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |

### B3 full U distribution (49 features, all screen candidates — raw `readmitted` undeclared)

| feature | role | U (raw) | n_distinct | distinct/rows | max count | min count |
|---|---|---|---|---|---|---|
| encounter_id | screen_candidate | 1.0000 | 101766 | 1.0000 | 1 | 1 |
| readmitted | screen_candidate | 1.0000 | 3 | 0.000029 | 54864 | 11357 |
| number_inpatient | screen_candidate | 0.031552 | 21 | 0.000206 | 67630 | 1 |
| diag_1 | screen_candidate | 0.024362 | 717 | 0.007046 | 6862 | 1 |
| diag_3 | screen_candidate | 0.021779 | 790 | 0.007763 | 11555 | 1 |
| discharge_disposition_id | screen_candidate | 0.021616 | 26 | 0.000255 | 60234 | 2 |
| diag_2 | screen_candidate | 0.020309 | 749 | 0.007360 | 6752 | 1 |
| number_emergency | screen_candidate | 0.006913 | 33 | 0.000324 | 90383 | 1 |
| medical_specialty | screen_candidate | 0.005260 | 73 | 0.000717 | 49949 | 1 |
| num_medications | screen_candidate | 0.004523 | 75 | 0.000737 | 6086 | 1 |
| time_in_hospital | screen_candidate | 0.004300 | 14 | 0.000138 | 17756 | 1042 |
| number_diagnoses | screen_candidate | 0.003840 | 16 | 0.000157 | 49474 | 7 |
| num_lab_procedures | screen_candidate | 0.002676 | 118 | 0.001160 | 3208 | 1 |
| insulin | screen_candidate | 0.002600 | 4 | 0.000039 | 47383 | 11316 |
| number_outpatient | screen_candidate | 0.002249 | 39 | 0.000383 | 85027 | 1 |
| age | screen_candidate | 0.001786 | 10 | 0.000098 | 26068 | 161 |
| payer_code | screen_candidate | 0.001466 | 18 | 0.000177 | 40256 | 1 |
| diabetesMed | screen_candidate | 0.001084 | 2 | 0.000020 | 78363 | 23403 |
| metformin | screen_candidate | 0.000857 | 4 | 0.000039 | 81778 | 575 |
| admission_source_id | screen_candidate | 0.000768 | 17 | 0.000167 | 57494 | 1 |
| num_procedures | screen_candidate | 0.000669 | 7 | 0.000069 | 46652 | 3078 |
| change | screen_candidate | 0.000543 | 2 | 0.000020 | 54755 | 47011 |
| A1Cresult | screen_candidate | 0.000518 | 4 | 0.000039 | 84748 | 3812 |
| admission_type_id | screen_candidate | 0.000413 | 8 | 0.000079 | 53990 | 10 |
| race | screen_candidate | 0.000390 | 6 | 0.000059 | 76099 | 641 |
| max_glu_serum | screen_candidate | 0.000208 | 4 | 0.000039 | 96420 | 1264 |
| repaglinide | screen_candidate | 0.000156 | 4 | 0.000039 | 100227 | 45 |
| glipizide | screen_candidate | 0.000149 | 4 | 0.000039 | 89080 | 560 |
| glimepiride | screen_candidate | 0.000100 | 4 | 0.000039 | 96575 | 194 |
| pioglitazone | screen_candidate | 0.000088 | 4 | 0.000039 | 94438 | 118 |
| rosiglitazone | screen_candidate | 0.000084 | 4 | 0.000039 | 95401 | 87 |
| weight | screen_candidate | 0.000071 | 10 | 0.000098 | 98569 | 3 |
| glyburide | screen_candidate | 0.000067 | 4 | 0.000039 | 91116 | 564 |
| miglitol | screen_candidate | 0.000057 | 4 | 0.000039 | 101728 | 2 |
| acarbose | screen_candidate | 0.000056 | 4 | 0.000039 | 101458 | 3 |
| chlorpropamide | screen_candidate | 0.000054 | 4 | 0.000039 | 101680 | 1 |
| glyburide-metformin | screen_candidate | 0.000029 | 4 | 0.000039 | 101060 | 6 |
| nateglinide | screen_candidate | 0.000024 | 4 | 0.000039 | 101063 | 11 |
| gender | screen_candidate | 0.000022 | 3 | 0.000029 | 54708 | 3 |
| tolbutamide | screen_candidate | 0.000019 | 2 | 0.000020 | 101743 | 23 |
| troglitazone | screen_candidate | 0.000010 | 2 | 0.000020 | 101763 | 3 |
| tolazamide | screen_candidate | 0.000010 | 3 | 0.000029 | 101727 | 1 |
| metformin-rosiglitazone | screen_candidate | 0.000007 | 2 | 0.000020 | 101764 | 2 |
| acetohexamide | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glimepiride-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| metformin-pioglitazone | screen_candidate | 0.000003 | 2 | 0.000020 | 101765 | 1 |
| glipizide-metformin | screen_candidate | 0.000002 | 2 | 0.000020 | 101753 | 13 |
| citoglipton | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |
| examide | screen_candidate | 0.000000 | 1 | 0.000010 | 101766 | 101766 |

</details>

## Named anchors — U at every pooling level

| frame | feature | role | U raw | U pool>=5 | U pool>=10 | U pool>=20 | U pool>=50 |
|---|---|---|---|---|---|---|---|
| B1 | encounter_id | screen_candidate | 1.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B2 | encounter_id | screen_candidate | 1.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B3 | encounter_id | screen_candidate | 1.0000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B3 | readmitted | screen_candidate (undeclared leak) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| B2 | planted_leak | known_leak_reference (Addition 3) | 0.641748 | 0.641748 | 0.641748 | 0.641748 | 0.641748 |
| B1 | number_inpatient | screen_candidate | 0.031552 | 0.031497 | 0.031355 | 0.031320 | 0.031275 |
| B2 | number_inpatient | screen_candidate | 0.031552 | 0.031497 | 0.031355 | 0.031320 | 0.031275 |
| B3 | number_inpatient | screen_candidate | 0.031552 | 0.031497 | 0.031355 | 0.031320 | 0.031275 |

**Per-anchor, which frame it was measured in** (Addition 2's explicit requirement): `readmitted` — **B-3 only** (absent from B-1/B-2). `planted_leak` — **B-2 only** (absent from B-1/B-3, and there labeled `known_leak_reference`, not a screen candidate, per Addition 3). `encounter_id` and `number_inpatient` — measured identically in **all three frames** (same underlying column values in all three).

### Falsification conditions, checked

1. **Raw `readmitted` must trigger (B-3) → holds.** `U = 1.0000` exactly, at every pooling level. `readmitted_lt30` is derived directly from it — a perfect value-to-label correspondence, as predicted.
2. **`encounter_id` must not trigger → holds, but only because of the guard.** Raw `U = 1.0000` in all three frames — it *would* trigger on raw U alone, exactly as the ID-trap section anticipated. At **every** pooling floor tested (5, 10, 20, 50), `U` collapses to **exactly 0.000000** in all three frames. This is because `encounter_id`'s per-value count is uniformly 1 (`max_count = min_count = 1` — literally unique on every row), so any floor ≥ 2 pools the entire column into one bucket, and one bucket's conditional entropy equals the marginal entropy by construction (`U = 0` exactly, not approximately). The falsification condition holds under the guard, not under the raw signal.
3. **`number_inpatient` must not trigger → holds, trivially and robustly.** `U` ranges from 0.031552 (raw) down to 0.031275 (pool≥50) — nowhere near either leak anchor at any pooling level. Margin to the *weaker* leak anchor (`planted_leak`, 0.641748) is >0.61 at every floor.

**No pre-registered assumption failed. No STOP condition triggered.**

## `diag_1` / `diag_2` / `diag_3` — own section, as required

Identical across all three frames (same underlying feature values). 700+-category claim from the pre-registration is **not confirmed at this row count** — actual counts here are 717 / 749 / 790 distinct values over 101,766 rows (the pre-registration's "700+" figure, sourced from an earlier, less precise description, is in the right neighborhood but should be corrected to these exact counts going forward).

| feature | U raw | U pool>=5 | U pool>=10 | U pool>=20 | U pool>=50 | n_distinct raw | n_distinct @5 | n_distinct @10 | n_distinct @20 | n_distinct @50 |
|---|---|---|---|---|---|---|---|---|---|---|
| diag_1 | 0.024362 | 0.021906 | 0.019831 | 0.018010 | 0.015760 | 717 | 510 | 397 | 316 | 211 |
| diag_2 | 0.020309 | 0.018008 | 0.016429 | 0.014583 | 0.012026 | 749 | 488 | 382 | 297 | 192 |
| diag_3 | 0.021779 | 0.018542 | 0.016569 | 0.014749 | 0.012526 | 790 | 523 | 398 | 296 | 198 |

Rare-code tail (fraction of all 101,766 rows sitting in values with raw count below each floor):

| feature | frac rows in values <5 | <10 | <20 | <50 |
|---|---|---|---|---|
| diag_1 | 0.42% | 1.16% | 2.24% | 5.49% |
| diag_2 | 0.50% | 1.19% | 2.29% | 5.45% |
| diag_3 | 0.53% | 1.34% | 2.74% | 5.69% |

**How the rare-code tail affects `U`, and whether the guard handles it correctly.** Pooling steadily *reduces* all three columns' `U` as the floor rises (diag_1: 0.0244 → 0.0158, a 35% relative drop by floor 50) — real, measurable erosion of apparent signal as more of the long tail gets merged away. This confirms the pre-registration's expectation that these columns carry a genuine rare-code tail (up to ~5.7% of rows sit in values occurring fewer than 50 times). **The guard handles this correctly on this data**: even the *most generous* reading (raw, unpooled, no guard applied at all) puts all three columns at `U ≈ 0.02–0.024` — more than 25x below the weaker leak anchor (`planted_leak`, 0.6417) and more than 40x below the stronger one (`readmitted`, 1.0). The rare-code tail is real, but nowhere near large or concentrated enough on this data to push any of the three into leak territory, pooled or not. This is a real test the two extreme anchors (`encounter_id` fully unique, `number_inpatient` low-cardinality) could not have provided on their own, and it passed.

## Measured margin — no overlap, at any pooling level

| pooling level | max honest-tail U (excl. `encounter_id`/`readmitted`) | min leak-anchor U | gap |
|---|---|---|---|
| raw | 0.031552 (`number_inpatient`) | 0.641748 (`planted_leak`) | **0.610195** |
| pool>=5 | 0.031497 | 0.641748 | 0.610251 |
| pool>=10 | 0.031355 | 0.641748 | 0.610392 |
| pool>=20 | 0.031320 | 0.641748 | 0.610427 |
| pool>=50 | 0.031275 | 0.641748 | 0.610473 |

**No overlap at any pooling level — the gap is ~0.61 wide throughout**, roughly **20x** the honest ceiling itself. This is a substantially cleaner separation than Upgrade 1's own AUC-based margin (0.237–0.392 there, vs. ~0.61 here) — expected, since a deterministic value/label mapping is a much stronger and more binary property than a model's discriminative score. Pooling can only ever narrow the honest ceiling further (merging values weakly increases conditional entropy for the merged group, per the math — confirmed empirically: every honest feature's `U` is monotonically non-increasing as the floor rises) or leave a well-supported leak anchor untouched (both `readmitted` and `planted_leak` have per-value counts in the tens of thousands, far above any floor tested, so pooling never touches them) — so the raw-distribution gap reported here is the *most conservative* (smallest possible) gap across all guard variants tested, and it is already unambiguous.

## Candidate guard, derived from the measured numbers

**1. Cardinality ceiling (distinct-value count / row count).** Measured honest-tail maximum: `diag_3` at **0.7763%**. `encounter_id` sits at **100%**. No feature in this data falls between roughly 0.8% and 100% — a wide, empty gap, the same shape of honest caveat `UPGRADE1_CALIBRATION.md` recorded for its own AUC floor: **the exact ceiling number is not tightly pinned by this data** (nothing observed sits close enough to any candidate cutoff to adjudicate the precise value), but any ceiling comfortably above ~1% and comfortably below ~50% would separate every honest feature measured here from `encounter_id` cleanly. A candidate value of **5%** is proposed for future work (comfortably above `diag_3`'s 0.78%, comfortably below anything ID-like) — offered as a starting point for wider calibration on more datasets, not a final number this single dataset can justify on its own.

**2. Minimum per-value support floor.** All four tested floors (5, 10, 20, 50) **fully neutralize** `encounter_id` (`U` drops from 1.0 to exactly 0.0 at the smallest floor already, since every one of its values has count 1) while leaving both leak anchors completely untouched (their rarest value still has 11,357+ occurrences, far above any floor tested) and only mildly eroding the genuine `diag_1`/`diag_2`/`diag_3` signal (a 25–35% relative reduction by floor 50, never enough to approach leak territory). **A floor of 20** is proposed — chosen for consistency with Upgrade 1's own existing `_MIN_MINORITY_CLASS_COUNT = 20` convention (`undeclared_feature_probe.py`) rather than because this data distinguishes it from 5, 10, or 50; all four performed identically on the falsification conditions here. Like the cardinality ceiling, the exact number is under-determined by this single dataset and should be treated as a starting point, not a final calibrated value.

**Both guard components independently would have separated `encounter_id` from every real anchor on this data** — the cardinality ceiling by exploiting the ~0.8%-vs-100% gap, the support floor by exploiting the fact that `encounter_id`'s support is the minimum possible (1) versus tens of thousands for both real leaks. Neither component alone was strictly necessary to pass the falsification conditions measured here; the pre-registration's requirement that both be present is not undermined by that — a future dataset with a moderate-cardinality-but-still-largely-unique column (not tested here) could plausibly need the ceiling where the support floor alone wouldn't catch it, or vice versa.

## What remains unvalidated (stated plainly, not glossed over)

- **The exact guard thresholds are not tightly pinned by this data.** As stated above for both components — a wide empty gap exists between the honest ceiling and `encounter_id`'s extreme values, but nothing in this dataset sits close enough to any candidate cutoff to determine the precise boundary. This is the same honest caveat this project's own calibration history has recorded before (`F2b_CALIBRATION.md`'s NSL=1.0 boundary, `UPGRADE1_CALIBRATION.md`'s NEAR_CERTAIN_AUC_FLOOR).
- **Only one real dataset (Diabetes-130) was used.** No synthetic fixture (comparable to Upgrade 1's F2b) was constructed or scored here to stress-test the cardinality ceiling or support floor against a wider range of honest high-cardinality features than Diabetes-130 alone provides. `diag_1`/`diag_2`/`diag_3` are the only moderately-high-cardinality honest features this data offers.
- **NaN handling was implemented but not exercised** — zero NaN values exist in any of the three frames' columns. Whether the `__NaN__`-as-category treatment behaves as intended on a real column with missingness (and whether missingness-driven partitions are a real risk in practice, not just in principle) was not tested here.
- **Per-period value locality was not measured.** The pre-registration flagged, as an open question, whether a value with adequate *total* support that is nonetheless concentrated in one narrow time window deserves a separate guard from the flat support floor. This calibration did not compute anything per-period — it is whole-frame only, as pre-registered — so this question remains exactly as open as it was before this calibration ran.
- **The guard's behavior on a genuinely continuous, near-unique real leak was not (and structurally cannot be) tested here** — no such fixture exists in Diabetes-130, and the pre-registration already named this as an accepted blind spot of the guard itself, not something calibration could resolve.
- **`known_strong_features`-style suppression was not exercised** — same status as Upgrade 1's own calibration: nothing to suppress since no allowlist contract field exists yet.

## Provenance

Raw evidence: `scratch/upgrade_h_calibration_distribution.csv` (146 rows: 48 B-1
screen candidates, 49 B-2 rows [48 screen candidates + 1 known-leak
reference], 49 B-3 screen candidates), `scratch/upgrade_h_calibration_raw.json`
(same data, JSON form). Both scratch-only, untracked, not committed.

Regenerate via the scratch-only script (not committed, same convention as
`scratch/upgrade1_calibration.py`):

```
.venv/Scripts/python.exe scratch/upgrade_h_calibration.py
```

Requires `testB1_specificity.csv`, `testB2_sensitivity.csv`, and
`testB3_honest_unknown.csv` (the full 101,766-row Test B frames) at
`<DATA_DIR>/` —
these are external to the repo, same convention as every prior Test B
calibration document.

## Addendum (2026-07-21) — null-inflation floor derivation, cardinality-ceiling necessity, U>=0.99 flag criterion

Three follow-up measurements, appended without editing anything above.
**Report, not tune** — no number below was searched for to make an anchor
pass; every figure is read directly off the measured simulation or
distribution.

### A. Null-feature chance-inflation ceiling (derives the support floor)

Synthetic categorical features **independent of the target by construction**
were generated against the real B-2 100k frame's actual `readmitted_lt30`
column (`n=101,766`, `n1=11,357`, `H(Y)=0.504721` bits): for each
`K ∈ {50, 200, 800, 2000, 5000, 20000}` distinct values, every row was
assigned a uniformly random value in `[0, K)`, independent of row order or
target by construction, 10 seeds per `K`. `U` was measured raw and pooled at
floors `{2, 5, 10, 20, 50, 100}`. Script: `scratch/upgrade_h_null_inflation.py`;
raw evidence: `scratch/upgrade_h_null_inflation.csv` (60 rows: 6 K values ×
10 seeds).

**Max U observed across the 10 seeds, per (K, floor) cell:**

| K | avg rows/value | U raw | pool>=2 | pool>=5 | pool>=10 | pool>=20 | pool>=50 | pool>=100 |
|---|---|---|---|---|---|---|---|---|
| 50 | 2,035.3 | 0.000789 | 0.000789 | 0.000789 | 0.000789 | 0.000789 | 0.000789 | 0.000789 |
| 200 | 508.8 | 0.003206 | 0.003206 | 0.003206 | 0.003206 | 0.003206 | 0.003206 | 0.003206 |
| 800 | 127.2 | 0.012482 | 0.012482 | 0.012482 | 0.012482 | 0.012482 | 0.012482 | 0.012377 |
| 2000 | 50.9 | 0.030414 | 0.030414 | 0.030414 | 0.030414 | 0.030414 | 0.016965 | 0.000000 |
| 5000 | 20.4 | 0.081837 | 0.081837 | 0.081837 | 0.081544 | 0.045570 | 0.000000 | 0.000000 |
| 20000 | 5.1 | 0.299403 | 0.292546 | 0.185204 | 0.012233 | 0.000000 | 0.000000 | 0.000000 |

**Every cell in this table is far short of the flag criterion (`U >= 0.99`)** — the single highest value measured anywhere, at any (K, floor) combination, is **0.299403** (K=20000, raw/unpooled). Nothing in this simulation approaches 0.99 at any floor. **No STOP condition triggers**: an independent feature does not reach high U at every tested floor — it reaches a moderate ceiling (~0.30) only at the highest K tested and only with little or no pooling, and that ceiling is driven to exactly 0 by floor 20 already (for every K up to 20,000).

**Which floors keep chance inflation below the flag criterion, and by what margin**: **every tested floor, including floor 2**, keeps chance inflation below 0.99 — worst case (K=20000, floor≥2) tops out at 0.292546, a margin of **0.697454** below the criterion. Relative to the flag criterion alone, the support floor is not doing much work — the gap between chance noise and 0.99 is wide even unpooled, because 0.99 is a very high absolute bar to clear by chance alone at these K values.

**A stricter, more informative bar than "below 0.99" is "does not exceed the real honest-feature ceiling measured in the main calibration above" (0.031552, `number_inpatient`)** — this is the bar that actually matters for not letting synthetic noise look *more* suspicious than genuine honest signal. By that bar: floor 2 and floor 5 **fail** for K=20000 (0.292546 and 0.185204, both far above 0.0316 — a truly independent, meaningless feature at high cardinality would outrank the strongest real honest feature in the data under those floors). **Floor 10 is the smallest tested floor that gets under the honest ceiling for every K tested** (K=20000 max = 0.012233 < 0.031552). **Floor 20 is the smallest tested floor that drives chance inflation to exactly 0.0 for every K tested**, the strongest and cleanest property measured here.

**Recommendation: floor >= 20**, derived independently from this simulation (the smallest floor producing an exact-zero chance-inflation ceiling across the full tested K range) — **not** chosen to match Upgrade 1's `_MIN_MINORITY_CLASS_COUNT = 20`, though it happens to coincide with it; that coincidence is noted, not the justification.

**Gap in this measurement, stated plainly**: `K` was tested only up to 20,000 (19.6% of `n`). `encounter_id` is the `K = n` extreme (101,766) and is a **deterministic** bijection there (`U = 1.0` exactly, not chance-driven) — this simulation does not establish the shape of the chance-inflation curve between `K=20,000` and `K=n`, i.e. whether a very-high-but-not-fully-unique cardinality (e.g. `K≈50,000–90,000`) could push chance-driven `U` above 0.99 before the floor engages. No feature in the real B-1/B-2/B-3 frames sits in that range (the highest real cardinality measured is `diag_3` at 790, `encounter_id` at 101,766 — nothing in between), so this gap could not be closed with real data here; it is recorded as unmeasured, not assumed safe.

### B. Is the cardinality ceiling load-bearing at all?

**With NO cardinality ceiling, support floor only** — re-checked across the full floor grid `{2, 5, 10, 20, 50, 100}` (the original report only covered `{5,10,20,50}`; floors 2 and 100 measured fresh here):

| frame | feature | U raw | pool>=2 | pool>=5 | pool>=10 | pool>=20 | pool>=50 | pool>=100 |
|---|---|---|---|---|---|---|---|---|
| B1 | encounter_id | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B2 | encounter_id | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B3 | encounter_id | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B3 | readmitted | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| B2 | planted_leak | 0.641748 | 0.641748 | 0.641748 | 0.641748 | 0.641748 | 0.641748 | 0.641748 |

**Yes, confirmed explicitly: `encounter_id` scores exactly 0.0 and `readmitted` scores exactly 1.0 in every frame, at every tested floor from 2 through 100, with no cardinality ceiling applied at all.** `encounter_id`'s per-value count is uniformly 1 in every frame, so any floor ≥ 2 pools its entire 101,766-value column into a single bucket, giving `U = 0` exactly by construction — not an approximation. **Plainly stated, as the task requires: the cardinality ceiling is not load-bearing on this data** — the support floor alone, at any tested value, fully separates `encounter_id` from both real leak anchors without it. This should not be read as "the ceiling is unnecessary in general" — see Part A's gap (untested `K` between 20,000 and `n`) and the ratio-ceiling analysis immediately below, which shows the ceiling has its own, different failure mode that the anchors in this dataset don't happen to expose either.

**Ratio-based ceiling of 5% — absolute distinct-value counts permitted, and whether `diag_1` survives, by `n`:**

| n | 5% ceiling (distinct values allowed) | `diag_1` measured distinct count | survives? |
|---|---|---|---|
| ~1,000 | 50 | **196–225** (5 seeds, mean ~209; measured via seeded subsample of `scratch/testB2_strat.csv`) | **NO — fails by 4x, would be wrongly excluded** |
| 5,000 (`scratch/testB2_5k.csv`, real subsample) | 250 | **374** | **NO — fails by ~1.5x, would be wrongly excluded** |
| 10,008 (`scratch/testB2_strat.csv`, real stratified sample) | 500 | **458** | Survives, but narrowly (~8.4% headroom) |
| 101,766 (full B-2/B-3 frame) | 5,000 | **717** | Survives comfortably (~86% headroom) |

**This is the n-sensitivity finding requested, on the record either way, and it is decisive against a flat-ratio ceiling**: `diag_1`'s distinct-value count does **not** shrink proportionally with `n` (ICD-9-style diagnosis codes have a long tail; even a 1,000-row sample still touches ~200 distinct codes, not ~7 as a naive `717 * 1000/101766` linear scaling would suggest) — so a **fixed percentage** ceiling penalizes smaller audits far more harshly than larger ones, and would **wrongly exclude a real, legitimate, moderately predictive categorical feature** at both of the two smaller scales actually measured here (n≈1,000 and n=5,000), only marginally clearing it at n≈10k, and only comfortably clearing it at the full 101,766-row scale this project has calibrated against so far. A ratio-based ceiling designed and validated only against 100k-scale data would silently misbehave on smaller real-world audits — worth recording plainly as a design risk for whenever the cardinality ceiling *is* implemented.

### C. Flag criterion at the near-bijection floor (`U >= 0.99`, support floor = 20 applied)

Applying the Part-A-derived floor (20) and re-checking `U >= 0.99` across every feature in all three frames (146 rows total, `scratch/upgrade_h_calibration_distribution.csv`, `U_pool20` column):

**Features triggering:**

| frame | feature | role | U raw | U (floor=20) |
|---|---|---|---|---|
| B-3 | `readmitted` | screen_candidate (undeclared leak) | 1.000000 | **1.000000** |

**Exactly one feature triggers, in exactly one frame.** Nothing in B-1 or B-2 crosses 0.99 at this floor.

**Highest non-triggering feature**: `planted_leak` (B-2, `known_leak_reference`), `U (floor=20) = 0.641748`. **Measured distance from the highest non-triggering feature to the criterion: `0.990000 - 0.641748 = 0.348252`.**

**Is it correct that `planted_leak` does not trigger, given (H) claims a deterministic partition and `confirmed=True`? Yes — this is correct, and it is not a coverage gap.** `planted_leak` is a genuine leak (Zekan's own B/C ablation attribution correctly measures and flags it, `+0.3089` AUC, `FAIL`), but it is **not a deterministic one** — its measured `U = 0.6417` means roughly a third of the target's uncertainty remains even knowing `planted_leak`'s value, consistent with this fixture's own design (a target-copy leak with injected label noise, not an exact copy). (H)'s claim, when it fires, is specifically "this feature's values determine the target as a fact about the data" (`confirmed=True`) — firing on `planted_leak` here would be an **incorrect, overconfident claim the data does not support**. Catching this shape of leak (real, strong, but imperfect) is explicitly **not (H)'s job** per its own pre-registered scope-honesty section — it is Upgrade 1's `NEAR_CERTAIN`/panel's job (an AUC-based, `confirmed=False` instrument, better suited to an imperfect signal), and Zekan's primary B/C ablation attribution's job (which already caught it, independent of either structural screen). Additionally, in normal operation (H) would never even see `planted_leak` as a candidate at all — it is declared `forbidden_after_prediction` in the real B-2 contract, and was only scored here, out-of-band, as a reference point per this task's own Addition 3; the "correct" question is not "should (H) have flagged a forbidden column" (structurally it never evaluates one) but "would (H) be expected to catch an *undeclared* leak of this same imperfect shape" — and the honest answer, consistent with the pre-registration, is no, by design.

### What this addendum leaves unresolved (stated plainly, not glossed over)

- **The chance-inflation curve between `K=20,000` and `K=n` (101,766) was not measured.** The real anchors available in this dataset don't fill that gap (`diag_3` at 790 is the highest real non-ID cardinality; `encounter_id` at 101,766 is the only data point at the extreme). Whether a hypothetical feature at, say, `K≈60,000` could push chance-driven `U` uncomfortably close to 0.99 before any reasonable floor engages remains unknown from this evidence.
- **The `n=1,000` `diag_1` cardinality figure is a subsample estimate** (5 seeds off `scratch/testB2_strat.csv`, range 196–225), not a real, independently-collected 1,000-row Test B frame — reported as a reasonable estimate for the n-sensitivity question, not as a separately-validated ground truth the way the 10k/100k figures are (those come from real, already-established Test B frames).
- **Neither guard component's exact number is fully pinned by this addendum either** — floor=20 is now supported by an independent chance-inflation argument (Part A), but the *upper* bound on how large a floor could safely go before it starts eroding genuine signal (e.g. `diag_1`/`diag_2`/`diag_3`'s real, if modest, categorical signal) was not swept here; only floors up to 100 were tested, and only against the specific features already in the main calibration.
- **The ratio-ceiling finding (Part B) argues against a flat-percentage design, but does not propose a replacement.** An n-adaptive cardinality rule (e.g., a floor on absolute distinct-value count rather than a ratio, or a curve fit to how categorical cardinality actually grows with `n` for this kind of data) is implied as a better direction but was not designed or measured here — flagged as follow-up, not solved.

## Correction (2026-07-21) — the previous addendum's floor-clears-the-honest-ceiling claims were wrong

**Found while re-verifying the full grid before writing the next addendum below — flagged before anything else, per this project's own standing discipline.** Two sentences in the addendum immediately above are incorrect and are corrected here, not silently edited:

- *"Floor 10 is the smallest tested floor that gets under the honest ceiling for every K tested (K=20000 max = 0.012233 < 0.031552)"* — **wrong**. That sentence only checked `K=20000`'s value. The actual max for `U_pool10` across the full originally-tested grid is **0.081544 at K=5000** (already present in the original table, just not compared against the ceiling at the time) — well above the 0.031552 honest ceiling, not under it.
- *"Floor 20 is the smallest tested floor that drives chance inflation to exactly 0.0 for every K tested"* — **wrong**. `U_pool20` at `K=5000` is **0.045570**, not `0.0` (also already present in the original table). Floor 20 reaches exactly `0.0` starting at `K=20000` and above, not for every `K` in the tested set — and its true max across the full grid, **0.045570 at K=5000, is itself above the 0.031552 honest ceiling**, not under it.

**Root cause of the error**: both claims implicitly assumed the worst case sits at the highest tested `K`, which is true for `U_raw` (monotonically increasing toward the deterministic `K=n` endpoint) but **false for a fixed pooling floor**, where the worst case sits near whichever `K` makes the average per-value row count (`n/K`) land close to the floor itself — a peak, not a monotonic trend. `U_pool10`'s peak is at `K=5000` (avg 20.4 rows/value); `U_pool20`'s peak is also at `K=5000` (same reason); at higher `K` the average count drops well below the floor and pooling absorbs nearly everything, driving `U` back down toward 0. This peaked (not monotonic) shape was visible in the original data but not checked properly against the ceiling at every `K` before the claim was written.

**Corrected, full-grid-verified result** (now covering `K` from 50 through 101,766 — see the K-gap closure immediately below, which supplied the missing high-`K` values used to build this complete table): **`floor >= 50` is the smallest tested floor whose worst case across every tested `K` (0.016965, at `K=2000`) stays under the real honest ceiling (0.031552).** `floor=20`'s true worst case (0.045570 at `K=5000`) does **not** clear that stricter bar — it is about 1.44x the honest ceiling, not comfortably under it as previously claimed.

**This does not change the floor=20 recommendation relative to the criterion (H) actually uses.** The 0.99 flag criterion is the number that matters for whether the guard produces a false `NEAR_CERTAIN`-shaped trigger — and floor=20's worst case anywhere in the fully-tested range (0.045570) is still **0.944 below 0.99**, an enormous margin. The error was in a secondary, stricter comparison (against the honest-tail ceiling, introduced as extra context, not as the actual design criterion) — it does not touch the primary falsification result (no floor tested ever approaches 0.99). It is corrected here because it was stated as fact and was wrong, not because it changes the bottom line.

## Addendum (2026-07-21) — Part A extension: closing the K gap

**Prediction on record, stated before this measurement ran**: as `K` rises toward `n`, per-value counts fall below the support floor, pooling absorbs them, and `U` should **drop** toward 0 — meaning the previously-unmeasured region (`K` between 20,000 and `n=101,766`) is **safer** than the measured one, not more dangerous.

`K ∈ {30000, 50000, 70000, 90000, 101766}` were added to the simulation, same 10 seeds, same floors, same target column and seeding formula as the original run. Script: `scratch/upgrade_h_null_inflation_extended.py`; raw evidence: `scratch/upgrade_h_null_inflation_extended.csv` (50 rows). Note on method: at `K=101766`, values are drawn **with replacement** from `[0, 101766)` — this does not reproduce `encounter_id`'s actual structure (a true bijection, every value used exactly once); it produces a realistic collision distribution instead (mean ~64,300 distinct values actually realized out of 101,766 possible, consistent with the birthday-paradox expectation `n(1-1/e)`). This is the right test for "very-high-but-not-perfectly-unique cardinality," not a literal `encounter_id` replica.

**Max U observed, extended range:**

| K | avg rows/value | n_distinct realized (mean) | U raw | pool>=2 | pool>=5 | pool>=10 | pool>=20 | pool>=50 | pool>=100 |
|---|---|---|---|---|---|---|---|---|---|
| 30000 | 3.39 | 28,977 | 0.396214 | 0.362176 | 0.122103 | 0.001673 | 0.000000 | 0.000000 | 0.000000 |
| 50000 | 2.04 | 43,460 | 0.532814 | 0.399421 | 0.044384 | 0.000134 | 0.000000 | 0.000000 | 0.000000 |
| 70000 | 1.45 | 53,677 | 0.620287 | 0.385765 | 0.018841 | 0.000033 | 0.000000 | 0.000000 | 0.000000 |
| 90000 | 1.13 | 60,914 | 0.680700 | 0.356899 | 0.009366 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 101766 | 1.00 | 64,299 | 0.707934 | 0.341723 | 0.006184 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Verdict, by floor — the prediction's answer depends on which floor, and this is reported plainly rather than collapsed into one yes/no:**

- **`floor >= 10`, and every floor recommended or considered above (20, 50, 100): prediction CONFIRMED, cleanly.** `U_pool10` continues the decline already visible at `K=20000` (0.012233) down through 0.001673, 0.000134, 0.000033, to exactly `0.0` by `K=90000`. `U_pool20`/`50`/`100` are already exactly `0.0` at `K=20000` and remain exactly `0.0` at every newly-tested `K` through `n`. **No STOP condition applies** — the newly-closed region is at least as safe as the previously-tested region, for every floor that was actually under consideration for the guard.
- **`floor = 2` and raw (unpooled): prediction CONTRADICTED, and reported honestly rather than hidden.** `U_raw` climbs monotonically all the way to `K=n` (0.396 → 0.708), consistent with converging toward the true deterministic bijection value (`1.0`, not reached exactly here only because of with-replacement collisions). `U_pool2` does not decline either — it peaks around `K=50000` (0.399) and stays substantial (0.34–0.40) through `K=101766`, because floor 2 only removes true singletons and a large fraction of values still land at count 1–2 by chance at these `K`. **Neither of these two settings was ever the floor under consideration** (the recommendation has been `>= 20` since the prior addendum, now corrected to `>= 50` against the stricter honest-ceiling bar above) — so this contradiction does not apply to the guard as actually being designed, but it is recorded because the instruction was to report a contradiction if the data showed one, not to only report it if it was inconvenient. **A weak floor (2, or none at all) would indeed make the high-`K` region more dangerous, not less — this is exactly why floor selection matters and low floors were never adopted.**

## DESIGN DECISIONS (2026-07-21) — cardinality ceiling DROPPED: DEPARTURE FROM THE PRE-REGISTRATION

`UPGRADE_H_PREREGISTRATION.md`'s guard was specified as two required components (a cardinality ceiling AND a minimum per-value support floor). Based on all calibration evidence gathered above, **the cardinality ceiling is dropped from the design** — this is a departure from what was pre-registered, labelled as such here, not silently narrowed.

**1. Flag criterion.** `U >= 0.99`, applied after the support floor's pooling. Across all three real frames (B-1, B-2, B-3), the **sole trigger is `readmitted` (B-3)**, `U = 1.0000`. The highest non-triggering feature is `planted_leak` (B-2, known-leak reference), `U = 0.6417` — a measured margin of **0.348** below the criterion.

**2. Support floor: 20.** Derived from the null-feature chance-inflation simulation (Part A, both the original run and this session's K-gap closure), independent of Upgrade 1's own convention. **Correcting the figure this decision was originally justified with** (see the Correction section above, written before this one): floor 20 does not drive chance inflation to exactly `0.0` at every tested `K` — its true worst case across the full now-tested range (`K` from 50 to 101,766) is **0.045570 (at K=5000)**. What holds, verified: **floor 20's worst case anywhere in the tested range stays 0.944 below the actual 0.99 flag criterion** — an enormous margin for the criterion this guard actually applies. (Floor 50 is the smallest tested floor that additionally clears the *stricter*, non-criterion honest-tail-ceiling comparison of 0.031552 at every `K`, with its own worst case at 0.016965 — recorded here as an alternative, more conservative option, in case a tighter margin than "0.944 below 0.99" is wanted; floor 20 is what is being recorded as the decision, per the instruction, since it comfortably satisfies the criterion that governs the actual guard.) Floors 2 and 5 fail even the loose criterion comparison in spirit (they leave the door open to a moderate false signal at high cardinality — floor 5's worst case is 0.185, floor 2's is 0.399 — though still short of 0.99 itself). **This floor value (20) matches Upgrade 1's `_MIN_MINORITY_CLASS_COUNT` by independent measurement, not by borrowing** — it was derived here from a chance-inflation simulation against (H)'s own criterion, and the coincidence with Upgrade 1's number is noted, not the justification.

**3. Cardinality ceiling: dropped from the design entirely.** Two independent, measured reasons:

   - **(a) Not load-bearing.** With no cardinality ceiling at all, the support floor alone drives `encounter_id` to exactly `U = 0.0` in all three frames (B-1, B-2, B-3), at every tested floor from 2 through 100 — `encounter_id`'s per-value count is uniformly 1, so any floor >= 2 pools its entire column into one bucket by construction. The ceiling adds nothing to the falsification conditions this project has actually tested.
   - **(b) A ratio-based ceiling is n-sensitive, and degrades the check hardest for the smallest audits — the opposite of what you'd want.** Measured directly: a flat 5% ceiling permits 50 distinct values at n=1,000, 250 at n=5,000, 500 at n=10,008, and 5,000 at n=101,766. `diag_1`'s measured distinct-value count at each of those scales is **196–225 (n~1,000, 5 seeds) — fails the cap by ~4x**; **374 (n=5,000, real subsample) — fails the cap by ~1.5x**; **458 (n=10,008, real stratified sample) — survives, but only ~8.4% headroom**; **717 (n=101,766, full frame) — survives comfortably, ~86% headroom**. A real, legitimate, moderately-predictive categorical feature would be **wrongly excluded** at exactly the two smaller scales measured, precisely because diagnosis-code cardinality does not shrink proportionally with row count. Shipping a ratio-based ceiling calibrated only against the 100k-scale evidence this project has would silently misbehave on smaller real audits.

   The support floor alone already carries the guard's entire measured protective value; the ceiling's only calibrated behavior found here is a failure mode (b), not a benefit. Dropping it is therefore a simplification supported by evidence, not an unexamined shortcut — but it remains a genuine departure from what `UPGRADE_H_PREREGISTRATION.md` specified, and any future evidence of a feature the floor alone doesn't catch (see the still-open `K` between-20,000-and-`n` chance-inflation shape, closed above only for the floors that matter, and the still-untested genuinely-continuous-near-unique-leak case) would be grounds to revisit this decision, not proof it was unconditionally safe to make.

**4. `planted_leak` (U=0.6417) not triggering is correct, not a miss.** (H) asserts a **deterministic** value-to-label partition and earns `confirmed=True` specifically because of that determinism. `planted_leak` is a real, measured leak (Zekan's own B/C ablation attribution catches it independently, `+0.3089` AUC, `FAIL`) but not a deterministic one — a third of the target's uncertainty remains even knowing its value. Firing `confirmed=True` on it would overclaim what the data supports. Catching a real-but-imperfect leak is Upgrade 1's `NEAR_CERTAIN`/panel's job (`confirmed=False`, AUC-based) and the ablation engine's job (which already caught this specific one) — not (H)'s. The two checks are complementary by design, not redundant, and this is exactly what the pre-registration's own scope-honesty section anticipated.

**5. Named residual limits, carried forward from all evidence gathered so far:**
   - **Single-dataset calibration.** Every number above (the flag criterion's margin, the floor's chance-inflation ceiling, the ratio-ceiling's n-sensitivity) comes from Diabetes-130 / Test B alone. No second real dataset or synthetic fixture (comparable to Upgrade 1's F2b) has been used to check whether these figures generalize.
   - **Single target base rate.** All measurements here used the one 11.16%-positive target (`readmitted_lt30`). Whether the chance-inflation ceiling or the honest-tail ceiling shift materially at a different base rate (e.g. a much rarer or much more balanced target) has not been tested.
   - **Untested under `--stability`.** Nothing in this calibration exercised Zekan's seed-stability re-verdicting path. Whether (H)'s output is stable, or interacts at all, under `--stability` remains exactly as unknown as it was before this calibration began.

## SUPPORT FLOOR RE-DERIVATION (2026-07-21)

The Design Decisions section's floor=20 justification rested on a comparison against the 0.031552 honest-tail ceiling — a bar the Correction section above has since retracted as the wrong operative question (nothing anywhere near 0.03 is ever flagged; the criterion that actually governs a trigger is `U >= 0.99`). This section re-derives the floor against the correct bar. No new simulation was run — every figure below is read from the existing grids already on disk (`scratch/upgrade_h_null_inflation.csv`, `scratch/upgrade_h_null_inflation_extended.csv`, `scratch/upgrade_h_calibration_distribution.csv`), plus two deterministic re-derivations (floor 2 and floor 100 for `diag_1`/`diag_2`/`diag_3`, not previously tabulated) computed directly from the same per-value joint-frequency tables already built for the main calibration — no new data pull, no new randomness. Neither the Correction section nor the Design Decisions section above is edited; both stay on the record as written.

### 1. Single maximum U achieved by a known-independent feature, across all K and all floors

Across the full null-inflation grid (`K ∈ {50,...,101766}`, all 10 seeds, pooled at any of the six candidate floors): **max U = 0.399421**, at `K=50000`, `floor=2`. **Margin to 0.99: 0.590579.**

(Context, not a candidate floor: the unpooled/raw maximum is 0.707934 at `K=101766` — margin 0.282066. This is why pooling matters at all; it is not one of the six candidate floors under consideration.)

### 2. Worst-case independent-feature U per candidate floor, across all K, and margin to 0.99

| floor | worst-case U (any K) | K where worst case occurs | margin to 0.99 |
|---|---|---|---|
| 2 | 0.399421 | 50,000 | 0.590579 |
| 5 | 0.185204 | 20,000 | 0.804796 |
| 10 | 0.081544 | 5,000 | 0.908456 |
| 20 | 0.045570 | 5,000 | 0.944430 |
| 50 | 0.016965 | 2,000 | 0.973035 |
| 100 | 0.012377 | 800 | 0.977623 |

**Every candidate floor clears the actual criterion by a wide margin** — even the weakest, floor 2, sits at 0.59 below 0.99. Raising the floor buys progressively more margin (0.59 → 0.98), but the criterion itself (0.99) was never close to being threatened at any floor tested.

### 3. Does `encounter_id` go to 0.0 in all three frames, per candidate floor?

| floor | B-1 | B-2 | B-3 |
|---|---|---|---|
| 2 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 0.000000 | 0.000000 | 0.000000 |
| 10 | 0.000000 | 0.000000 | 0.000000 |
| 20 | 0.000000 | 0.000000 | 0.000000 |
| 50 | 0.000000 | 0.000000 | 0.000000 |
| 100 | 0.000000 | 0.000000 | 0.000000 |

**Yes, at every candidate floor, in every frame** — exactly, not approximately, because `encounter_id`'s per-value count is uniformly 1 in all three frames, so any floor >= 2 pools its entire column into a single bucket. (Floor 1, i.e. no pooling at all, is excluded from the candidate set for exactly this reason: it is the one value that would fail this requirement, since a value with count 1 is only pooled once the floor reaches >= 2.)

### 4. `diag_1`/`diag_2`/`diag_3` U per candidate floor, and erosion from raw (the real cost of a higher floor)

Identical across all three frames (same underlying feature values, per the main calibration).

| feature | raw | f2 (erosion) | f5 (erosion) | f10 (erosion) | f20 (erosion) | f50 (erosion) | f100 (erosion) |
|---|---|---|---|---|---|---|---|
| diag_1 | 0.024362 | 0.023690 (2.8%) | 0.021906 (10.1%) | 0.019831 (18.6%) | 0.018010 (26.1%) | 0.015760 (35.3%) | 0.013434 (44.9%) |
| diag_2 | 0.020309 | 0.019476 (4.1%) | 0.018008 (11.3%) | 0.016429 (19.1%) | 0.014583 (28.2%) | 0.012026 (40.8%) | 0.010349 (49.0%) |
| diag_3 | 0.021779 | 0.020297 (6.8%) | 0.018542 (14.9%) | 0.016569 (23.9%) | 0.014749 (32.3%) | 0.012526 (42.5%) | 0.011203 (48.6%) |

**Erosion rises steadily and substantially with the floor** — from ~3–7% at floor 2 to ~45–49% at floor 100. This is the real, measured cost side of the tradeoff: every increment of floor beyond the minimum needed buys additional margin against a criterion that was never in danger, while steadily discarding real (if modest) categorical signal from exactly the features this project's own evidence (`diag_1`'s Pearson pre-rank failure, `UPGRADE1_CALIBRATION.md`'s cost investigation) has already flagged as legitimate and worth preserving.

### Which floors satisfy both hard requirements, and the recommendation

**Both hard requirements — independent-feature worst case far below 0.99 (requirement 1/2 above), and `encounter_id -> 0.0` in all three frames (requirement 3) — are satisfied by every one of the six candidate floors tested: 2, 5, 10, 20, 50, and 100, with no exception.** Stated plainly, as instructed: there is no discriminating evidence among them on the two hard requirements — all six pass both, including the lowest one tested.

**Recommendation: floor = 2.** Among floors that are equivalent on both hard requirements, the lowest is preferred on the stated grounds that a higher floor buys no additional safety against the criterion that actually governs a trigger (`U >= 0.99`) while steadily eroding legitimate categorical signal. The measured evidence for this specific choice: floor 2's independent-feature worst case (0.399421) is still 0.59 below 0.99 — an enormous margin for the criterion in play — while its erosion of `diag_1`/`diag_2`/`diag_3` (2.8–6.8%) is the smallest of any candidate floor by a wide margin (the next step up, floor 5, already costs 10–15%; floor 20, the previously-recorded decision, costs 26–32%; floor 100 costs nearly half the signal). This is a preference stated as a preference, following the instruction, not dressed up as a finding the data alone forced — the data shows six floors tied on safety and one clear ranking on cost, and the lowest-cost option among tied-safety options is recommended on that basis.

**This supersedes the Design Decisions section's floor=20 choice**, which was derived against the wrong bar (see Correction, above) — it does not retroactively make floor=20 unsafe (0.944 margin is still enormous), only unnecessarily costly relative to floor=2, which is equally safe against the criterion that actually matters and preserves more real signal.

## NEAR-ID ANCHOR: patient_nbr (2026-07-21)

The calibration has never scored `patient_nbr` — the declared `entity_id` in every Test B contract, structurally excluded from every screen candidate set (it is in the `role` exclusion set, alongside `period_ordinal`/`readmitted_lt30`, in every frame's contract). It is the real near-ID case the guard needs to survive: unlike `encounter_id` (unique on every row, `max_count = min_count = 1`), `patient_nbr` repeats — the same patient can have multiple encounters — so it is **not** fully absorbed by a floor of 2 the way `encounter_id` is. This is the strongest available real-data stress test of the floor=2 recommendation. No simulation; deterministic counting on the same three real frames, reusing the same functions as every other section.

**Identical across all three frames** (same underlying `patient_nbr` column in B-1/B-2/B-3):

| | value |
|---|---|
| n_rows | 101,766 |
| n_distinct | 71,518 |
| distinct/rows ratio | 0.702769 (70.3%) |
| max per-value count | 40 |
| min per-value count | 1 |

**Per-value count distribution** (71,518 distinct patients):

| count == 1 | count == 2 | count == 3 | count 4–9 | count >= 10 |
|---|---|---|---|---|
| 54,745 | 10,434 | 3,328 | 2,872 | 139 |

**76.5% of distinct values are singletons** (54,745 / 71,518) — this is the real-world shape a floor of 2 has to handle: a large but not total majority of values still get pooled at floor 2, while the remaining ~16,773 values (count >= 2, covering roughly 47,000 rows) survive floor-2 pooling and continue contributing to `U`.

**U at raw and each candidate floor (identical across B-1, B-2, B-3):**

| | U |
|---|---|
| raw (unpooled) | 0.609177 |
| floor >= 2 | 0.353062 |
| floor >= 5 | 0.069571 |
| floor >= 10 | 0.018020 |
| floor >= 20 | 0.004592 |
| floor >= 50 | 0.000000 |
| floor >= 100 | 0.000000 |

**Does `U >= 0.99` at any floor, in any frame? No — not at raw, and not at any of the six candidate floors, in any of the three frames.**

### Does floor=2 survive this anchor?

**Yes, clearly.** At floor 2, `patient_nbr` scores `U = 0.353062` — a margin of **0.636938** below the 0.99 criterion. This is not a near-miss; it is not even close. The measurement does not contradict the floor=2 recommendation, so it is not superseded here — per the instruction, the floor was not picked to make this anchor come out a particular way, and the anchor came out clearly in favor of the existing recommendation without needing to move it.

**Worth recording regardless, since it bears directly on how close this real case actually runs**: `patient_nbr`'s **raw, unpooled** `U` (0.609177) is higher than any other feature this project has measured in Test B **except** `readmitted` and `encounter_id`, and lands close to `planted_leak`'s 0.641748. The two numbers being close is coincidental, not comparable in kind: `planted_leak` has only 2 distinct values, so its 0.641748 is its true, uninflated signal, with no cardinality artifact to worry about; `patient_nbr`'s 0.609177 is almost entirely an ID-shaped artifact of its high cardinality (70.3% distinct/rows), not real predictive signal. A raw `U` reading alone cannot tell these two cases apart — which is exactly the surface-level confusion the support floor exists to resolve. Floor 2 alone cuts `patient_nbr`'s apparent signal by 42% (0.609 → 0.353); by floor 10 it is already below the honest-tail ceiling (0.0316) recorded earlier in this document. The reason none of this ever becomes a real risk in practice is structural, not just numeric: `patient_nbr` is the declared `entity_id` and is excluded from every screen candidate set by construction, the same way `planted_leak` (forbidden) and the role columns are — this anchor was scored here only as an out-of-band stress test, exactly as `planted_leak` was in the Design Decisions section, not because (H) would ever actually evaluate it in normal operation.

**Floor=2 stands, confirmed by the strongest real near-ID anchor available in this dataset.**
