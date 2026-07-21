"""Shared message constants for text_view and html_view.

Single source of truth for all translation lines, action lines, per-feature
explanations, and scope footers. Both formatters import from here so the
wording cannot diverge between the two skins.
"""

from __future__ import annotations

# ── Per-verdict translation lines (what to BELIEVE) ──────────────────────────

TRANSLATION_TRUSTED = (
    "Zekan found no evidence of leakage within the declared audit scope."
)

TRANSLATION_RISKY = (
    "Your model will likely perform noticeably worse in the real world "
    "than these test results suggest."
)

TRANSLATION_FAILED = (
    "These test results cannot be trusted as a guide to real-world "
    "performance — the measured leakage is large."
)

TRANSLATION_INCONCLUSIVE = (
    "Zekan could not certify this result. The measured signal is too "
    "uncertain or unavailable under the current audit setup."
)

# ── Action lines (what to DO) ─────────────────────────────────────────────────

# Call .format(top_feature=...) before using.
ACTION_RISKY_FAILED_TOP = (
    "Start by removing or rebuilding '{top_feature}', then address the other "
    "flagged features above. Re-run Zekan and inspect how each was created "
    "before trusting this model."
)

ACTION_RISKY_FAILED_NO_ATTR = (
    "Investigate the flagged leakage before trusting this model."
)

ACTION_INCONCLUSIVE_OPENER = "Do not treat this as trusted."

# Cause-specific diagnosis lines — branch on bool(annotations) at render time.
# Rendered as: ACTION_INCONCLUSIVE_OPENER + one of the two lines below.
ACTION_INCONCLUSIVE_STRUCTURAL = (
    "Adding more data will not resolve this: the permutation test is blind "
    "to a feature that is constant within each entity. Inspect how this "
    "feature was created."
)

ACTION_INCONCLUSIVE_STATISTICAL = (
    "A large effect was measured but could not be statistically confirmed. "
    "This usually means too few time periods, or a signal the permutation "
    "test could not separate from noise. Add more time periods, or inspect "
    "the flagged feature before trusting this result."
)

# ── Per-worst-feature plain-English explanation ───────────────────────────────

# Call .format(top_feature=...) before using. For confirmed states (RISKY/FAILED).
FEATURE_TRANSLATION = (
    "In plain terms: '{top_feature}' appears to already contain information "
    "about the answer — that is why the results look better than they really are."
)

# Call .format(feature=...) before using. For INCONCLUSIVE (permutation null not reached).
FEATURE_TRANSLATION_UNCONFIRMED = (
    "In plain terms: '{feature}' shows a pattern that may mean it already "
    "contains information about the answer — but this could not be "
    "statistically confirmed."
)

# ── Scope boundary footers (what Zekan did NOT prove) ────────────────────────

FOOTER_TRUSTED = (
    "Scope note: Zekan v1 checks the dataset and declared contract. It does not "
    "inspect undeclared features, notebooks, user split code, or external "
    "train/test artifacts. TRUSTED means no issue was found within this scope "
    "— not that the whole pipeline is guaranteed clean."
)

FOOTER_RISKY_FAILED = (
    "Scope note: This audit is based on the dataset and declared contract. "
    "Additional issues may exist in undeclared features or user code."
)

FOOTER_INCONCLUSIVE = (
    "Scope note: INCONCLUSIVE is not a pass. Zekan v1 does not inspect "
    "undeclared features, notebooks, user split code, or external "
    "train/test artifacts."
)

# ── Across-entity detection (spec 1) ──────────────────────────────────────────
# Shown only when the across-entity permutation null fired (detection_channel
# is "across_entity" or "both") — names the different leak class it catches.

ACROSS_ENTITY_DETECTED = (
    "Leakage detected via across-entity structure — a forbidden feature appears "
    "to carry entity-level information that a within-entity check cannot see."
)

# ── Structural finding constants ──────────────────────────────────────────────
# Shared by all four verdict states and both render surfaces (drift-lock).

STRUCTURAL_FINDING_HEADING = "STRUCTURAL FINDING"

# Lead-in for TRUSTED verdicts: frames the block as additional, not contradictory.
STRUCTURAL_FINDING_TRUSTED_LEAD = "Zekan also noticed:"

# ── Undeclared-feature screen: NEAR_CERTAIN (Upgrade 1 step 1f) ──────────────
# Rendered PROMINENTLY, immediately after the verdict/headline block, BEFORE
# the damage/fix sections -- in ALL verdict states, including TRUSTED/PASS.
# Motivating case (TEST_B_RESULTS.md, B-3): a column with AUC~1.0 must never
# again sit silently beside a confident "no evidence of leakage" headline.
# Call .format(feature=..., auc=...) on NEAR_CERTAIN_LEAD and .format(feature=...)
# on NEAR_CERTAIN_ACTION before using.

NEAR_CERTAIN_HEADING = "NEAR-CERTAIN UNDECLARED LEAK"

NEAR_CERTAIN_LEAD = (
    "'{feature}' predicts the answer almost perfectly (AUC {auc:.4f} on temporal "
    "folds) and was NOT declared forbidden. The verdict above does not account for it."
)

NEAR_CERTAIN_ACTION = (
    "Add '{feature}' to forbidden_after_prediction and re-run to measure its true impact."
)

# ── Structural check: NEAR_BIJECTION (Upgrade H) ─────────────────────────────
# Rendered in the SAME prominent slot as NEAR_CERTAIN above, at least equal
# prominence -- confirmed=True (a deterministic counting fact) is the
# STRONGER claim than NEAR_CERTAIN's confirmed=False model inference. When
# both fire on the same feature, NEAR_BIJECTION renders as the primary
# statement and NEAR_BIJECTION_CORROBORATED_BY_NEAR_CERTAIN below adds a
# short corroboration line instead of a second full restatement (no
# duplicated claim -- see text_view.py/html_view.py's merge logic).
# Call .format(feature=..., theil_u=..., threshold_compared_against=...) on
# NEAR_BIJECTION_LEAD, .format(feature=...) on NEAR_BIJECTION_ACTION, and
# .format(feature=..., auc=...) on NEAR_BIJECTION_CORROBORATED_BY_NEAR_CERTAIN.

NEAR_BIJECTION_HEADING = "NEAR-BIJECTION UNDECLARED LEAK"

NEAR_BIJECTION_LEAD = (
    "The values of '{feature}' determine the target almost exactly (Theil's U "
    "{theil_u:.4f}, flagged at >= {threshold_compared_against:.2f}, after "
    "pooling rare values) and it was NOT declared forbidden. The verdict "
    "above does not account for it."
)

NEAR_BIJECTION_CORROBORATED_BY_NEAR_CERTAIN = (
    "Zekan's separate model-based screen flagged '{feature}' too (univariate "
    "AUC {auc:.4f}) -- two independent checks agree."
)

NEAR_BIJECTION_ACTION = (
    "Add '{feature}' to forbidden_after_prediction and re-run to measure its true impact."
)

# ── Undeclared-feature screen: ranked informational panel ────────────────────
# Rendered in the same trailing position as STRUCTURAL FINDING. No threshold,
# no suspicion claim -- this is a candidate list, never a flag. Wording here
# must never use "suspicious"/"flagged"/threshold language (parity-tested).
# Call .format(screened=..., total=...) on RANKED_PANEL_SCREENED and
# .format(count=...) on RANKED_PANEL_NOT_SCREENABLE before using.

RANKED_PANEL_HEADING = "FEATURES RANKED BY ASSOCIATION WITH THE TARGET"

RANKED_PANEL_LEAD = (
    "Zekan also measured how strongly each non-forbidden feature predicts the "
    "target on its own (temporal folds). This is a candidate list, not a finding."
)

RANKED_PANEL_SCREENED = "Screened {screened} of {total} non-forbidden features."

RANKED_PANEL_NOT_SCREENABLE = (
    "{count} feature(s) could not be screened (insufficient usable data) and "
    "are not included above."
)

RANKED_PANEL_ACTION = (
    "To measure any of these, add it to forbidden_after_prediction and re-run."
)

# ── Scope footers, revised for when the undeclared-feature screen ran ────────
# Selected instead of the base FOOTER_* constant above whenever a NEAR_CERTAIN
# finding or the ranked panel is present -- the base footers' unqualified "does
# not inspect undeclared features" claim would otherwise be false the moment
# this screen has run. Every phrase the footer-parity tests key on
# ("not that the whole pipeline is guaranteed clean", "Additional issues may
# exist", "not a pass") is preserved verbatim.

FOOTER_TRUSTED_WITH_SCREEN = (
    "Scope note: Zekan v1 checks the dataset and declared contract, and also "
    "screens non-forbidden features for a univariate signal against the target "
    "(see above) — that screen does not catch combined or multi-feature leakage. "
    "It does not inspect notebooks, user split code, or external train/test "
    "artifacts. TRUSTED means no issue was found within this scope — not that "
    "the whole pipeline is guaranteed clean."
)

FOOTER_RISKY_FAILED_WITH_SCREEN = (
    "Scope note: This audit is based on the dataset and declared contract, and "
    "also includes a univariate screen of non-forbidden features (see above) — "
    "that screen does not catch combined or multi-feature leakage. Additional "
    "issues may exist beyond this screen, or in user code."
)

FOOTER_INCONCLUSIVE_WITH_SCREEN = (
    "Scope note: INCONCLUSIVE is not a pass. Zekan v1 also screens non-forbidden "
    "features for a univariate signal (see above) — that screen does not catch "
    "combined or multi-feature leakage. It does not inspect notebooks, user "
    "split code, or external train/test artifacts."
)
