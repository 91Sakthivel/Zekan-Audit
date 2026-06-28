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

# ── Structural finding constants ──────────────────────────────────────────────
# Shared by all four verdict states and both render surfaces (drift-lock).

STRUCTURAL_FINDING_HEADING = "STRUCTURAL FINDING"

# Lead-in for TRUSTED verdicts: frames the block as additional, not contradictory.
STRUCTURAL_FINDING_TRUSTED_LEAD = "Zekan also noticed:"
