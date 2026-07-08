"""Pure diff of two audit JSON artifacts (schema version "1").

No I/O — callers load the dicts; this module only computes.
"""

from __future__ import annotations

from typing import Any


def diff_reports(old: dict, new: dict) -> dict:
    """Compare two verdict_to_dict outputs and return a structured diff.

    fixable_leakage is already null-safe (NaN → None via json_export._coerce).
    When either side is None, fl_delta is None and direction is
    UNVERIFIABLE_CHANGE — no delta is fabricated.

    schema_version mismatch is flagged in "schema_mismatch" but comparison
    continues on whatever fields are present.
    """
    schema_old = old.get("schema_version")
    schema_new = new.get("schema_version")

    summary_old: dict = old.get("summary") or {}
    summary_new: dict = new.get("summary") or {}

    fl_old: Any = summary_old.get("fixable_leakage")
    fl_new: Any = summary_new.get("fixable_leakage")

    if fl_old is None or fl_new is None:
        fl_delta: Any = None
        direction = "UNVERIFIABLE_CHANGE"
    else:
        # round to 10 dp to suppress floating-point noise in exact comparisons
        fl_delta = round(fl_new - fl_old, 10)
        if fl_delta < 0:
            direction = "IMPROVED"
        elif fl_delta > 0:
            direction = "REGRESSED"
        else:
            direction = "UNCHANGED"

    verdict_old = summary_old.get("verdict")
    verdict_new = summary_new.get("verdict")
    headline_old = summary_old.get("headline")
    headline_new = summary_new.get("headline")

    top_feature_old = summary_old.get("top_feature")
    top_feature_new = summary_new.get("top_feature")

    if top_feature_old is None and top_feature_new is None:
        top_feature_status = "none"
    elif top_feature_old is not None and top_feature_new is None:
        top_feature_status = "fixed"
    elif top_feature_old is None and top_feature_new is not None:
        top_feature_status = "appeared"
    elif top_feature_old == top_feature_new:
        top_feature_status = "same"
    else:
        top_feature_status = "changed"

    result: dict = {
        "direction": direction,
        "fl_delta": fl_delta,
        "fl_new": fl_new,
        "fl_old": fl_old,
        "headline_new": headline_new,
        "headline_old": headline_old,
        "top_feature_new": top_feature_new,
        "top_feature_old": top_feature_old,
        "top_feature_status": top_feature_status,
        "verdict_changed": verdict_old != verdict_new,
        "verdict_new": verdict_new,
        "verdict_old": verdict_old,
    }

    if schema_old != schema_new:
        result["schema_mismatch"] = f"old={schema_old!r} new={schema_new!r}"

    return result
