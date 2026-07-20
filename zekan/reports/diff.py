"""Pure diff of two audit JSON artifacts (schema version "1").

No I/O — callers load the dicts; this module only computes.
"""

from __future__ import annotations

from typing import Any


def _null_scheme(d: dict) -> str:
    """Extract the permutation-null seeding scheme from a provenance block.

    JSON produced before F2a carries no null_scheme at all (provenance may even
    be absent entirely, e.g. artifacts from commands that don't attach it) —
    that absence is treated as the retired "serial_v1" scheme, never guessed
    to match the other side.
    """
    seed_block = (d.get("provenance") or {}).get("seed") or {}
    return seed_block.get("null_scheme") or "serial_v1"


def _null_stopping(d: dict) -> str:
    """Extract the permutation-null stopping scheme from a provenance block.

    JSON produced before Tier 2 carries no null_stopping at all — that absence
    is treated as the original "fixed_v1" scheme, never guessed to match the
    other side. Mirrors _null_scheme's exact pattern.
    """
    seed_block = (d.get("provenance") or {}).get("seed") or {}
    return seed_block.get("null_stopping") or "fixed_v1"


def _undeclared_screen(d: dict) -> str:
    """Extract the undeclared-feature screen version from a provenance block.

    JSON produced before Upgrade 1 (step 1e) carries no undeclared_screen at
    all -- that absence is treated as "none" (no screen ran), never guessed
    to match the other side. Mirrors _null_scheme/_null_stopping's exact
    pattern. Top-level provenance key (not nested under "seed" -- see
    build_provenance's own docstring for why).
    """
    provenance = d.get("provenance") or {}
    return provenance.get("undeclared_screen") or "none"


def _annotation_identity(ann: dict) -> tuple[Any, Any]:
    """(issue_type, feature) identity for one structural_annotations entry.

    feature comes from evidence.structural_detail.feature when the detail
    struct carries one (e.g. NEAR_CERTAIN_UNDECLARED_LEAK) -- None otherwise.
    Distinguishing by feature (not issue_type alone) matters specifically
    for NEAR_CERTAIN: the screen flags EVERY feature clearing the absolute
    criterion, so two artifacts can each carry a NEAR_CERTAIN_UNDECLARED_LEAK
    annotation for a DIFFERENT feature -- collapsing on issue_type alone
    would wrongly read that as "unchanged."
    """
    issue_type = ann.get("issue_type")
    detail = (ann.get("evidence") or {}).get("structural_detail") or {}
    feature = detail.get("feature")
    return (issue_type, feature)


def _annotation_ids(d: dict) -> set[tuple[Any, Any]]:
    return {_annotation_identity(a) for a in (d.get("structural_annotations") or [])}


def _format_annotation_id(identity: tuple[Any, Any]) -> str:
    issue_type, feature = identity
    return f"{issue_type} ({feature})" if feature is not None else str(issue_type)


def _estimator_identity(d: dict) -> str:
    """Extract the estimator identity from a provenance block.

    estimator_identity has always been a required field on build_provenance
    (unlike the additive null_scheme/null_stopping), so there is no historical
    "field didn't exist yet" case to guess at -- only a missing provenance
    block entirely (e.g. artifacts from commands that don't attach one) falls
    back to "unknown".
    """
    provenance = d.get("provenance") or {}
    return provenance.get("estimator_identity") or "unknown"


def diff_reports(old: dict, new: dict) -> dict:
    """Compare two verdict_to_dict outputs and return a structured diff.

    fixable_leakage is already null-safe (NaN → None via json_export._coerce).
    When either side is None, fl_delta is None and direction is
    UNVERIFIABLE_CHANGE — no delta is fabricated.  fixable_leakage does not
    depend on the permutation-null stream, so this comparison is unaffected by
    a null-scheme change on either side.

    schema_version mismatch is flagged in "schema_mismatch" but comparison
    continues on whatever fields are present.

    Null-derived scalars (p_value, nsl, null_iqr, ...) are NOT compared by this
    function today — nothing here reads them.  When the two inputs' provenance
    carries a different null_scheme (F2a spawn_v2 vs the retired serial_v1, or
    either side missing null_scheme entirely), "null_scheme_notice" is set so
    callers can surface that null statistics are not comparable across schemes
    even if a future field starts comparing them.  Same pattern for
    null_stopping (Tier 2 fixed_v1 vs sequential_v1): "null_stopping_notice" is
    set when they differ, since the two schemes draw a different number of
    permutations and are not directly comparable either.

    estimator_identity is different from the two notices above: a different
    estimator changes what fixable_leakage itself means (it's not just a
    null-derived statistic), so when estimator_identity differs, this function
    goes further than a notice -- fl_delta is forced to None and direction to
    "UNVERIFIABLE_CHANGE", refusing the comparison rather than merely flagging it.
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

    scheme_old = _null_scheme(old)
    scheme_new = _null_scheme(new)
    if scheme_old != scheme_new:
        result["null_scheme_notice"] = (
            f"null scheme differs ({scheme_old} vs {scheme_new}): null statistics "
            "are not comparable across schemes; fixable_leakage comparison is unaffected."
        )

    stopping_old = _null_stopping(old)
    stopping_new = _null_stopping(new)
    if stopping_old != stopping_new:
        result["null_stopping_notice"] = (
            f"null stopping scheme differs ({stopping_old} vs {stopping_new}): "
            "the number of permutations drawn may differ; null statistics are not "
            "directly comparable across stopping schemes; fixable_leakage comparison "
            "is unaffected."
        )

    estimator_old = _estimator_identity(old)
    estimator_new = _estimator_identity(new)
    if estimator_old != estimator_new:
        result["estimator_identity_notice"] = (
            f"estimator differs ({estimator_old} vs {estimator_new}): cross-estimator "
            "leakage numbers are not comparable -- fl_delta/direction are refused, not computed."
        )
        result["fl_delta"] = None
        result["direction"] = "UNVERIFIABLE_CHANGE"

    screen_old = _undeclared_screen(old)
    screen_new = _undeclared_screen(new)
    if screen_old != screen_new:
        result["undeclared_screen_notice"] = (
            f"undeclared-feature screen version differs ({screen_old} vs {screen_new}): "
            "screen scores/panels are not directly comparable across versions; "
            "fixable_leakage comparison is unaffected."
        )

    old_ann_ids = _annotation_ids(old)
    new_ann_ids = _annotation_ids(new)
    appeared = sorted(_format_annotation_id(i) for i in (new_ann_ids - old_ann_ids))
    resolved = sorted(_format_annotation_id(i) for i in (old_ann_ids - new_ann_ids))
    if appeared:
        result["new_annotations"] = appeared
    if resolved:
        result["resolved_annotations"] = resolved

    return result
