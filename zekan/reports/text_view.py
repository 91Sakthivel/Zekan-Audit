"""Plain-text formatter for VerdictReport.

Shared core renderer reused by CLI, notebook, and HTML skins.
Pure function: same report in -> same string out, no side effects.
"""

from __future__ import annotations

import math
from typing import Optional

from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.verdict import FoldCI, VerdictReport
from zekan.reports.markers import marker as _marker, sanitize as _sanitize
import zekan.reports.messages as _MSG


def render_verdict(report: VerdictReport, stream=None) -> str:
    """Render a VerdictReport as a plain-language string.

    stream: stream whose .encoding selects glyph vs ASCII markers.
            Defaults to sys.stdout when None.
    """
    verdict = report.policy_decision.verdict
    fl = report.measured_damage.fixable_leakage
    attribution = report.measured_damage.feature_attribution
    detection_channel = report.engine_detection.detection_channel
    panel = report.undeclared_feature_panel
    near_certain, near_bijection, annotations = _split_structural_prominent(
        report.structural_annotations
    )

    fold_ci = report.fold_ci
    fold_inert_columns = report.fold_inert_columns
    fold_feature_coverage = report.fold_feature_coverage

    if verdict in ("PASS", "NOTE"):
        text = _render_trusted(
            _marker("trusted", stream), annotations=annotations, fold_ci=fold_ci,
            detection_channel=detection_channel, near_certain=near_certain,
            near_bijection=near_bijection, panel=panel,
            fold_inert_columns=fold_inert_columns, fold_feature_coverage=fold_feature_coverage,
        )
    elif verdict == "WARN":
        text = _render_actionable(
            marker=_marker("risky", stream),
            translation=_MSG.TRANSLATION_RISKY,
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
            detection_channel=detection_channel,
            near_certain=near_certain,
            near_bijection=near_bijection,
            panel=panel,
            fold_inert_columns=fold_inert_columns,
            fold_feature_coverage=fold_feature_coverage,
        )
    elif verdict == "FAIL":
        text = _render_actionable(
            marker=_marker("failed", stream),
            translation=_MSG.TRANSLATION_FAILED,
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
            detection_channel=detection_channel,
            near_certain=near_certain,
            near_bijection=near_bijection,
            panel=panel,
            fold_inert_columns=fold_inert_columns,
            fold_feature_coverage=fold_feature_coverage,
        )
    elif verdict == "UNCONFIRMED_HIGH_DAMAGE":
        text = _render_inconclusive(
            _marker("inconclusive", stream),
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
            near_certain=near_certain,
            near_bijection=near_bijection,
            panel=panel,
            fold_inert_columns=fold_inert_columns,
            fold_feature_coverage=fold_feature_coverage,
        )
    else:
        text = _render_trusted(
            _marker("trusted", stream), annotations=annotations,
            near_certain=near_certain, near_bijection=near_bijection, panel=panel,
            fold_inert_columns=fold_inert_columns, fold_feature_coverage=fold_feature_coverage,
        )
    return _sanitize(text, stream)


# ── Render states ─────────────────────────────────────────────────────────────

_DIVIDER = "─" * 60


def _render_trusted(
    banner: str, annotations=None, fold_ci: Optional[FoldCI] = None,
    detection_channel: str = "", near_certain=None, near_bijection=None, panel=None,
    fold_inert_columns=None, fold_feature_coverage=None,
) -> str:
    lines = [banner, _MSG.TRANSLATION_TRUSTED, ""]
    if detection_channel in ("across_entity", "both"):
        lines.append(_MSG.ACROSS_ENTITY_DETECTED)
        lines.append("")
    _append_block(lines, _prominent_lines(near_certain, near_bijection))
    _append_block(lines, _fold_coverage_lines(fold_inert_columns, fold_feature_coverage))
    if fold_ci is not None and fold_ci.stability_seeds_checked > 0 and not fold_ci.seed_instability_note:
        lines.append(
            f"  Stability: verdict consistent across {fold_ci.stability_seeds_checked} null seeds."
        )
        lines.append("")
    if annotations:
        lines.append(_MSG.STRUCTURAL_FINDING_HEADING)
        lines.append(f"  {_MSG.STRUCTURAL_FINDING_TRUSTED_LEAD}")
        for ann in annotations:
            lines.append(f"  {ann.what}")
        lines.append("")
    _append_block(lines, _panel_lines(panel))
    _ensure_blank(lines)
    lines.append(_DIVIDER)
    lines.append(_footer_for(
        _MSG.FOOTER_TRUSTED, _MSG.FOOTER_TRUSTED_WITH_SCREEN, near_certain, near_bijection, panel
    ))
    return "\n".join(lines)


def _render_actionable(
    marker: str,
    translation: str,
    fl: float,
    attribution: Optional[AblationSummary],
    annotations=None,
    fold_ci: Optional[FoldCI] = None,
    detection_channel: str = "",
    near_certain=None,
    near_bijection=None,
    panel=None,
    fold_inert_columns=None,
    fold_feature_coverage=None,
) -> str:
    lines: list[str] = [marker, translation, ""]
    if detection_channel in ("across_entity", "both"):
        lines.append(_MSG.ACROSS_ENTITY_DETECTED)
        lines.append("")
    _append_block(lines, _prominent_lines(near_certain, near_bijection))
    _append_block(lines, _fold_coverage_lines(fold_inert_columns, fold_feature_coverage))

    lines.append("THE DAMAGE")
    lines.append("  Your reported accuracy is inflated — real performance will be lower.")
    lines.append(f"    inflation: {fl:+.4f} AUC (B−C fixable leakage)")
    if fold_ci is not None and fold_ci.folds_skipped > 0:
        lines.append(
            f"    note: {fold_ci.folds_skipped} fold(s) skipped "
            f"({fold_ci.folds_evaluated} of "
            f"{fold_ci.folds_evaluated + fold_ci.folds_skipped} evaluated)"
            + (f" — {fold_ci.skip_reasons[0]}" if fold_ci.skip_reasons else "")
            + "."
        )
    if fold_ci is not None and fold_ci.stability_seeds_checked > 0 and not fold_ci.seed_instability_note:
        lines.append(
            f"    Stability: verdict consistent across {fold_ci.stability_seeds_checked} null seeds."
        )

    ablated = _sorted_ablated(attribution)
    if ablated:
        lines.append("")
        lines.append("WHAT TO FIX FIRST")
        for i, entry in enumerate(ablated, 1):
            tag = "— biggest cause" if i == 1 else f"— #{i}"
            lines.append(f"  {i}. {entry.feature} {tag}")
            lines.append(f"       estimated inflation: {entry.leakage_estimate:+.4f} AUC")
        if attribution is not None and attribution.one_at_a_time_understates:
            lines.append(
                "  These features overlap — fixing one may not be enough; "
                "address them together."
            )
            for entry in ablated:
                ap = entry.apportioned_leakage
                if not math.isnan(ap):
                    lines.append(
                        f"    feature '{entry.feature}': alone shows "
                        f"{entry.leakage_estimate:+.4f}, but contributes {ap:+.4f} "
                        "once correlated features are removed (they were masking it)."
                    )

        top_feature = ablated[0].feature
        lines.append("")
        lines.append(_MSG.FEATURE_TRANSLATION.format(top_feature=top_feature))
        lines.append("")
        lines.append(_MSG.ACTION_RISKY_FAILED_TOP.format(top_feature=top_feature))
    else:
        lines.append("")
        lines.append(_MSG.ACTION_RISKY_FAILED_NO_ATTR)

    if annotations:
        lines.append("")
        lines.append(_MSG.STRUCTURAL_FINDING_HEADING)
        for ann in annotations:
            lines.append(f"  {ann.what}")

    _append_block(lines, _panel_lines(panel))

    _ensure_blank(lines)
    lines.append(_DIVIDER)
    lines.append(_footer_for(
        _MSG.FOOTER_RISKY_FAILED, _MSG.FOOTER_RISKY_FAILED_WITH_SCREEN,
        near_certain, near_bijection, panel,
    ))

    return "\n".join(lines)


def _render_inconclusive(
    banner: str,
    fl: float,
    attribution: Optional[AblationSummary],
    annotations: list | None = None,
    fold_ci: Optional[FoldCI] = None,
    near_certain=None,
    near_bijection=None,
    panel=None,
    fold_inert_columns=None,
    fold_feature_coverage=None,
) -> str:
    lines: list[str] = [banner, _MSG.TRANSLATION_INCONCLUSIVE, ""]

    if fold_ci is not None and fold_ci.seed_instability_note:
        lines.append(f"  Stability: {fold_ci.seed_instability_note}.")
        lines.append(
            "  Downgraded to INCONCLUSIVE — the audit verdict depends on the random seed."
        )
        lines.append("")
    elif fold_ci is not None and fold_ci.folds_skipped > 0:
        lines.append(
            f"  Fold coverage: {fold_ci.folds_skipped} fold(s) skipped "
            f"({fold_ci.folds_evaluated} of "
            f"{fold_ci.folds_evaluated + fold_ci.folds_skipped} evaluated)"
            + (f" — {fold_ci.skip_reasons[0]}" if fold_ci.skip_reasons else "")
            + "."
        )
        lines.append("")

    _append_block(lines, _prominent_lines(near_certain, near_bijection))
    _append_block(lines, _fold_coverage_lines(fold_inert_columns, fold_feature_coverage))

    lines.extend([
        "THE DAMAGE",
        (
            f"  Possible inflation: {fl:+.4f} AUC "
            "(unconfirmed — permutation null did not reach significance)"
        ),
    ])

    ablated = _sorted_ablated(attribution)
    if ablated:
        lines.append("")
        lines.append("WHAT TO FIX FIRST (unconfirmed — treat with caution)")
        for i, entry in enumerate(ablated, 1):
            lines.append(f"  {i}. {entry.feature} — largest contributor")
            lines.append(f"       estimated inflation: {entry.leakage_estimate:+.4f} AUC")
        if attribution is not None and attribution.one_at_a_time_understates:
            lines.append(
                "  These features overlap — fixing one may not be enough; "
                "address them together."
            )
            for entry in ablated:
                ap = entry.apportioned_leakage
                if not math.isnan(ap):
                    lines.append(
                        f"    feature '{entry.feature}': alone shows "
                        f"{entry.leakage_estimate:+.4f}, but contributes {ap:+.4f} "
                        "once correlated features are removed (they were masking it)."
                    )

        top_feature = ablated[0].feature
        lines.append("")
        lines.append(_MSG.FEATURE_TRANSLATION_UNCONFIRMED.format(feature=top_feature))

    if annotations:
        lines.append("")
        lines.append(_MSG.STRUCTURAL_FINDING_HEADING)
        for ann in annotations:
            lines.append(f"  {ann.what}")

    _append_block(lines, _panel_lines(panel))

    _ensure_blank(lines)
    lines.append(_MSG.ACTION_INCONCLUSIVE_OPENER)
    if annotations:
        lines.append(_MSG.ACTION_INCONCLUSIVE_STRUCTURAL)
    else:
        lines.append(_MSG.ACTION_INCONCLUSIVE_STATISTICAL)
    _ensure_blank(lines)
    lines.append(_DIVIDER)
    lines.append(_footer_for(
        _MSG.FOOTER_INCONCLUSIVE, _MSG.FOOTER_INCONCLUSIVE_WITH_SCREEN,
        near_certain, near_bijection, panel,
    ))

    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_structural_prominent(annotations: list) -> tuple[list, list, list]:
    """Pull NEAR_CERTAIN_UNDECLARED_LEAK (Upgrade 1) and
    NEAR_BIJECTION_UNDECLARED_LEAK (Upgrade H) findings out of the generic
    structural_annotations list for prominent rendering, beside the verdict
    -- everything else still renders exactly as before, in the trailing
    STRUCTURAL FINDING position. Returns (near_certain, near_bijection, rest).
    """
    if not annotations:
        return [], [], annotations
    near_certain = [a for a in annotations if _is_near_certain(a)]
    near_bijection = [a for a in annotations if _is_near_bijection(a)]
    if not near_certain and not near_bijection:
        return [], [], annotations
    rest = [
        a for a in annotations
        if not _is_near_certain(a) and not _is_near_bijection(a)
    ]
    return near_certain, near_bijection, rest


def _is_near_certain(ann) -> bool:
    issue_type = getattr(ann, "issue_type", None)
    value = getattr(issue_type, "value", issue_type)  # duck-typed-fake-friendly
    return value == "near_certain_undeclared_leak"


def _is_near_bijection(ann) -> bool:
    issue_type = getattr(ann, "issue_type", None)
    value = getattr(issue_type, "value", issue_type)  # duck-typed-fake-friendly
    return value == "near_bijection_undeclared_leak"


def _prominent_lines(near_certain: list | None, near_bijection: list | None) -> list[str]:
    """PROMINENT block content for NEAR_CERTAIN (Upgrade 1) and NEAR_BIJECTION
    (Upgrade H) findings, rendered near the top of every verdict state,
    including TRUSTED (see this module's render_verdict: inserted before THE
    DAMAGE/fix sections, right after the headline).

    NEAR_BIJECTION is confirmed=True -- a deterministic counting fact -- and
    is therefore the STRONGER claim whenever it and NEAR_CERTAIN
    (confirmed=False, a model inference) name the SAME feature: it renders
    first, as the primary statement, with a short corroboration line naming
    NEAR_CERTAIN's result instead of NEAR_CERTAIN restating its own block in
    full (no duplicated claim about one feature). NEAR_CERTAIN findings on a
    feature NOT also flagged by NEAR_BIJECTION still render standalone,
    unchanged, exactly as before.

    No leading/trailing blank padding -- callers use _append_block so
    exactly one blank line separates this from whatever precedes/follows it,
    never zero, never doubled (FIX 4 output hygiene, tests/test_cli.py::
    test_no_consecutive_blank_lines_with_gate)."""
    near_bijection = near_bijection or []
    near_certain = near_certain or []
    consumed_features = set()
    lines: list[str] = []

    for ann in near_bijection:
        detail = ann.evidence.structural_detail
        match = next(
            (c for c in near_certain
             if c.evidence.structural_detail.feature == detail.feature),
            None,
        )
        lines.append(_MSG.NEAR_BIJECTION_HEADING)
        lines.append(
            f"  {_MSG.NEAR_BIJECTION_LEAD.format(feature=detail.feature, theil_u=detail.theil_u, threshold_compared_against=detail.threshold_compared_against)}"
        )
        if match is not None:
            consumed_features.add(detail.feature)
            match_detail = match.evidence.structural_detail
            lines.append(
                f"  {_MSG.NEAR_BIJECTION_CORROBORATED_BY_NEAR_CERTAIN.format(feature=detail.feature, auc=match_detail.univariate_auc)}"
            )
        lines.append(f"  {_MSG.NEAR_BIJECTION_ACTION.format(feature=detail.feature)}")

    for ann in near_certain:
        detail = ann.evidence.structural_detail
        if detail.feature in consumed_features:
            continue
        lines.append(_MSG.NEAR_CERTAIN_HEADING)
        lines.append(
            f"  {_MSG.NEAR_CERTAIN_LEAD.format(feature=detail.feature, auc=detail.univariate_auc)}"
        )
        lines.append(f"  {_MSG.NEAR_CERTAIN_ACTION.format(feature=detail.feature)}")

    return lines


def _panel_lines(panel) -> list[str]:
    """Ranked informational panel content -- in the annotation position.
    Renders whenever the screen ran (panel is not None), independent of
    whether structural_annotations/near_certain are present, so "screened X
    of Y" is never silently absent. No leading/trailing blank padding -- see
    _prominent_lines' docstring; same _append_block convention."""
    if panel is None:
        return []
    lines = [_MSG.RANKED_PANEL_HEADING, f"  {_MSG.RANKED_PANEL_LEAD}"]
    for i, entry in enumerate(panel.entries, 1):
        lines.append(f"  {i}. {entry.feature} — univariate AUC {entry.univariate_auc:.4f}")
    lines.append(
        f"  {_MSG.RANKED_PANEL_SCREENED.format(screened=panel.screened_count, total=panel.total_features)}"
    )
    if panel.not_screenable:
        lines.append(
            f"  {_MSG.RANKED_PANEL_NOT_SCREENABLE.format(count=len(panel.not_screenable))}"
        )
    lines.append(f"  {_MSG.RANKED_PANEL_ACTION}")
    return lines


def _fold_coverage_lines(
    fold_inert_columns: Optional[dict] = None,
    fold_feature_coverage: Optional[dict] = None,
) -> list[str]:
    """FEATURE COVERAGE block -- FOLD_INERT_FEATURES_PREREGISTRATION.md
    section 8, amended by FOLD_INERT_ADDENDUM_01_ZK_EST_04_UNTESTABLE.md
    section 6: this is the only mechanism by which a user learns fold-local
    inerting occurred. Empty (no block) when fold_inert_columns is empty/None
    -- the common case, including Test B. Reports "active from fold N"
    per column (section 7 monotonicity makes that a complete summary, not a
    per-fold list) and states coverage is PARTIAL for those columns --
    never full. Deliberately says nothing about structural probes (Upgrade
    H/1, section 9): severity coverage and structural coverage are kept
    separate here, same as everywhere else in this renderer.

    No leading/trailing blank padding -- see _prominent_lines' docstring;
    same _append_block convention.
    """
    if not fold_inert_columns:
        return []
    lines = ["FEATURE COVERAGE"]

    # Keys are strings (JSON-friendly) -- sort numerically, not lexicographically,
    # so this stays correct once fold indices reach two digits (e.g. "10" vs "2").
    worst_fold = min(fold_feature_coverage, key=int) if fold_feature_coverage else None
    worst = fold_feature_coverage.get(worst_fold) if worst_fold is not None else None
    n_inert = len(fold_inert_columns)
    if worst is not None:
        lines.append(
            f"  {n_inert} of {worst['active'] + worst['inert']} declared feature(s) had "
            f"no training data in fold {worst_fold} ({worst['active']} active)."
        )
    else:
        lines.append(f"  {n_inert} declared feature(s) had no training data in an early fold.")

    items = sorted(
        fold_inert_columns.items(),
        key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else -1, kv[0]),
    )
    shown, rest = items[:5], items[5:]
    for name, first_active in shown:
        if first_active is None:
            lines.append(f"    {name!r} — never accumulated training data in this audit")
        else:
            lines.append(f"    {name!r} — active from fold {first_active}")
    if rest:
        lines.append(f"    ...and {len(rest)} more")

    lines.append(
        "  Severity for these feature(s) is PARTIAL — measured only from the fold "
        "each became active, not the full fold set other features were measured on."
    )
    return lines


def _ensure_blank(lines: list[str]) -> None:
    """Append a single blank line, unless `lines` is empty or already ends
    with one. Idempotent separator -- never produces a double blank."""
    if lines and lines[-1] != "":
        lines.append("")


def _append_block(lines: list[str], block: list[str]) -> None:
    """Append `block` (from _prominent_lines/_panel_lines) with exactly
    one blank line of separation before it and one trailing blank after --
    regardless of what already precedes it. No-op when `block` is empty (no
    padding introduced for a block with nothing to say)."""
    if not block:
        return
    _ensure_blank(lines)
    lines.extend(block)
    lines.append("")


def _footer_for(base: str, with_screen: str, near_certain, near_bijection, panel) -> str:
    """Select the screen-honest footer variant whenever a NEAR_CERTAIN or
    NEAR_BIJECTION finding, or the ranked panel, is present -- the base
    footer's unqualified "does not inspect undeclared features" claim would
    otherwise be false."""
    if near_certain or near_bijection or panel is not None:
        return with_screen
    return base


def _sorted_ablated(attribution: Optional[AblationSummary]) -> list[AblationEntry]:
    """Ablated entries sorted by leakage_estimate descending; NaN entries excluded."""
    if attribution is None:
        return []
    valid = [
        e for e in attribution.individual
        if e.ablated and not math.isnan(e.leakage_estimate)
    ]
    return sorted(valid, key=lambda e: e.leakage_estimate, reverse=True)
