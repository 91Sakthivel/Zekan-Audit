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
    annotations = report.structural_annotations

    fold_ci = report.fold_ci

    if verdict in ("PASS", "NOTE"):
        text = _render_trusted(_marker("trusted", stream), annotations=annotations, fold_ci=fold_ci)
    elif verdict == "WARN":
        text = _render_actionable(
            marker=_marker("risky", stream),
            translation=_MSG.TRANSLATION_RISKY,
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
        )
    elif verdict == "FAIL":
        text = _render_actionable(
            marker=_marker("failed", stream),
            translation=_MSG.TRANSLATION_FAILED,
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
        )
    elif verdict == "UNCONFIRMED_HIGH_DAMAGE":
        text = _render_inconclusive(
            _marker("inconclusive", stream),
            fl=fl,
            attribution=attribution,
            annotations=annotations,
            fold_ci=fold_ci,
        )
    else:
        text = _render_trusted(_marker("trusted", stream), annotations=annotations)
    return _sanitize(text, stream)


# ── Render states ─────────────────────────────────────────────────────────────

_DIVIDER = "─" * 60


def _render_trusted(banner: str, annotations=None, fold_ci: Optional[FoldCI] = None) -> str:
    lines = [banner, _MSG.TRANSLATION_TRUSTED, ""]
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
    lines.extend([_DIVIDER, _MSG.FOOTER_TRUSTED])
    return "\n".join(lines)


def _render_actionable(
    marker: str,
    translation: str,
    fl: float,
    attribution: Optional[AblationSummary],
    annotations=None,
    fold_ci: Optional[FoldCI] = None,
) -> str:
    lines: list[str] = [marker, translation, ""]

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

    lines.append("")
    lines.append(_DIVIDER)
    lines.append(_MSG.FOOTER_RISKY_FAILED)

    return "\n".join(lines)


def _render_inconclusive(
    banner: str,
    fl: float,
    attribution: Optional[AblationSummary],
    annotations: list | None = None,
    fold_ci: Optional[FoldCI] = None,
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

    lines.append("")
    lines.append(_MSG.ACTION_INCONCLUSIVE_OPENER)
    if annotations:
        lines.append(_MSG.ACTION_INCONCLUSIVE_STRUCTURAL)
    else:
        lines.append(_MSG.ACTION_INCONCLUSIVE_STATISTICAL)
    lines.append("")
    lines.append(_DIVIDER)
    lines.append(_MSG.FOOTER_INCONCLUSIVE)

    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sorted_ablated(attribution: Optional[AblationSummary]) -> list[AblationEntry]:
    """Ablated entries sorted by leakage_estimate descending; NaN entries excluded."""
    if attribution is None:
        return []
    valid = [
        e for e in attribution.individual
        if e.ablated and not math.isnan(e.leakage_estimate)
    ]
    return sorted(valid, key=lambda e: e.leakage_estimate, reverse=True)
