"""HTML formatter for VerdictReport — Jupyter _repr_html_ backend.

Returns a self-contained <div> with inline styles only.  No external CSS,
no JS, no CDN resources — renders identically offline and when emailed.

State detection mirrors text_view.py exactly:
  verdict in ("PASS", "NOTE")        -> TRUSTED
  verdict == "WARN"                  -> RISKY
  verdict == "FAIL"                  -> FAILED
  verdict == "UNCONFIRMED_HIGH_DAMAGE" -> INCONCLUSIVE
  fallback                           -> TRUSTED

Every user-derived string (feature names) is passed through html.escape()
before interpolation. Numbers are formatted via Python and are safe as-is.
"""

from __future__ import annotations

import html
import math
from typing import Optional

from zekan.severity.ablation import AblationEntry, AblationSummary
from zekan.severity.verdict import VerdictReport
import zekan.reports.messages as _MSG


# ── Inline style constants ────────────────────────────────────────────────────

_OUTER = (
    "font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;"
    "max-width:680px;margin:8px 0;color:#1a1a1a;line-height:1.5;"
)
_BODY = (
    "padding:14px 18px;border:1px solid #e5e7eb;border-top:none;"
    "border-radius:0 0 6px 6px;"
)

_BANNER_BASE = (
    "padding:12px 18px;display:flex;align-items:center;gap:10px;"
    "border-radius:6px 6px 0 0;"
)
_GREEN = _BANNER_BASE + "background:#dcfce7;border-left:4px solid #16a34a;"
_AMBER = _BANNER_BASE + "background:#fef9c3;border-left:4px solid #ca8a04;"
_RED   = _BANNER_BASE + "background:#fee2e2;border-left:4px solid #dc2626;"
_GREY  = _BANNER_BASE + "background:#f3f4f6;border-left:4px solid #6b7280;"

_S_ICON    = "font-size:1.25em;line-height:1;flex-shrink:0;"
_S_WORD    = "font-weight:700;font-size:1.05em;letter-spacing:0.03em;"
_S_LEAD    = "margin:0 0 6px;font-size:0.95em;"
_S_MUTED   = (
    "font-size:0.82em;color:#6b7280;margin:2px 0 12px;"
    "padding-left:14px;"
)
_S_SECTION = (
    "font-weight:600;font-size:0.72em;letter-spacing:0.07em;"
    "text-transform:uppercase;color:#6b7280;margin:16px 0 6px 0;"
)
_S_OL      = "margin:0;padding-left:20px;"
_S_LI      = "margin:5px 0;font-size:0.92em;"
_S_FEAT    = "font-weight:600;"
_S_TAG     = "color:#374151;"
_S_AMT     = "color:#6b7280;font-size:0.85em;display:block;margin-left:2px;"
_S_NOTE    = "font-size:0.85em;color:#6b7280;margin-top:8px;font-style:italic;"
_S_TRANSLATION = "margin:0 0 10px;font-size:1.0em;font-weight:500;"
_S_ACTION  = "margin:12px 0 6px;font-size:0.9em;font-weight:500;"
_S_FEAT_EXPLAIN = "font-size:0.88em;color:#374151;margin:8px 0;font-style:italic;"
_S_FOOTER  = (
    "font-size:0.78em;color:#6b7280;margin-top:16px;padding-top:10px;"
    "border-top:1px solid #e5e7eb;"
)


# ── Public API ────────────────────────────────────────────────────────────────

def render_verdict_html(report: VerdictReport) -> str:
    """Render a VerdictReport as a self-contained HTML fragment."""
    verdict = report.policy_decision.verdict
    fl = report.measured_damage.fixable_leakage
    attribution = report.measured_damage.feature_attribution

    if verdict in ("PASS", "NOTE"):
        return _html_trusted()
    if verdict == "WARN":
        return _html_actionable(
            banner_style=_AMBER,
            icon="⚠",
            word="RISKY",
            translation=_MSG.TRANSLATION_RISKY,
            fl=fl,
            attribution=attribution,
        )
    if verdict == "FAIL":
        return _html_actionable(
            banner_style=_RED,
            icon="✗",
            word="FAILED",
            translation=_MSG.TRANSLATION_FAILED,
            fl=fl,
            attribution=attribution,
        )
    if verdict == "UNCONFIRMED_HIGH_DAMAGE":
        return _html_inconclusive(
            fl=fl,
            attribution=attribution,
            annotations=report.structural_annotations,
        )
    return _html_trusted()


# ── Render states ─────────────────────────────────────────────────────────────

def _html_trusted() -> str:
    banner = _banner(_GREEN, "✓", "TRUSTED")
    body = _wrap_body(
        _p(_MSG.TRANSLATION_TRUSTED, _S_LEAD)
        + _footer_block(_MSG.FOOTER_TRUSTED)
    )
    return _wrap_outer(banner + body)


def _html_actionable(
    banner_style: str,
    icon: str,
    word: str,
    translation: str,
    fl: float,
    attribution: Optional[AblationSummary],
) -> str:
    banner = _banner(banner_style, icon, word)

    parts: list[str] = []
    parts.append(_p(translation, _S_TRANSLATION))
    parts.append(_p("THE DAMAGE", _S_SECTION))
    parts.append(_p(
        "Your reported accuracy is inflated — real performance will be lower.",
        _S_LEAD,
    ))
    parts.append(_p(f"inflation: {fl:+.4f} AUC (B−C fixable leakage)", _S_MUTED))

    ablated = _sorted_ablated(attribution)
    if ablated:
        parts.append(_p("WHAT TO FIX FIRST", _S_SECTION))
        parts.append(_fix_list(ablated, attribution, confirmed=True))

        top_feature = html.escape(ablated[0].feature)
        parts.append(_p(
            _MSG.FEATURE_TRANSLATION.format(top_feature=top_feature),
            _S_FEAT_EXPLAIN,
        ))
        parts.append(_p(
            _MSG.ACTION_RISKY_FAILED_TOP.format(top_feature=top_feature),
            _S_ACTION,
        ))
    else:
        parts.append(_p(_MSG.ACTION_RISKY_FAILED_NO_ATTR, _S_ACTION))

    parts.append(_footer_block(_MSG.FOOTER_RISKY_FAILED))

    return _wrap_outer(banner + _wrap_body("".join(parts)))


def _html_inconclusive(
    fl: float,
    attribution: Optional[AblationSummary],
    annotations: list | None = None,
) -> str:
    banner = _banner(_GREY, "⚠", "INCONCLUSIVE")

    parts: list[str] = []
    parts.append(_p(_MSG.TRANSLATION_INCONCLUSIVE, _S_TRANSLATION))
    parts.append(_p("THE DAMAGE", _S_SECTION))
    parts.append(_p(
        f"Possible inflation: {fl:+.4f} AUC "
        "(unconfirmed — permutation null did not reach significance)",
        _S_MUTED,
    ))

    ablated = _sorted_ablated(attribution)
    if ablated:
        parts.append(_p(
            "WHAT TO FIX FIRST (unconfirmed — treat with caution)",
            _S_SECTION,
        ))
        parts.append(_fix_list(ablated, attribution, confirmed=False))

        top_feature = html.escape(ablated[0].feature)
        parts.append(_p(
            _MSG.FEATURE_TRANSLATION_UNCONFIRMED.format(feature=top_feature),
            _S_FEAT_EXPLAIN,
        ))

    if annotations:
        parts.append(_p("STRUCTURAL FINDING", _S_SECTION))
        for ann in annotations:
            parts.append(_p(html.escape(ann.what), _S_LEAD))

    parts.append(_p(_MSG.ACTION_INCONCLUSIVE, _S_ACTION))
    parts.append(_footer_block(_MSG.FOOTER_INCONCLUSIVE))

    return _wrap_outer(banner + _wrap_body("".join(parts)))


# ── HTML building blocks ──────────────────────────────────────────────────────

def _wrap_outer(content: str) -> str:
    return f'<div style="{_OUTER}">{content}</div>'


def _wrap_body(content: str) -> str:
    return f'<div style="{_BODY}">{content}</div>'


def _banner(style: str, icon: str, word: str) -> str:
    return (
        f'<div style="{style}">'
        f'<span style="{_S_ICON}">{icon}</span>'
        f'<span style="{_S_WORD}">{word}</span>'
        f'</div>'
    )


def _p(text: str, style: str) -> str:
    return f'<p style="{style}">{text}</p>'


def _fix_list(
    ablated: list[AblationEntry],
    attribution: Optional[AblationSummary],
    confirmed: bool,
) -> str:
    items: list[str] = []
    for i, entry in enumerate(ablated, 1):
        feat = html.escape(entry.feature)
        tag = "biggest cause" if i == 1 else f"#{i}"
        amt = f"{entry.leakage_estimate:+.4f} AUC"
        items.append(
            f'<li style="{_S_LI}">'
            f'<span style="{_S_FEAT}">{feat}</span>'
            f'<span style="{_S_TAG}"> — {tag}</span>'
            f'<span style="{_S_AMT}">estimated inflation: {amt}</span>'
            f'</li>'
        )
    ol = f'<ol style="{_S_OL}">{"".join(items)}</ol>'

    note = ""
    if attribution is not None and attribution.one_at_a_time_understates:
        note = (
            f'<p style="{_S_NOTE}">'
            "These features overlap — fixing one may not be enough; "
            "address them together."
            "</p>"
        )

    return ol + note


def _footer_block(text: str) -> str:
    return f'<p style="{_S_FOOTER}">{text}</p>'


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
