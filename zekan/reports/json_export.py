"""Exports audit results as structured JSON for downstream tooling."""

from __future__ import annotations

import json
import math
from typing import Any

from zekan.severity.verdict import VerdictReport


def _coerce(obj: Any) -> Any:
    """Walk a nested dict/list and coerce non-JSON-serializable scalars.

    NaN/Inf floats → None.  numpy scalar types → Python native, then re-checked.
    All other values pass through unchanged.
    """
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            obj = obj.item()
    except ImportError:
        pass

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    return obj


def _summary_top_feature(report: VerdictReport) -> str | None:
    """Top feature by leakage_estimate descending — reuses text_view's sort."""
    from zekan.reports.text_view import _sorted_ablated
    ablated = _sorted_ablated(report.measured_damage.feature_attribution)
    return ablated[0].feature if ablated else None


def _summary_headline(report: VerdictReport) -> str:
    """Verdict headline — reuses message constants from text_view."""
    import zekan.reports.messages as _MSG
    verdict = report.policy_decision.verdict
    if verdict in ("PASS", "NOTE"):
        return _MSG.TRANSLATION_TRUSTED
    if verdict == "WARN":
        return _MSG.TRANSLATION_RISKY
    if verdict == "FAIL":
        return _MSG.TRANSLATION_FAILED
    if verdict == "UNCONFIRMED_HIGH_DAMAGE":
        return _MSG.TRANSLATION_INCONCLUSIVE
    return _MSG.TRANSLATION_TRUSTED


def _undeclared_feature_panel_dict(report: VerdictReport) -> dict | None:
    """Hand-serialize undeclared_feature_panel (a plain dataclass, same
    reason structural_annotations gets manual treatment: the field is typed
    loosely -- Optional[Any] -- on VerdictReport to avoid a circular import,
    so Pydantic's automatic model_dump() can't introspect its contents).
    None when the screen did not run or found nothing to report.
    """
    panel = report.undeclared_feature_panel
    if panel is None:
        return None
    import dataclasses
    return dataclasses.asdict(panel)


def verdict_to_dict(report: VerdictReport) -> dict:
    """Serialize a VerdictReport to a JSON-safe dict.

    - Keys are NOT sorted here; call verdict_to_json for sorted output.
    - NaN/Inf are coerced to None.
    - numpy scalar types are coerced to Python native.
    - structural_annotations serialized explicitly (bare list — no Pydantic schema).
    - undeclared_feature_panel serialized explicitly (plain dataclass — no
      Pydantic schema), same reason as structural_annotations. None when the
      screen did not run or found nothing to report.
    - feature_attribution stays nested under measured_damage (not hoisted).
    - categorical_encoding / categorical_unseen_counts / fold_inert_columns /
      fold_feature_coverage are plain dicts (no custom Pydantic schema
      needed) so they come straight out of `raw` -- unlike
      structural_annotations/undeclared_feature_panel, no manual
      serialization step, just an explicit key in `out` so they're not
      silently dropped by this function's own allowlist below.
    - Every documented top-level key is always present; None when absent.
    """
    raw = report.model_dump(
        mode="json", exclude={"structural_annotations", "undeclared_feature_panel"}
    )

    annotations = [ann.model_dump(mode="json") for ann in report.structural_annotations]

    summary = {
        "fixable_leakage": report.measured_damage.fixable_leakage,
        "headline": _summary_headline(report),
        "top_feature": _summary_top_feature(report),
        "verdict": report.policy_decision.verdict,
    }

    out: dict = {
        "categorical_encoding": raw["categorical_encoding"],
        "categorical_unseen_counts": raw["categorical_unseen_counts"],
        "engine_detection": raw["engine_detection"],
        "fold_ci": raw["fold_ci"],
        "fold_inert_columns": raw["fold_inert_columns"],
        "fold_feature_coverage": raw["fold_feature_coverage"],
        "measured_damage": raw["measured_damage"],
        "policy_decision": raw["policy_decision"],
        "schema_version": "1",
        "structural_annotations": annotations,
        "summary": summary,
        "undeclared_feature_panel": _undeclared_feature_panel_dict(report),
    }

    return _coerce(out)


def verdict_to_json(report: VerdictReport, *, indent: int | None = None) -> str:
    """Serialize a VerdictReport to a deterministic JSON string with sorted keys."""
    return json.dumps(verdict_to_dict(report), sort_keys=True, indent=indent)
