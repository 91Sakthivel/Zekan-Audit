"""Pure helpers for the `zekan init` wizard — no typer, no stdin, fully unit-testable."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def resolve_column(raw: str, cols: list[str]) -> Optional[int]:
    """Resolve user-typed input to a column index: exact name, then unambiguous
    case-insensitive name, then integer index. No fuzzy/partial matching --
    returns None (never a guess) whenever nothing resolves unambiguously,
    including when a case-insensitive match hits more than one column.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw in cols:
        return cols.index(raw)
    ci_matches = [i for i, c in enumerate(cols) if c.lower() == raw.lower()]
    if len(ci_matches) == 1:
        return ci_matches[0]
    if len(ci_matches) > 1:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 0 <= idx < len(cols):
        return idx
    return None


def build_contract_mapping(
    prediction_problem: str,
    entity_id: str,
    prediction_time: str,
    target: str,
    available_features_until: str,
    forbidden_after_prediction: list[str],
    data_path: Optional[str] = None,
) -> dict:
    """Return an ordered mapping {"contract": {...}} for yaml.safe_dump.

    Field order matches PredictionContract's required fields exactly.
    forbidden_after_prediction is always included, even when empty.
    When data_path is given, a top-level "data" key is included alongside
    "contract" so the written config declares its own dataset path.
    """
    mapping = {
        "contract": {
            "prediction_problem": prediction_problem,
            "entity_id": entity_id,
            "prediction_time": prediction_time,
            "target": target,
            "available_features_until": available_features_until,
            "forbidden_after_prediction": forbidden_after_prediction,
        }
    }
    if data_path is not None:
        mapping["data"] = data_path
    return mapping


def validate_mapping(mapping: dict) -> None:
    """Construct ZekanConfig(**mapping) to validate; lets ValidationError propagate."""
    from zekan.config.schema import ZekanConfig
    ZekanConfig(**mapping)


def write_config(mapping: dict, path: str | Path) -> None:
    """Write mapping as BOM-free UTF-8 YAML with field order preserved."""
    text = yaml.safe_dump(
        mapping,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    Path(path).write_text(text, encoding="utf-8")
