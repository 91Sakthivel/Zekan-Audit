"""Pure helpers for the `zekan init` wizard — no typer, no stdin, fully unit-testable."""

from __future__ import annotations

from pathlib import Path

import yaml


def build_contract_mapping(
    prediction_problem: str,
    entity_id: str,
    prediction_time: str,
    target: str,
    available_features_until: str,
    forbidden_after_prediction: list[str],
) -> dict:
    """Return an ordered mapping {"contract": {...}} for yaml.safe_dump.

    Field order matches PredictionContract's required fields exactly.
    forbidden_after_prediction is always included, even when empty.
    """
    return {
        "contract": {
            "prediction_problem": prediction_problem,
            "entity_id": entity_id,
            "prediction_time": prediction_time,
            "target": target,
            "available_features_until": available_features_until,
            "forbidden_after_prediction": forbidden_after_prediction,
        }
    }


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
