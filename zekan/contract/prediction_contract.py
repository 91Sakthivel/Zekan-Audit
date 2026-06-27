"""Prediction-time contract: declares what a model is allowed to know at inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CostModel(BaseModel):
    """Business cost parameters for ranking leakage fixes."""

    decision: str
    discount_cost: tuple[float, float]
    customer_value: tuple[float, float]
    conversion_lift: tuple[float, float]


class SeverityThresholds(BaseModel):
    """AUC-inflation bands that gate trust levels."""

    metric: str = "roc_auc"
    low_max: float = 0.02
    moderate_max: float = 0.05
    high_max: float = 0.10


class PredictionContract(BaseModel):
    """Declares the temporal boundary a model must respect at prediction time.

    Every field is a column name or a scalar configuration value — not data.
    The severity engine consults this to know what is and is not allowable.
    """

    prediction_problem: str
    entity_id: str
    prediction_time: str
    target: str
    available_features_until: str
    forbidden_after_prediction: list[str] = Field(default_factory=list)
    cost_model: Optional[CostModel] = None
    severity_thresholds: Optional[SeverityThresholds] = None
    schema_version: str = "1"
    zekan_version: str = "0.1.0"


def load_contract(path: str | Path) -> PredictionContract:
    """Load a PredictionContract from a YAML file.

    Accepts either a bare contract YAML (fields at the top level) or a full
    zekan.yml (in which case the 'contract' key is extracted automatically).
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if "contract" in data:
        data = data["contract"]
    return PredictionContract(**data)
