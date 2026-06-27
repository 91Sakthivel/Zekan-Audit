"""Top-level zekan.yml schema: contract + model spec + split policy."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from zekan.contract.prediction_contract import PredictionContract


# ── Model spec ────────────────────────────────────────────────────────────────

class ModelSpec(BaseModel):
    """Holds the model specification.

    Exactly one of 'type' (sklearn class path) or 'factory' (file.py:func_name)
    must be set. Params are passed through as-is; nothing is imported here.
    """

    type: Optional[str] = None
    factory: Optional[str] = None
    params: dict[str, Any] = {}

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ModelSpec":
        if self.type is None and self.factory is None:
            raise ValueError("model spec must have either 'type' or 'factory'")
        if self.type is not None and self.factory is not None:
            raise ValueError("model spec cannot have both 'type' and 'factory'")
        return self


# ── Split policy ──────────────────────────────────────────────────────────────

class RandomBaselineStrategy(str, Enum):
    GROUPED_CV = "grouped_cv"
    PLAIN_CV = "plain_cv"


class TemporalStrategy(str, Enum):
    EXPANDING_WINDOW = "expanding_window"
    SLIDING_WINDOW = "sliding_window"


class TemporalEntityOverlap(str, Enum):
    ALLOW_WITH_DIAGNOSTICS = "allow_with_diagnostics"
    FORBID = "forbid"


class SplitPolicy(BaseModel):
    """Controls how zekan creates train/test folds for both protocols.

    Only grouped_cv + expanding_window + allow_with_diagnostics are implemented
    in v1. The other enum values parse without error but raise at runtime.
    """

    random_baseline: RandomBaselineStrategy = RandomBaselineStrategy.GROUPED_CV
    temporal_strategy: TemporalStrategy = TemporalStrategy.EXPANDING_WINDOW
    temporal_entity_overlap: TemporalEntityOverlap = (
        TemporalEntityOverlap.ALLOW_WITH_DIAGNOSTICS
    )
    n_splits: int = 5
    min_valid_folds: int = 3
    min_test_rows_per_fold: int = 100
    min_positive_cases_per_fold: int = 20
    min_negative_cases_per_fold: int = 20
    leak_lookahead: int = 1  # k future periods required; folds where test_time_max is within k of end are excluded from fixable_leakage median

    @model_validator(mode="after")
    def _reject_unimplemented(self) -> "SplitPolicy":
        if self.random_baseline == RandomBaselineStrategy.PLAIN_CV:
            raise ValueError(
                "random_baseline='plain_cv' is not implemented in v1; use 'grouped_cv'"
            )
        if self.temporal_strategy == TemporalStrategy.SLIDING_WINDOW:
            raise ValueError(
                "temporal_strategy='sliding_window' is not implemented in v1; "
                "use 'expanding_window'"
            )
        if self.temporal_entity_overlap == TemporalEntityOverlap.FORBID:
            raise ValueError(
                "temporal_entity_overlap='forbid' is not implemented in v1; "
                "use 'allow_with_diagnostics'"
            )
        return self


# ── Severity config ───────────────────────────────────────────────────────────

class SeverityConfig(BaseModel):
    """Controls the A/B/C performance decomposition and feature ablation."""

    ablation_method: str = "retrain_without"
    top_k_ablation: int = 10


# ── Top-level config ──────────────────────────────────────────────────────────

class ZekanConfig(BaseModel):
    """Top-level configuration for a zekan audit run."""

    contract: PredictionContract
    model: Optional[ModelSpec] = None
    split_policy: SplitPolicy = Field(default_factory=SplitPolicy)
    severity: SeverityConfig = Field(default_factory=SeverityConfig)


_KNOWN_WRONG_KEYS = {"prediction_contract", "zekan", "config", "gotcha"}


def load_config(path: str | Path) -> ZekanConfig:
    """Load a ZekanConfig from a zekan.yml file."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read().lstrip('﻿')
    data = yaml.safe_load(text)
    if isinstance(data, dict) and "contract" not in data:
        found = next((k for k in data if k in _KNOWN_WRONG_KEYS), None)
        if found is not None:
            raise ValueError(
                f"unknown top-level key '{found}' — did you mean 'contract'?"
            )
    return ZekanConfig(**data)
