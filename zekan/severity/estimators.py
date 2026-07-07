"""Estimator allowlist and factory builder for the --estimator CLI flag."""

from __future__ import annotations

from typing import Any, Callable

import typer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

_ESTIMATOR_ALLOWLIST: dict[str, type] = {
    "rf": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gbm": GradientBoostingClassifier,
    "logistic": LogisticRegression,
}


def _build_factory(name: str) -> Callable[[], Any]:
    """Return a zero-arg factory for the named estimator, seed 42 baked in.

    Raises typer.BadParameter for unknown names so the CLI gets a clean error.
    """
    cls = _ESTIMATOR_ALLOWLIST.get(name)
    if cls is None:
        valid = ", ".join(sorted(_ESTIMATOR_ALLOWLIST))
        raise typer.BadParameter(
            f"Unknown estimator '{name}'. Valid choices: {valid}"
        )
    if cls is LogisticRegression:
        return lambda: cls(random_state=42, max_iter=1000)
    if cls is GradientBoostingClassifier:
        # GradientBoostingClassifier does not accept n_jobs
        return lambda: cls(n_estimators=200, random_state=42)
    # RandomForestClassifier and ExtraTreesClassifier accept n_jobs
    return lambda: cls(n_estimators=200, random_state=42, n_jobs=1)
