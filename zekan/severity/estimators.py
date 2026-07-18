"""Estimator allowlist and factory builder for the --estimator CLI flag."""

from __future__ import annotations

from typing import Any, Callable

import typer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

_ESTIMATOR_ALLOWLIST: dict[str, type] = {
    "rf": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gbm": GradientBoostingClassifier,
    "histgb": HistGradientBoostingClassifier,
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
    if cls is HistGradientBoostingClassifier:
        # early_stopping=False (NOT the sklearn default 'auto'): 'auto' silently
        # enables an internal validation split above 10k rows, making the
        # model's fit procedure discontinuous in row count purely because n
        # crossed 10,000 -- unacceptable for a trust gate that must mean the
        # same thing at every scale. random_state=42 for the same reason every
        # other allowlisted estimator has one: with early_stopping off, HistGB
        # still has internal randomness (histogram binning tie-breaks etc.),
        # and leaving random_state=None would make repeat audits on identical
        # data non-reproducible. No other non-default params.
        return lambda: cls(random_state=42, early_stopping=False)
    # RandomForestClassifier and ExtraTreesClassifier accept n_jobs
    return lambda: cls(n_estimators=200, random_state=42, n_jobs=1)
