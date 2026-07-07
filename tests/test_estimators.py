"""Unit tests for the estimator allowlist and factory builder."""

from __future__ import annotations

import pytest
import typer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from zekan.severity.estimators import _ESTIMATOR_ALLOWLIST, _build_factory


def test_allowlist_contains_expected_keys():
    assert set(_ESTIMATOR_ALLOWLIST) == {"rf", "extra_trees", "gbm", "logistic"}


def test_rf_factory_returns_seeded_rf():
    est = _build_factory("rf")()
    assert isinstance(est, RandomForestClassifier)
    assert est.random_state == 42
    assert est.n_estimators == 200
    assert est.n_jobs == 1


def test_extra_trees_factory_returns_seeded_extra_trees():
    est = _build_factory("extra_trees")()
    assert isinstance(est, ExtraTreesClassifier)
    assert est.random_state == 42
    assert est.n_estimators == 200
    assert est.n_jobs == 1


def test_gbm_factory_returns_seeded_gbm():
    est = _build_factory("gbm")()
    assert isinstance(est, GradientBoostingClassifier)
    assert est.random_state == 42
    assert est.n_estimators == 200


def test_logistic_factory_returns_seeded_logistic():
    est = _build_factory("logistic")()
    assert isinstance(est, LogisticRegression)
    assert est.random_state == 42
    assert est.max_iter == 1000


def test_factory_returns_fresh_instance_each_call():
    factory = _build_factory("rf")
    assert factory() is not factory()


def test_unknown_name_raises_bad_parameter_listing_choices():
    with pytest.raises(typer.BadParameter) as exc_info:
        _build_factory("xgboost")
    msg = str(exc_info.value)
    assert "xgboost" in msg
    # All valid choices must appear in the error
    for key in _ESTIMATOR_ALLOWLIST:
        assert key in msg


def test_unknown_name_empty_string_raises_bad_parameter():
    with pytest.raises(typer.BadParameter):
        _build_factory("")
