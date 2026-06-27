"""Integration tests for run_audit() — chains run_severity_analysis → build_verdict.

Three verdict paths are tested end-to-end:
  PASS                   : clean dataset, null not run
  UNCONFIRMED_HIGH_DAMAGE: strong label-proxy leak, null not run (n_permutations=0)
  FAIL                   : strong label-proxy leak, 100 permutations confirm detection

Each path runs the engine exactly once (module-scoped fixture) to keep the suite fast.
The FAIL path uses RF(n_estimators=5) to minimise permutation wall time.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import inject_label_proxy
from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.audit import run_audit
from zekan.severity.verdict import VerdictReport


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fast_clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


def _noise_contract():
    return PredictionContract(
        prediction_problem="integration-test-pass",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["noise_forbidden"],
    )


def _proxy_contract():
    return PredictionContract(
        prediction_problem="integration-test-proxy",
        entity_id="entity_id",
        prediction_time="prediction_time",
        target="target",
        available_features_until="prediction_time",
        forbidden_after_prediction=["leaky_label_proxy"],
    )


# ── Module-scoped fixtures — each engine run executes once ────────────────────

@pytest.fixture(scope="module")
def pass_report():
    df = make_clean_dataset(n_entities=100, seed=0)
    df["noise_forbidden"] = np.random.default_rng(42).standard_normal(size=len(df))
    contract = _noise_contract()
    return run_audit(
        df, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=0,
    )


@pytest.fixture(scope="module")
def unconfirmed_report():
    """Label-proxy + n_permutations=0 → detected=False + fl>>warn_floor → UNCONFIRMED_HIGH_DAMAGE."""
    df_clean = make_clean_dataset(n_entities=100, seed=0)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = _proxy_contract()
    return run_audit(
        df_leaky, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=0,
    )


@pytest.fixture(scope="module")
def fail_report():
    """Label-proxy + n_permutations=100 → p=1/101<0.01, NSL>>1, fl>>0.15 → FAIL."""
    df_clean = make_clean_dataset(n_entities=100, seed=0)
    df_leaky, _ = inject_label_proxy(df_clean, flip_rate=0.05, seed=0)
    contract = _proxy_contract()
    return run_audit(
        df_leaky, contract, ZekanConfig(contract=contract),
        model_factory=_fast_clf, n_permutations=100,
    )


# ── PASS path ─────────────────────────────────────────────────────────────────

class TestPassPath:
    def test_returns_verdict_report(self, pass_report):
        assert isinstance(pass_report, VerdictReport)

    def test_verdict_is_pass(self, pass_report):
        assert pass_report.policy_decision.verdict == "PASS"

    def test_not_detected(self, pass_report):
        assert pass_report.engine_detection.detected is False

    def test_fixable_leakage_near_zero(self, pass_report):
        assert pass_report.measured_damage.fixable_leakage < 0.10

    def test_all_three_blocks_present(self, pass_report):
        assert pass_report.engine_detection is not None
        assert pass_report.measured_damage is not None
        assert pass_report.policy_decision is not None


# ── UNCONFIRMED_HIGH_DAMAGE path ──────────────────────────────────────────────

class TestUnconfirmedHighDamagePath:
    def test_verdict_is_unconfirmed(self, unconfirmed_report):
        assert unconfirmed_report.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"

    def test_not_detected(self, unconfirmed_report):
        """p=None when null not run — detection gate cannot fire."""
        assert unconfirmed_report.engine_detection.detected is False

    def test_large_fixable_leakage(self, unconfirmed_report):
        """Label-proxy fl is well above warn_floor=0.10 (temporal ceiling ~0.096)."""
        assert unconfirmed_report.measured_damage.fixable_leakage >= 0.10

    def test_interpretation_not_clean_pass(self, unconfirmed_report):
        assert "not a clean PASS" in unconfirmed_report.policy_decision.interpretation


# ── FAIL path ─────────────────────────────────────────────────────────────────

class TestFailPath:
    def test_verdict_is_fail(self, fail_report):
        assert fail_report.policy_decision.verdict == "FAIL"

    def test_detected(self, fail_report):
        """With 100 permutations and strong proxy, best p=1/101≈0.0099 < 0.01."""
        assert fail_report.engine_detection.detected is True

    def test_p_value_below_alpha(self, fail_report):
        assert fail_report.engine_detection.p_value is not None
        assert fail_report.engine_detection.p_value < 0.01

    def test_fixable_leakage_above_fail_floor(self, fail_report):
        assert fail_report.measured_damage.fixable_leakage >= 0.15
