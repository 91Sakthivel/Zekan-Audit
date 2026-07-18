"""Tests for Tier 2: sequential/adaptive permutation stopping (Besag-Clifford
style with decision-stability early stop).

Spec (pre-registered, locked): h=10, N_min=30, N_max=500.

Layout:
  (a) unit tests on the isolated helper functions (order-statistic quantile
      CI, conservative IQR bound, NSL decision-stability check) -- hand
      verifiable, no monkeypatching.
  (b) unit tests on the sequential stopping loop in estimate_fixable_leakage_null,
      via the SAME monkeypatch pattern test_f2a_parallel_null.py already uses
      for _null_permutation_once, so the exact stop point is fully controlled
      and known in advance.
  (c) determinism: same seed -> same draws -> same stop point -> byte-identical
      output, across n_jobs values (mirrors the F2a acceptance bar).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm as scipy_norm

from zekan.severity.null_baseline import (
    NullResult,
    _SEQ_H,
    _SEQ_N_MAX,
    _SEQ_N_MIN,
    _conservative_iqr_bound,
    _nsl_decision_stable,
    _quantile_order_stat_indices,
    estimate_fixable_leakage_null,
)

from tests.test_f2a_parallel_null import (
    _fast_clf,
    _leaky_config,
    _leaky_contract,
    _leaky_panel,
)


# ── (a) Unit: order-statistic quantile CI ──────────────────────────────────────

def test_quantile_order_stat_indices_matches_hand_formula():
    """Direct check against the normal-approximation formula, hand-computed."""
    n, p, confidence = 100, 0.25, 0.99
    z = float(scipy_norm.ppf((1 + confidence) / 2))
    se = float(np.sqrt(n * p * (1 - p)))
    expected_lo = max(0, min(int(np.floor(n * p - z * se)), n - 1))
    expected_hi = max(0, min(int(np.ceil(n * p + z * se)), n - 1))
    lo, hi = _quantile_order_stat_indices(n, p, confidence)
    assert (lo, hi) == (expected_lo, expected_hi)


def test_quantile_order_stat_indices_widens_with_higher_confidence():
    """A wider (more conservative) confidence level must never narrow the bound."""
    lo_95, hi_95 = _quantile_order_stat_indices(50, 0.75, 0.95)
    lo_99, hi_99 = _quantile_order_stat_indices(50, 0.75, 0.99)
    assert lo_99 <= lo_95
    assert hi_99 >= hi_95


def test_quantile_order_stat_indices_narrows_with_more_samples():
    """More samples -> tighter RELATIVE bound around n*p (the whole point of
    the sequential rule: more draws should narrow the plausible-quantile band)."""
    lo_30, hi_30 = _quantile_order_stat_indices(30, 0.75, 0.99)
    lo_300, hi_300 = _quantile_order_stat_indices(300, 0.75, 0.99)
    rel_width_30 = (hi_30 - lo_30) / 30
    rel_width_300 = (hi_300 - lo_300) / 300
    assert rel_width_300 < rel_width_30


def test_quantile_order_stat_indices_clipped_to_valid_range():
    lo, hi = _quantile_order_stat_indices(5, 0.75, 0.999999)
    assert 0 <= lo <= 4
    assert 0 <= hi <= 4


# ── (a) Unit: conservative IQR bound ────────────────────────────────────────────

def test_conservative_iqr_bound_contains_point_estimate():
    """The [iqr_low, iqr_high] bound must bracket the plain sample IQR."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 1.0, 200)
    point_iqr = float(np.percentile(samples, 75) - np.percentile(samples, 25))
    iqr_low, iqr_high = _conservative_iqr_bound(samples, 0.99)
    assert iqr_low <= point_iqr <= iqr_high


def test_conservative_iqr_bound_narrows_with_more_samples():
    rng = np.random.default_rng(1)
    small = rng.normal(0.0, 1.0, 30)
    large = np.concatenate([small, rng.normal(0.0, 1.0, 270)])
    lo_s, hi_s = _conservative_iqr_bound(small, 0.99)
    lo_l, hi_l = _conservative_iqr_bound(large, 0.99)
    assert (hi_l - lo_l) < (hi_s - lo_s)


# ── (a) Unit: NSL decision stability ────────────────────────────────────────────

def test_nsl_decision_stable_when_both_ends_agree():
    """observed far above null_99th, IQR bound entirely small -> both ends >= 1.0."""
    assert _nsl_decision_stable(
        observed_fixable_leakage=1.0, null_99th=0.0, iqr_low=0.01, iqr_high=0.05,
    ) is True


def test_nsl_decision_unstable_when_bound_straddles_boundary():
    """Construct iqr_low/iqr_high so nsl_worst >= 1.0 but nsl_best < 1.0."""
    observed, null_99th = 1.0, 0.0
    # nsl = (observed - null_99th) / iqr = 1.0 / iqr
    # iqr_low=0.9 -> nsl_worst = 1.111 (>=1.0); iqr_high=1.5 -> nsl_best = 0.667 (<1.0)
    assert _nsl_decision_stable(observed, null_99th, iqr_low=0.9, iqr_high=1.5) is False


def test_nsl_decision_stable_when_both_ends_below_one():
    assert _nsl_decision_stable(
        observed_fixable_leakage=0.5, null_99th=0.0, iqr_low=0.9, iqr_high=2.0,
    ) is True


# ── (b) Unit: sequential loop via monkeypatched _null_permutation_once ────────

def test_sequential_stops_exactly_on_h_exceedance_besag_clifford():
    """9 exceedances in draws 1-9, then none until draw 35 (the 10th) -- the
    loop must continue past N_MIN=30 (only 9 exceedances there, and the
    running p is nowhere near alpha) and stop the instant h=10 is reached,
    with the exact Besag-Clifford p = h/n_drawn."""
    import zekan.severity.null_baseline as nb

    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 9:
            return 1.0   # exceedance (>= observed=0.5)
        if calls["n"] == 35:
            return 1.0   # the 10th exceedance
        return 0.1       # non-exceedance

    import unittest.mock as mock
    with mock.patch.object(nb, "_null_permutation_once", _fake_unit):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=0.5,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    assert calls["n"] == 35
    assert result.n_permutations == 35
    assert result.stopped_early is True
    assert result.stopping == "sequential_v1"
    assert result.p_value == pytest.approx(_SEQ_H / 35)


def test_sequential_never_stops_before_n_min_even_if_h_reached_early():
    """All 30 draws are exceedances (count would hit h=10 at draw 10), but
    N_MIN=30 must still be enforced before ANY stop check runs."""
    import zekan.severity.null_baseline as nb
    import unittest.mock as mock

    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        return 1.0  # every draw is an exceedance

    with mock.patch.object(nb, "_null_permutation_once", _fake_unit):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=0.5,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    # First check point is draw 30 (== _SEQ_N_MIN); count_gte=30 >= h=10 there.
    assert calls["n"] == _SEQ_N_MIN == 30
    assert result.n_permutations == 30
    assert result.stopped_early is True
    assert result.p_value == pytest.approx(_SEQ_H / 30)


def test_sequential_runs_to_n_max_when_neither_rule_fires():
    """No exceedances ever, but keep the running p just above alpha long enough
    that it never dips below 0.01 before N_MAX (with 0 exceedances, p=1/(n+1),
    which only drops below 0.01 at n=100) AND arrange for decision-stability
    to never fire either (a large, genuinely wide null spread) -- must run all
    the way to _SEQ_N_MAX and use the Laplace-corrected p-value."""
    import zekan.severity.null_baseline as nb
    import unittest.mock as mock

    # Force the "neither rule resolves" path deterministically: 0 exceedances
    # forever (so h=10 is never reached) and decision-stability mocked to
    # always report "not yet stable" (so it never fires either). This isolates
    # exactly the code path under test -- the loop's forced-N_MAX behavior --
    # without depending on where a real value sequence happens to stabilize
    # (which is data-dependent and not the point of this test).
    rng = np.random.default_rng(42)
    fake_values = rng.normal(0.0, 1.0, _SEQ_N_MAX).tolist()
    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        return fake_values[calls["n"] - 1]

    with mock.patch.object(nb, "_null_permutation_once", _fake_unit), \
         mock.patch.object(nb, "_nsl_decision_stable", return_value=False):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=1000.0,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    assert calls["n"] == _SEQ_N_MAX
    assert result.n_permutations == _SEQ_N_MAX
    assert result.stopped_early is False
    # Laplace formula, not Besag-Clifford (count_gte == 0 here).
    assert result.p_value == pytest.approx(1 / (_SEQ_N_MAX + 1))


def test_sequential_decision_stability_can_defer_stop_past_n_min():
    """A case with 0 exceedances (p_running < alpha only once n>=100) and a
    real (non-degenerate) spread: confirm the loop actually reaches a point
    beyond N_MIN before considering a stop tied to decision-stability, and
    that whenever it DOES stop early via decision-stability, the reported
    null_iqr-based NSL decision is provably stable (re-derived independently
    here, not just trusted from the loop's own internal check)."""
    import zekan.severity.null_baseline as nb
    import unittest.mock as mock

    rng = np.random.default_rng(7)
    fake_values = rng.normal(0.0, 1.0, _SEQ_N_MAX).tolist()
    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        return fake_values[calls["n"] - 1]

    # observed chosen near the upper tail of a Normal(0,1) sample -- plausible
    # borderline NSL regime, but with 0 exceedances virtually certain.
    observed = 4.0

    with mock.patch.object(nb, "_null_permutation_once", _fake_unit):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=observed,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    # p=1/(n+1) cannot go below alpha=0.01 before n=100 with 0 exceedances --
    # the loop must have run at least that far (whether it stopped there via
    # decision-stability or continued further).
    assert result.n_permutations >= 100
    if result.stopped_early:
        # Independently re-verify the decision-stability claim using the
        # ACTUAL draws consumed, not trusting the loop's internal bookkeeping.
        arr = np.array(fake_values[: result.n_permutations])
        iqr_lo, iqr_hi = _conservative_iqr_bound(arr, 0.99)
        null_99th = float(np.percentile(arr, 99))
        assert _nsl_decision_stable(observed, null_99th, iqr_lo, iqr_hi) is True


# ── (c) Determinism: same seed -> same draws -> byte-identical, across n_jobs ──

@pytest.mark.parametrize("n_jobs", [1, 2])
def test_sequential_determinism_same_seed_same_n_jobs(n_jobs):
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    from tests.test_f2a_parallel_null import _observed_fl
    fl = _observed_fl(df, contract, config)

    r1 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=n_jobs, stopping="sequential_v1",
    )
    r2 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=n_jobs, stopping="sequential_v1",
    )
    assert r1.n_permutations == r2.n_permutations
    assert r1.stopped_early == r2.stopped_early
    assert r1.p_value == r2.p_value
    np.testing.assert_array_equal(r1.null_samples, r2.null_samples)


def test_sequential_n_jobs_independence_byte_identical():
    """The whole point of spawning children up front for _SEQ_N_MAX: draw i
    must be identical regardless of n_jobs, so the stop point (and everything
    derived from it) must be identical too."""
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    from tests.test_f2a_parallel_null import _observed_fl
    fl = _observed_fl(df, contract, config)

    r_serial = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
    )
    r_parallel = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=2, stopping="sequential_v1",
    )
    assert r_serial.n_permutations == r_parallel.n_permutations
    assert r_serial.stopped_early == r_parallel.stopped_early
    assert r_serial.p_value == r_parallel.p_value
    np.testing.assert_array_equal(r_serial.null_samples, r_parallel.null_samples)


def test_fixed_v1_default_unaffected_by_stopping_param_presence():
    """Sanity: NOT passing stopping= at all (or passing 'fixed_v1' explicitly)
    must give byte-identical results -- the new parameter must be a true no-op
    for every existing caller."""
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    from tests.test_f2a_parallel_null import _observed_fl
    fl = _observed_fl(df, contract, config)

    r_default = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=20, seed=0, method="within_entity", n_jobs=1,
    )
    r_explicit = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        n_permutations=20, seed=0, method="within_entity", n_jobs=1,
        stopping="fixed_v1",
    )
    assert r_default.stopping == "fixed_v1"
    assert r_default.stopped_early is False
    assert r_default.p_value == r_explicit.p_value
    np.testing.assert_array_equal(r_default.null_samples, r_explicit.null_samples)


def test_unsupported_stopping_value_raises():
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    with pytest.raises(ValueError, match="unsupported stopping"):
        estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=0.5,
            seed=0, method="within_entity", n_jobs=1, stopping="bogus_mode",
        )
