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
    _ALPHA_FLOOR_DRAWS,
    _SEQ_ALPHA,
    _SEQ_H,
    _SEQ_N_MAX,
    _SEQ_N_MIN,
    _conservative_iqr_bound,
    _conservative_quantile_bound,
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
    """observed far above null_99th, IQR bound entirely small -> both ends >= 1.0.

    Mechanical signature update only (Tier 2b-final): _nsl_decision_stable now
    takes a [null_99th_lo, null_99th_hi] bound instead of one null_99th point,
    to also propagate null_99th's own sampling uncertainty (FIX A). Passing
    the same point value for both bounds reproduces the original degenerate
    (point-estimate) case exactly -- the assertion's intent is unchanged.
    """
    assert _nsl_decision_stable(
        observed_fixable_leakage=1.0, null_99th_lo=0.0, null_99th_hi=0.0,
        iqr_low=0.01, iqr_high=0.05,
    ) is True


def test_nsl_decision_unstable_when_bound_straddles_boundary():
    """Construct iqr_low/iqr_high so nsl_worst >= 1.0 but nsl_best < 1.0.

    Mechanical signature update only -- see test_nsl_decision_stable_when_both_ends_agree.
    """
    observed, null_99th = 1.0, 0.0
    # nsl = (observed - null_99th) / iqr = 1.0 / iqr
    # iqr_low=0.9 -> nsl_worst = 1.111 (>=1.0); iqr_high=1.5 -> nsl_best = 0.667 (<1.0)
    assert _nsl_decision_stable(
        observed, null_99th_lo=null_99th, null_99th_hi=null_99th, iqr_low=0.9, iqr_high=1.5
    ) is False


def test_nsl_decision_stable_when_both_ends_below_one():
    """Mechanical signature update only -- see test_nsl_decision_stable_when_both_ends_agree."""
    assert _nsl_decision_stable(
        observed_fixable_leakage=0.5, null_99th_lo=0.0, null_99th_hi=0.0,
        iqr_low=0.9, iqr_high=2.0,
    ) is True


# ── (b) Unit: sequential loop via monkeypatched _null_permutation_once ────────

def test_sequential_stops_exactly_on_h_exceedance_besag_clifford():
    """9 exceedances in draws 1-9, then none until draw 35 (the 10th) -- the
    loop must continue past N_MIN=30 and stop the instant h=10 is reached,
    with the exact Besag-Clifford p = h/n_drawn.

    RE-BASELINED (Tier 2b-final, approved scope). With this exact fixture,
    Tier 2b's distance-aware decision-stability check (unconditional past
    N_MIN, no longer gated behind "running p already looks significant")
    legitimately resolves at draw 30, in the NOT-DETECTED direction: the top
    of the sample is dominated by the 9 early exceedances (all == 1.0), which
    pins null_99th's conservative bound at exactly [1.0, 1.0], making NSL
    provably negative against observed=0.5. That is a CORRECT distance-aware
    conclusion for this specific data, not a weakened rule -- verified
    empirically before this test was touched (see the Phase-C-style
    discussion in the session transcript). It just means this fixture no
    longer isolates the Besag-Clifford counting mechanic on its own.

    So decision-stability is mocked out here (return_value=False, the same
    pattern test_sequential_runs_to_n_max_when_neither_rule_fires already
    uses) specifically to isolate h-exceedance counting from the OTHER
    stopping mechanism, rather than hand-tuning floating-point fixture values
    to coincidentally avoid triggering it -- more robust against future
    changes to the CI method. See
    test_sequential_decision_stability_fires_early_not_detected_direction
    below for the fixture's original behavior, now tested directly instead of
    incidentally."""
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
    with mock.patch.object(nb, "_null_permutation_once", _fake_unit), \
         mock.patch.object(nb, "_nsl_decision_stable", return_value=False):
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
    assert result.p_is_upper_bound is False  # count_gte=10 > 0, a real count


def test_sequential_decision_stability_fires_early_not_detected_direction():
    """NEW (Tier 2b-final): decision-stability ALONE, with no help from
    Besag-Clifford, stops the loop early in the NOT-DETECTED direction.

    Same 9-early-exceedances-then-flat fixture that used to (incidentally)
    make test_sequential_stops_exactly_on_h_exceedance_besag_clifford resolve
    at draw 30 instead of 35 -- tested directly here instead. count_gte caps
    at 9 forever (h=10 is never reached), so ONLY decision-stability can be
    responsible for stopping at n=30: the top of the sample (the 9 early
    exceedances, all == 1.0) pins null_99th's conservative bound at exactly
    [1.0, 1.0], making the NSL interval provably negative (well below 1.0)
    against observed=0.5 -- "this is not a leak" is supportable at N_MIN with
    no floor, per Tier 2b-final's asymmetric design (Addendum 5)."""
    import zekan.severity.null_baseline as nb
    import unittest.mock as mock

    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 9:
            return 1.0   # exceedance
        return 0.1       # non-exceedance forever -- h=10 is never reached

    with mock.patch.object(nb, "_null_permutation_once", _fake_unit):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=0.5,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    assert calls["n"] == _SEQ_N_MIN == 30
    assert result.n_permutations == 30
    assert result.stopped_early is True
    # Laplace formula with count_gte=9 -- NOT the Besag-Clifford h/n formula
    # (count_gte=9 < h=10, so _p_value_override was never set).
    assert result.p_value == pytest.approx((9 + 1) / (30 + 1))
    assert result.p_is_upper_bound is False  # count_gte=9 > 0, a real count


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
    """A zero-exceedance DETECTED case with a real (non-degenerate) spread:
    confirm the loop does not stop before the honest alpha floor, and does
    not materially overshoot it once cleared.

    RE-BASELINED (Tier 2b-final, approved scope). The OLD assertion
    (`n_permutations >= 100`) was an ARTIFACT of the pre-Tier-2b-final gate
    (decision-stability could only be attempted once running p already
    looked significant, which for 0 exceedances required n>=100 as a side
    effect) -- not a real invariant about this data. Confirmed by re-deriving
    independently, outside the loop, on the raw fake_values: with FIX A alone
    (symmetric, no alpha floor), the NSL interval for this exact seed/observed
    first becomes stable at n=41 (see the empirical sweep run before this
    edit) -- i.e. the OLD code's n>=100 floor was never about this data being
    ambiguous past n=41, only about the removed p_running gate.

    Under Tier 2b-final's asymmetric design, this is a DETECTED-direction
    case (NSL stable and >= 1.0), so it now correctly waits for
    _ALPHA_FLOOR_DRAWS (100) rather than stopping at 41 -- restoring the
    verdict-flip fix Addendum 5 recorded. Confirmed empirically to stop at
    n=101 for this seed (stability holds at n=41..99, is momentarily marginal
    at exactly n=100, and clears again the next draw) -- not overshooting the
    floor by more than one batch."""
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

    # Never stops before the honest alpha floor for a zero-exceedance
    # DETECTED case, and never overshoots it by much once cleared.
    assert result.n_permutations >= _ALPHA_FLOOR_DRAWS
    assert result.n_permutations < _ALPHA_FLOOR_DRAWS + 20
    assert result.stopped_early is True
    assert result.p_value < _SEQ_ALPHA
    assert result.p_is_upper_bound is True  # count_gte == 0 throughout for this fixture

    # Independently re-verify the decision-stability claim using the ACTUAL
    # draws consumed, not trusting the loop's internal bookkeeping.
    arr = np.array(fake_values[: result.n_permutations])
    iqr_lo, iqr_hi = _conservative_iqr_bound(arr, 0.99)
    null_99th_lo, null_99th_hi = _conservative_quantile_bound(arr, 0.99, 0.99)
    assert _nsl_decision_stable(observed, null_99th_lo, null_99th_hi, iqr_lo, iqr_hi) is True


# ── (b2) Tier 2b-final: the alpha floor and asymmetric stopping ──────────────

def test_alpha_floor_draws_is_100_not_99():
    """Derived, not hardcoded (see _derive_alpha_floor_draws) -- and confirms
    Addendum 5's stated 99 was off by one."""
    assert _ALPHA_FLOOR_DRAWS == 100


def test_alpha_floor_boundary_n99_fails_n100_passes():
    """The exact arithmetic _ALPHA_FLOOR_DRAWS is derived from: detection
    needs p STRICTLY less than alpha (engine.py: `if p_value >= _NULL_ALPHA:
    not-detected`). At n=99, the zero-exceedance Laplace p is exactly
    alpha (0.01) -- not strictly less -- so it fails the gate. At n=100, it's
    just under alpha and passes."""
    p_99 = 1.0 / (99 + 1)
    p_100 = 1.0 / (100 + 1)
    assert p_99 == _SEQ_ALPHA
    assert not (p_99 < _SEQ_ALPHA)
    assert p_100 < _SEQ_ALPHA


def test_sequential_detected_direction_waits_for_alpha_floor():
    """NEW (Tier 2b-final): a zero-exceedance DETECTED case (NSL interval
    provably >= 1.0 from the first checkpoint onward, since every draw is
    identically 0.0 against observed=1000.0) must NOT stop at N_MIN=30 --
    it has to wait for _ALPHA_FLOOR_DRAWS, because engine.py's reality gate
    needs p_value < alpha and the Laplace floor 1/(n+1) can't cross alpha
    before n=100 with zero exceedances, regardless of how obvious the NSL
    signal already is. This is the exact case Addendum 5 recorded a verdict
    flip on; this test locks the fix."""
    import zekan.severity.null_baseline as nb
    import unittest.mock as mock

    calls = {"n": 0}

    def _fake_unit(child_seed, *args, **kwargs):
        calls["n"] += 1
        return 0.0  # every draw identical, always below observed -- 0 exceedances ever

    with mock.patch.object(nb, "_null_permutation_once", _fake_unit):
        df = _leaky_panel()
        contract = _leaky_contract()
        config = _leaky_config()
        result = nb.estimate_fixable_leakage_null(
            df, contract, config, _fast_clf, observed_fixable_leakage=1000.0,
            seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
        )

    assert calls["n"] == _ALPHA_FLOOR_DRAWS == 100
    assert result.n_permutations == 100
    assert result.stopped_early is True
    assert result.p_value == pytest.approx(1.0 / 101)
    assert result.p_value < _SEQ_ALPHA
    assert result.p_is_upper_bound is True  # count_gte == 0 throughout


# ── (c) Determinism: same seed -> same draws -> byte-identical, across n_jobs ──

@pytest.mark.parametrize("n_jobs", [1, 2, 12])
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


def test_sequential_draw_values_identical_across_n_jobs_even_when_stop_point_differs():
    """NEW (Tier 2b-final Part 4d): the guaranteed invariant across n_jobs is
    that draw i's VALUE is byte-identical (spawn(N_MAX) is precomputed once,
    up front, regardless of batch size) -- NOT that the stop point itself is
    identical. The stopping decision is only evaluated at batch boundaries
    (batch_size = n_jobs when n_jobs > 1), so a batch size that doesn't
    evenly divide the natural checkpoint (e.g. 12 doesn't divide the n=30
    checkpoint that n_jobs=1/2 land on for this data) can and does make the
    loop stop at a LATER batch boundary. That's expected, inherent to any
    batch-checked sequential design -- not a bug, and not introduced by this
    change (test_sequential_n_jobs_independence_byte_identical's n_jobs=1 vs 2
    comparison only matches today because N_MIN=30 happens to be divisible by
    both 1 and 2 for this data; it does not generalize to n_jobs=12, which is
    exactly what this test checks instead)."""
    df = _leaky_panel()
    contract = _leaky_contract()
    config = _leaky_config()
    from tests.test_f2a_parallel_null import _observed_fl
    fl = _observed_fl(df, contract, config)

    r1 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=1, stopping="sequential_v1",
    )
    r12 = estimate_fixable_leakage_null(
        df, contract, config, _fast_clf, observed_fixable_leakage=fl,
        seed=0, method="within_entity", n_jobs=12, stopping="sequential_v1",
    )
    # Stop points legitimately differ due to batch granularity (documented
    # above) -- assert that plainly rather than assuming equality.
    assert r1.n_permutations != r12.n_permutations
    # But draw i's value must still be identical: r1's entire array must
    # equal the SAME PREFIX of r12's longer array.
    np.testing.assert_array_equal(r1.null_samples, r12.null_samples[: r1.n_permutations])


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
