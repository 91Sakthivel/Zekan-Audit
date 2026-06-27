"""Tests for zekan.severity.power.estimate_n_required.

estimate_n_required is verified correct and tested here, but is currently
NOT wired to any rendered output.  It is reserved for a future
INCONCLUSIVE-below-warn_floor path (fl < 0.10), which is not reachable in
the current verdict ladder.  Do not remove or skip these tests.
"""

from __future__ import annotations

from zekan.severity.power import MIN_ESTIMABLE_N, estimate_n_required


# ── Known values (hand-computed) ──────────────────────────────────────────────

def test_fl_clamps_to_min():
    # (0.352 / 0.0352)^2 = 10^2 = 100 < MIN_ESTIMABLE_N → clamps to 500
    assert estimate_n_required(0.0352) == MIN_ESTIMABLE_N


def test_fl_above_min():
    # (0.352 / 0.00352)^2 = 100^2 = 10000 > MIN_ESTIMABLE_N → 10000
    assert estimate_n_required(0.00352) == 10_000


def test_very_large_fl_clamps_to_min():
    # Any fl large enough to make n < 500 must return MIN_ESTIMABLE_N
    assert estimate_n_required(1.0) == MIN_ESTIMABLE_N
    assert estimate_n_required(0.5) == MIN_ESTIMABLE_N


# ── Zero and negative inputs → None ──────────────────────────────────────────

def test_zero_returns_none():
    assert estimate_n_required(0.0) is None


def test_negative_returns_none():
    assert estimate_n_required(-0.05) is None


# ── Monotonic: smaller fl → larger n ─────────────────────────────────────────

def test_monotonic():
    n_large_fl = estimate_n_required(0.0352)   # clamps to 500
    n_small_fl = estimate_n_required(0.00352)  # 10000
    assert n_large_fl < n_small_fl


def test_monotonic_mid_range():
    n_hi = estimate_n_required(0.005)
    n_lo = estimate_n_required(0.001)
    assert n_hi is not None and n_lo is not None
    assert n_hi < n_lo
