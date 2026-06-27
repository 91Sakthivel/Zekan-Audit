"""Power-number estimator: n_required to statistically confirm a measured fl.

Grounded in the measured 1/sqrt(n) null scaling law (clean-DGP, 10-period
fixture, within_entity permutation, n_estimators=10, N=100 perms, n>=500):
  null_iqr(n) ~= 0.176 / sqrt(n)   [R2=0.978, stabilized 6-point fit]

DETECTION_COEFF = 2 * 0.176 = 0.352 encodes the two-gate detection condition:
  p-gate:  fl must exceed null_99th  (~1 IQR unit above the null center)
  NSL-gate: fl >= null_99th + null_iqr  (+1 IQR unit more; NSL >= 1.0 required)
  => detection threshold ~= 2 * null_iqr ~= 0.352 / sqrt(n)

Inverting: n_required = ceil((0.352 / fl)^2), clamped to MIN_ESTIMABLE_N=500
because the 1/sqrt(n) law was validated only for n >= 500 on this DGP.

Assumptions: similar data structure, 10 periods, roughly balanced target.
This is an ESTIMATE, not a guarantee. Real detection depends on actual null
geometry, fold structure, and model quality.
"""

from __future__ import annotations

import math

DETECTION_COEFF: float = 0.352
"""2 * 0.176; anchors the power formula on the two-gate detection condition."""

MIN_ESTIMABLE_N: int = 500
"""Minimum n for which the 1/sqrt(n) law was validated; clamp floor."""


def estimate_n_required(fl: float) -> int | None:
    """Estimate sample size needed to statistically confirm fixable_leakage=fl.

    Returns None when fl <= 0 (no effect to confirm).
    Returns at least MIN_ESTIMABLE_N when formula yields a smaller value,
    because the underlying scaling law was not validated below n=500.
    """
    if fl <= 0:
        return None
    n = math.ceil((DETECTION_COEFF / fl) ** 2)
    return max(n, MIN_ESTIMABLE_N)
