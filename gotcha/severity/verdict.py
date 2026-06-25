"""Phase 4 + 5 three-block verdict layer with fold-CI confidence.

Takes a SeverityResult from the engine and produces a VerdictReport with three
independent readable blocks plus a fold-CI stability assessment:

  engine_detection  -- did the null-standardized gate fire?
  measured_damage   -- what is the fl magnitude, with optimism decomposition?
  policy_decision   -- what is the operational verdict given policy floors?
  fold_ci           -- how stable is the fold-level leakage estimate? (Phase 5)

Detection is NOT severity.
NSL encodes statistical power (how far above the permutation null the observed
fixable_leakage sits, in units of the null IQR) — it does not measure the
physical magnitude of the damage.

Confidence (fold_ci.confidence_tier) describes the stability of the fold-level
leakage estimate: how precisely the K interior temporal fold differences pin down
the pooled OOF fixable_leakage.  It is NOT severity.  A high-confidence low fl
is still a PASS; a low-confidence high fl is UNCONFIRMED_HIGH_DAMAGE.

Tier calibration (earned from 5-scenario Option A sweep, seeds 0-4):
  high   (ci_ratio < 0.33):  calibrated — proxy/large-n clearly separated
  medium (ci_ratio < 0.67):  provisional_awaiting_intermediate_regime — idle
                              in 5-scenario sweep; needs n=500/alpha=0.30 data
  low    (ci_ratio >= 0.67): provisional_awaiting_intermediate_regime

Policy floors (policy_decision block only — engine.py is unchanged):
  warn_floor = 0.10  anchored above the worst-case mild-temporal-leak ceiling
                     observed in the calibration suite.
  fail_floor = 0.15  documented operational blocking default, not engine law.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import t as _scipy_t
from pydantic import BaseModel, ConfigDict

from gotcha.severity.engine import (
    SeverityResult,
    _NULL_ALPHA,
    _NSL_NOTE_THRESHOLD,
)

# ── Policy floor defaults ────────────────────────────────────────────────────

_DEFAULT_WARN_FLOOR: float = 0.10
"""fl floor for WARN.

Anchored above the worst-case mild-temporal-leak ceiling observed in the
calibration suite (~0.096 at n=200, alpha=1.60).  Exceeding this does NOT prove
a target-copy mechanism — it places the observation above the regime where
rho-bounded temporal leakage saturates.
"""

_DEFAULT_FAIL_FLOOR: float = 0.15
"""fl floor for FAIL.

A documented operational blocking default, not an engine-derived law.  The
calibration study showed that real label-proxy leaks (flip_rate <= 0.20) land
above this level (fl_med >= 0.162), while rho-bounded temporal leaks top out
below it.  Overridable by domain.
"""

# ── Fold-CI constant ──────────────────────────────────────────────────────────

_CI_EPS: float = 0.001
"""Denominator floor for ci_ratio = ci_half / max(abs(fl_mean), _CI_EPS).

Prevents division by zero when fl_mean is near zero (clean/null data).
At fl_mean=0 and ci_half=0.015, ci_ratio=15 -> 'low' (correctly: wide CI
relative to a negligible signal).
"""


# ── Sub-models ────────────────────────────────────────────────────────────────

class OptimismDecomposition(BaseModel):
    """A/B/C triangle components from the engine's decomposition."""

    model_config = ConfigDict(extra="forbid")

    naive_auc: float
    """A: random grouped CV, all features — optimistic upper bound."""

    temporal_all_auc: float
    """B: temporal CV, all features (mean across non-skipped folds)."""

    deployable_auc: float
    """C: temporal CV, safe features only — estimated deployed performance."""


class EngineDetection(BaseModel):
    """Null-standardized detection gate result.

    detected = (p_value < alpha) AND (nsl >= 1.0).

    NSL (null-standardized leakage) measures how many null-IQR units the
    observed fixable_leakage sits above the permutation 99th-percentile.
    It reflects statistical power / detection confidence — NOT physical damage
    magnitude.
    """

    model_config = ConfigDict(extra="forbid")

    detected: bool
    p_value: Optional[float]
    nsl: Optional[float]
    alpha: float
    confidence: str
    """Fold-level CI confidence tier: 'high' / 'medium' / 'low' / 'unavailable'.

    Mirrors fold_ci.confidence_tier.  Driven by the relative CI width of the
    fold-level leakage estimate (ci_ratio = ci_half / max(|fl_mean|, _CI_EPS)):
      high   (ci_ratio < 0.33): calibrated; proxy/large-n clearly separated
      medium (ci_ratio < 0.67): provisional_awaiting_intermediate_regime
      low    (ci_ratio >= 0.67): wide CI relative to signal; estimate imprecise

    Confidence describes measurement stability, NOT severity.
    See VerdictReport.fold_ci for the full CI computation.
    """
    interpretation: str


class MeasuredDamage(BaseModel):
    """Pure measurement block: how much AUC inflation was observed?

    No verdict, no band field.  fixable_leakage (B - C pooled OOF) is the
    magnitude of the declared-feature inflation.  The policy_decision block owns
    the floor comparison and the verdict.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    fixable_leakage: float
    optimism_decomposition: OptimismDecomposition
    interpretation: str


class PolicyDecision(BaseModel):
    """Operational verdict under the active policy profile.

    Verdict ladder:
      not detected + fl < warn_floor          -> PASS
      not detected + fl >= warn_floor         -> UNCONFIRMED_HIGH_DAMAGE
      detected + fl < warn_floor              -> NOTE
      detected + warn_floor <= fl < fail_floor -> WARN
      detected + fl >= fail_floor             -> FAIL

    UNCONFIRMED_HIGH_DAMAGE is load-bearing: at small or underpowered n the
    permutation null is too wide for NSL to clear 1.0, yet the measured fl can
    be genuinely large.  That is not a clean PASS.
    """

    model_config = ConfigDict(extra="forbid")

    policy_profile: str
    warn_floor: float
    fail_floor: float
    user_overridable: bool
    verdict: str
    policy_default_used: bool
    interpretation: str


class FoldCI(BaseModel):
    """Phase 5 fold-level CI stability block.

    Confidence describes stability of the fold-level leakage estimate, NOT
    severity.  Pooled OOF fixable_leakage remains the primary damage estimate;
    this block quantifies how precisely the K interior temporal fold differences
    agree on that estimate.

    CI construction (t-interval, df = n_interior_folds - 1):
      interior folds: per_fold entries where not pf.is_terminal
      fl_mean  = mean(fold_leakage_i  for interior folds)
      fl_std   = sample std (ddof=1)
      fl_se    = fl_std / sqrt(n_interior_folds)
      t_crit   = scipy.stats.t.ppf(0.975, df=n_interior_folds - 1)
      ci_half  = t_crit * fl_se
      CI       = [fl_mean - ci_half, fl_mean + ci_half]
      ci_ratio = ci_half / max(abs(fl_mean), _CI_EPS)

    Tier calibration (earned from 5-scenario Option A sweep, seeds 0-4):
      high   (ci_ratio < 0.33): calibrated; clean separation proxy/large-n vs small-n
      medium (ci_ratio < 0.67): provisional_awaiting_intermediate_regime
      low    (ci_ratio >= 0.67): provisional_awaiting_intermediate_regime
    """

    model_config = ConfigDict(extra="forbid")

    fixable_leakage_pooled: float
    """Pooled OOF fl — the primary damage estimate (unchanged from engine)."""

    fixable_leakage_fold_mean: Optional[float]
    """Mean of interior fold leakages.  Close to, but not identical to, pooled OOF."""

    fl_fold_std: Optional[float]
    """Sample std (ddof=1) of interior fold leakages."""

    fl_fold_se: Optional[float]
    """Standard error: fl_fold_std / sqrt(n_interior_folds)."""

    ci_center: str
    """Always 'fl_mean'.  CI is centered on the fold-level mean, not pooled OOF."""

    ci_half: Optional[float]
    """t_crit * fl_fold_se.  Half-width of the 95% t-interval."""

    ci_low: Optional[float]
    """Lower bound: fl_fold_mean - ci_half."""

    ci_high: Optional[float]
    """Upper bound: fl_fold_mean + ci_half."""

    ci_ratio: Optional[float]
    """ci_half / max(abs(fl_fold_mean), _CI_EPS).  Relative CI width drives the tier."""

    confidence_tier: str
    """Stability tier: 'high' / 'medium' / 'low' / 'unavailable'."""

    confidence_calibration_status: str
    """Calibration status of the tier boundary used:
      'calibrated'                               -- high boundary (0.33) earned from data
      'provisional_awaiting_intermediate_regime' -- medium/low boundary (0.67) idle
      'unavailable: ...'                         -- guard condition; tier defaulted to low
    """

    pooled_vs_fold_gap: Optional[float]
    """pooled_fl - fl_fold_mean.  Non-zero reflects estimator differences (normal)."""

    instability_note: str
    """Empty when estimate is stable.  Non-empty flags:
      'insufficient folds for interval (df<=0)'
      'fl below action floor; precision not meaningful'
    """


class VerdictReport(BaseModel):
    """Four-block verdict for one leakage analysis run."""

    model_config = ConfigDict(extra="forbid")

    engine_detection: EngineDetection
    measured_damage: MeasuredDamage
    policy_decision: PolicyDecision
    fold_ci: FoldCI


# ── Fold-CI computation (Phase 5) ─────────────────────────────────────────────

def _compute_fold_ci(result: SeverityResult, warn_floor: float) -> FoldCI:
    """Compute fold-level t-CI and assign confidence tier.

    Confidence reflects how stably the K interior temporal folds agree on the
    leakage magnitude.  It is NOT severity: the tier describes precision, not
    whether fl is large or harmful.

    Pooled OOF fixable_leakage is kept as the primary damage estimate throughout.
    """
    pooled = result.fixable_leakage
    interior = [pf.fold_leakage for pf in result.per_fold if not pf.is_terminal]
    k = len(interior)

    if k < 2:
        fl_mean_1 = interior[0] if k == 1 else None
        gap_1 = (pooled - interior[0]) if k == 1 else None
        return FoldCI(
            fixable_leakage_pooled=pooled,
            fixable_leakage_fold_mean=fl_mean_1,
            fl_fold_std=None,
            fl_fold_se=None,
            ci_center="fl_mean",
            ci_half=None,
            ci_low=None,
            ci_high=None,
            ci_ratio=None,
            confidence_tier="low",
            confidence_calibration_status="unavailable: insufficient folds for interval",
            pooled_vs_fold_gap=gap_1,
            instability_note="insufficient folds for interval (df<=0)",
        )

    arr = np.array(interior, dtype=float)
    fl_mean = float(arr.mean())
    fl_std = float(arr.std(ddof=1))
    fl_se = fl_std / float(np.sqrt(k))
    t_crit = float(_scipy_t.ppf(0.975, df=k - 1))
    ci_half = t_crit * fl_se
    ci_low = fl_mean - ci_half
    ci_high = fl_mean + ci_half
    ci_ratio = ci_half / max(abs(fl_mean), _CI_EPS)
    gap = pooled - fl_mean

    if ci_ratio < 0.33:
        tier = "high"
        cal_status = "calibrated"
    elif ci_ratio < 0.67:
        tier = "medium"
        cal_status = "provisional_awaiting_intermediate_regime"
    else:
        tier = "low"
        cal_status = "provisional_awaiting_intermediate_regime"

    note = ""
    if abs(fl_mean) < warn_floor:
        note = "fl below action floor; precision not meaningful"

    return FoldCI(
        fixable_leakage_pooled=pooled,
        fixable_leakage_fold_mean=fl_mean,
        fl_fold_std=fl_std,
        fl_fold_se=fl_se,
        ci_center="fl_mean",
        ci_half=ci_half,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_ratio=ci_ratio,
        confidence_tier=tier,
        confidence_calibration_status=cal_status,
        pooled_vs_fold_gap=gap,
        instability_note=note,
    )


# ── Verdict ladder ────────────────────────────────────────────────────────────

def _policy_verdict(
    detected: bool,
    fl: float,
    nsl: Optional[float],
    warn_floor: float,
    fail_floor: float,
) -> tuple[str, str]:
    """Return (verdict, interpretation) from the five-branch ladder."""
    nsl_str = f"NSL={nsl:.2f}" if nsl is not None else "NSL=n/a"

    if not detected:
        if fl >= warn_floor:
            interp = (
                f"large inflation measured ({fl:+.4f} AUC) but not statistically "
                f"confirmed at this sample size ({nsl_str} < 1.0) — not a clean "
                "PASS; increase n before trusting."
            )
            return "UNCONFIRMED_HIGH_DAMAGE", interp
        interp = (
            f"fixable_leakage = {fl:+.4f} is below warn_floor "
            f"({warn_floor:.2f}) and not statistically confirmed; "
            "no evidence of meaningful temporal leakage."
        )
        return "PASS", interp

    if fl < warn_floor:
        interp = (
            f"leakage statistically confirmed ({nsl_str} >= 1.0, "
            f"p < {_NULL_ALPHA}) but magnitude is below warn_floor "
            f"({warn_floor:.2f}): fixable_leakage = {fl:+.4f}. "
            "Small effect; monitor for drift."
        )
        return "NOTE", interp
    if fl < fail_floor:
        interp = (
            f"leakage confirmed ({nsl_str}, p < {_NULL_ALPHA}) and "
            f"fixable_leakage = {fl:+.4f} is at or above warn_floor "
            f"({warn_floor:.2f}). Remove or lag the forbidden feature."
        )
        return "WARN", interp
    interp = (
        f"leakage confirmed ({nsl_str}, p < {_NULL_ALPHA}) and "
        f"fixable_leakage = {fl:+.4f} is at or above fail_floor ({fail_floor:.2f}; "
        "a documented operational blocking default, not an engine-derived law; "
        "overridable by domain). Remove the forbidden feature before deployment."
    )
    return "FAIL", interp


# ── Public API ────────────────────────────────────────────────────────────────

def build_verdict(
    result: SeverityResult,
    warn_floor: float = _DEFAULT_WARN_FLOOR,
    fail_floor: float = _DEFAULT_FAIL_FLOOR,
    policy_profile: str = "default_auc",
) -> VerdictReport:
    """Assemble the four-block VerdictReport from a SeverityResult.

    Parameters
    ----------
    result
        Output of run_severity_analysis().  When result.status == 'unavailable'
        all blocks reflect that gracefully.
    warn_floor
        fl threshold for WARN verdict; default 0.10.
    fail_floor
        fl threshold for FAIL verdict; default 0.15.
    policy_profile
        Descriptive name for the policy block.
    """
    policy_default_used = (
        warn_floor == _DEFAULT_WARN_FLOOR and fail_floor == _DEFAULT_FAIL_FLOOR
    )

    # ── unavailable engine result ─────────────────────────────────────────────
    if result.status == "unavailable":
        reason = result.unavailable_reason or "Severity engine could not run."
        _nan = float("nan")
        return VerdictReport(
            engine_detection=EngineDetection(
                detected=False,
                p_value=None,
                nsl=None,
                alpha=_NULL_ALPHA,
                confidence="unavailable",
                interpretation=f"Engine unavailable: {reason}",
            ),
            measured_damage=MeasuredDamage(
                metric="roc_auc",
                fixable_leakage=_nan,
                optimism_decomposition=OptimismDecomposition(
                    naive_auc=_nan,
                    temporal_all_auc=_nan,
                    deployable_auc=_nan,
                ),
                interpretation=f"Engine unavailable: {reason}",
            ),
            policy_decision=PolicyDecision(
                policy_profile=policy_profile,
                warn_floor=warn_floor,
                fail_floor=fail_floor,
                user_overridable=True,
                verdict="PASS",
                policy_default_used=policy_default_used,
                interpretation=f"Engine unavailable — no verdict possible: {reason}",
            ),
            fold_ci=FoldCI(
                fixable_leakage_pooled=_nan,
                fixable_leakage_fold_mean=None,
                fl_fold_std=None,
                fl_fold_se=None,
                ci_center="fl_mean",
                ci_half=None,
                ci_low=None,
                ci_high=None,
                ci_ratio=None,
                confidence_tier="unavailable",
                confidence_calibration_status="unavailable: engine unavailable",
                pooled_vs_fold_gap=None,
                instability_note=f"Engine unavailable: {reason}",
            ),
        )

    # ── fold-CI (Phase 5 confidence layer) ───────────────────────────────────
    fold_ci = _compute_fold_ci(result, warn_floor)

    # ── engine_detection block ────────────────────────────────────────────────
    detected = bool(
        result.p_value is not None
        and result.p_value < _NULL_ALPHA
        and result.nsl is not None
        and result.nsl >= _NSL_NOTE_THRESHOLD
    )

    if detected:
        nsl_str = f"NSL={result.nsl:.2f}"
        det_interp = (
            f"Temporal leakage statistically confirmed: "
            f"p={result.p_value:.4f} < {_NULL_ALPHA} and {nsl_str} >= 1.0.  "
            "NSL measures how many null-IQR units the observed fixable_leakage "
            "sits above the permutation 99th-percentile — it reflects statistical "
            "power and detection confidence, not physical damage magnitude."
        )
    elif result.p_value is None:
        det_interp = (
            "No permutation null was run (n_permutations=0).  "
            "Detection gate requires n_permutations >= 100."
        )
    elif result.p_value >= _NULL_ALPHA:
        det_interp = (
            f"fixable_leakage (p={result.p_value:.4f}) is indistinguishable from "
            "the permutation null — no statistical evidence of temporal leakage "
            "from the declared forbidden features at this sample size."
        )
    else:
        nsl_str = f"NSL={result.nsl:.2f}" if result.nsl is not None else "NSL=n/a"
        det_interp = (
            f"p={result.p_value:.4f} < {_NULL_ALPHA} (leakage is outside the null) "
            f"but {nsl_str} < 1.0 — effect size is small relative to null "
            "uncertainty at this sample size; detection gate did not fire."
        )

    detection = EngineDetection(
        detected=detected,
        p_value=result.p_value,
        nsl=result.nsl,
        alpha=_NULL_ALPHA,
        confidence=fold_ci.confidence_tier,
        interpretation=det_interp,
    )

    # ── measured_damage block ─────────────────────────────────────────────────
    fl = result.fixable_leakage
    mean_b_auc = result.naive_auc - result.nonfixable_optimism

    damage = MeasuredDamage(
        metric="roc_auc",
        fixable_leakage=fl,
        optimism_decomposition=OptimismDecomposition(
            naive_auc=result.naive_auc,
            temporal_all_auc=mean_b_auc,
            deployable_auc=result.estimated_deployable_auc,
        ),
        interpretation=(
            f"Forbidden feature(s) inflated offline AUC by {fl:+.4f} "
            f"(fixable_leakage = B − C = "
            f"{mean_b_auc:.4f} − {result.estimated_deployable_auc:.4f})."
        ),
    )

    # ── policy_decision block ─────────────────────────────────────────────────
    verdict, policy_interp = _policy_verdict(
        detected=detected,
        fl=fl,
        nsl=result.nsl,
        warn_floor=warn_floor,
        fail_floor=fail_floor,
    )

    decision = PolicyDecision(
        policy_profile=policy_profile,
        warn_floor=warn_floor,
        fail_floor=fail_floor,
        user_overridable=True,
        verdict=verdict,
        policy_default_used=policy_default_used,
        interpretation=policy_interp,
    )

    return VerdictReport(
        engine_detection=detection,
        measured_damage=damage,
        policy_decision=decision,
        fold_ci=fold_ci,
    )
