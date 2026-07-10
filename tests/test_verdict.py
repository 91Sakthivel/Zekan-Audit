"""Tests for the Phase 4 + 5 verdict assembler (verdict.py).

Covers all five ladder branches including UNCONFIRMED_HIGH_DAMAGE, edge cases
(unavailable engine, null not run), Phase 5 fold-CI confidence tiers, custom
floors, and structural assertions on every block.
"""

import math

import numpy as np
import pytest

from zekan.severity.engine import PerFoldSeverity, SeverityResult
from zekan.severity.verdict import (
    _DEFAULT_FAIL_FLOOR,
    _DEFAULT_WARN_FLOOR,
    FoldCI,
    VerdictReport,
    build_verdict,
)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _interior_folds(leakages: list[float]) -> list[PerFoldSeverity]:
    """Build interior PerFoldSeverity objects from a list of fold leakage values."""
    return [
        PerFoldSeverity(
            fold_idx=i,
            auc_with_forbidden=0.5 + fl,
            auc_without_forbidden=0.5,
            is_terminal=False,
        )
        for i, fl in enumerate(leakages)
    ]


def _result(
    fixable_leakage: float = 0.05,
    p_value: float | None = 0.50,
    nsl: float | None = -1.0,
    null_iqr: float | None = 0.010,
    null_99th: float | None = 0.040,
    p_value_across: float | None = None,
    nsl_across: float | None = None,
    null_iqr_across: float | None = None,
    null_99th_across: float | None = None,
    naive_auc: float = 0.80,
    estimated_deployable_auc: float = 0.75,
    nonfixable_optimism: float = 0.02,
    engine_status: str = "pass",
    unavailable_reason: str | None = None,
    n_permutations_run: int = 100,
    per_fold: list | None = None,
) -> SeverityResult:
    if per_fold is None:
        per_fold = []
    return SeverityResult(
        status=engine_status,
        metric="roc_auc",
        naive_auc=naive_auc,
        estimated_deployable_auc=estimated_deployable_auc,
        total_optimism=naive_auc - estimated_deployable_auc,
        fixable_leakage=fixable_leakage,
        fixable_leakage_range=(fixable_leakage * 0.8, fixable_leakage * 1.2),
        nonfixable_optimism=nonfixable_optimism,
        per_fold=per_fold,
        caveat="test",
        null_iqr=null_iqr,
        null_99th=null_99th,
        p_value=p_value,
        nsl=nsl,
        null_iqr_across=null_iqr_across,
        null_99th_across=null_99th_across,
        p_value_across=p_value_across,
        nsl_across=nsl_across,
        n_permutations_across=100 if p_value_across is not None else 0,
        unavailable_reason=unavailable_reason,
        n_permutations_run=n_permutations_run,
    )


# ── Ladder branch tests ───────────────────────────────────────────────────────

class TestVerdictLadder:
    """All five policy_decision verdict values must be reachable."""

    def test_pass_not_detected_low_fl(self):
        """Not detected (p >= alpha) + fl below warn_floor -> PASS."""
        r = build_verdict(_result(fixable_leakage=0.03, p_value=0.50, nsl=-1.5))
        assert r.policy_decision.verdict == "PASS"
        assert r.engine_detection.detected is False

    def test_unconfirmed_high_damage_p_not_significant(self):
        """Not detected (p >= alpha) + fl >= warn_floor -> UNCONFIRMED_HIGH_DAMAGE.

        This is the load-bearing case: at small/underpowered n the null is wide
        enough that NSL < 1.0 while fl is genuinely large.  Must NOT be PASS.
        """
        r = build_verdict(_result(fixable_leakage=0.12, p_value=0.50, nsl=-0.5))
        assert r.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"
        assert r.engine_detection.detected is False
        interp = r.policy_decision.interpretation
        assert "not a clean PASS" in interp
        assert "increase n" in interp

    def test_unconfirmed_high_damage_nsl_below_gate(self):
        """Not detected (p fires but NSL < 1.0) + fl >= warn_floor -> UNCONFIRMED_HIGH_DAMAGE.

        Explicitly constructs the high-fl + NSL<1.0 case: p is significant but
        the null-standardized effect size is below the detection threshold.
        The high measured fl still prevents a clean PASS verdict.
        """
        r = build_verdict(_result(fixable_leakage=0.12, p_value=0.009, nsl=0.5))
        assert r.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"
        assert r.engine_detection.detected is False
        assert "not a clean PASS" in r.policy_decision.interpretation

    def test_note_detected_low_fl(self):
        """Detected (p < alpha AND NSL >= 1.0) + fl below warn_floor -> NOTE."""
        r = build_verdict(_result(fixable_leakage=0.05, p_value=0.009, nsl=1.5))
        assert r.policy_decision.verdict == "NOTE"
        assert r.engine_detection.detected is True

    def test_warn_detected_mid_fl(self):
        """Detected + warn_floor <= fl < fail_floor -> WARN."""
        r = build_verdict(_result(fixable_leakage=0.12, p_value=0.009, nsl=3.0))
        assert r.policy_decision.verdict == "WARN"
        assert r.engine_detection.detected is True

    def test_fail_detected_high_fl(self):
        """Detected + fl >= fail_floor -> FAIL."""
        r = build_verdict(_result(fixable_leakage=0.20, p_value=0.009, nsl=8.0))
        assert r.policy_decision.verdict == "FAIL"
        assert r.engine_detection.detected is True

    def test_fl_at_warn_floor_boundary_is_warn(self):
        """fl == warn_floor exactly triggers WARN (not NOTE) when detected."""
        r = build_verdict(
            _result(fixable_leakage=_DEFAULT_WARN_FLOOR, p_value=0.009, nsl=1.5)
        )
        assert r.policy_decision.verdict == "WARN"

    def test_fl_at_fail_floor_boundary_is_fail(self):
        """fl == fail_floor exactly triggers FAIL when detected."""
        r = build_verdict(
            _result(fixable_leakage=_DEFAULT_FAIL_FLOOR, p_value=0.009, nsl=5.0)
        )
        assert r.policy_decision.verdict == "FAIL"

    def test_fl_just_below_warn_floor_is_note(self):
        """fl just under warn_floor stays NOTE when detected."""
        r = build_verdict(
            _result(fixable_leakage=_DEFAULT_WARN_FLOOR - 0.001, p_value=0.009, nsl=1.5)
        )
        assert r.policy_decision.verdict == "NOTE"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unavailable_engine_result(self):
        """Unavailable SeverityResult propagates cleanly through all blocks."""
        sr = _result(
            engine_status="unavailable",
            unavailable_reason="Insufficient rows.",
            p_value=None,
            nsl=None,
            null_iqr=None,
            n_permutations_run=0,
        )
        r = build_verdict(sr)
        assert r.engine_detection.detected is False
        assert r.engine_detection.confidence == "unavailable"
        assert math.isnan(r.measured_damage.fixable_leakage)
        assert r.policy_decision.verdict == "PASS"
        assert "unavailable" in r.policy_decision.interpretation.lower()

    def test_null_not_run_low_fl_is_pass(self):
        """n_permutations=0 + low fl: not detected -> PASS."""
        r = build_verdict(
            _result(
                fixable_leakage=0.05,
                p_value=None,
                nsl=None,
                null_iqr=None,
                n_permutations_run=0,
            )
        )
        assert r.engine_detection.detected is False
        assert r.policy_decision.verdict == "PASS"

    def test_null_not_run_high_fl_is_unconfirmed(self):
        """n_permutations=0 + high fl: not detected but fl above floor -> UNCONFIRMED_HIGH_DAMAGE."""
        r = build_verdict(
            _result(
                fixable_leakage=0.20,
                p_value=None,
                nsl=None,
                null_iqr=None,
                n_permutations_run=0,
            )
        )
        assert r.policy_decision.verdict == "UNCONFIRMED_HIGH_DAMAGE"
        assert "not a clean PASS" in r.policy_decision.interpretation

    def test_nsl_exactly_one_is_detected(self):
        """NSL == 1.0 (the boundary value) counts as detected."""
        r = build_verdict(_result(fixable_leakage=0.05, p_value=0.009, nsl=1.0))
        assert r.engine_detection.detected is True

    def test_nsl_just_below_one_is_not_detected(self):
        """NSL < 1.0 does not fire the detection gate even when p < alpha."""
        r = build_verdict(_result(fixable_leakage=0.05, p_value=0.009, nsl=0.999))
        assert r.engine_detection.detected is False


# ── Phase 5: fold-CI confidence ───────────────────────────────────────────────

class TestFoldCI:
    """Confidence is driven by fold-level CI width (Phase 5), not null_iqr."""

    def test_proxy_like_gives_high_tier(self):
        """Tight fold leakages (proxy-like) -> ci_ratio << 0.33 -> 'high'."""
        # fl_mean=0.275, fl_std~0.002 -> ci_half~0.004 -> ci_ratio~0.014
        per_fold = _interior_folds([0.275, 0.278, 0.272, 0.275])
        r = build_verdict(_result(fixable_leakage=0.275, per_fold=per_fold))
        assert r.engine_detection.confidence == "high"
        assert r.fold_ci.confidence_tier == "high"
        assert r.fold_ci.confidence_calibration_status == "calibrated"

    def test_temporal_n200_like_gives_low_tier(self):
        """Wide fold leakages (temporal-n200-like) -> ci_ratio > 0.67 -> 'low'."""
        # fl_mean=0.14, fl_std~0.063 -> ci_half~0.100 -> ci_ratio~0.72
        per_fold = _interior_folds([0.10, 0.22, 0.08, 0.16])
        r = build_verdict(_result(fixable_leakage=0.14, per_fold=per_fold))
        assert r.engine_detection.confidence == "low"
        assert r.fold_ci.confidence_tier == "low"
        assert "provisional" in r.fold_ci.confidence_calibration_status

    def test_medium_tier_assigned(self):
        """ci_ratio in [0.33, 0.67) -> 'medium' with provisional status."""
        # fl_mean=0.12, fl_std~0.037 -> ci_half~0.058 -> ci_ratio~0.484
        per_fold = _interior_folds([0.08, 0.10, 0.14, 0.16])
        r = build_verdict(_result(fixable_leakage=0.12, per_fold=per_fold))
        assert r.fold_ci.confidence_tier == "medium"
        assert "provisional" in r.fold_ci.confidence_calibration_status

    def test_insufficient_folds_guard_gives_low(self):
        """Empty per_fold -> guard (k<2) -> tier='low', ci_ratio=None, note set."""
        r = build_verdict(_result(per_fold=[]))
        assert r.engine_detection.confidence == "low"
        assert r.fold_ci.ci_ratio is None
        assert r.fold_ci.ci_half is None
        assert "insufficient folds" in r.fold_ci.instability_note

    def test_sub_floor_fl_note_emitted(self):
        """fl_mean < warn_floor -> instability_note present even if tier is high."""
        # fl_mean=0.05 < 0.10 warn_floor; identical folds -> ci_ratio=0 -> 'high'
        per_fold = _interior_folds([0.05, 0.05, 0.05, 0.05])
        r = build_verdict(_result(fixable_leakage=0.05, per_fold=per_fold))
        assert "action floor" in r.fold_ci.instability_note

    def test_high_tier_no_note_when_fl_above_floor(self):
        """Tight folds with fl_mean >= warn_floor: no instability note."""
        per_fold = _interior_folds([0.12, 0.12, 0.12, 0.12])
        r = build_verdict(_result(fixable_leakage=0.12, per_fold=per_fold))
        assert r.fold_ci.confidence_tier == "high"
        assert r.fold_ci.instability_note == ""

    def test_pooled_vs_fold_gap_computed(self):
        """pooled_vs_fold_gap = pooled_fl - fl_fold_mean, reported correctly."""
        per_fold = _interior_folds([0.10, 0.12, 0.11, 0.10])
        # fl_fold_mean = 0.1075; pooled deliberately differs by 0.0025
        pooled = 0.11
        r = build_verdict(_result(fixable_leakage=pooled, per_fold=per_fold))
        expected_gap = pooled - float(np.mean([0.10, 0.12, 0.11, 0.10]))
        assert r.fold_ci.pooled_vs_fold_gap == pytest.approx(expected_gap, abs=1e-6)

    def test_fold_ci_fields_all_present(self):
        """FoldCI block on VerdictReport carries all required fields."""
        per_fold = _interior_folds([0.10, 0.11, 0.10, 0.11])
        r = build_verdict(_result(fixable_leakage=0.105, per_fold=per_fold))
        fc = r.fold_ci
        assert isinstance(fc, FoldCI)
        assert fc.fixable_leakage_pooled == pytest.approx(0.105)
        assert fc.ci_center == "fl_mean"
        assert fc.fixable_leakage_fold_mean is not None
        assert fc.fl_fold_std is not None
        assert fc.fl_fold_se is not None
        assert fc.ci_half is not None
        assert fc.ci_low is not None
        assert fc.ci_high is not None
        assert fc.ci_ratio is not None
        assert fc.confidence_tier in ("high", "medium", "low")
        assert fc.confidence_calibration_status
        assert isinstance(fc.instability_note, str)


# ── Phase 5: t-CI coverage ────────────────────────────────────────────────────

class TestCIcoverage:
    def test_t_ci_covers_zero_on_null_folds(self):
        """With K=4 N(0, sigma) fold leakages, 0 falls inside the 95% t-CI.

        5 independent seeds: P(>=4/5 covered) = 0.977 when nominal coverage=95%.
        """
        rng = np.random.default_rng(42)
        n_covered = 0
        for _ in range(5):
            leakages = rng.normal(0.0, 0.020, size=4).tolist()
            per_fold = _interior_folds(leakages)
            fl_mean = float(np.mean(leakages))
            r = build_verdict(_result(fixable_leakage=fl_mean, per_fold=per_fold))
            fc = r.fold_ci
            if fc.ci_low is not None and fc.ci_high is not None:
                if fc.ci_low <= 0.0 <= fc.ci_high:
                    n_covered += 1
        assert n_covered >= 4, (
            f"t-CI covered 0 in {n_covered}/5 seeds; expected >= 4 (95% nominal)"
        )


# ── Custom floors and policy profile ─────────────────────────────────────────

class TestCustomFloors:
    def test_custom_floors_shift_verdict(self):
        """Custom warn/fail floors override defaults correctly."""
        # With defaults: fl=0.08 + detected -> NOTE (0.08 < 0.10)
        # With custom floors (warn=0.05, fail=0.10): fl=0.08 -> WARN
        r = build_verdict(
            _result(fixable_leakage=0.08, p_value=0.009, nsl=2.0),
            warn_floor=0.05,
            fail_floor=0.10,
        )
        assert r.policy_decision.verdict == "WARN"
        assert r.policy_decision.warn_floor == 0.05
        assert r.policy_decision.fail_floor == 0.10

    def test_policy_default_used_true_with_defaults(self):
        r = build_verdict(_result())
        assert r.policy_decision.policy_default_used is True

    def test_policy_default_used_false_with_custom_floors(self):
        r = build_verdict(_result(), warn_floor=0.05, fail_floor=0.12)
        assert r.policy_decision.policy_default_used is False

    def test_custom_profile_name(self):
        r = build_verdict(_result(), policy_profile="retail_churn_v2")
        assert r.policy_decision.policy_profile == "retail_churn_v2"

    def test_user_overridable_always_true(self):
        """Floors are always marked overridable — policy, not engine law."""
        r = build_verdict(_result())
        assert r.policy_decision.user_overridable is True


# ── Block structure and field contracts ───────────────────────────────────────

class TestBlockStructure:
    def test_returns_verdict_report(self):
        assert isinstance(build_verdict(_result()), VerdictReport)

    def test_engine_detection_fields_populated(self):
        # Identical fold leakages (fl_std=0) -> ci_ratio=0 -> 'high'
        per_fold = _interior_folds([0.12, 0.12, 0.12, 0.12])
        r = build_verdict(
            _result(p_value=0.009, nsl=3.0, fixable_leakage=0.12, per_fold=per_fold)
        )
        d = r.engine_detection
        assert d.detected is True
        assert d.alpha == pytest.approx(0.01)
        assert d.p_value == pytest.approx(0.009)
        assert d.nsl == pytest.approx(3.0)
        assert d.confidence == "high"
        assert d.interpretation

    def test_detection_interpretation_frames_nsl_as_power(self):
        """EngineDetection.interpretation must frame NSL as power, not severity."""
        r = build_verdict(_result(p_value=0.009, nsl=8.5, null_iqr=0.002))
        interp = r.engine_detection.interpretation.lower()
        assert any(w in interp for w in ("power", "statistical", "confidence", "null"))
        assert "severity" not in interp

    def test_measured_damage_fields(self):
        r = build_verdict(_result(
            fixable_leakage=0.07,
            naive_auc=0.80,
            estimated_deployable_auc=0.73,
            nonfixable_optimism=0.02,
        ))
        m = r.measured_damage
        assert m.metric == "roc_auc"
        assert m.fixable_leakage == pytest.approx(0.07)
        assert m.optimism_decomposition.naive_auc == pytest.approx(0.80)
        assert m.optimism_decomposition.deployable_auc == pytest.approx(0.73)
        # B = naive_auc - nonfixable_optimism = 0.80 - 0.02 = 0.78
        assert m.optimism_decomposition.temporal_all_auc == pytest.approx(0.78)
        assert m.interpretation

    def test_measured_damage_interpretation_contains_fl(self):
        r = build_verdict(_result(fixable_leakage=0.087))
        assert "0.0870" in r.measured_damage.interpretation or "+0.087" in r.measured_damage.interpretation

    def test_policy_decision_floor_defaults(self):
        r = build_verdict(_result())
        pd = r.policy_decision
        assert pd.warn_floor == _DEFAULT_WARN_FLOOR
        assert pd.fail_floor == _DEFAULT_FAIL_FLOOR
        assert pd.policy_profile == "default_auc"

    def test_no_band_field_on_measured_damage(self):
        """measured_damage must not have a 'band' field."""
        r = build_verdict(_result())
        assert not hasattr(r.measured_damage, "band")

    def test_json_round_trip(self):
        """VerdictReport serialises and deserialises without loss."""
        per_fold = _interior_folds([0.12, 0.13, 0.11, 0.12])
        r = build_verdict(
            _result(fixable_leakage=0.12, p_value=0.009, nsl=5.0, per_fold=per_fold)
        )
        raw = r.model_dump()
        r2 = VerdictReport.model_validate(raw)
        assert r2.policy_decision.verdict == r.policy_decision.verdict
        assert r2.engine_detection.detected == r.engine_detection.detected
        assert r2.fold_ci.confidence_tier == r.fold_ci.confidence_tier


# ── Spec 1: across-entity null OR-combination + channel attribution ───────────
# Pure synthetic SeverityResult inputs (no real permutation run) — fast unit
# coverage of the combination logic in build_verdict.  The real permutation
# behaviour (does across-entity actually catch what within-entity misses on
# real data) is covered separately in tests/test_across_entity_null.py.

class TestAcrossEntityDetection:
    def test_within_only_channel(self):
        """across defaults to not-run (None) -> detected via within alone."""
        r = _result(p_value=0.005, nsl=2.0)
        d = build_verdict(r).engine_detection
        assert d.detected is True
        assert d.detection_channel == "within_entity"

    def test_across_only_channel(self):
        r = _result(p_value=0.5, nsl=-1.0, p_value_across=0.005, nsl_across=2.0)
        d = build_verdict(r).engine_detection
        assert d.detected is True
        assert d.detection_channel == "across_entity"

    def test_both_channel(self):
        r = _result(p_value=0.005, nsl=2.0, p_value_across=0.002, nsl_across=3.0)
        d = build_verdict(r).engine_detection
        assert d.detected is True
        assert d.detection_channel == "both"

    def test_neither_channel(self):
        r = _result(p_value=0.5, nsl=-1.0, p_value_across=0.5, nsl_across=-1.0)
        d = build_verdict(r).engine_detection
        assert d.detected is False
        assert d.detection_channel == ""

    def test_across_none_is_fail_safe_never_manufactures_detection(self):
        """A not-run/None across null must contribute False only — it can never
        make detected True on its own, and must not error."""
        r = _result(p_value=0.5, nsl=-1.0)  # within not detected; across not run
        d = build_verdict(r).engine_detection
        assert d.detected is False
        assert d.detection_channel == ""

    def test_policy_verdict_reaches_warn_via_across_only_detection(self):
        """An across-only detection with fl >= warn_floor must reach a
        detected-branch verdict (NOTE/WARN/FAIL) via the OR, not
        UNCONFIRMED_HIGH_DAMAGE (which would happen if only within-entity fed
        the detected gate, pre-spec-1 behavior)."""
        r = _result(
            fixable_leakage=0.12, p_value=0.5, nsl=-1.0,
            p_value_across=0.005, nsl_across=2.0,
        )
        v = build_verdict(r)
        assert v.engine_detection.detected is True
        assert v.policy_decision.verdict in ("NOTE", "WARN", "FAIL")

    def test_within_entity_only_case_unchanged_from_pre_spec1(self):
        """Within-entity-alone detection (across not run) must produce the exact
        same detected/verdict outcome as before spec 1 — the within path and its
        calibration are untouched."""
        r = _result(fixable_leakage=0.12, p_value=0.009, nsl=3.0)
        v = build_verdict(r)
        assert v.engine_detection.detected is True
        assert v.policy_decision.verdict == "WARN"
        assert v.engine_detection.p_value == pytest.approx(0.009)
        assert v.engine_detection.nsl == pytest.approx(3.0)
