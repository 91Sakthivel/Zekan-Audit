"""Phase 4 orchestrator: chains engine → verdict into a single callable.

Usage
-----
    from zekan.severity.audit import run_audit

    report = run_audit(df, contract, config, n_permutations=100)
    report.engine_detection   # detection gate result
    report.measured_damage    # fl magnitude + A/B/C decomposition
    report.policy_decision    # operational verdict (PASS/NOTE/WARN/FAIL/UNCONFIRMED_HIGH_DAMAGE)

All three blocks are on the returned VerdictReport for full traceability.
The caller never needs to call run_severity_analysis or build_verdict directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from zekan.config.schema import ZekanConfig
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.engine import run_severity_analysis
from zekan.severity.verdict import (
    VerdictReport,
    _DEFAULT_FAIL_FLOOR,
    _DEFAULT_WARN_FLOOR,
    build_verdict,
)


@dataclass
class _ProbeSpec:
    """Registry entry for one structural probe."""
    fn: Callable
    needs_folds: bool = False


def _run_structural_probes(
    df: pd.DataFrame,
    contract: PredictionContract,
    folds: Optional[list] = None,
) -> list:
    """Run all registered structural probes; return non-pass IssueRecords only.

    Return type is a plain list to avoid importing IssueRecord here (would create
    a module-level import of detectors into the audit orchestrator).

    Probes with needs_folds=True are silently skipped when folds is None.
    Each probe may return either a list[IssueRecord] or a single IssueRecord;
    both are handled uniformly.
    """
    from zekan.detectors.entity_aggregate_probe import probe_forbidden_entity_level_aggregate
    from zekan.detectors.duplicate_probe import probe_raw_duplicates, probe_cross_fold_duplicates

    _PROBES: list[_ProbeSpec] = [
        _ProbeSpec(fn=probe_forbidden_entity_level_aggregate, needs_folds=False),
        _ProbeSpec(fn=probe_raw_duplicates, needs_folds=False),
        _ProbeSpec(fn=probe_cross_fold_duplicates, needs_folds=True),
    ]

    found: list = []
    for spec in _PROBES:
        if spec.needs_folds and not folds:
            continue
        result = spec.fn(df, contract, folds) if spec.needs_folds else spec.fn(df, contract)
        records: list = result if isinstance(result, list) else [result]
        for rec in records:
            if rec.status != "pass":
                found.append(rec)
    return found


def run_audit(
    df: pd.DataFrame,
    contract: PredictionContract,
    config: ZekanConfig,
    model_factory: Optional[Callable[[], Any]] = None,
    n_permutations: int = 100,
    null_seed: int = 0,
    warn_floor: float = _DEFAULT_WARN_FLOOR,
    fail_floor: float = _DEFAULT_FAIL_FLOOR,
    policy_profile: str = "default_auc",
    n_jobs: int = 1,
) -> VerdictReport:
    """Chain run_severity_analysis → build_verdict, returning all three verdict blocks.

    Parameters
    ----------
    df
        Dataset to audit.
    contract
        PredictionContract describing entity/time/target/forbidden columns.
    config
        ZekanConfig with SplitPolicy.
    model_factory
        Callable returning a fresh sklearn-compatible classifier.  When None the
        engine uses its internal default (RandomForestClassifier).
    n_permutations
        Permutation null draws.  Default 100 (the recommended minimum for a
        defensible detection verdict).  Pass 0 to skip the null — detection will
        be False and high fl will appear as UNCONFIRMED_HIGH_DAMAGE.
    null_seed
        RNG seed for the permutation null.
    warn_floor
        fl threshold for WARN in policy_decision; default 0.10.
    fail_floor
        fl threshold for FAIL in policy_decision; default 0.15.
    policy_profile
        Label for the policy block; use "default_auc" or a domain string.

    Returns
    -------
    VerdictReport
        Three blocks accessible as attributes:
          .engine_detection  — detected flag, p_value, nsl, confidence
          .measured_damage   — fixable_leakage, A/B/C decomposition
          .policy_decision   — verdict, floors, interpretation
    """
    severity_result = run_severity_analysis(
        df,
        contract,
        config,
        model_factory=model_factory,
        n_permutations=n_permutations,
        null_seed=null_seed,
        ablation_warn_floor=warn_floor,
        n_jobs=n_jobs,
    )
    report = build_verdict(
        severity_result,
        warn_floor=warn_floor,
        fail_floor=fail_floor,
        policy_profile=policy_profile,
    )

    annotations = _run_structural_probes(df, contract, folds=severity_result.folds)
    if annotations:
        report = report.model_copy(update={"structural_annotations": annotations})

    return report
