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

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel

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
    """Registry entry for one structural probe.

    Capability flags are declarative -- _run_structural_probes only builds and
    passes the kwargs a given probe actually declared it needs, so adding a
    flag here is opt-in and never changes the call shape for probes that
    don't set it (Upgrade 1 step 1b: widened for probes that need to fit
    models, e.g. the future undeclared-feature screen -- none of the three
    probes registered today set any of the new flags).
    """
    fn: Callable
    needs_folds: bool = False
    needs_model: bool = False    # receives model_factory=, n_jobs=
    needs_matrix: bool = False   # receives X_all=, y_all=, col_pos=
    needs_budget: bool = False   # receives deadline= (monotonic timestamp, or None)


class _ProbeFailure(BaseModel):
    """Placeholder for a probe that raised during execution (Upgrade 1 step 1b
    resilience requirement) -- NOT a real IssueRecord.

    Deliberately does NOT use zekan.detectors.schema.IssueType/_REGISTRY: this
    step is wiring + hardening only ("do NOT invent a registry row in this
    step" -- that's step 1c's job, once the probes that can actually fail in
    interesting ways exist). This placeholder is safe to sit inside
    VerdictReport.structural_annotations today because every current renderer
    of that list only reads two things: `.what` (zekan/reports/text_view.py,
    html_view.py -- both do `for ann in annotations: ... ann.what`, nothing
    else) and `.model_dump()` (zekan/reports/json_export.py, for every entry
    uniformly) -- both are satisfied here as a plain Pydantic BaseModel with a
    `what` field.

    1c should replace this with a real PROBE_EXECUTION_FAILED (or similar)
    IssueType + _REGISTRY row + typed ProbeDetail, wire status/severity/
    evidence properly, and delete this class.
    """
    what: str
    probe_name: str
    exception_type: str
    exception_message: str
    traceback_text: str


def _build_probe_registry() -> list[_ProbeSpec]:
    """The list of registered structural probes.

    Separated out from _run_structural_probes so tests have a clean seam to
    inject a fake probe (mock.patch("zekan.severity.audit._build_probe_registry",
    return_value=[...])) and exercise the needs_model/needs_matrix/needs_budget/
    exception-isolation wiring directly, without needing a real model-fitting
    probe to exist yet (that's step 1c).
    """
    from zekan.detectors.entity_aggregate_probe import probe_forbidden_entity_level_aggregate
    from zekan.detectors.duplicate_probe import probe_raw_duplicates, probe_cross_fold_duplicates

    return [
        _ProbeSpec(fn=probe_forbidden_entity_level_aggregate, needs_folds=False),
        _ProbeSpec(fn=probe_raw_duplicates, needs_folds=False),
        _ProbeSpec(fn=probe_cross_fold_duplicates, needs_folds=True),
    ]


def _run_structural_probes(
    df: pd.DataFrame,
    contract: PredictionContract,
    folds: Optional[list] = None,
    model_factory: Optional[Callable[[], Any]] = None,
    n_jobs: int = 1,
    X_all: Optional[Any] = None,
    y_all: Optional[Any] = None,
    all_features: Optional[list[str]] = None,
    time_budget_seconds: Optional[float] = None,
) -> list:
    """Run all registered structural probes; return non-pass IssueRecords
    (plus any _ProbeFailure placeholders for probes that raised) only.

    Return type is a plain list to avoid importing IssueRecord here (would create
    a module-level import of detectors into the audit orchestrator).

    Probes with needs_folds=True are silently skipped when folds is None --
    unchanged from before 1b. needs_model/needs_matrix/needs_budget probes are
    NOT skipped when their inputs are None (unlike needs_folds): those kwargs
    are still passed through as None, and it's the probe's own job to decide
    whether it can do useful work without them. No registered probe declares
    any of these three flags yet, so this path is exercised only by tests
    until step 1c adds a probe that does.

    Each probe may return either a list[IssueRecord] or a single IssueRecord;
    both are handled uniformly. A probe that raises is isolated -- caught here,
    turned into a _ProbeFailure record naming the probe and the exception, and
    the remaining probes still run. One probe's bug can never take down the
    whole audit.

    col_pos (feature name -> column position in `all_features`/`X_all`) is
    built once here, by POSITION not by any implicit column-name matching, so
    a needs_matrix probe can slice a single column via
    `X_all[:, [col_pos[name]]]` -- boolean masks are deliberately not offered
    here: some subset-selection patterns reorder columns relative to
    `all_features` (see the ablation.py matrix-hoist work), and column order
    is not interchangeable for every allowlisted estimator (RandomForest's
    fitted predictions change under a column reorder even with a fixed
    random_state; verified empirically before that hoist was implemented).

    time_budget_seconds, when given, becomes an absolute monotonic deadline
    passed to needs_budget probes. This is a SOFT, cooperative budget -- there
    is no preemptive kill here; a probe that ignores its deadline simply runs
    to completion. The mechanism is wired now, ahead of any probe using it,
    per UPGRADE1_PREREGISTRATION.md's resilience requirements.
    """
    _PROBES = _build_probe_registry()

    col_pos = {f: i for i, f in enumerate(all_features)} if all_features is not None else {}
    deadline = (
        time.monotonic() + time_budget_seconds if time_budget_seconds is not None else None
    )
    # A needs_model probe must receive a USABLE factory, never a bare None --
    # resolve the same way metrics._default_model_factory /
    # ablation._default_factory already do, so "no --estimator given" means
    # the same concrete estimator here as it does everywhere else in the audit.
    if model_factory is None and any(spec.needs_model for spec in _PROBES):
        from zekan.severity.metrics import _default_model_factory
        model_factory = _default_model_factory

    found: list = []
    for spec in _PROBES:
        if spec.needs_folds and not folds:
            continue

        kwargs: dict = {}
        if spec.needs_folds:
            kwargs["folds"] = folds
        if spec.needs_model:
            kwargs["model_factory"] = model_factory
            kwargs["n_jobs"] = n_jobs
        if spec.needs_matrix:
            kwargs["X_all"] = X_all
            kwargs["y_all"] = y_all
            kwargs["col_pos"] = col_pos
        if spec.needs_budget:
            kwargs["deadline"] = deadline

        try:
            result = spec.fn(df, contract, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- isolation is the point; any probe, any exception
            probe_name = getattr(spec.fn, "__name__", repr(spec.fn))
            found.append(_ProbeFailure(
                what=(
                    f"Structural probe '{probe_name}' failed to run "
                    f"({type(exc).__name__}: {exc}) -- this finding was skipped, "
                    "not a data property that was checked and found clean."
                ),
                probe_name=probe_name,
                exception_type=type(exc).__name__,
                exception_message=str(exc) or repr(exc),
                # Captured in the record itself rather than routed to a logger --
                # this project has no logging convention yet, and inventing one
                # is out of scope for this step. Not rendered in text/HTML today
                # (only `.what` is), but present in the JSON for 1c to build on.
                traceback_text=traceback.format_exc(),
            ))
            continue

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
    null_stopping: str = "fixed_v1",
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
        engine uses its internal default (see estimators.DEFAULT_ESTIMATOR_NAME;
        histgb as of Tier 3 Phase C).
    n_permutations
        Permutation null draws.  Default 100 (the recommended minimum for a
        defensible detection verdict).  Pass 0 to skip the null — detection will
        be False and high fl will appear as UNCONFIRMED_HIGH_DAMAGE. Ignored when
        null_stopping="sequential_v1" (see below).
    null_seed
        RNG seed for the permutation null.
    warn_floor
        fl threshold for WARN in policy_decision; default 0.10.
    fail_floor
        fl threshold for FAIL in policy_decision; default 0.15.
    policy_profile
        Label for the policy block; use "default_auc" or a domain string.
    null_stopping
        "fixed_v1" (default) or "sequential_v1" (Tier 2 Besag-Clifford +
        decision-stability adaptive stopping) — see
        engine.run_severity_analysis / null_baseline.estimate_fixable_leakage_null.

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
        null_stopping=null_stopping,
    )
    report = build_verdict(
        severity_result,
        warn_floor=warn_floor,
        fail_floor=fail_floor,
        policy_profile=policy_profile,
        min_valid_folds=config.split_policy.min_valid_folds,
    )

    annotations = _run_structural_probes(
        df, contract, folds=severity_result.folds,
        model_factory=model_factory, n_jobs=n_jobs,
        X_all=severity_result.X_all, y_all=severity_result.y_all,
        all_features=severity_result.all_features,
    )
    if annotations:
        report = report.model_copy(update={"structural_annotations": annotations})

    return report
