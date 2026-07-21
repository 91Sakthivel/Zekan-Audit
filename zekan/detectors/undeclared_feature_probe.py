"""NEAR_CERTAIN_UNDECLARED_LEAK probe + ranked informational panel.

Upgrade 1 step 1e. Screens every non-forbidden feature's univariate AUC on
TEMPORAL folds only (never random, never in-sample -- the pre-registered
invariant; see UPGRADE1_PREREGISTRATION.md's "Method, locked before
calibration" section). Two outputs, one scoring pass:

  1. NEAR_CERTAIN_UNDECLARED_LEAK IssueRecords -- the earned hard signal.
     Absolute, tie-robust: EVERY evaluated fold's univariate AUC must clear
     NEAR_CERTAIN_AUC_FLOOR. Flags every feature meeting the criterion, not
     just the top-ranked one -- a dataset can contain more than one
     target-adjacent column (see UPGRADE1_PREREGISTRATION.md's own B-3
     motivation).
  2. A ranked informational panel (UndeclaredFeaturePanel) -- NOT an
     IssueRecord; no issue is asserted. This is what replaces the
     SUSPECTED_UNDECLARED_LEAK tier: UPGRADE1_CALIBRATION.md's step-1d
     measurement found that tier's pre-registered design (Benjamini-Hochberg
     FDR over per-feature univariate-AUC-vs-0.5 p-values) fails its own
     pre-committed falsification condition -- BH-FDR flags number_inpatient
     (a genuinely honest, strong predictor) at every tested q, at both n,
     structurally (it is always the single most significant p-value, and BH
     always includes that rank whenever it flags anything at all). That tier
     is therefore DEFERRED, not implemented here. SuspectedUndeclaredLeakDetail
     stays registered in schema.py, unused, for a future calibrated redesign.
     The panel is a plain top-N-by-AUC candidate list, carrying no threshold
     and no suspicion claim -- a starting point for a human to review,
     optionally declare forbidden, and re-run through Zekan's own B/C
     severity engine, not a verdict.

Name-pattern corroboration (never a gate on its own): reuses ablation.py's
existing _name_score/_SUSPICIOUS_PATTERNS verbatim, carried on NEAR_CERTAIN
detail records and on every panel entry. Note: this machinery matches a
FIXED set of suspicious keyword shapes (final_*, days_to_*, future_*, next_*
prefixes; _after_/_future/_next_period/_lag0/_t0 suffixes) -- it does NOT
detect feature-vs-target name similarity. B-3's own anchor ('readmitted' vs
target 'readmitted_lt30') scores name_pattern_score=0.0 under this exact
machinery -- verified empirically before use here, not assumed. It is
reused as-is (not extended) per the task's explicit instruction to reuse
the EXISTING machinery; this gap is recorded, not silently patched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from zekan.contract.contract_checks import candidate_features
from zekan.contract.prediction_contract import PredictionContract
from zekan.detectors.schema import (
    Evidence,
    IssueRecord,
    IssueType,
    NearCertainUndeclaredLeakDetail,
)
from zekan.severity.ablation import _name_score
from zekan.severity.metrics import evaluate_folds
from zekan.severity.splitters import FoldIndices

SCREEN_VERSION: str = "univariate_v1"
"""Provenance version string for this screen (threaded like null_scheme /
null_stopping -- see reports/provenance.py). Bump when the scoring method,
the NEAR_CERTAIN criterion, or the screenability/pre-rank rules change in a
way that makes two runs' screen output not directly comparable."""

# NEAR_CERTAIN absolute criterion: every evaluated fold's univariate AUC must
# be >= this floor. Derived from UPGRADE1_CALIBRATION.md (Upgrade 1 step 1d):
# the honest-tail ceiling (real B-2 Diabetes-130 features + synthetic F2b
# fixtures, both n=10,008 and n=101,766) topped out at 0.7630
# (F2b's feature_0); the B-3 leak anchor ('readmitted' vs 'readmitted_lt30')
# scored EXACTLY 1.0 in every one of 5 folds, at both n. 0.99 sits deep
# inside that ~0.24-wide, completely empty gap -- comfortably above every
# honest score ever observed (real or synthetic), comfortably below 1.0 with
# room for a near-but-imperfect duplicate leak (e.g. B-2's own planted_leak,
# which carries 5% label noise). Per the calibration doc's own honest
# caveat: not tightly pinned by this data (nothing sits close enough to 0.99
# to adjudicate the exact cut), but safely inside a wide empty gap on both
# sides -- same epistemic status F2b_CALIBRATION.md recorded for NSL=1.0.
NEAR_CERTAIN_AUC_FLOOR: float = 0.99

# Screenability gate: a feature is only scored if it clears a minimum-
# information floor (UPGRADE1_PREREGISTRATION.md's "Screenability gate").
# Constants mirror SplitPolicy's own default floors (min_test_rows_per_fold
# =100, min_positive/negative_cases_per_fold=20) -- same order of magnitude,
# reused rather than inventing new numbers -- but applied here to a
# FEATURE's own non-missing subset, not the whole fold (see
# _screenability_reason). Not calibrated against real freak-AUC evidence
# (UPGRADE1_CALIBRATION.md's "what remains unvalidated": this gate was not
# stress-tested there -- Diabetes-130's 97%-missing `weight` column happens
# to already be ordinal-encoded with no real NaNs by the time it reaches the
# audit, so it never exercised this path).
_MIN_NONMISSING_COUNT: int = 100
_MIN_MINORITY_CLASS_COUNT: int = 20

# Wide-data cap: when there are more screenable candidates than this, a
# cheap pre-rank (absolute Pearson correlation with target -- no model fit)
# selects the top WIDE_DATA_CAP for full temporal-fold scoring; the rest are
# excluded by capacity, not screenability, and are never silently dropped --
# UndeclaredFeaturePanel.total_features / screened_count always account for
# them. Chosen above Diabetes-130's 48-feature B-2/B-3 frames (the only
# calibration evidence this screen has), so the calibrated frames always get
# full scoring, uncapped.
WIDE_DATA_CAP: int = 50
PRE_RANK_METHOD: str = "abs_pearson_correlation_with_target"

# Ranked panel size -- matches run_ablation's own top_k=10 default
# (zekan/severity/ablation.py), reused for consistency rather than picking a
# new unrelated number.
PANEL_TOP_N: int = 10


# ── Panel data structures (NOT IssueRecords -- see module docstring) ──────────

@dataclass
class NotScreenableEntry:
    """One feature that failed the screenability gate. Always reported with
    its reason -- never silently dropped from total_features accounting."""
    feature: str
    reason: str


@dataclass
class PanelEntry:
    """One row of the ranked informational panel. No threshold, no
    suspicion claim -- a bare score plus corroboration context."""
    feature: str
    univariate_auc: float
    name_pattern_score: float
    n_folds_evaluated: int


@dataclass
class UndeclaredFeaturePanel:
    """Ranked informational panel for the undeclared-feature screen.

    NOT an IssueRecord -- no issue is asserted here; this is a candidate
    list for a human to review, optionally declare forbidden_after_prediction,
    and re-run (feeding Zekan's own B/C severity engine), not a verdict.
    See VerdictReport.undeclared_feature_panel (severity/verdict.py) for
    where this attaches to the report, and this module's docstring for why.

    screened_count / total_features / not_screenable together are the
    "screened X of Y" accounting UPGRADE1_PREREGISTRATION.md requires so
    silence about a feature is never mistaken for that feature having been
    cleared -- entries only holds the top PANEL_TOP_N by score, but these
    three fields always describe the FULL candidate set.
    """
    entries: list[PanelEntry]
    screened_count: int
    total_features: int
    not_screenable: list[NotScreenableEntry] = field(default_factory=list)
    pre_rank_applied: bool = False
    pre_rank_method: Optional[str] = None
    screen_version: str = SCREEN_VERSION


# ── Screenability gate ──────────────────────────────────────────────────────

def _screenability_reason(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    folds: list[FoldIndices],
) -> Optional[str]:
    """Return a reason string if `feature` fails the minimum-information
    floor, else None.

    Two checks:
      1. Dataset-wide non-missing count >= _MIN_NONMISSING_COUNT.
      2. At least one non-skipped temporal test fold has, among that fold's
         rows where THIS feature is non-missing, >= _MIN_MINORITY_CLASS_COUNT
         of BOTH target classes. A feature can clear check 1 (enough total
         non-missing rows) yet still have every fold's non-missing subset be
         too thin -- e.g. non-missing rows concentrated in one time window --
         to produce a meaningful univariate AUC; this catches that case.
    """
    non_missing = df[feature].notna()
    n_non_missing = int(non_missing.sum())
    if n_non_missing < _MIN_NONMISSING_COUNT:
        return f"only {n_non_missing} non-missing values (< {_MIN_NONMISSING_COUNT})"

    non_missing_arr = non_missing.to_numpy()
    y_arr = df[target_col].to_numpy()

    for fold in folds:
        if fold.meta.skipped:
            continue
        mask = non_missing_arr[fold.test_idx]
        y_test = y_arr[fold.test_idx][mask]
        if len(y_test) == 0:
            continue
        y_test_int = pd.Series(y_test).astype(int)
        n_pos = int((y_test_int == 1).sum())
        n_neg = len(y_test_int) - n_pos
        if n_pos >= _MIN_MINORITY_CLASS_COUNT and n_neg >= _MIN_MINORITY_CLASS_COUNT:
            return None  # at least one fold is viable for this feature

    return (
        f"no temporal test fold has >= {_MIN_MINORITY_CLASS_COUNT} of both "
        f"target classes among this feature's non-missing values"
    )


# ── Wide-data cap: cheap pre-rank ───────────────────────────────────────────

def _pre_rank(df: pd.DataFrame, candidates: list[str], target_col: str) -> list[str]:
    """Cheap correlation-style pass (no model fit): absolute Pearson
    correlation with the target, pairwise-complete (pandas .corr() drops
    NaN pairwise, no imputation). Returns the top WIDE_DATA_CAP candidates,
    descending by |correlation|."""
    y = pd.to_numeric(df[target_col], errors="coerce")
    scores: dict[str, float] = {}
    for feat in candidates:
        x = pd.to_numeric(df[feat], errors="coerce")
        c = x.corr(y)
        scores[feat] = abs(c) if pd.notna(c) else 0.0
    ranked = sorted(candidates, key=lambda f: scores[f], reverse=True)
    return ranked[:WIDE_DATA_CAP]


# ── Per-feature scoring ──────────────────────────────────────────────────────

def _score_one_feature(
    feature: str,
    df: pd.DataFrame,
    target_col: str,
    folds: list[FoldIndices],
    model_factory: Callable[[], Any],
    X_sub: Optional[np.ndarray],
    y_all: Optional[np.ndarray],
) -> tuple[str, float, list[float], int]:
    """Module-level (joblib-picklable, mirrors ablation.py's _ablate_one /
    _marginal_one pattern) -- univariate AUC for one feature on TEMPORAL
    folds only. Returns (feature, mean_auc, per_fold_aucs, n_valid_folds).
    """
    result = evaluate_folds(df, [feature], target_col, folds, model_factory,
                             X_all=X_sub, y_all=y_all)
    fold_aucs = [fe.auc for fe in result.fold_evals]
    return (feature, result.mean_auc, fold_aucs, result.n_valid_folds)


def _build_near_certain_record(
    feature: str,
    univariate_auc: float,
    n_folds_evaluated: int,
    screened_count: int,
    total_features: int,
    name_pattern_score: float,
) -> IssueRecord:
    detail = NearCertainUndeclaredLeakDetail(
        feature=feature,
        univariate_auc=univariate_auc,
        threshold_compared_against=NEAR_CERTAIN_AUC_FLOOR,
        n_folds_evaluated=n_folds_evaluated,
        screened_count=screened_count,
        total_features=total_features,
        name_pattern_score=name_pattern_score,
    )
    return IssueRecord(
        issue_type=IssueType.NEAR_CERTAIN_UNDECLARED_LEAK,
        status="fail",
        what=(
            f"Feature '{feature}' scores univariate AUC={univariate_auc:.4f}; "
            f"EVERY evaluated temporal fold cleared {NEAR_CERTAIN_AUC_FLOOR:.2f} "
            "-- functionally a copy of the target, not a declared forbidden feature."
        ),
        why=(
            "A non-forbidden feature this strongly associated with the target, "
            "measured on TEMPORAL folds (never random, never in-sample), is not "
            "something a legitimate honest predictor produces -- see "
            "UPGRADE1_CALIBRATION.md: the honest-feature ceiling measured across "
            "real (Diabetes-130) and synthetic (F2b) data tops out at 0.7630, far "
            f"below the {NEAR_CERTAIN_AUC_FLOOR:.2f} floor."
        ),
        how_much=(
            f"univariate_auc={univariate_auc:.4f}, every one of "
            f"{n_folds_evaluated} evaluated fold(s) >= {NEAR_CERTAIN_AUC_FLOOR:.2f}; "
            f"screened {screened_count} of {total_features} non-forbidden features "
            f"this run (name_pattern_score={name_pattern_score:.1f}, corroboration "
            "only -- never a gate)."
        ),
        next_fix=(
            f"Inspect '{feature}': if it encodes the target directly or indirectly "
            "(e.g. a near-duplicate column, or derived from post-outcome data), add "
            "it to the contract's forbidden_after_prediction and re-run."
        ),
        evidence=Evidence(
            measured_value=univariate_auc,
            threshold=NEAR_CERTAIN_AUC_FLOOR,
            metric_name="univariate_auc",
            structural_detail=detail,
        ),
    )


# ── Public probe entrypoint (matches the _ProbeSpec calling convention) ───────

def probe_undeclared_feature_screen(
    df: pd.DataFrame,
    contract: PredictionContract,
    folds: Optional[list[FoldIndices]] = None,
    model_factory: Optional[Callable[[], Any]] = None,
    n_jobs: int = 1,
    X_all: Optional[np.ndarray] = None,
    y_all: Optional[np.ndarray] = None,
    col_pos: Optional[dict] = None,
    deadline: Optional[float] = None,
    side_channel: Optional[dict] = None,
) -> list[IssueRecord]:
    """Undeclared-feature screen: TEMPORAL folds only, never rand_folds,
    never in-sample (the pre-registered invariant).

    Returns only NEAR_CERTAIN_UNDECLARED_LEAK IssueRecords (the annotate-only,
    non-verdict-changing hard signal). The ranked informational panel is NOT
    an IssueRecord and cannot flow through this return value -- when
    `side_channel` is given (audit._run_structural_probes passes one through
    for any probe declaring needs_side_channel=True), this probe writes
    side_channel["undeclared_feature_panel"] = UndeclaredFeaturePanel(...)
    before returning. `side_channel=None` (e.g. a direct/test call outside
    the full audit pipeline) means the panel is simply not captured anywhere
    -- the NEAR_CERTAIN records are still returned normally.

    deadline (soft, cooperative time budget) is accepted for calling-
    convention compatibility with needs_budget but not consulted here --
    this screen does not currently self-limit against it. Documented, not
    silently ignored: a future step may wire this in if the wide-data cap
    alone proves insufficient for pathological cases.
    """
    if not folds:
        return []

    if model_factory is None:
        from zekan.severity.metrics import _default_model_factory
        model_factory = _default_model_factory

    candidates = candidate_features(contract, df)
    total_features = len(candidates)

    if total_features == 0:
        if side_channel is not None:
            side_channel["undeclared_feature_panel"] = UndeclaredFeaturePanel(
                entries=[], screened_count=0, total_features=0,
            )
        return []

    # ── screenability gate ────────────────────────────────────────────────────
    not_screenable: list[NotScreenableEntry] = []
    screenable: list[str] = []
    for feat in candidates:
        reason = _screenability_reason(df, feat, contract.target, folds)
        if reason is not None:
            not_screenable.append(NotScreenableEntry(feature=feat, reason=reason))
        else:
            screenable.append(feat)

    # ── wide-data cap: cheap pre-rank when too many screenable candidates ────
    pre_rank_applied = False
    to_score = screenable
    if len(screenable) > WIDE_DATA_CAP:
        to_score = _pre_rank(df, screenable, contract.target)
        pre_rank_applied = True

    y_all_local = y_all if y_all is not None else df[contract.target].to_numpy()
    _col_pos = col_pos or {}

    def _slice(feat: str) -> Optional[np.ndarray]:
        # By explicit column POSITION list, never a boolean mask -- see
        # audit._run_structural_probes' own docstring on why (column order
        # is not interchangeable for every allowlisted estimator).
        if X_all is not None and feat in _col_pos:
            return X_all[:, [_col_pos[feat]]]
        return None  # evaluate_folds rebuilds from df when X_sub is None

    if to_score:
        if n_jobs == 1:
            raw = [
                _score_one_feature(feat, df, contract.target, folds, model_factory,
                                    _slice(feat), y_all_local)
                for feat in to_score
            ]
        else:
            from joblib import Parallel, delayed  # noqa: PLC0415
            raw = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_score_one_feature)(
                    feat, df, contract.target, folds, model_factory,
                    _slice(feat), y_all_local,
                )
                for feat in to_score
            )
    else:
        raw = []

    screened_count = len(raw)

    # ── NEAR_CERTAIN (absolute, tie-robust) + panel entries ──────────────────
    near_certain: list[IssueRecord] = []
    panel_entries: list[PanelEntry] = []
    for feature, mean_auc, fold_aucs, n_folds_evaluated in raw:
        name_score = _name_score(feature)
        panel_entries.append(PanelEntry(
            feature=feature,
            univariate_auc=mean_auc,
            name_pattern_score=name_score,
            n_folds_evaluated=n_folds_evaluated,
        ))
        if n_folds_evaluated > 0 and all(a >= NEAR_CERTAIN_AUC_FLOOR for a in fold_aucs):
            near_certain.append(_build_near_certain_record(
                feature=feature,
                univariate_auc=mean_auc,
                n_folds_evaluated=n_folds_evaluated,
                screened_count=screened_count,
                total_features=total_features,
                name_pattern_score=name_score,
            ))

    panel_entries.sort(key=lambda e: e.univariate_auc, reverse=True)

    if side_channel is not None:
        side_channel["undeclared_feature_panel"] = UndeclaredFeaturePanel(
            entries=panel_entries[:PANEL_TOP_N],
            screened_count=screened_count,
            total_features=total_features,
            not_screenable=not_screenable,
            pre_rank_applied=pre_rank_applied,
            pre_rank_method=PRE_RANK_METHOD if pre_rank_applied else None,
        )

    return near_certain
