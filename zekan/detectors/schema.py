"""Normalized issue record emitted by every detector, probe, and the severity engine.

Every audit finding — whether a measured AUC gap, a structural data property, or a
declared scope gap — is one IssueRecord.  The report layer consumes only this type;
detectors and the engine produce only this type.

Severity is intrinsic to the check class, not to the per-run outcome
----------------------------------------------------------------------
severity encodes "how serious is this category of finding" — independent of whether
the check fired on this dataset.  A passing TEMPORAL_LEAKAGE record carries
severity=HIGH, status=pass: the stakes of the check are high, and it was clean.
This is more informative than a synthetic INFO level (which would discard the class
stake information from passing records).

Magnitude for ENGINE_MEASURED issues lives in evidence.measured_value and how_much,
not in severity.  CRITICAL is reserved for structural contamination (ENTITY_CONTAMINATION)
where the model literally memorised test-label-bearing entities — categorically worse
than any measured AUC gap.  This gives CRITICAL unambiguous meaning.

source_layer and severity are derived, not settable
----------------------------------------------------
Both fields are @computed_field properties on IssueRecord, derived from issue_type via
_REGISTRY.  Callers never pass them; passing them as kwargs is silently ignored
(model_config extra='ignore').  JSON round-trips correctly: serialization includes the
derived values; deserialization re-derives them from issue_type (the serialized values
are extra fields, ignored).

Incoherence is therefore impossible at construction time — not documented convention.
The two free dimensions per run are: status (gate outcome) and evidence (measurements).

Structural detail is typed, not a free dict
-------------------------------------------
Each structural detail type is a small Pydantic model with a Literal 'kind'
discriminator.  Dicts with a docstring convention are the Case-4-pattern disease
(soft contract that lets a silently wrong implementation pass).  With three Phase-3
probes, the typing cost is ~20 lines; the drift prevention is permanent.

Reserved fields
---------------
null_95th, p_value  : permutation null baseline — not implemented in v1; reserved to
                      avoid retrofitting when the engine hardening pass arrives.
code_location       : reserved for v1.1 AST scan.  Adding Optional[CodeLocation] = None
                      to Evidence is a non-breaking Pydantic addition.

--verbose rendering assumption
-------------------------------
The four narrative slots (what, why, how_much, next_fix) are pre-rendered strings.
This is sound only if --verbose adds evidence fields to the display without rewording
the narrative.  If --verbose must reword (different wording for manager vs engineer),
the schema needs what_short/what_long pairs instead.  Lock the --verbose spec first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ── Classification enums ──────────────────────────────────────────────────────

class IssueType(str, Enum):
    """Check category.  Each value is one row in the coverage matrix.

    Adding a new probe requires adding an entry here AND in _REGISTRY before writing
    any probe logic — the enum is the scope-declaration forcing function.
    """
    # Engine-measured (A/B/C decomposition)
    TEMPORAL_LEAKAGE = "temporal_leakage"
    # Ablation heuristic flag
    CORRELATED_LEAK_PAIR = "correlated_leak_pair"
    # Phase-3 data probes
    ROW_DUPLICATION = "row_duplication"
    CROSS_FOLD_DUPLICATE = "cross_fold_duplicate"
    ENTITY_CONTAMINATION = "entity_contamination"
    ENTITY_CONTAMINATION_RISK = "entity_contamination_risk"
    WRONG_SPLIT_STRATEGY = "wrong_split_strategy"
    # Structural aggregate probe
    FORBIDDEN_ENTITY_LEVEL_AGGREGATE = "forbidden_entity_level_aggregate"
    # Upgrade 1: undeclared-feature screen (univariate AUC, annotate-only).
    # Both tiers are FLAGGED_SUSPICIOUS/confirmed=False -- a univariate score is
    # suggestive, not a confirmed statistical gate the way the permutation-null
    # backed TEMPORAL_LEAKAGE is. See UPGRADE1_PREREGISTRATION.md.
    SUSPECTED_UNDECLARED_LEAK = "suspected_undeclared_leak"
    NEAR_CERTAIN_UNDECLARED_LEAK = "near_certain_undeclared_leak"
    # Internal integrity self-check (Zekan's own splitter, not a user finding)
    SPLITTER_CONTRACT_VIOLATION = "splitter_contract_violation"
    # Internal integrity self-check: a structural probe raised during execution
    # and was isolated (audit._run_structural_probes) rather than propagating.
    # Zekan reporting its own failure, not a finding about the user's data --
    # same category as SPLITTER_CONTRACT_VIOLATION.
    PROBE_FAILED = "probe_failed"
    # v1.1 scope (emitted as OUT_OF_SCOPE for coverage-matrix transparency)
    CODE_STRUCTURAL_LEAK = "code_structural_leak"


class SourceLayer(str, Enum):
    """Which layer of the audit produced this record.

    ENGINE_MEASURED      A/B/C statistical measurement (fixable_leakage, etc.)
    DETECTED_STRUCTURAL  Deterministic data-probe finding (binary structural fact)
    FLAGGED_SUSPICIOUS   Heuristic flag: suggestive but not a hard statistical gate
    OUT_OF_SCOPE         Category not checked in this version; emitted for coverage matrix
    """
    ENGINE_MEASURED = "engine_measured"
    DETECTED_STRUCTURAL = "detected_structural"
    FLAGGED_SUSPICIOUS = "flagged_suspicious"
    ZEKAN_INTEGRITY = "zekan_integrity"
    OUT_OF_SCOPE = "out_of_scope"


class IssueSeverity(str, Enum):
    """Intrinsic severity of the check class.  Derived from issue_type; not per-run.

    CRITICAL  Structural contamination (entity contamination): model memorised test labels.
              Categorically worse than any measured AUC gap.
    HIGH      Clear-leak-band checks (temporal leakage, row duplication, wrong split,
              code structural leak).  Serious; requires remediation.
    MEDIUM    Heuristic flags (correlated leak pair): evidence is suggestive, not definitive.
    LOW       Currently unused in v1; reserved for near-noise informational findings.

    Magnitude for ENGINE_MEASURED issues lives in evidence.measured_value and how_much.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    # INFO intentionally absent: "no finding" is status=pass, not a severity level.


class EvidenceScope(str, Enum):
    """Where the evidence for an issue class originates.

    DATA_ONLY       Derived entirely from the dataset's structure or content (no folds needed).
    CROSS_FOLD      Requires comparing train and test partitions across evaluation folds.
    ENGINE_MEASURED Derived from statistical engine output (AUC gaps, ablation runs).
    CODE_ANALYSIS   Requires inspecting pipeline source code (v1.1+).
    OUT_OF_SCOPE    Not available in this version.
    """
    DATA_ONLY = "data_only"
    CROSS_FOLD = "cross_fold"
    ENGINE_MEASURED = "engine_measured"
    CODE_ANALYSIS = "code_analysis"
    SELF_CHECK = "self_check"
    OUT_OF_SCOPE = "out_of_scope"


class ImpactType(str, Enum):
    """Category of impact this check class guards against.

    MEASUREMENT_ERROR  Check guards against inflated or invalid performance estimates.
    STRUCTURAL_RISK    Check guards against a data or pipeline structure that creates
                       leakage risk, even if no direct measurement error is confirmed.
    HEURISTIC          Check flags a suspicious pattern; causation is not confirmed.
    OUT_OF_SCOPE       Not applicable in this version.
    """
    MEASUREMENT_ERROR = "measurement_error"
    STRUCTURAL_RISK = "structural_risk"
    HEURISTIC = "heuristic"
    OUT_OF_SCOPE = "out_of_scope"


# ── Issue class registry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class _IssueClass:
    """Frozen per-class properties derived from issue_type."""
    source_layer: SourceLayer
    severity: IssueSeverity
    evidence_scope: EvidenceScope
    impact_type: ImpactType
    confirmed: bool
    # confirmed=False marks advisory probes: the probe confirms a structural data
    # property (e.g. longitudinal recurrence) but cannot assert that an error was made.


_REGISTRY: dict[IssueType, _IssueClass] = {
    # fmt: off
    IssueType.TEMPORAL_LEAKAGE:          _IssueClass(SourceLayer.ENGINE_MEASURED,     IssueSeverity.HIGH,     EvidenceScope.ENGINE_MEASURED, ImpactType.MEASUREMENT_ERROR, True),
    IssueType.CORRELATED_LEAK_PAIR:      _IssueClass(SourceLayer.FLAGGED_SUSPICIOUS,  IssueSeverity.MEDIUM,   EvidenceScope.ENGINE_MEASURED, ImpactType.HEURISTIC,         False),
    IssueType.ROW_DUPLICATION:           _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.HIGH,     EvidenceScope.DATA_ONLY,       ImpactType.MEASUREMENT_ERROR, True),
    IssueType.CROSS_FOLD_DUPLICATE:      _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.HIGH,     EvidenceScope.CROSS_FOLD,      ImpactType.MEASUREMENT_ERROR, True),
    IssueType.ENTITY_CONTAMINATION:      _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.CRITICAL, EvidenceScope.CROSS_FOLD,      ImpactType.STRUCTURAL_RISK,   True),
    IssueType.ENTITY_CONTAMINATION_RISK: _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.HIGH,     EvidenceScope.DATA_ONLY,       ImpactType.STRUCTURAL_RISK,   False),
    IssueType.WRONG_SPLIT_STRATEGY:                  _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.HIGH,     EvidenceScope.DATA_ONLY,       ImpactType.STRUCTURAL_RISK,   True),
    IssueType.FORBIDDEN_ENTITY_LEVEL_AGGREGATE:      _IssueClass(SourceLayer.DETECTED_STRUCTURAL, IssueSeverity.HIGH,     EvidenceScope.DATA_ONLY,       ImpactType.STRUCTURAL_RISK,   True),
    # Upgrade 1 -- both mirror CORRELATED_LEAK_PAIR on source_layer/evidence_scope/
    # impact_type/confirmed: same mechanism (fits a model via the engine's own
    # evaluation harness, produces a suggestive not causally-confirmed AUC-based
    # signal). Differ only on severity, mirroring how ENTITY_CONTAMINATION sits
    # above ENTITY_CONTAMINATION_RISK -- NEAR_CERTAIN is HIGH, not CRITICAL:
    # CRITICAL is reserved for silent-corruption-class findings (see the class
    # docstring above); an undeclared-leak flag is loud and annotate-only, it
    # never silently corrupts engine_detection/measured_damage/policy_decision.
    IssueType.SUSPECTED_UNDECLARED_LEAK:              _IssueClass(SourceLayer.FLAGGED_SUSPICIOUS, IssueSeverity.MEDIUM,   EvidenceScope.ENGINE_MEASURED, ImpactType.HEURISTIC,         False),
    IssueType.NEAR_CERTAIN_UNDECLARED_LEAK:           _IssueClass(SourceLayer.FLAGGED_SUSPICIOUS, IssueSeverity.HIGH,     EvidenceScope.ENGINE_MEASURED, ImpactType.HEURISTIC,         False),
    IssueType.SPLITTER_CONTRACT_VIOLATION:           _IssueClass(SourceLayer.ZEKAN_INTEGRITY,    IssueSeverity.CRITICAL, EvidenceScope.SELF_CHECK,      ImpactType.MEASUREMENT_ERROR, True),
    # Mirrors SPLITTER_CONTRACT_VIOLATION on source_layer (ZEKAN_INTEGRITY --
    # this is Zekan reporting its own execution failure, not a user-data
    # finding) and evidence_scope (SELF_CHECK) and confirmed (True -- "this
    # probe raised this exact exception" is a deterministic fact, not a
    # heuristic or an advisory). Diverges on severity (HIGH, not CRITICAL --
    # a probe crash is loud, isolated, and annotate-only; it never silently
    # corrupts another measurement the way an undetected splitter contract
    # violation would, so it doesn't earn CRITICAL's reserved meaning) and on
    # impact_type (STRUCTURAL_RISK, not MEASUREMENT_ERROR -- a probe crash
    # doesn't touch engine-computed performance numbers at all; it means a
    # structural-risk check didn't run this time, the same impact_type every
    # other structural-risk probe in this registry already uses).
    IssueType.PROBE_FAILED:                           _IssueClass(SourceLayer.ZEKAN_INTEGRITY,    IssueSeverity.HIGH,     EvidenceScope.SELF_CHECK,      ImpactType.STRUCTURAL_RISK,   True),
    IssueType.CODE_STRUCTURAL_LEAK:                  _IssueClass(SourceLayer.OUT_OF_SCOPE,        IssueSeverity.HIGH,     EvidenceScope.CODE_ANALYSIS,   ImpactType.STRUCTURAL_RISK,   True),
    # fmt: on
}


# ── Typed structural detail models ────────────────────────────────────────────
# Each model carries a Literal 'kind' discriminator so Pydantic can deserialise
# ProbeDetail unambiguously without left-to-right trial-and-error.

class RowDuplicationDetail(BaseModel):
    """Evidence for ROW_DUPLICATION: excess copies in the raw dataset."""
    kind: Literal["row_duplication"] = "row_duplication"
    duplicate_rows: int          # excess copies (N identical rows -> N-1 excess)
    duplicate_fraction: float    # duplicate_rows / total_rows


class CrossFoldDuplicateDetail(BaseModel):
    """Evidence for CROSS_FOLD_DUPLICATE: test rows whose content also appears in their fold's train set."""
    kind: Literal["cross_fold_duplicate"] = "cross_fold_duplicate"
    duplicate_rows: int              # total test rows whose content-hash appears in their fold's train set
    duplicate_fraction: float        # duplicate_rows / total test rows across non-skipped folds
    affected_folds: list[int]        # fold indices where contamination was detected
    rows_per_fold: dict[int, int]    # fold_idx -> contaminated test row count


class EntityContaminationDetail(BaseModel):
    """Evidence for ENTITY_CONTAMINATION: entities that appear in both train and test."""
    kind: Literal["entity_contamination"] = "entity_contamination"
    contaminated_entities: int      # count of entities that appear in both partitions
    contamination_fraction: float   # contaminated_entities / total_entities
    affected_folds: list[int]       # fold indices where contamination was detected


class EntityContaminationRiskDetail(BaseModel):
    """Evidence for ENTITY_CONTAMINATION_RISK: longitudinal data structure advisory.

    This probe confirms a structural data property (entities recur across periods),
    not that the user applied a wrong split.  confirmed=False in the registry reflects
    that distinction.  All fields are measurable directly from the dataset.
    """
    kind: Literal["entity_contamination_risk"] = "entity_contamination_risk"
    recurring_entities: int              # entities appearing in >1 distinct prediction period
    total_entities: int                  # total distinct entities in the dataset
    entity_recurrence_fraction: float    # recurring_entities / total_entities
    median_obs_per_entity: float         # median observation count per entity
    max_obs_per_entity: int              # maximum observation count across all entities
    distinct_periods: int                # number of distinct prediction_time values


class WrongSplitStrategyDetail(BaseModel):
    """Evidence for WRONG_SPLIT_STRATEGY: CV strategy mismatched to data structure."""
    kind: Literal["wrong_split_strategy"] = "wrong_split_strategy"
    detected_strategy: str      # what the user is doing (e.g. "random_grouped_cv")
    recommended_strategy: str   # what they should do (e.g. "temporal_expanding_cv")
    reason: str                 # one sentence explaining the mismatch


class CorrelatedLeakPairDetail(BaseModel):
    """Evidence for CORRELATED_LEAK_PAIR: ablation understatement from shared latent source."""
    kind: Literal["correlated_leak_pair"] = "correlated_leak_pair"
    individual_leakages: list[float]   # per-feature leakage_estimate from individual ablation
    cumulative_leakage: float          # leakage when all flagged features are dropped together
    understatement_ratio: float        # cumulative_leakage / max(individual_leakages)


class SplitterContractViolationDetail(BaseModel):
    """Evidence for SPLITTER_CONTRACT_VIOLATION: Zekan's own grouped splitter self-check.

    This probe checks Zekan's internal invariant: the grouped splitter must never
    place the same entity in both train and test within the same fold.  If it fires,
    the bug is in Zekan, not the user's data.
    """
    kind: Literal["splitter_contract_violation"] = "splitter_contract_violation"
    affected_folds: list[int]                  # fold indices where entity appears in both partitions
    spanning_entities_per_fold: dict[int, int] # fold_idx -> count of entities that span train+test
    total_folds_checked: int                   # non-skipped folds evaluated
    total_entities_checked: int                # distinct entities in the dataset


class ForbiddenEntityLevelAggregateDetail(BaseModel):
    """Evidence for FORBIDDEN_ENTITY_LEVEL_AGGREGATE: constant-within-entity forbidden feature.

    Confirmed structural fact: the feature takes exactly one unique value within every
    repeated entity AND varies across entities.  This is consistent with an entity-level
    aggregate or pre-split artifact whose leakage the within-entity permutation null
    cannot statistically confirm (the null is a no-op on constant-within-entity values).
    confirmed=True in the registry reflects that the STRUCTURAL PATTERN is certain;
    metric impact is annotate-only — the policy verdict is never changed.
    """
    kind: Literal["forbidden_entity_level_aggregate"] = "forbidden_entity_level_aggregate"
    feature: str                        # name of the forbidden feature
    entity_col: str                     # entity_id column from the contract
    eligible_entities: int              # count of repeated entities (>= 2 observations)
    within_entity_constant: bool        # always True when the probe fires
    between_entity_unique_count: int    # distinct feature values across entity representatives
    statistical_confirmation: str       # always "not_required"
    verdict_effect: str                 # always "annotate_only"


class _UndeclaredLeakDetailBase(BaseModel):
    """Shared fields between SuspectedUndeclaredLeakDetail and
    NearCertainUndeclaredLeakDetail (Upgrade 1 -- the undeclared-feature
    screen; see UPGRADE1_PREREGISTRATION.md). NOT itself a ProbeDetail Union
    member -- only the two concrete subclasses below are. Introduced purely to
    avoid duplicating six field definitions across two closely related
    structs; `kind` stays a fixed, per-subclass Literal on each concrete
    class (never a field defined here), so the discriminator can never
    disagree with which subclass -- and therefore which IssueType -- a record
    actually carries.
    """
    feature: str                        # the non-forbidden feature that was scored
    univariate_auc: float               # measured univariate AUC on temporal folds
    threshold_compared_against: float   # the calibrated threshold this score was judged against
    n_folds_evaluated: int              # temporal folds actually used for this feature's score
    screened_count: int                 # features screened this run ("screened X of Y")
    total_features: int                 # Y in "screened X of Y"


class SuspectedUndeclaredLeakDetail(_UndeclaredLeakDetailBase):
    """Evidence for SUSPECTED_UNDECLARED_LEAK: a non-forbidden feature's
    univariate AUC clears the Benjamini-Hochberg FDR-controlled threshold.
    Suggestive, not confirmed -- confirmed=False in the registry.
    """
    kind: Literal["suspected_undeclared_leak"] = "suspected_undeclared_leak"
    suppressed_by_known_strong_features: bool
    # True when this feature is named in the contract's known_strong_features
    # allowlist: the record still fires (coverage-matrix completeness -- every
    # screened feature gets a record) but with status="pass" instead of a live
    # flag, per UPGRADE1_PREREGISTRATION.md. False for every genuinely-flagged
    # SUSPECTED record.


class NearCertainUndeclaredLeakDetail(_UndeclaredLeakDetailBase):
    """Evidence for NEAR_CERTAIN_UNDECLARED_LEAK: a non-forbidden feature's
    univariate AUC clears the absolute (non-FDR, non-percentile) near-1.0
    criterion -- the regime where the feature is functionally a copy of the
    target. Never suppressible by known_strong_features (nothing legitimate
    scores this high), so there is deliberately no waiver field here.
    """
    kind: Literal["near_certain_undeclared_leak"] = "near_certain_undeclared_leak"
    name_pattern_score: float
    # Corroboration only (Upgrade 1 step 1e) -- reuses ablation.py's existing
    # _name_score/_SUSPICIOUS_PATTERNS machinery verbatim (suspicious
    # temporal-keyword patterns: final_*, days_to_*, future_*, next_*
    # prefixes; _after_/_future/_next_period/_lag0/_t0 suffixes). 0.0 or 1.0.
    # Never gates the finding -- NEAR_CERTAIN already fired on the AUC
    # criterion alone before this is even computed. NOTE: this does NOT
    # detect feature-vs-target NAME similarity (e.g. B-3's 'readmitted' vs
    # 'readmitted_lt30' scores 0.0 here -- verified empirically, see
    # UPGRADE1_CALIBRATION.md's step-1e read-first notes); it only matches a
    # fixed list of suspicious keyword shapes.


class ProbeFailedDetail(BaseModel):
    """Evidence for PROBE_FAILED: a structural probe raised during execution
    and was isolated (audit._run_structural_probes) rather than propagating.
    This is Zekan reporting its own execution failure, not a finding about
    the user's data -- see SplitterContractViolationDetail for the same
    distinction.

    Deliberately no raw-traceback field: no existing detail struct in this
    schema carries an unstructured diagnostic blob (this schema's own stated
    principle is against exactly this "Case-4-pattern" free-text soft
    contract -- see the module docstring), so this stays with short,
    structured fields only, matching every other detail struct here.
    """
    kind: Literal["probe_failed"] = "probe_failed"
    probe_name: str
    exception_type: str
    message: str


ProbeDetail = Annotated[
    Union[
        RowDuplicationDetail,
        CrossFoldDuplicateDetail,
        EntityContaminationDetail,
        EntityContaminationRiskDetail,
        WrongSplitStrategyDetail,
        CorrelatedLeakPairDetail,
        SplitterContractViolationDetail,
        ForbiddenEntityLevelAggregateDetail,
        SuspectedUndeclaredLeakDetail,
        NearCertainUndeclaredLeakDetail,
        ProbeFailedDetail,
    ],
    Field(discriminator="kind"),
]


# ── Evidence container ────────────────────────────────────────────────────────

class Evidence(BaseModel):
    """Backing data for one IssueRecord.  Not all fields apply to every issue type.

    Fields are additive and all Optional.  The report renderer checks which fields
    are populated to decide how to display the finding.
    """
    model_config = ConfigDict(extra="forbid")

    # Quantitative (ENGINE_MEASURED issues)
    measured_value: Optional[float] = None
    # The scalar that drove the gate decision (e.g. fixable_leakage = 0.087).

    threshold: Optional[float] = None
    # The gate value measured_value was compared against (e.g. 0.04 clear-leak floor).

    metric_name: Optional[str] = None
    # Human-readable metric name (e.g. "fixable_leakage", "roc_auc").

    # Permutation null baseline — reserved; not implemented in v1.
    null_95th: Optional[float] = None
    # 95th percentile of the null distribution.  When populated, measured_value below
    # null_95th is indistinguishable from chance.

    p_value: Optional[float] = None
    # p(observed >= measured_value | null true).  One-tailed; companion to null_95th.

    # Structural (DETECTED_STRUCTURAL / FLAGGED_SUSPICIOUS issues)
    structural_detail: Optional[ProbeDetail] = None
    # Typed detail model; kind discriminator identifies which probe emitted it.

    # code_location: Optional[CodeLocation] = None
    # Reserved for v1.1 AST scan.  Non-breaking addition when the time comes.


# ── Issue record ──────────────────────────────────────────────────────────────

class IssueRecord(BaseModel):
    """One audit finding.  Emitted by detectors, probes, and the severity engine alike.

    source_layer and severity are @computed_field properties derived from issue_type
    via _REGISTRY.  They cannot be meaningfully set by the caller: the before-validator
    strips every computed-field name from the input before Pydantic's extra='forbid'
    sees it, so round-trip JSON (which includes these fields) loads cleanly, while any
    genuine typo on a non-computed name raises ValidationError.

    The before-validator reads cls.model_computed_fields — it is self-maintaining.
    Adding a new @computed_field to IssueRecord automatically extends what gets stripped;
    no manual list to update.

    Every check that runs produces one IssueRecord — including checks that pass.
    Passing checks: status='pass', severity = class severity (not INFO).
    Out-of-scope checks: status='unavailable', source_layer=OUT_OF_SCOPE.
    This makes the coverage matrix complete from a single pass over IssueRecord list.
    """
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _strip_computed(cls, data: Any) -> Any:
        """Strip computed-field names from the input dict before extra='forbid' validates.

        This is the only way to have both:
          - extra='forbid'  (typos raise ValidationError)
          - @computed_field  (fields appear in JSON output and need round-trip)

        Passing a computed-field name as a kwarg is silently discarded; the registry
        value always wins.  This is correct: callers should never set derived fields.
        """
        if isinstance(data, dict):
            computed = cls.model_computed_fields
            if computed:
                data = {k: v for k, v in data.items() if k not in computed}
        return data

    # -- Settable per-run fields ----------------------------------------------

    issue_type: IssueType
    # Identifies the check category; drives both computed fields and the coverage matrix.

    status: Literal["pass", "note", "warn", "fail", "internal_fail", "unavailable"]
    # Gate outcome for this run.
    # 'note'       : statistically confirmed (p < 0.01) but effect size is small (NSL < 1.0).
    # 'unavailable': either insufficient data to run, or source_layer == OUT_OF_SCOPE.

    what: str
    # One sentence: what was found (or confirmed absent, for passing records).

    why: str
    # One sentence: why this matters for deployable prediction quality.

    how_much: str
    # Human-readable quantification.
    # ENGINE_MEASURED pass:  "fixable_leakage = +0.008; null_99th = +0.018; p=0.4231 >= 0.01 (inside null — PASS)"
    # ENGINE_MEASURED fail:  "fixable_leakage = +0.087; null_99th = +0.015; p=0.0099 < 0.01 (outside null); NSL=9.60"
    # STRUCTURAL fail:       "47 of 500 entities (9.4%) appear in both train and test sets"

    next_fix: str
    # Concrete action.  "No action required." for passing records.

    evidence: Evidence = Field(default_factory=Evidence)
    # Numeric and structural backing data.  Always present; fields within it are Optional.

    # -- Derived fields (not settable) ----------------------------------------

    @computed_field
    @property
    def source_layer(self) -> SourceLayer:
        """Derived from issue_type via _REGISTRY.  Cannot be overridden."""
        return _REGISTRY[self.issue_type].source_layer

    @computed_field
    @property
    def severity(self) -> IssueSeverity:
        """Intrinsic class severity, derived from issue_type via _REGISTRY.

        Reflects the stakes of the check category, independent of whether it fired.
        Magnitude for ENGINE_MEASURED issues lives in evidence.measured_value.
        """
        return _REGISTRY[self.issue_type].severity

    @computed_field
    @property
    def evidence_scope(self) -> EvidenceScope:
        """Where this check's evidence originates.  Derived from issue_type via _REGISTRY."""
        return _REGISTRY[self.issue_type].evidence_scope

    @computed_field
    @property
    def impact_type(self) -> ImpactType:
        """Category of risk this check guards against.  Derived from issue_type via _REGISTRY."""
        return _REGISTRY[self.issue_type].impact_type

    @computed_field
    @property
    def confirmed(self) -> bool:
        """True if the probe confirms an error occurred; False if it is a risk advisory.

        Only ENTITY_CONTAMINATION_RISK is confirmed=False in v1: it observes that the
        dataset is longitudinal (a measurable data property) but cannot observe whether
        the user applied a random split.  Every other probe either confirms a measured
        AUC gap, a content-hash collision, or a structural data fact.
        """
        return _REGISTRY[self.issue_type].confirmed
