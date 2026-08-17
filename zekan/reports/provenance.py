"""Pure provenance helpers — side-effect-free except write_manifest.

No typer, no stdin.  All functions are independently unit-testable.
The timestamp appears ONLY in build_manifest / write_manifest; it is
never injected into the determinism-checked --json body.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pandas as pd
    from zekan.contract.prediction_contract import PredictionContract


def hash_dataframe(df: "pd.DataFrame") -> str:
    """SHA-256 hex of pandas row-hashes.

    Uses pd.util.hash_pandas_object (stable for identical content within the
    same pandas version).  Includes the index so row reorders produce a
    different hash.  Do NOT use to_parquet — not byte-stable across versions.
    """
    import pandas as pd  # noqa: PLC0415

    raw: bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def hash_contract(contract: "PredictionContract") -> str:
    """SHA-256 hex of the contract's Pydantic v2 JSON serialization."""
    raw: bytes = contract.model_dump_json().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def capture_versions() -> dict:
    """Return library versions keyed by package alias.

    Uses importlib.metadata for numpy/pandas/scikit-learn; falls back to
    'unknown' on PackageNotFoundError.  zekan version is read from
    zekan.__version__ directly.
    """
    import importlib.metadata

    import zekan as _zekan  # noqa: PLC0415

    def _ver(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "scikit_learn": _ver("scikit-learn"),
        "zekan": _zekan.__version__,
    }


def read_estimator_random_state(model_factory: Optional[Any]) -> Optional[int]:
    """Instantiate model_factory() and read its random_state attribute.

    Returns None when model_factory is None (caller used the internal default)
    or when instantiation fails for any reason.  Never raises into the audit path.
    """
    if model_factory is None:
        return None
    try:
        est = model_factory()
        rs = getattr(est, "random_state", None)
        return int(rs) if rs is not None else None
    except Exception:
        return None


def build_provenance(
    data_hash: str,
    contract_hash: str,
    versions: dict,
    null_seed: int,
    estimator_identity: str,
    estimator_random_state: Optional[int],
    null_scheme: str = "spawn_v2",
    null_stopping: str = "fixed_v1",
    undeclared_screen: str = "univariate_v1",
    categorical_encoding: Optional[dict] = None,
) -> dict:
    """Return a deterministic provenance dict — NO timestamp.

    null_scheme
        The permutation-null seeding scheme (F2a).  "spawn_v2" is the current
        (SeedSequence-based, n_jobs-independent) scheme.  Additive field — JSON
        produced before F2a has no null_scheme key at all; `zekan diff` treats
        that absence as the retired "serial_v1" scheme.
    null_stopping
        The permutation-null stopping scheme (Tier 2).  "fixed_v1" is the
        original (draw exactly n_permutations) scheme; "sequential_v1" is the
        Besag-Clifford + decision-stability adaptive scheme.  Additive field —
        JSON produced before Tier 2 has no null_stopping key at all; `zekan
        diff` treats that absence as "fixed_v1" (never guessed to match the
        other side).
    undeclared_screen
        The undeclared-feature screen version (Upgrade 1 step 1e).
        "univariate_v1" is the current (univariate-AUC-on-temporal-folds,
        NEAR_CERTAIN-only) screen -- see
        zekan.detectors.undeclared_feature_probe.SCREEN_VERSION, the single
        source of truth this default should track.  Additive field, threaded
        the same way as null_scheme/null_stopping -- JSON produced before
        Upgrade 1 has no undeclared_screen key at all; `zekan diff` treats
        that absence as "none" (no screen ran), never guessed to match the
        other side.  A top-level provenance key (not nested under "seed" --
        unlike null_scheme/null_stopping, this isn't a seeding concern).
    categorical_encoding
        The {column: {"codes": {raw_value: code}, "nan_sentinel": str}}
        ordinal-encoding map actually applied to declared
        categorical_features for this audit (see
        contract_checks.build_categorical_mapping), or None when no
        categorical column was declared/encoded (the default -- every caller
        before CATEGORICAL_SUPPORT_PREREGISTRATION.md step 3 wiring passes
        nothing here).  The per-column nan_sentinel is included alongside
        codes because it is required to reproduce the encoding exactly: it
        is picked per column (collision-avoided against that column's own
        real values, see contract_checks._pick_nan_sentinel), so a reader
        cannot assume a fixed sentinel string -- reproducing which raw value
        a given code came from requires the sentinel actually used for that
        column, not a guessed default.  Additive field -- JSON produced
        before categorical support has no categorical_encoding key at all;
        `zekan diff` should treat that absence as "no encoding was applied"
        (equivalent to None), never guessed to match the other side.
        Without this, a --json result computed from encoded categorical
        columns is not independently reproducible: a reader has no way to
        map the audited float32 codes back to the original category values.
    """
    return {
        "contract_sha256": contract_hash,
        "data_sha256": data_hash,
        "estimator_identity": estimator_identity,
        "seed": {
            "estimator_random_state": estimator_random_state,
            "null_seed": null_seed,
            "null_scheme": null_scheme,
            "null_stopping": null_stopping,
        },
        "undeclared_screen": undeclared_screen,
        "categorical_encoding": categorical_encoding,
        "versions": versions,
    }


def build_manifest(provenance: dict) -> dict:
    """Wrap provenance with a UTC timestamp.

    This is the ONLY place a timestamp appears — never in the determinism-
    checked --json body.
    """
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    """Write manifest as BOM-free UTF-8 JSON with sorted keys."""
    text = json.dumps(manifest, sort_keys=True, indent=2)
    Path(path).write_text(text, encoding="utf-8")
