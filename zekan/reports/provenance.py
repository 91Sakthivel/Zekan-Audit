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
) -> dict:
    """Return a deterministic provenance dict — NO timestamp."""
    return {
        "contract_sha256": contract_hash,
        "data_sha256": data_hash,
        "estimator_identity": estimator_identity,
        "seed": {
            "estimator_random_state": estimator_random_state,
            "null_seed": null_seed,
        },
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
