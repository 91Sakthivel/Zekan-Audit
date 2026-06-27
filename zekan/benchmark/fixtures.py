"""Synthetic benchmark datasets for zekan validation.

All datasets use abstract column names (entity_id, prediction_time, feature_*, target)
so tests are not coupled to any real-world schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_Z_SEED_OFFSET = 99_999  # decouples z sub-seed from the main data RNG stream


def _make_z_mat(n_entities: int, n_periods: int, dataset_seed: int) -> np.ndarray:
    """AR(1) idiosyncratic factor, shape (n_entities, n_periods).

    Parameters (locked): rho=0.80, sigma_z=0.7.
    Uses dataset_seed + _Z_SEED_OFFSET so z is reproducible independently of
    how many RNG calls precede it in make_clean_dataset.
    """
    rng = np.random.default_rng(dataset_seed + _Z_SEED_OFFSET)
    sigma_z, rho = 0.7, 0.80
    z = np.empty((n_entities, n_periods))
    z[:, 0] = rng.normal(0.0, sigma_z, n_entities)
    innov = np.sqrt(1.0 - rho ** 2) * sigma_z
    for t in range(1, n_periods):
        z[:, t] = rho * z[:, t - 1] + innov * rng.normal(0.0, 1.0, n_entities)
    return z


def make_clean_dataset(
    n_entities: int = 2000,
    snapshots_per_entity: int = 5,
    n_features: int = 6,
    base_rate: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Entity-snapshot panel with genuine moderate signal and zero leakage.

    Structure: n_entities x n_periods rows (all entities at all periods).
    n_periods = max(snapshots_per_entity, 10) to guarantee >= 8 distinct time points.

    Signal design:
    - features 0..n_signal-1: strong entity-level signal (coeff 0.7 on entity_effect)
    - features n_signal..n_features//2: moderate signal (coeff 0.3)
    - features n_features//2..: pure noise
    Target is binary, generated via a logistic model on entity_effect + time trend.
    Expected ROC-AUC with a reasonable classifier: 0.65 - 0.82.
    """
    rng = np.random.default_rng(seed)
    n_periods = max(snapshots_per_entity, 10)

    # Latent entity-level propensity toward the positive class
    entity_effects = rng.normal(0.0, 1.0, n_entities)

    base = pd.Timestamp("2022-01-01")
    period_strs = [
        (base + pd.DateOffset(months=p)).strftime("%Y-%m-%d")
        for p in range(n_periods)
    ]

    # Build the full entity x period grid
    n_rows = n_entities * n_periods
    entity_ids = np.repeat([f"entity_{i:05d}" for i in range(n_entities)], n_periods)
    period_ids = np.tile(period_strs, n_entities)
    eff = np.repeat(entity_effects, n_periods)           # entity effect per row
    t_rank = np.tile(                                     # time rank 0..1 per row
        np.linspace(0.0, 1.0, n_periods), n_entities
    )

    n_signal = max(1, n_features // 3)
    n_moderate = max(0, n_features // 2 - n_signal)

    feature_dict: dict[str, np.ndarray] = {}
    for i in range(n_signal):
        feature_dict[f"feature_{i}"] = (
            0.7 * eff + 0.2 * t_rank + rng.normal(0.0, 0.6, n_rows)
        )
    for i in range(n_signal, n_signal + n_moderate):
        feature_dict[f"feature_{i}"] = (
            0.3 * eff + rng.normal(0.0, 1.0, n_rows)
        )
    for i in range(n_signal + n_moderate, n_features):
        feature_dict[f"feature_{i}"] = rng.normal(0.0, 1.0, n_rows)

    # Latent idiosyncratic AR(1) factor — contributes to target, excluded from
    # every feature column.  Parameters (locked): rho=0.80, sigma_z=0.7, z_coeff=0.8.
    # z ≈ 25% of total logit variance; entity_effect ≈ 53%; epsilon ≈ 20%.
    # Generated via _make_z_mat so injectors can reconstruct the same z from seed.
    _z_flat = _make_z_mat(n_entities, n_periods, seed).flatten()

    logit = 0.8 * eff + 0.3 * t_rank + 0.8 * _z_flat + rng.normal(0.0, 0.5, n_rows)
    probs = 1.0 / (1.0 + np.exp(-logit))
    threshold = float(np.quantile(probs, 1.0 - base_rate))
    target = (probs > threshold).astype(int)

    return pd.DataFrame(
        {
            "entity_id": entity_ids,
            "prediction_time": period_ids,
            **feature_dict,
            "target": target,
        }
    )
