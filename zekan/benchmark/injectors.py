"""Synthetic leakage and drift injectors for benchmark scenarios.

Each injector takes a clean DataFrame (from make_clean_dataset) and returns
(modified_df, InjectionRecord). Leakage injectors add columns that encode
target information unavailable at prediction time. Drift injectors add columns
that shift in distribution or relationship WITHOUT leaking target information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class InjectionRecord:
    """Describes what was planted into the dataset."""

    kind: str
    planted_columns: list[str]
    description: str
    is_leak: bool


# ── Leakage injectors ─────────────────────────────────────────────────────────

def inject_label_proxy(
    df: pd.DataFrame,
    target_col: str = "target",
    flip_rate: float = 0.05,
    seed: int = 1,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Near-copy of the target label (5 % random flips) — massive leak."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    col = "leaky_label_proxy"
    proxy = df[target_col].values.copy()
    flip_mask = rng.random(len(df)) < flip_rate
    proxy[flip_mask] = 1 - proxy[flip_mask]
    df[col] = proxy
    return df, InjectionRecord(
        kind="label_proxy",
        planted_columns=[col],
        description=(
            f"Binary near-copy of '{target_col}' with {flip_rate*100:.0f}% random flips. "
            "Encodes the label directly — maximal leakage."
        ),
        is_leak=True,
    )


def inject_future_feature(
    df: pd.DataFrame,
    time_col: str = "prediction_time",
    entity_col: str = "entity_id",
    target_col: str = "target",
    seed: int = 2,
    dataset_seed: int = 42,
    window_size: int = 1,
    _probe_current_period: bool = False,
    _probe_noise_sigma: float = 0.35,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Forward-window latent z mean + Gaussian noise — orthogonal temporal leak.

    For each (entity E, period T) row with a full k-step future window:

        future_z_latent = mean(z[E, T+1], ..., z[E, T+k]) + N(0, sigma)

    For rows where the full k-step window is not available (T + k > T_max):

        future_z_latent = 0.0 + N(0, sigma)   [fallback: z unconditional mean]

    The k-step mean reduces per-fold variance relative to a single-step draw
    while remaining genuinely future, row-level, and orthogonal.

    sigma is chosen on an SNR=2 basis:
        Var(mean) = sigma_z^2 * (k + 2*sum((k-j)*rho^j, j=1..k-1)) / k^2
        sigma_noise = sqrt(Var(mean)) / 2
    For k=3: Var(mean) = 0.4072, sigma_noise = 0.32.
    For k=1: sigma_noise = 0.35 (sigma_z/2).

    Internal parameters _probe_* are used by the alignment sanity probe only.
    """
    from zekan.benchmark.fixtures import _make_z_mat

    rng = np.random.default_rng(seed)
    df = df.copy()
    col = "future_z_latent"

    sorted_periods: list = sorted(df[time_col].unique(), key=lambda x: pd.to_datetime(x))
    sorted_entities: list = sorted(df[entity_col].unique())
    n_periods = len(sorted_periods)
    n_entities = len(sorted_entities)

    z_mat = _make_z_mat(n_entities, n_periods, dataset_seed)
    entity_idx: dict = {e: i for i, e in enumerate(sorted_entities)}
    period_idx: dict = {p: i for i, p in enumerate(sorted_periods)}

    # SNR=2 sigma for the chosen window_size k
    k = window_size
    rho_z, sigma_z = 0.80, 0.7
    signal_var = (sigma_z ** 2 / k ** 2) * (
        k + 2 * sum((k - j) * rho_z ** j for j in range(1, k))
    )
    sigma_noise = float(np.sqrt(signal_var) / 2)

    if _probe_current_period:
        # Alignment probe: leak z[T] (current period).
        values = np.array([
            z_mat[entity_idx[eid], period_idx[pid]]
            for eid, pid in zip(df[entity_col].values, df[time_col].values)
        ], dtype=float)
        df[col] = values + rng.normal(0.0, _probe_noise_sigma, len(df))
    else:
        # Production: mean of z[T+1..T+k]; fall back to 0.0 when full window unavailable.
        values = np.array([
            float(np.mean([
                z_mat[entity_idx[eid], period_idx[pid] + step]
                for step in range(1, k + 1)
            ]))
            if period_idx[pid] + k <= n_periods - 1
            else 0.0
            for eid, pid in zip(df[entity_col].values, df[time_col].values)
        ], dtype=float)
        df[col] = values + rng.normal(0.0, sigma_noise, len(df))

    return df, InjectionRecord(
        kind="future_feature",
        planted_columns=[col],
        description=(
            f"Each row receives the mean of the entity's next {k} latent z values "
            f"(mean(z[E,T+1],...,z[E,T+{k}])) plus N(0, {sigma_noise:.4f}). "
            "Orthogonal to existing features; strictly future; row-level. "
            f"Rows where the full {k}-step window is unavailable fall back to 0.0 + noise."
        ),
        is_leak=True,
    )


def inject_presplit_artifact(
    df: pd.DataFrame,
    entity_col: str = "entity_id",
    target_col: str = "target",
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Entity-level target mean encoding computed on the FULL dataset.

    In a correct pipeline this would be computed only on training data.
    Using full-dataset statistics leaks test-set target information.
    """
    df = df.copy()
    col = "entity_target_mean_encoded"
    entity_means = df.groupby(entity_col)[target_col].transform("mean")
    global_mean = float(df[target_col].mean())
    df[col] = entity_means - global_mean   # centred for cleaner downstream use
    return df, InjectionRecord(
        kind="presplit_artifact",
        planted_columns=[col],
        description=(
            f"Entity-level target mean encoding (centred by global mean) computed "
            "on the full dataset before any split — presplit artifact / mild leakage."
        ),
        is_leak=True,
    )


def inject_correlated_leaks(
    df: pd.DataFrame,
    time_col: str = "prediction_time",
    entity_col: str = "entity_id",
    target_col: str = "target",
    seed: int = 5,
    dataset_seed: int = 42,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Two noisy copies of z[entity, T+1] — correlated leakage pair from a shared source.

    shared_source = z[entity, T+1]  (same AR(1) latent factor as inject_future_feature)
    corr_leak_alpha = shared_source + N(0, sigma_noise)
    corr_leak_beta  = shared_source + N(0, sigma_noise)

    With sigma_noise = sigma_z / 3 ≈ 0.233, corr(alpha, beta) ≈ 0.90.
    Each column carries genuine row-level signal. Dropping one alone shows minimal
    leakage because the partner compensates; dropping both removes the shared source
    entirely and reveals the true cumulative leakage — the understatement pattern.

    Rows at the last period (no T+1 available) fall back to shared_source = 0.0.
    """
    from zekan.benchmark.fixtures import _make_z_mat

    rng = np.random.default_rng(seed)
    df = df.copy()

    sorted_periods: list = sorted(df[time_col].unique(), key=lambda x: pd.to_datetime(x))
    sorted_entities: list = sorted(df[entity_col].unique())
    n_periods = len(sorted_periods)
    n_entities = len(sorted_entities)

    z_mat = _make_z_mat(n_entities, n_periods, dataset_seed)
    entity_idx: dict = {e: i for i, e in enumerate(sorted_entities)}
    period_idx: dict = {p: i for i, p in enumerate(sorted_periods)}

    sigma_z = 0.7
    sigma_noise = sigma_z / 3.0  # corr(alpha,beta) = sigma_z^2 / (sigma_z^2 + sigma_noise^2) ≈ 0.90

    shared = np.array([
        z_mat[entity_idx[eid], period_idx[pid] + 1]
        if period_idx[pid] < n_periods - 1
        else 0.0
        for eid, pid in zip(df[entity_col].values, df[time_col].values)
    ], dtype=float)

    col_a, col_b = "corr_leak_alpha", "corr_leak_beta"
    df[col_a] = shared + rng.normal(0.0, sigma_noise, len(df))
    df[col_b] = shared + rng.normal(0.0, sigma_noise, len(df))

    return df, InjectionRecord(
        kind="correlated_leaks",
        planted_columns=[col_a, col_b],
        description=(
            f"Two noisy copies of z[entity, T+1] with sigma_noise={sigma_noise:.3f} "
            f"(corr(alpha,beta) ≈ 0.90). Dropping either alone shows minimal leakage "
            "(the partner compensates); dropping both reveals the full shared-source leakage."
        ),
        is_leak=True,
    )


def inject_graded_future_leak(
    df: pd.DataFrame,
    alpha: float = 1.0,
    noise_sigma: float = 0.35,
    time_col: str = "prediction_time",
    entity_col: str = "entity_id",
    seed: int = 2,
    dataset_seed: int = 42,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Graded future leak: forbidden = alpha * z[entity, T+1] + N(0, noise_sigma).

    Continuous severity knob for benchmark sweeps:
      alpha=0.0                 → pure noise; no future information; is_leak=False
      alpha=1.0, sigma=0.35     → same SNR=2 signal as inject_future_feature(window_size=1)
      alpha > 1.0               → higher SNR; stronger leakage contribution

    SNR = alpha * sigma_z / noise_sigma = alpha * 0.7 / noise_sigma.
    The relationship is monotone: higher alpha → larger fixable_leakage → higher NSL.

    Rows at the last period (no T+1 available) fall back to z_next = 0.0.
    dataset_seed must match the seed used in make_clean_dataset so the z-matrix
    is the same AR(1) process that generated the target signal.
    """
    from zekan.benchmark.fixtures import _make_z_mat

    rng = np.random.default_rng(seed)
    df = df.copy()
    col = "graded_future_leak"

    sorted_periods: list = sorted(df[time_col].unique(), key=lambda x: pd.to_datetime(x))
    sorted_entities: list = sorted(df[entity_col].unique())
    n_periods = len(sorted_periods)
    n_entities = len(sorted_entities)

    z_mat = _make_z_mat(n_entities, n_periods, dataset_seed)
    entity_idx: dict = {e: i for i, e in enumerate(sorted_entities)}
    period_idx: dict = {p: i for i, p in enumerate(sorted_periods)}

    z_next = np.array([
        z_mat[entity_idx[eid], period_idx[pid] + 1]
        if period_idx[pid] + 1 < n_periods
        else 0.0
        for eid, pid in zip(df[entity_col].values, df[time_col].values)
    ], dtype=float)

    df[col] = alpha * z_next + rng.normal(0.0, noise_sigma, len(df))

    snr = (alpha * 0.7 / noise_sigma) if noise_sigma > 0.0 else float("inf")
    return df, InjectionRecord(
        kind="graded_future_leak",
        planted_columns=[col],
        description=(
            f"forbidden = {alpha:.3f} * z[entity,T+1] + N(0, {noise_sigma:.3f}); "
            f"SNR = {snr:.2f}. "
            "alpha=0 is pure noise; alpha=1, sigma=0.35 reproduces inject_future_feature(window_size=1). "
            "Rows at last period fall back to z_next=0."
        ),
        is_leak=alpha > 0.0,
    )


# ── Drift injectors (NO leakage) ──────────────────────────────────────────────

def inject_covariate_drift(
    df: pd.DataFrame,
    strength: float = 0.5,
    time_col: str = "prediction_time",
    seed: int = 7,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Add a feature whose DISTRIBUTION shifts over time; relationship with target is fixed.

    The feature mean increases linearly with time (covariate drift / dataset shift).
    It contains no future information about the target — not a leak.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    times = pd.to_datetime(df[time_col])
    sorted_periods = sorted(times.unique())
    n_p = len(sorted_periods)
    rank_map = {p: i / max(1, n_p - 1) for i, p in enumerate(sorted_periods)}
    time_rank = times.map(rank_map).values

    col = "covariate_drift_feat"
    df[col] = rng.normal(0.0, 1.0, len(df)) + time_rank * strength * 2.0

    return df, InjectionRecord(
        kind="covariate_drift",
        planted_columns=[col],
        description=(
            f"Feature distribution mean increases linearly over time (strength={strength}). "
            "Feature-target relationship is constant across periods. Not a leak."
        ),
        is_leak=False,
    )


def inject_concept_drift(
    df: pd.DataFrame,
    strength: float = 2.0,
    time_col: str = "prediction_time",
    entity_col: str = "entity_id",
    seed: int = 8,
) -> tuple[pd.DataFrame, InjectionRecord]:
    """Add a feature whose RELATIONSHIP with the target reverses across periods.

    The feature is built from the entity-level GROUP MEAN of the first signal
    feature column, z-scored.  Averaging across all periods for each entity
    removes per-row noise, giving a clean proxy for the latent entity_effect
    that drives the target.

    The time-varying coefficient transitions from +strength (earliest period)
    to -strength (latest period).

    Why grouped CV is unaffected: each fold sees the same entity at all periods,
    so the +/- swings cancel in expectation and the RF assigns near-zero importance
    to this feature — naive_auc (evaluation A) stays close to the clean baseline.

    Why temporal CV suffers: the RF trained on early periods learns a large positive
    coefficient for this feature.  In late test periods the true coefficient is
    strongly negative, so the model's predictions are systematically reversed for
    the most informative entities — temporal AUC drops substantially, widening
    total_optimism relative to the clean baseline.

    Not a leak: no future target information is encoded.
    """
    rng = np.random.default_rng(seed)  # used only in fallback
    df = df.copy()
    times = pd.to_datetime(df[time_col])
    sorted_periods = sorted(times.unique())
    n_p = len(sorted_periods)
    rank_map = {p: i / max(1, n_p - 1) for i, p in enumerate(sorted_periods)}
    time_rank = times.map(rank_map).values

    # Entity group mean of first signal feature — noise-free entity propensity proxy.
    # Averaging across all observed periods cancels per-row noise and
    # t_rank contributions, leaving a stable signal proportional to entity_effect.
    base_cols = sorted(c for c in df.columns if c.startswith("feature_"))
    if base_cols:
        entity_mean_vals = (
            df.groupby(entity_col)[base_cols[0]].transform("mean").values.astype(float)
        )
        base = (entity_mean_vals - entity_mean_vals.mean()) / (entity_mean_vals.std() + 1e-8)
    else:
        base = rng.normal(0.0, 1.0, len(df))  # fallback when no feature columns

    # Coefficient transitions: +strength (early) -> 0 (mid) -> -strength (late)
    coeff = strength * (1.0 - 2.0 * time_rank)
    col = "concept_drift_feat"
    df[col] = base * coeff

    return df, InjectionRecord(
        kind="concept_drift",
        planted_columns=[col],
        description=(
            f"Feature-target relationship transitions from +{strength} (early periods) "
            f"to -{strength} (late periods). Built from entity group-mean signal so "
            "the reversal is genuine and measurable. Not a leak."
        ),
        is_leak=False,
    )
