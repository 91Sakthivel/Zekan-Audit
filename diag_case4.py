import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

from zekan.benchmark.fixtures import make_clean_dataset
from zekan.benchmark.injectors import (
    inject_correlated_leaks, inject_future_feature,
    inject_covariate_drift, inject_concept_drift,
)
from zekan.config.schema import ZekanConfig, SplitPolicy
from zekan.contract.prediction_contract import PredictionContract
from zekan.severity.ablation import run_ablation
from zekan.severity.engine import run_severity_analysis, FIXABLE_LEAKAGE_CLEAR_LEAK
from zekan.severity.metrics import evaluate_folds
from zekan.severity.splitters import temporal_expanding_folds

SEP = "=" * 72


def _fast_clf():
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _make_contract(**kwargs):
    defaults = dict(
        prediction_problem="benchmark churn",
        entity_id="entity_id", prediction_time="prediction_time",
        target="target", available_features_until="prediction_time",
    )
    defaults.update(kwargs)
    return PredictionContract(**defaults)


def _make_config(contract, leak_lookahead=1):
    return ZekanConfig(
        contract=contract,
        split_policy=SplitPolicy(
            n_splits=5, min_test_rows_per_fold=50,
            min_positive_cases_per_fold=10, min_negative_cases_per_fold=10,
            leak_lookahead=leak_lookahead,
        ),
    )


def _temp_folds(df):
    return temporal_expanding_folds(
        df, time_col="prediction_time", entity_col="entity_id",
        target_col="target", n_splits=5, min_test_rows=50, min_pos=10, min_neg=10,
    )


SEEDS = [1, 2, 3, 7, 13, 17, 42, 99, 123, 999]

# ── SECTION 1: Case 4 default-seed per-fold table ────────────────────────────
print(SEP)
print("SECTION 1: Case 4 per-fold AUC breakdown (seed=42)")
print(SEP)

base_df = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=42)
df_leak, record = inject_correlated_leaks(base_df, dataset_seed=42)
forbidden = record.planted_columns
contract4 = _make_contract(forbidden_after_prediction=forbidden)
temp_folds = _temp_folds(df_leak)

excluded = {"entity_id", "prediction_time", "target"}
all_features   = [c for c in df_leak.columns if c not in excluded]
safe_features  = [f for f in all_features if f not in set(forbidden)]
feat_a, feat_b = forbidden[0], forbidden[1]
features_minus_a = [f for f in all_features if f != feat_a]
features_minus_b = [f for f in all_features if f != feat_b]

print(f"{'Fold':>4}  {'all':>7}  {'drop_a':>7}  {'drop_b':>7}  {'drop_both':>9}  "
      f"{'lkg_a':>8}  {'lkg_b':>8}  {'lkg_cum':>8}")

fold_all, fold_da, fold_db, fold_dab = [], [], [], []
for fold in temp_folds:
    if fold.meta.skipped:
        continue
    fi = fold.meta.fold_idx

    def _eval(feats, _fold=fold):
        X_tr = df_leak.iloc[_fold.train_idx][feats].to_numpy(dtype=float)
        y_tr = df_leak.iloc[_fold.train_idx]["target"].to_numpy()
        X_te = df_leak.iloc[_fold.test_idx][feats].to_numpy(dtype=float)
        y_te = df_leak.iloc[_fold.test_idx]["target"].to_numpy()
        m = _fast_clf()
        m.fit(X_tr, y_tr)
        return float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))

    a   = _eval(all_features)
    da  = _eval(features_minus_a)
    db  = _eval(features_minus_b)
    dab = _eval(safe_features)
    fold_all.append(a); fold_da.append(da); fold_db.append(db); fold_dab.append(dab)
    print(f"{fi:>4}  {a:>7.4f}  {da:>7.4f}  {db:>7.4f}  {dab:>9.4f}  "
          f"{a-da:>+8.4f}  {a-db:>+8.4f}  {a-dab:>+8.4f}")

bl = float(np.mean(fold_all))
print(f"Mean  {bl:.4f}  {np.mean(fold_da):.4f}  {np.mean(fold_db):.4f}  "
      f"{np.mean(fold_dab):.4f}  "
      f"{bl-np.mean(fold_da):+.4f}  {bl-np.mean(fold_db):+.4f}  "
      f"{bl-np.mean(fold_dab):+.4f}")

# ── SECTION 2: Correlations ───────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 2: Correlations")
print(SEP)
sigma_z = 0.7
sigma_noise = sigma_z / 3.0
print(f"sigma_noise used    : {sigma_noise:.4f}  (sigma_z/3, designed for corr~0.90)")
print(f"theoretical corr    : {sigma_z**2 / (sigma_z**2 + sigma_noise**2):.4f}")
print(f"corr(alpha, beta)   : {df_leak['corr_leak_alpha'].corr(df_leak['corr_leak_beta']):.6f}")
print(f"corr(alpha, target) : {df_leak['corr_leak_alpha'].corr(df_leak['target']):.4f}")
print(f"corr(beta,  target) : {df_leak['corr_leak_beta'].corr(df_leak['target']):.4f}")

# ── SECTION 3: Ablation run + flag condition evaluation ───────────────────────
print()
print(SEP)
print("SECTION 3: Ablation run + understatement flag (seed=42)")
print(SEP)

baseline_eval = evaluate_folds(df_leak, all_features, "target", temp_folds, _fast_clf)
baseline_auc  = baseline_eval.mean_auc
summary = run_ablation(
    df_leak, contract4, baseline_auc, temp_folds, top_k=10, model_factory=_fast_clf
)

ind     = summary.individual
cum     = summary.cumulative
max_ind = max(e.leakage_estimate for e in ind if e.ablated)
cum_lkg = cum.cumulative_leakage

print(f"baseline_auc            : {baseline_auc:.4f}")
print(f"alpha  auc_without      : {ind[0].auc_without:.4f}  leakage={ind[0].leakage_estimate:+.4f}")
print(f"beta   auc_without      : {ind[1].auc_without:.4f}  leakage={ind[1].leakage_estimate:+.4f}")
print(f"cumul  auc_without      : {cum.auc_without:.4f}  leakage={cum_lkg:+.4f}")
print(f"max_individual          : {max_ind:+.4f}")
print(f"\nFlag conditions (ablation.py lines 191-192):")
print(f"  cum ({cum_lkg:+.4f}) > max_ind+0.02 ({max_ind+0.02:+.4f}) -> {cum_lkg > max_ind + 0.02}")
print(f"  cum ({cum_lkg:+.4f}) > max_ind*1.3  ({max_ind*1.3:+.4f}) -> {cum_lkg > max_ind * 1.3}")
print(f"one_at_a_time_understates: {summary.one_at_a_time_understates}")

# ── SECTION 4: 10-seed robustness ────────────────────────────────────────────
print()
print(SEP)
print("SECTION 4: Case 4 flag across 10 seeds")
print(SEP)
print(f"{'seed':>5}  {'baseline':>8}  {'lkg_a':>8}  {'lkg_b':>8}  "
      f"{'max_ind':>8}  {'lkg_cum':>8}  {'flag':>6}")

c4_flags = []
for seed in SEEDS:
    bdf  = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    dfl, rec = inject_correlated_leaks(bdf, dataset_seed=seed)
    forb = rec.planted_columns
    c4c  = _make_contract(forbidden_after_prediction=forb)
    tf   = _temp_folds(dfl)
    af   = [c for c in dfl.columns if c not in {"entity_id", "prediction_time", "target"}]
    bl_e = evaluate_folds(dfl, af, "target", tf, _fast_clf)
    s    = run_ablation(dfl, c4c, bl_e.mean_auc, tf, top_k=10, model_factory=_fast_clf)
    mi   = max(e.leakage_estimate for e in s.individual if e.ablated)
    cl   = s.cumulative.cumulative_leakage
    flag = s.one_at_a_time_understates
    c4_flags.append(flag)
    print(f"{seed:>5}  {bl_e.mean_auc:.4f}    "
          f"{s.individual[0].leakage_estimate:+.4f}  "
          f"{s.individual[1].leakage_estimate:+.4f}  "
          f"{mi:+.4f}  {cl:+.4f}  "
          f"{'FIRE' if flag else 'MISS'}")

print(f"\nFlag fired: {sum(c4_flags)}/{len(SEEDS)}")

# ── SECTION 5: Cases 1 / 2 / 3 untouched ────────────────────────────────────
print()
print(SEP)
print("SECTION 5: Cases 1 / 2a / 2b / Case 3 unchanged")
print(SEP)

base_df = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=42)

r1 = run_severity_analysis(
    base_df, _make_contract(), _make_config(_make_contract()), _fast_clf
)
df2a, _ = inject_covariate_drift(base_df)
r2a = run_severity_analysis(
    df2a, _make_contract(), _make_config(_make_contract()), _fast_clf
)
df2b, _ = inject_concept_drift(base_df)
r2b = run_severity_analysis(
    df2b, _make_contract(), _make_config(_make_contract()), _fast_clf
)

print(f"Case 1  fixable_leakage={r1.fixable_leakage:+.4f}   {'OK' if abs(r1.fixable_leakage) <= 0.02 else 'FAIL'} (want <=0.02)")
print(f"Case 2a fixable_leakage={r2a.fixable_leakage:+.4f}   {'OK' if abs(r2a.fixable_leakage) <= 0.02 else 'FAIL'} (want <=0.02)")
print(f"Case 2b fixable_leakage={r2b.fixable_leakage:+.4f}   {'OK' if abs(r2b.fixable_leakage) <= 0.02 else 'FAIL'} (want <=0.02)")
print(f"Case 2b optimism gap vs 1: {r2b.total_optimism - r1.total_optimism:+.4f}  "
      f"{'OK' if r2b.total_optimism - r1.total_optimism >= 0.03 else 'FAIL'} (want >=0.03)")

print(f"\nCase 3 10-seed sweep (gate={FIXABLE_LEAKAGE_CLEAR_LEAK}):")
c3_results = []
for seed in SEEDS:
    bdf  = make_clean_dataset(n_entities=500, snapshots_per_entity=5, seed=seed)
    dfl, rec = inject_future_feature(bdf, dataset_seed=seed)
    c3c  = _make_contract(forbidden_after_prediction=rec.planted_columns)
    r3   = run_severity_analysis(dfl, c3c, _make_config(c3c), _fast_clf)
    c3_results.append(r3.fixable_leakage)
    gate = "PASS" if r3.fixable_leakage >= FIXABLE_LEAKAGE_CLEAR_LEAK else "FAIL"
    print(f"  seed={seed:>4}  fixable_leakage={r3.fixable_leakage:+.4f}  [{gate}]")

c3_passes = sum(1 for v in c3_results if v >= FIXABLE_LEAKAGE_CLEAR_LEAK)
print(f"  Case 3: {c3_passes}/{len(SEEDS)} pass  "
      f"min={min(c3_results):+.4f}  mean={float(np.mean(c3_results)):+.4f}")
