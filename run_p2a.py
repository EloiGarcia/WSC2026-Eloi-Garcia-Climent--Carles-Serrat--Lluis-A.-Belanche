#!/usr/bin/env python3
"""
run_p2a.py -- Reviewer 3 (2a): the AUC anomaly.

The fairness-constrained student beats the baseline on AUC for Bank Marketing
(0.8959->0.9066) and UCI Credit Card (0.7276->0.7486). This script determines why.

Key structural fact (confirmed from the notebooks): the student's RISK head is
trained on the ground-truth labels via BCE, NOT on the teacher's predictions.
Only the EXPLANATION head distills SHAP. So "student beats teacher on AUC" is a
comparison between a fresh label-trained MLP and an XGBoost teacher -- and the
teacher was never tuned.

We therefore:
  1. Report the FIXED (submitted) teacher AUC.
  2. Run a proper randomized hyperparameter search for the teacher; report tuned AUC.
  3. Ablation isolating student capacity vs the fairness regularizer:
       - student lambda=0  (capacity only)
       - student lambda=baseline (capacity + fairness)
     across >=5 seeds, mean +/- 95% CI.
This attributes the anomaly to (teacher undertuning + label-trained student),
not to fairness improving accuracy.

Usage: python run_p2a.py [german|creditcard|bank|all]
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler
import xgboost as xgb

from rtxfair_sim.data import load_dataset
from rtxfair_sim.model import train_teacher, teacher_shap
from rtxfair_sim.config import DATASETS, ScenarioConfig, SEEDS, PENALTY_DP
from rtxfair_sim.train import train_student

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

PARAM_DIST = dict(
    n_estimators=[100, 200, 300, 500, 800],
    max_depth=[2, 3, 4, 5, 6, 8],
    learning_rate=[0.01, 0.02, 0.05, 0.1, 0.2],
    subsample=[0.6, 0.8, 1.0],
    colsample_bytree=[0.6, 0.8, 1.0],
    min_child_weight=[1, 3, 5],
    gamma=[0, 0.1, 0.5],
)


def ci95(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return np.nan
    return stats.sem(vals) * stats.t.ppf(0.975, len(vals) - 1)


def tune_teacher(X, y, seed=42, n_iter=40):
    """Manual randomized search with 3-fold CV scored by AUC via predict_proba
    (avoids sklearn<->xgboost estimator-type incompatibilities)."""
    sampler = list(ParameterSampler(PARAM_DIST, n_iter=n_iter, random_state=seed))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best, best_auc, best_params = None, -1.0, None
    for params in sampler:
        aucs = []
        for tr, va in skf.split(X, y):
            m = xgb.XGBClassifier(random_state=seed, n_jobs=4,
                                  eval_metric="logloss", **params)
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        a = float(np.mean(aucs))
        if a > best_auc:
            best_auc, best_params = a, params
    best = xgb.XGBClassifier(random_state=seed, n_jobs=4,
                             eval_metric="logloss", **best_params)
    best.fit(X, y)
    return best, best_params, best_auc


def students_at_lambdas(bundle, sv_tr, sv_te, lambdas, seeds=SEEDS):
    cfg = DATASETS[bundle.name]
    rows = []
    for lam in lambdas:
        for sd in seeds:
            sc = ScenarioConfig(dataset=bundle.name, seed=sd, lambda_fair=lam,
                                epochs=cfg.epochs, batch_size=cfg.batch_size,
                                lr=cfg.lr, penalty=PENALTY_DP)
            rec, _ = train_student(bundle.X_train, bundle.y_train, sv_tr,
                                   bundle.sensitive_train, bundle.X_test,
                                   bundle.y_test, sv_te, bundle.sensitive_test, sc)
            rows.append(rec)
    return pd.DataFrame(rows)


def run_dataset(name):
    print(f"\n=== P2a {DATASETS[name].pretty} ===")
    b = load_dataset(name)

    # 1) fixed teacher
    t_fixed = train_teacher(b.X_train, b.y_train, 100, 4, 0.1)
    auc_fixed = roc_auc_score(b.y_test, t_fixed.predict_proba(b.X_test)[:, 1])

    # 2) tuned teacher
    t_tuned, best_params, cv_auc = tune_teacher(b.X_train, b.y_train)
    auc_tuned = roc_auc_score(b.y_test, t_tuned.predict_proba(b.X_test)[:, 1])
    print(f"  teacher AUC fixed={auc_fixed:.4f}  tuned={auc_tuned:.4f}  (cv={cv_auc:.4f})")
    print(f"  best params: {best_params}")

    # 3) ablation: student at lambda=0 and lambda=baseline (fixed-teacher SHAP)
    sv_tr, sv_te, _ = teacher_shap(t_fixed, b.X_train, b.X_test)
    lam_base = DATASETS[name].lambda_fair
    df = students_at_lambdas(b, sv_tr, sv_te, [0.0, lam_base])
    summ = (df.groupby("lambda_fair")
              .agg(AUC_mean=("AUC", "mean"), DP_mean=("DP_gap", "mean"),
                   AUC_std=("AUC", "std"))
              .reset_index())
    print(summ.to_string(index=False))

    # assemble comparison row
    s0 = df[df.lambda_fair == 0.0]["AUC"]
    sb = df[df.lambda_fair == lam_base]["AUC"]
    out = dict(
        dataset=name, teacher_AUC_fixed=auc_fixed, teacher_AUC_tuned=auc_tuned,
        student_AUC_lambda0_mean=s0.mean(), student_AUC_lambda0_ci95=ci95(s0),
        student_AUC_lambdaBase_mean=sb.mean(), student_AUC_lambdaBase_ci95=ci95(sb),
        lambda_base=lam_base, best_params=str(best_params), cv_auc=cv_auc,
    )
    df.to_csv(os.path.join(RES, f"{name}_p2a_student_ablation.csv"), index=False)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DATASETS) if which == "all" else [which]
    rows = [run_dataset(n) for n in names]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "p2a_summary.csv"), index=False)
    print("\nP2a summary written.")
