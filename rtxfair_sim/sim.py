# revision_outputs/rtxfair_sim/sim.py
"""
Scenario-based simulation driver (P1).

Treats the distillation pipeline as the system under study. Builds scenario lists
(OFAT sweeps + one 2-factor factorial), runs each scenario across N_SEEDS
replications, records a tidy row per (scenario x seed), and aggregates to
mean +/- 95% CI across replications.

Teacher+TreeSHAP are cached per teacher-capacity key so they are not recomputed
for every seed/lambda/batch cell.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import itertools
import numpy as np
import pandas as pd

from .config import (DATASETS, ScenarioConfig, GRID, FACTORIAL, SEEDS,
                     PENALTY_DP, GLOBAL_SEED)
from .data import load_dataset, minority_resample
from .model import train_teacher, teacher_shap
from .train import train_student

RESPONSES = ["AUC", "SHAP_R2", "DP_gap", "mean_score_gap", "EO_gap",
             "EOdds_gap", "grad_norm_mean", "grad_norm_max", "p99_latency_ms",
             "masked_batch_frac", "train_seconds"]


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "results", "_teacher_cache")


class TeacherCache:
    """Caches (shap_train, shap_test, teacher_AUC) per (dataset,n_estimators,
    max_depth,lr), in-process AND on disk so chunked runs don't recompute the
    teacher+TreeSHAP every call. (The teacher object itself is not needed
    downstream, so only the SHAP targets and AUC are persisted.)"""
    def __init__(self, bundle):
        self.b = bundle
        self._c = {}
        os.makedirs(_CACHE_DIR, exist_ok=True)

    def _path(self, n, d, lr):
        return os.path.join(_CACHE_DIR, f"{self.b.name}_n{n}_d{d}_lr{lr}.npz")

    def get(self, n_estimators, max_depth, lr=0.1, seed=GLOBAL_SEED):
        key = (n_estimators, max_depth, lr)
        if key in self._c:
            return self._c[key]
        p = self._path(n_estimators, max_depth, lr)
        if os.path.exists(p):
            z = np.load(p)
            val = (None, z["sv_tr"], z["sv_te"], float(z["t_auc"]))
            self._c[key] = val
            return val
        t = train_teacher(self.b.X_train, self.b.y_train, n_estimators,
                          max_depth, lr, seed)
        sv_tr, sv_te, _ = teacher_shap(t, self.b.X_train, self.b.X_test)
        from sklearn.metrics import roc_auc_score
        t_auc = float(roc_auc_score(self.b.y_test, t.predict_proba(self.b.X_test)[:, 1]))
        np.savez(p, sv_tr=sv_tr, sv_te=sv_te, t_auc=t_auc)
        val = (t, sv_tr, sv_te, t_auc)
        self._c[key] = val
        return val


def run_one(bundle, cache, scenario):
    t, sv_tr, sv_te, t_auc = cache.get(
        scenario.teacher_n_estimators, scenario.teacher_max_depth,
        scenario.teacher_learning_rate, scenario.seed)
    Xtr, ytr, shtr, sens = bundle.X_train, bundle.y_train, sv_tr, bundle.sensitive_train
    if scenario.minority_fraction is not None:
        Xtr, ytr, shtr, sens = minority_resample(
            Xtr, ytr, shtr, sens, scenario.minority_fraction, seed=scenario.seed)
    rec, _ = train_student(Xtr, ytr, shtr, sens,
                           bundle.X_test, bundle.y_test, sv_te,
                           bundle.sensitive_test, scenario)
    rec["teacher_AUC"] = t_auc
    return rec


def _base_scenario(dataset, **over):
    cfg = DATASETS[dataset]
    kw = dict(dataset=dataset, epochs=cfg.epochs, batch_size=cfg.batch_size,
              lambda_fair=cfg.lambda_fair, lr=cfg.lr, penalty=PENALTY_DP)
    kw.update(over)
    return ScenarioConfig(**kw)


def build_ofat_scenarios(dataset, seeds=SEEDS):
    """One-factor-at-a-time around each dataset's baseline anchor."""
    scs = []
    factors = {
        "teacher_n_estimators": GRID["teacher_n_estimators"],
        "teacher_max_depth": GRID["teacher_max_depth"],
        "lambda_fair": GRID["lambda_fair"],
        "batch_size": GRID["batch_size"],
        "minority_fraction": GRID["minority_fraction"],
    }
    for fac, values in factors.items():
        for v in values:
            for s in seeds:
                scs.append(_base_scenario(dataset, seed=s, label=f"OFAT:{fac}={v}",
                                          **{fac: v}))
    return scs


def build_core_scenarios(dataset, seeds=SEEDS):
    """Focused OFAT grid (lambda, batch, minority) using only the baseline teacher.
    Used for the heavy datasets where the full 1200-epoch grid is intractable in a
    constrained sandbox; the full grid + factorial + teacher-capacity sweep is
    carried by German Credit."""
    scs = []
    factors = {
        "lambda_fair": GRID["lambda_fair"],
        "batch_size": GRID["batch_size"],
        "minority_fraction": GRID["minority_fraction"],
    }
    for fac, values in factors.items():
        for v in values:
            for s in seeds:
                scs.append(_base_scenario(dataset, seed=s, label=f"OFAT:{fac}={v}",
                                          **{fac: v}))
    return scs


def build_factorial_scenarios(dataset, seeds=SEEDS):
    keys = list(FACTORIAL.keys())
    scs = []
    for combo in itertools.product(*FACTORIAL.values()):
        over = dict(zip(keys, combo))
        for s in seeds:
            scs.append(_base_scenario(dataset, seed=s,
                                      label="FACT:" + ",".join(f"{k}={v}" for k, v in over.items()),
                                      **over))
    return scs


def run_scenarios(dataset, scenarios, out_csv=None, verbose=True):
    bundle = load_dataset(dataset)
    cache = TeacherCache(bundle)
    rows = []
    for i, sc in enumerate(scenarios, 1):
        rec = run_one(bundle, cache, sc)
        rows.append(rec)
        if verbose:
            print(f"[{dataset} {i}/{len(scenarios)}] {sc.label} seed={sc.seed} "
                  f"AUC={rec['AUC']:.4f} DPgap={rec['DP_gap']:.4f} "
                  f"EOgap={rec['EO_gap']:.4f} div={rec['diverged']}")
        if out_csv:
            pd.DataFrame(rows).to_csv(out_csv, index=False)  # incremental save
    return pd.DataFrame(rows)


def aggregate(df, factor_cols, responses=RESPONSES):
    """Mean +/- 95% CI (t-interval) across replications, grouped by factor_cols."""
    from scipy import stats
    out = []
    for keys, g in df.groupby(factor_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(factor_cols, keys))
        row["n_rep"] = len(g)
        for r in responses:
            vals = g[r].dropna().values.astype(float)
            m = float(np.mean(vals)) if len(vals) else np.nan
            if len(vals) > 1:
                se = stats.sem(vals)
                h = se * stats.t.ppf(0.975, len(vals) - 1)
            else:
                h = np.nan
            row[f"{r}_mean"] = m
            row[f"{r}_ci95"] = float(h) if h == h else np.nan
        out.append(row)
    return pd.DataFrame(out)
