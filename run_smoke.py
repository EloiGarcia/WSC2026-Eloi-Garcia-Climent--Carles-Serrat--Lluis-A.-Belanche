#!/usr/bin/env python3
"""
run_smoke.py -- offline end-to-end correctness check on a synthetic surrogate.
Tiny epochs / few seeds. Proves the whole pipeline runs and produces tidy CSVs,
CI aggregates and figures BEFORE the real datasets are present. No network.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from rtxfair_sim.data import make_synthetic, minority_resample
from rtxfair_sim.model import train_teacher, teacher_shap
from rtxfair_sim.config import ScenarioConfig, PENALTY_DP, PENALTY_EO
from rtxfair_sim.train import train_student
from rtxfair_sim import sim, figures, metrics as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke")
os.makedirs(OUT, exist_ok=True)

print("1) synthetic data + teacher + SHAP")
b = make_synthetic(n=2000, d=12)
t = train_teacher(b.X_train, b.y_train, n_estimators=40, max_depth=3)
sv_tr, sv_te, _ = teacher_shap(t, b.X_train, b.X_test)
print("   shap shapes", sv_tr.shape, sv_te.shape)

print("2) one student (DP) + one (EO), tiny epochs")
for pen in (PENALTY_DP, PENALTY_EO):
    sc = ScenarioConfig(dataset="synthetic", seed=42, epochs=8, batch_size=256,
                        lambda_fair=2.5, penalty=pen)
    rec, model = train_student(b.X_train, b.y_train, sv_tr, b.sensitive_train,
                               b.X_test, b.y_test, sv_te, b.sensitive_test, sc)
    print(f"   penalty={pen}: AUC={rec['AUC']:.3f} R2={rec['SHAP_R2']:.3f} "
          f"DP={rec['DP_gap']:.3f} EO={rec['EO_gap']:.3f} EOdds={rec['EOdds_gap']:.3f} "
          f"gnmax={rec['grad_norm_max']:.2f} div={rec['diverged']}")

print("3) minority resample sanity")
for f in (0.01, 0.05, 0.10):
    X2, y2, s2, se2 = minority_resample(b.X_train, b.y_train, sv_tr, b.sensitive_train, f)
    frac = se2.mean() if se2.mean() <= 0.5 else 1 - se2.mean()
    print(f"   target={f} -> realized minority frac={frac:.3f} (n={len(y2)})")

print("4) batched throughput")
rows = M.batched_throughput(model, b.X_test, batch_sizes=(1, 32, 128, 512), repeats=5)
for r in rows:
    print(f"   bs={r['batch_size']:5d}  {r['throughput_sps']:.0f} sps  "
          f"{r['per_sample_ms']:.4f} ms/sample")

print("5) mini scenario sweep (lambda x 2 seeds) + aggregate + figure")
scs = []
for lam in (0.0, 1.0, 2.5):
    for sd in (42, 43):
        scs.append(ScenarioConfig(dataset="synthetic", seed=sd, epochs=6,
                                  batch_size=256, lambda_fair=lam, penalty=PENALTY_DP))
# run by hand (bypass load_dataset; reuse synthetic bundle + cached teacher)
recs = []
for sc in scs:
    if sc.minority_fraction is not None:
        pass
    rec, _ = train_student(b.X_train, b.y_train, sv_tr, b.sensitive_train,
                           b.X_test, b.y_test, sv_te, b.sensitive_test, sc)
    recs.append(rec)
df = pd.DataFrame(recs)
df.to_csv(os.path.join(OUT, "smoke_results.csv"), index=False)
agg = sim.aggregate(df, ["lambda_fair"])
agg.to_csv(os.path.join(OUT, "smoke_agg.csv"), index=False)
png = figures.dual_response_vs_lambda(agg, os.path.join(OUT, "smoke_lambda"))
print("   wrote", png)
print("\nSMOKE TEST PASSED")
