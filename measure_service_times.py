#!/usr/bin/env python3
"""
measure_service_times.py -- measure per-request service-time distributions used by
the discrete-event deployment simulation (run_des.py).

Records single-instance latency (ms) for three serving routes and saves them to
results/des_service_times_<dataset>.npz:
  - student        : distilled MLP forward pass
  - teacher        : submitted XGBoost (100 trees, depth 4) + TreeSHAP
  - teacher_large  : accurate XGBoost (500 trees, depth 8) + TreeSHAP

Usage: python measure_service_times.py [creditcard] [n_samples]
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch, shap
from rtxfair_sim.data import load_dataset
from rtxfair_sim.model import train_teacher, FairStudentXAI

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)


def time_route(fn, N):
    out = []
    for i in range(N):
        t0 = time.perf_counter(); fn(i); out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)


def main(name="creditcard", N=250):
    torch.set_num_threads(1)  # per-request server == single request
    b = load_dataset(name)
    teacher = train_teacher(b.X_train, b.y_train, 100, 4, 0.1)
    big = train_teacher(b.X_train, b.y_train, 500, 8, 0.02)
    ex_s, ex_l = shap.TreeExplainer(teacher), shap.TreeExplainer(big)
    model = FairStudentXAI(b.X_train.shape[1], b.X_train.shape[1]).eval()
    with torch.no_grad():
        _ = model(torch.tensor(b.X_test[:1]).float())
        student = time_route(lambda i: model(torch.tensor(b.X_test[i:i+1]).float()), N)
    teach = time_route(lambda i: (teacher.predict_proba(b.X_test[i:i+1]),
                                  ex_s.shap_values(b.X_test[i:i+1])), N)
    teach_l = time_route(lambda i: (big.predict_proba(b.X_test[i:i+1]),
                                    ex_l.shap_values(b.X_test[i:i+1])), N)
    out = os.path.join(RES, f"des_service_times_{name}.npz")
    np.savez(out, student=student, teacher=teach, teacher_large=teach_l)
    for nm, a in [("student", student), ("teacher", teach), ("teacher_large", teach_l)]:
        print(f"{nm:14s} mean={a.mean():.3f} ms  p99={np.percentile(a,99):.3f} ms")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "creditcard",
         int(sys.argv[2]) if len(sys.argv) > 2 else 250)
