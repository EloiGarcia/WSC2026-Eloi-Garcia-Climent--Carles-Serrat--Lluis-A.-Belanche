#!/usr/bin/env python3
"""
run_p2c.py -- Reviewer 3 (2c): batched-inference throughput benchmark.

Adds a batched benchmark on top of the existing sequential single-sample CPU
latency test. Sweeps batch sizes {1,32,128,512,4096}, measures throughput
(samples/sec) and per-sample latency. Device is detected and reported
(MPS/CUDA if available, else CPU). Produces a supplementary table.

Usage: python run_p2c.py [german|creditcard|bank|all]
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, torch
from rtxfair_sim.data import load_dataset
from rtxfair_sim.model import train_teacher, teacher_shap
from rtxfair_sim.config import DATASETS, ScenarioConfig
from rtxfair_sim.train import train_student
from rtxfair_sim import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_dataset(name, device):
    cfg = DATASETS[name]
    b = load_dataset(name)
    t = train_teacher(b.X_train, b.y_train, 100, 4, 0.1)
    sv_tr, sv_te, _ = teacher_shap(t, b.X_train, b.X_test)
    sc = ScenarioConfig(dataset=name, seed=42, lambda_fair=cfg.lambda_fair,
                        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
    _, model = train_student(b.X_train, b.y_train, sv_tr, b.sensitive_train,
                             b.X_test, b.y_test, sv_te, b.sensitive_test, sc)
    seq = M.sequential_p99_latency(model, b.X_test, device=device)
    rows = M.batched_throughput(model, b.X_test,
                                batch_sizes=(1, 32, 128, 512, 4096), device=device)
    for r in rows:
        r["dataset"] = name
        r["seq_p99_ms"] = seq["p99_ms"]
    print(f"[{cfg.pretty}] device={device} seq P99={seq['p99_ms']:.3f} ms")
    for r in rows:
        print(f"   bs={r['batch_size']:5d}  {r['throughput_sps']:>12.0f} sps  "
              f"{r['per_sample_ms']:.5f} ms/sample")
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DATASETS) if which == "all" else [which]
    device = detect_device()
    print(f"Benchmark device: {device}")
    allrows = []
    for n in names:
        allrows += run_dataset(n, device)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(RES, "p2c_throughput.csv"), index=False)
    print("\nP2c throughput table written.")
