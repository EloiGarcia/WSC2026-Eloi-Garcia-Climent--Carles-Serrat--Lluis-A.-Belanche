#!/usr/bin/env python3
"""
run_des.py -- discrete-event DEPLOYMENT simulation (optional P1 add-on).

Uses measured per-request service-time distributions (results/des_service_times_*.npz)
for three serving routes:
  - student            : distilled dual-head MLP forward pass
  - teacher_fixed      : submitted XGBoost (100 trees, depth 4) + TreeSHAP
  - teacher_large      : accurate/tuned XGBoost (500 trees, depth 8) + TreeSHAP
Poisson arrivals are swept across a load range; for each route x rate x seed we
record queue wait, p99 sojourn, throughput, utilisation and SLA-violation rate.

Usage: python run_des.py [creditcard] [sla_ms] [n_servers]
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rtxfair_sim import dessim
from rtxfair_sim.figures import _save, TEAL, RED, ORANGE

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")

plt.rcParams.update({"savefig.dpi": 300, "font.size": 15, "axes.titlesize": 18,
                     "axes.labelsize": 16, "legend.fontsize": 13, "lines.linewidth": 2.5,
                     "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"student": TEAL, "teacher_fixed": ORANGE, "teacher_large": RED}
LABELS = {"student": "Student (MLP)", "teacher_fixed": "Teacher+SHAP (100,d4)",
          "teacher_large": "Teacher+SHAP (500,d8)"}


def main(name="creditcard", sla_ms=50.0, n_servers=1):
    z = dict(np.load(os.path.join(RES, f"des_service_times_{name}.npz")))
    routes = {"student": z["student"], "teacher_fixed": z["teacher"],
              "teacher_large": z["teacher_large"]}
    # log-spaced arrival rates spanning both bottlenecks (req/s)
    lambdas = np.unique(np.round(np.logspace(np.log10(20), np.log10(6500), 16))).astype(int)
    seeds = [0, 1, 2, 3, 4]
    rows = dessim.sweep(routes, lambdas, seeds, n_servers=n_servers,
                        sla_ms=sla_ms, n_requests=20000)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"des_{name}_sweep.csv"), index=False)

    # aggregate mean +/- 95% CI
    from scipy import stats
    def agg(metric):
        out = []
        for (route, lam), g in df.groupby(["route", "lam"]):
            v = g[metric].values
            h = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
            out.append(dict(route=route, lam=lam, mean=np.mean(v), ci=h))
        return pd.DataFrame(out)

    # capacity table (1/mean service)
    cap = {r: 1000.0 / np.mean(s) for r, s in routes.items()}
    print("Single-server capacity (req/s):", {k: round(v) for k, v in cap.items()})

    # Figure 1: SLA-violation rate vs arrival rate
    a = agg("sla_violation_rate")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for r in routes:
        d = a[a.route == r].sort_values("lam")
        ax.plot(d.lam, d["mean"], "-o", color=COLORS[r], label=LABELS[r])
        ax.fill_between(d.lam, d["mean"] - d["ci"], d["mean"] + d["ci"],
                        color=COLORS[r], alpha=0.18)
    ax.set_xscale("log"); ax.set_xlabel("Arrival rate (requests/s)")
    ax.set_ylabel(f"SLA-violation rate (>{sla_ms:.0f} ms)")
    ax.set_title(f"Deployment under load: SLA violations  ({name}, c={n_servers})")
    ax.legend()
    _save(fig, os.path.join(FIG, f"des_{name}_sla"))

    # Figure 2: p99 sojourn vs arrival rate
    a2 = agg("p99_sojourn_ms")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for r in routes:
        d = a2[a2.route == r].sort_values("lam")
        ax.plot(d.lam, d["mean"], "-o", color=COLORS[r], label=LABELS[r])
        ax.fill_between(d.lam, d["mean"] - d["ci"], d["mean"] + d["ci"],
                        color=COLORS[r], alpha=0.18)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(sla_ms, ls="--", color="gray", label=f"SLA {sla_ms:.0f} ms")
    ax.set_xlabel("Arrival rate (requests/s)"); ax.set_ylabel("P99 sojourn latency (ms)")
    ax.set_title(f"Deployment under load: tail latency  ({name}, c={n_servers})")
    ax.legend()
    _save(fig, os.path.join(FIG, f"des_{name}_p99"))
    return df, cap


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "creditcard"
    sla = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    c = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    df, cap = main(name, sla, c)
    print("\nSLA-violation rate at selected rates:")
    piv = (df.groupby(["route", "lam"]).sla_violation_rate.mean().reset_index()
             .pivot(index="lam", columns="route", values="sla_violation_rate"))
    print(piv.round(3).to_string())
