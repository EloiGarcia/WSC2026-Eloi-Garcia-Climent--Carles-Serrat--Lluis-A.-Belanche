#!/usr/bin/env python3
"""
run_study.py -- PRIORITY 1 scenario-based simulation study on the real datasets.

Runs OFAT sweeps (teacher capacity, lambda_fair, batch size, minority fraction)
plus one 2-factor factorial (lambda x minority), >=5 seeds each, at each dataset's
baseline epoch budget. Writes tidy per-run CSVs and mean+/-95%CI aggregates, and
the scenario-response figures.

Usage:
    python run_study.py [german|creditcard|bank|all]
Requires revision_outputs/data/<name>_{X,y}.csv  (see export_data.py).
Long-running: saves incrementally so partial progress is preserved.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from rtxfair_sim import sim, figures
from rtxfair_sim.config import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def run_dataset(name):
    pretty = DATASETS[name].pretty
    t0 = time.time()
    print(f"\n=== {pretty} : OFAT scenarios ===")
    ofat = sim.build_ofat_scenarios(name)
    df_ofat = sim.run_scenarios(name, ofat,
                                out_csv=os.path.join(RES, f"{name}_ofat_runs.csv"))
    print(f"=== {pretty} : factorial scenarios ===")
    fact = sim.build_factorial_scenarios(name)
    df_fact = sim.run_scenarios(name, fact,
                                out_csv=os.path.join(RES, f"{name}_factorial_runs.csv"))

    # aggregates (mean +/- 95% CI)
    agg_lambda = sim.aggregate(df_ofat[df_ofat["label"].str.startswith("OFAT:lambda_fair")],
                               ["lambda_fair"])
    agg_batch = sim.aggregate(df_ofat[df_ofat["label"].str.startswith("OFAT:batch_size")],
                              ["batch_size"])
    agg_minor = sim.aggregate(df_ofat[df_ofat["label"].str.startswith("OFAT:minority_fraction")],
                              ["minority_fraction"])
    agg_tn = sim.aggregate(df_ofat[df_ofat["label"].str.startswith("OFAT:teacher_n_estimators")],
                           ["teacher_n_estimators"])
    agg_td = sim.aggregate(df_ofat[df_ofat["label"].str.startswith("OFAT:teacher_max_depth")],
                           ["teacher_max_depth"])
    agg_fact = sim.aggregate(df_fact, ["lambda_fair", "minority_fraction"])

    for tag, a in [("lambda", agg_lambda), ("batch", agg_batch),
                   ("minority", agg_minor), ("teacher_nest", agg_tn),
                   ("teacher_depth", agg_td), ("factorial", agg_fact)]:
        a.to_csv(os.path.join(RES, f"{name}_agg_{tag}.csv"), index=False)

    # figures
    figures.dual_response_vs_lambda(agg_lambda, os.path.join(FIG, f"{name}_lambda_response"), pretty)
    figures.pareto_auc_dpgap(agg_lambda, os.path.join(FIG, f"{name}_pareto"), pretty)
    figures.stability_vs_minority(agg_minor, os.path.join(FIG, f"{name}_stability_minority"), pretty)
    figures.response_vs_factor(agg_batch, "batch_size", "DP_gap_mean", "DP_gap_ci95",
                               os.path.join(FIG, f"{name}_batch_dpgap"),
                               title=f"DP gap vs batch size  {pretty}",
                               xlabel="Batch size", ylabel="DP gap", logx=True)
    print(f"[{pretty}] done in {(time.time()-t0)/60:.1f} min")
    return df_ofat, df_fact


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DATASETS) if which == "all" else [which]
    for nm in names:
        run_dataset(nm)
    print("\nALL DONE")
