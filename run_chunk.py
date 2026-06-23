#!/usr/bin/env python3
"""
run_chunk.py -- RESUMABLE, time-boxed scenario runner.

Runs as many remaining (scenario x seed) cells as fit in a wall-clock budget,
appending to results/<name>_runs.csv, skipping cells already present. Safe to
call repeatedly (e.g. to work within a sandbox's per-call time limit) or once
on a normal machine.

Usage:
    python run_chunk.py <dataset> [budget_seconds] [epochs_override]
e.g. python run_chunk.py german 38
     python run_chunk.py creditcard 40 300     # reduced sweep budget
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
torch.set_num_threads(4)  # 4-core box; deterministic, ~25% faster than 1 thread
from rtxfair_sim import sim
from rtxfair_sim.config import DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

def main(name, budget=38.0, epochs_override=None, mode="all", nseeds=None):
    out = os.path.join(RES, f"{name}_runs.csv")
    from rtxfair_sim.config import SEEDS
    seeds = SEEDS[:nseeds] if nseeds else SEEDS
    # deterministic scenario list -> stable integer id 'sid' is the dedup key
    if mode == "core":
        scenarios = sim.build_core_scenarios(name, seeds=seeds)
    else:
        scenarios = (sim.build_ofat_scenarios(name, seeds=seeds)
                     + sim.build_factorial_scenarios(name, seeds=seeds))
    if epochs_override:
        for s in scenarios:
            s.epochs = int(epochs_override)

    target_epochs = int(epochs_override) if epochs_override else None
    done = set()
    rows = []
    if os.path.exists(out):
        prev = pd.read_csv(out)
        same_epochs = (target_epochs is None
                       or ("epochs" in prev.columns
                           and set(prev["epochs"].unique()) <= {target_epochs}))
        if "sid" in prev.columns and same_epochs:   # resume only if schema+budget match
            rows = prev.to_dict("records")
            done = set(int(s) for s in prev["sid"].tolist())
        # else: stale / different-epoch file -> start fresh (overwrite)

    bundle = sim.load_dataset(name)
    cache = sim.TeacherCache(bundle)
    todo = [(i, s) for i, s in enumerate(scenarios) if i not in done]
    print(f"[{name}] total={len(scenarios)} done={len(done)} remaining={len(todo)}")

    t0 = time.time()
    ran = 0
    last = 0.0
    for i, s in todo:
        # predictive stop: don't start a run that would likely overflow the budget
        if ran > 0 and (time.time() - t0) + last > budget:
            break
        ts = time.time()
        rec = sim.run_one(bundle, cache, s)
        last = time.time() - ts
        rec["sid"] = i
        rows.append(rec)
        ran += 1
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{name}] ran {ran} this chunk in {time.time()-t0:.1f}s; "
          f"total now {len(rows)}/{len(scenarios)}")
    return len(rows), len(scenarios)


if __name__ == "__main__":
    name = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
    epo = int(sys.argv[3]) if len(sys.argv) > 3 else None
    mode = sys.argv[4] if len(sys.argv) > 4 else "all"
    nseeds = int(sys.argv[5]) if len(sys.argv) > 5 else None
    n, tot = main(name, budget, epo, mode, nseeds)
    print("REMAINING", tot - n)
