#!/usr/bin/env python3
"""
export_data.py  --  RUN THIS ONCE ON A MACHINE WITH INTERNET (UCI reachable).

It calls ucimlrepo.fetch_ucirepo for the three WSC datasets exactly as the
notebooks do, and dumps the *raw* feature/target frames to CSV. This guarantees
the simulation runner reproduces the identical schema, column order, and
sensitive-attribute columns used in the submitted paper (Table 1).

Output (next to this file):
    revision_outputs/data/german_X.csv     german_y.csv
    revision_outputs/data/creditcard_X.csv  creditcard_y.csv
    revision_outputs/data/bank_X.csv        bank_y.csv

Usage:
    pip install ucimlrepo pandas
    python export_data.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from ucimlrepo import fetch_ucirepo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

JOBS = {"german": 144, "creditcard": 350, "bank": 222}

for name, uci_id in JOBS.items():
    print(f"Fetching {name} (UCI id={uci_id}) ...")
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    X.to_csv(os.path.join(OUT, f"{name}_X.csv"), index=False)
    y.to_csv(os.path.join(OUT, f"{name}_y.csv"), index=False)
    print(f"  -> {name}_X.csv {X.shape}  |  {name}_y.csv {y.shape}")
    print(f"     columns: {list(X.columns)}")

print("\nDone. Move the revision_outputs/data folder into the project if needed.")
