#!/usr/bin/env python3
"""Generate the clearly-flagged *_REVISION.ipynb notebooks (one per dataset).
Each notebook is a thin, reproducible driver over revision_outputs/rtxfair_sim/.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "notebooks"); os.makedirs(NB, exist_ok=True)
DATASETS = {"german": "German Credit", "creditcard": "UCI Credit Card", "bank": "Bank Marketing"}


def md(*t): return {"cell_type": "markdown", "metadata": {}, "source": list(t)}
def code(*t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": list(t)}


def notebook(name, pretty):
    cells = [
        md(f"# WSC 2026 — REVISION notebook: {pretty}\n",
           "\n",
           "**This is the revised pipeline for the major-revision response.** It reuses the\n",
           "original seeds (42), stratified 80/20 split, and architecture ([256,128,64], ReLU,\n",
           "dropout 0.2) so results stay comparable to Table 1. New: scenario-based simulation\n",
           "study (P1), teacher tuning + AUC-anomaly ablation (P2a), EO/EOdds metrics + EO penalty\n",
           "(P2b), batched throughput (P2c). All logic lives in `revision_outputs/rtxfair_sim/`.\n"),
        code("import os\n",
             'os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")\n',
             'os.environ.setdefault("OMP_NUM_THREADS", "1")\n',
             "import sys; sys.path.insert(0, os.path.abspath('..'))\n",
             "import pandas as pd, numpy as np\n",
             "from rtxfair_sim import sim, figures, metrics as M\n",
             "from rtxfair_sim.data import load_dataset\n",
             "from rtxfair_sim.model import train_teacher, teacher_shap, FairStudentXAI\n",
             "from rtxfair_sim.train import train_student\n",
             "from rtxfair_sim.config import DATASETS, ScenarioConfig, PENALTY_DP, PENALTY_EO\n",
             f'NAME = "{name}"\n',
             "cfg = DATASETS[NAME]; print(cfg)"),
        md("## 0. Data (reads csvs/ raw files; reconstructs the ucimlrepo schema)"),
        code("b = load_dataset(NAME)\n",
             "print('train', b.X_train.shape, 'protected g1=', int(b.sensitive_train.sum()),\n",
             "      'g0=', int((b.sensitive_train==0).sum()))"),
        md("## 1. Baseline reproduction (teacher + baseline student), with EO/EOdds added"),
        code("teacher = train_teacher(b.X_train, b.y_train, 100, 4, 0.1)\n",
             "from sklearn.metrics import roc_auc_score\n",
             "print('fixed teacher AUC', roc_auc_score(b.y_test, teacher.predict_proba(b.X_test)[:,1]))\n",
             "sv_tr, sv_te, _ = teacher_shap(teacher, b.X_train, b.X_test)\n",
             "sc = ScenarioConfig(dataset=NAME, seed=42, lambda_fair=cfg.lambda_fair,\n",
             "                    epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)\n",
             "rec, model = train_student(b.X_train, b.y_train, sv_tr, b.sensitive_train,\n",
             "                           b.X_test, b.y_test, sv_te, b.sensitive_test, sc)\n",
             "print({k: round(rec[k],4) for k in ['AUC','SHAP_R2','DP_gap','EO_gap','EOdds_gap','p99_latency_ms']})"),
        md("## P1. Scenario-based simulation study (OFAT + factorial, 5 seeds)\n",
           "Run from the shell for long jobs: `python ../run_chunk.py %s` (repeat until REMAINING 0).\n"
           "On a normal multi-core/GPU machine you can run inline:" % name),
        code("# Inline run (German finishes in ~10 min; for creditcard/bank prefer run_chunk.py).\n",
             "scenarios = sim.build_ofat_scenarios(NAME) + sim.build_factorial_scenarios(NAME)\n",
             "# df = sim.run_scenarios(NAME, scenarios, out_csv=f'../results/{NAME}_runs.csv')\n",
             "df = pd.read_csv(f'../results/{NAME}_runs.csv')  # load precomputed if present\n",
             "print(len(df), 'runs')"),
        code("def sub(p): return df[df.label.str.startswith(p)]\n",
             "agg_lambda = sim.aggregate(sub('OFAT:lambda_fair'), ['lambda_fair'])\n",
             "agg_minor  = sim.aggregate(sub('OFAT:minority_fraction'), ['minority_fraction'])\n",
             "agg_batch  = sim.aggregate(sub('OFAT:batch_size'), ['batch_size'])\n",
             "display(agg_lambda[['lambda_fair','n_rep','AUC_mean','AUC_ci95','DP_gap_mean','EO_gap_mean']].round(4))"),
        code("figures.dual_response_vs_lambda(agg_lambda, f'../figures/{NAME}_lambda_response', cfg.pretty)\n",
             "figures.pareto_auc_dpgap(agg_lambda, f'../figures/{NAME}_pareto', cfg.pretty)\n",
             "figures.stability_vs_minority(agg_minor, f'../figures/{NAME}_stability_minority', cfg.pretty)\n",
             "print('figures written to ../figures/')"),
        md("## P2a. Teacher tuning + AUC-anomaly ablation\n",
           "`python ../run_p2a.py %s`  (RandomizedSearch via manual CV; multi-seed λ=0 vs λ=baseline)." % name),
        md("## P2b. EO / Equalized-Odds penalty variant\n",
           "Same setup, `penalty=PENALTY_EO` (TPR-difference surrogate, same safety net)."),
        code("sc_eo = ScenarioConfig(dataset=NAME, seed=42, lambda_fair=cfg.lambda_fair,\n",
             "                       epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,\n",
             "                       penalty=PENALTY_EO)\n",
             "rec_eo, _ = train_student(b.X_train, b.y_train, sv_tr, b.sensitive_train,\n",
             "                          b.X_test, b.y_test, sv_te, b.sensitive_test, sc_eo)\n",
             "print('DP-penalty :', {k: round(rec[k],4)    for k in ['AUC','DP_gap','EO_gap','EOdds_gap']})\n",
             "print('EO-penalty :', {k: round(rec_eo[k],4) for k in ['AUC','DP_gap','EO_gap','EOdds_gap']})"),
        md("## P2c. Batched-inference throughput (device noted)"),
        code("rows = M.batched_throughput(model, b.X_test, batch_sizes=(1,32,128,512,4096), device='cpu')\n",
             "display(pd.DataFrame(rows))"),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                         "name": "python3"},
                          "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


for name, pretty in DATASETS.items():
    p = os.path.join(NB, f"{name}_REVISION.ipynb")
    with open(p, "w") as f:
        json.dump(notebook(name, pretty), f, indent=1)
    print("wrote", p)
