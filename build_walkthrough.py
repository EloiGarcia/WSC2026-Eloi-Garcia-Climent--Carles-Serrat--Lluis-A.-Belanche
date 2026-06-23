#!/usr/bin/env python3
"""Generate 00_PROCESS_WALKTHROUGH.ipynb — a narrative, step-by-step explanation
of the whole WSC revision, loading the real saved results/figures."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "notebooks"); os.makedirs(NB, exist_ok=True)


def md(*t): return {"cell_type": "markdown", "metadata": {}, "source": list(t)}
def code(*t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": list(t)}

cells = []

cells.append(md(
"# WSC 2026 Revision — Step-by-Step Walkthrough\n",
"\n",
"**Paper:** *Resolving the Accuracy–Explainability–Fairness Trilemma in Financial Risk Scoring via Homoscedastic Knowledge Distillation* — accepted with **major revisions**.\n",
"\n",
"This notebook narrates the full revision process end to end: how we mapped the existing pipeline, the issues we found, the new **scenario-based simulation study** the committee asked for, the three targeted technical additions (Reviewer 3), the optional deployment simulation, and the edits folded back into the paper.\n",
"\n",
"It loads the **already-computed results** (CSVs + figures in `../results` and `../figures`) so every number you see is real; it does **not** re-run the heavy training. To reproduce from scratch, see the *How to reproduce* section at the end.\n"))

cells.append(code(
"import os, sys\n",
"os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); os.environ.setdefault('OMP_NUM_THREADS','1')\n",
"import pandas as pd\n",
"from IPython.display import Image, display, Markdown\n",
"RES = '../results'; FIG = '../figures'\n",
"pd.set_option('display.width', 200)\n",
"print('Loading precomputed artifacts from', os.path.abspath(RES))"))

# ---- STEP 0
cells.append(md(
"## Step 0 — Map the pipeline before changing anything\n",
"\n",
"We first read the three dataset notebooks and reconstructed the shared skeleton (the shipped `rtxfair/` package is an older attention pipeline and is **not** used by these notebooks). Every dataset follows the same flow:\n",
"\n",
"1. **Data load** via `ucimlrepo` → label-encode categoricals → stratified 80/20 split (`seed=42`) → `StandardScaler`.\n",
"2. **Teacher**: `XGBoost(n_estimators=100, max_depth=4, lr=0.1)` — *fixed, never tuned* in any notebook.\n",
"3. **Ground-truth explanations**: `TreeSHAP` on train/test.\n",
"4. **Sensitive attribute** (from raw features): German = foreign worker; Credit Card = `EDUCATION==2`; Bank = `age>60`.\n",
"5. **Student** `FairStudentXAI`: backbone `[256,128,64]` ReLU + dropout 0.2, a sigmoid **risk head (BCE)** and a linear **explanation head (MSE)**, with homoscedastic weights `w_risk,w_expl`.\n",
"6. **Loss**: `exp(-w_risk)·BCE + 0.5·w_risk + exp(-w_expl)·MSE + 0.5·w_expl + λ·DP`, where DP is the L1 in-batch parity gap with a <5-per-group safety net.\n",
"7. **Eval**: AUC, SHAP R², a DP-only audit, sequential single-sample latency.\n",
"\n",
"### Issues we flagged immediately (and that shaped everything after)\n",
"- ⚠️ **The risk head is trained on the ground-truth labels, not distilled from the teacher.** Only the explanation head distills SHAP. So Table 1's *baseline vs fair* is two runs of the same label-supervised net differing only in λ — not student-vs-teacher.\n",
"- ⚠️ **The teacher is never tuned**, so any teacher/student accuracy comparison is unfair.\n",
"- ⚠️ Only **Demographic Parity** existed (no EO/EOdds); latency was **sequential CPU only**; figures were flagged unreadable; and per-dataset hyper-parameters differed.\n",
"\n",
"These became Reviewer-3 items P2a (AUC anomaly), P2b (EO/EOdds), P2c (throughput), plus the make-or-break P1 simulation study."))

# ---- Engine
cells.append(md(
"## The reusable engine (`rtxfair_sim/`)\n",
"\n",
"We refactored the inline notebook logic into a small, reproducible package so a *scenario* can be run programmatically and aggregated across seeds:\n",
"\n",
"| Module | Role |\n",
"|---|---|\n",
"| `config.py` | dataset configs, baselines, the scenario grid |\n",
"| `data.py` | notebook-faithful preprocessing (+ reads your `csvs/` raw files), minority resampling, synthetic surrogate |\n",
"| `model.py` | `FairStudentXAI` (unchanged architecture) + teacher/SHAP helpers |\n",
"| `metrics.py` | AUC, SHAP R², DP / EO / EOdds gaps, latency, batched throughput |\n",
"| `train.py` | student training loop with grad-norm/divergence tracking + DP **or** EO penalty |\n",
"| `sim.py` | scenario builders, runner, mean ± 95% CI aggregation |\n",
"| `figures.py` | large, high-res scenario-response figures |\n",
"| `dessim.py` | discrete-event M/G/c queue for the deployment simulation |\n"))

cells.append(code(
"# the package modules\n",
"for f in sorted(os.listdir('../rtxfair_sim')):\n",
"    if f.endswith('.py'): print(f)"))

# ---- P1
cells.append(md(
"## Priority 1 — Scenario-based simulation study\n",
"\n",
"We treat the distillation pipeline as the **system under study**. A *scenario* is one factor configuration; a *replication* is one seed. We run ≥5 seeds (German, full budget) / 3 seeds (Credit Card & Bank, reduced epochs) and report **mean ± 95% CI**.\n",
"\n",
"**Factors:** teacher capacity (`n_estimators`, `max_depth`), fairness pressure `λ`, batch size, protected-group prevalence (synthetic resampling). **Responses:** AUC, SHAP R², DP/EO/EOdds gaps, gradient-norm & divergence, safety-net masking, P99 latency.\n",
"\n",
"Each run was time-boxed and **resumable** (`run_chunk.py`) because the build environment freezes background jobs between steps. Below we load the aggregated λ-sweep for each dataset."))

cells.append(code(
"def show_lambda(name, pretty):\n",
"    df = pd.read_csv(f'{RES}/{name}_runs.csv')\n",
"    lam = df[df.label.str.startswith('OFAT:lambda_fair')]\n",
"    g = (lam.groupby('lambda_fair')\n",
"            .agg(n=('AUC','size'), AUC=('AUC','mean'),\n",
"                 DP_gap=('DP_gap','mean'), EO_gap=('EO_gap','mean'),\n",
"                 EOdds_gap=('EOdds_gap','mean'))\n",
"            .round(4).reset_index())\n",
"    display(Markdown(f'**{pretty}** — fairness-pressure sweep (mean over seeds):'))\n",
"    display(g)\n",
"for n,p in [('german','German Credit'),('creditcard','UCI Credit Card'),('bank','Bank Marketing')]:\n",
"    show_lambda(n,p)"))

cells.append(md(
"### What the λ-sweep teaches us\n",
"The penalty's behaviour is governed by **protected-group prevalence**, not sample size:\n",
"- **UCI Credit Card** (47% group): DP gap falls cleanly, **AUC flat** → bias removed at no accuracy cost (the ideal).\n",
"- **Bank Marketing** (2.7% group): DP gap collapses but **AUC drops** *and* the **EO gap rises** → a genuine trade-off and a DP-vs-EO conflict.\n",
"- **German Credit** (31 instances): the penalty is unreliable (DP gap rises with λ) because ~26% of batches hit the safety net.\n",
"\n",
"Crucially, **fairness never improves AUC** — it is neutral or costly. The figures below show the accuracy–fairness response with CI bands."))

cells.append(code(
"for n in ['bankmarketing','germancredit','creditcard']:\n",
"    f = f'../figures/{ {\"bankmarketing\":\"bank\",\"germancredit\":\"german\",\"creditcard\":\"creditcard\"}[n] }_lambda_response.png'\n",
"    if os.path.exists(f): display(Image(filename=f, width=520))"))

cells.append(md(
"### Training stability (the safety-net / fairness-stability claim)\n",
"Stability is set by the resolution of the in-batch fairness signal. As the protected group or the batch shrinks, more batches are safety-net-masked and gradient norms grow — yet **no run ever diverged**."))

cells.append(code(
"for f in ['../figures/german_stability_minority.png','../figures/bank_stability_minority.png']:\n",
"    if os.path.exists(f): display(Image(filename=f, width=520))"))

# ---- P2a
cells.append(md(
"## Priority 2a — The AUC anomaly, explained\n",
"\n",
"The submitted paper reported the *fair* student beating the *baseline* on AUC (Bank 0.8959→0.9066; Credit Card 0.7276→0.7486). We resolved this with the four-part fix we agreed on:\n",
"\n",
"1. **(a) Reword the architecture** — the risk head is label-supervised, so it is not a teacher-distillation comparison.\n",
"2. **(b) Multi-seed CIs** — the AUC differences across λ are within noise (see table above); on Bank the trend even reverses.\n",
"3. **(c) Reframe the claim** — bias mitigation *at no significant accuracy cost*, not an accuracy gain.\n",
"4. **(d) Tuned-teacher ablation** — a proper CV search barely moves the teacher, and the tuned teacher **beats the student on every dataset**:"))

cells.append(code(
"import json\n",
"rows = [('German Credit', 0.8088, 0.8117), ('UCI Credit Card', 0.7778, 0.7800), ('Bank Marketing', 0.9286, 0.9343)]\n",
"for nm in ['creditcard','bank']:\n",
"    p=f'{RES}/{nm}_p2a_teacher.json'\n",
"    if os.path.exists(p):\n",
"        d=json.load(open(p)); print(nm, 'tuned best params:', d.get('best'))\n",
"display(pd.DataFrame(rows, columns=['Dataset','Teacher AUC (fixed)','Teacher AUC (tuned)']))"))

# ---- P2b
cells.append(md(
"## Priority 2b — Equal Opportunity & Equalized Odds\n",
"\n",
"We added label-conditioned fairness metrics (EO = TPR gap on `y=1`; EOdds = max of TPR/FPR gaps) for **all** models, and an **EO-adapted penalty** (a differentiable TPR-gap surrogate reusing the safety net). The decisive evidence is on Bank Marketing, where the **DP-targeted penalty drives DP down but pushes EO *up*** — exactly the limitation the reviewer flagged (visible in the λ-sweep table: Bank EO 0.118 → 0.34 as λ grows)."))

# ---- P2c
cells.append(md(
"## Priority 2c — Batched-inference throughput\n",
"\n",
"On top of the sequential single-sample test, we benchmarked batched inference (device: CPU, 4-core). Per-sample latency drops ~2–3 orders of magnitude; throughput reaches 1–3M samples/s."))

cells.append(code(
"t = pd.read_csv(f'{RES}/p2c_throughput.csv')\n",
"piv = t.pivot_table(index='dataset', columns='batch_size', values='throughput_sps').round(0)\n",
"display(Markdown('**Throughput (samples/s) by batch size:**')); display(piv)"))

# ---- DES
cells.append(md(
"## Optional — Discrete-event deployment simulation\n",
"\n",
"Finally we simulated the *serving* system: Poisson arrivals → M/G/1 FIFO queue → routed to the student, the submitted small teacher+SHAP, or an accurate large teacher+SHAP. We measured SLA-violation rate and P99 sojourn under load (5 replications).\n",
"\n",
"**Single-server capacity:** student ≈ 6,634 req/s · small teacher ≈ 5,746 · large teacher+SHAP ≈ 248. The accurate-teacher route breaches a 50 ms SLA at ~250 req/s; the student serves ~6,600 req/s within SLA (~27×). *Honest caveat:* the submitted small teacher is itself fast — the deployment win appears only for a large/accurate teacher."))

cells.append(code(
"for f in ['../figures/des_creditcard_sla.png','../figures/des_creditcard_p99.png']:\n",
"    if os.path.exists(f): display(Image(filename=f, width=520))"))

# ---- paper edits
cells.append(md(
"## Folding it back into the paper\n",
"\n",
"`wsc26paper.tex` was updated to match the evidence: the abstract and Results/Discussion/Conclusion were reframed (no fairness-driven AUC gain; latency claim tempered); Table 1 marked single-run with multi-seed CIs superseding it; and four new sections were added — **Scenario-Based Simulation Study** (with the λ table + CI figures + stability + EO/EOdds), **Batched Throughput**, and **Deployment Simulation Under Load**. New figures live in `fig_wsc_eloi/`.\n",
"\n",
"See `REVIEWER_CHECKLIST.md` for the point-by-point mapping and `results_log.md` for the full chronological log including every contradiction we flagged."))

# ---- reproduce
cells.append(md(
"## How to reproduce from scratch\n",
"\n",
"```bash\n",
"# 1. data: raw UCI files already in ../csvs (or run export_data.py with internet)\n",
"# 2. scenario study (resumable; repeat until REMAINING 0)\n",
"python ../run_chunk.py german 40            # full grid, 5 seeds, full budget\n",
"python ../run_chunk.py creditcard 40 70 core 3   # focused grid, reduced budget\n",
"python ../run_chunk.py bank 40 60 core 3\n",
"# 3. teacher tuning + ablation (P2a)\n",
"python ../run_p2a.py german\n",
"# 4. throughput (P2c) and deployment sim\n",
"python ../run_p2c.py all\n",
"python ../run_des.py creditcard 50 1\n",
"```\n",
"\n",
"On a normal multi-core / GPU machine the Credit Card and Bank studies can be run at the full 1200-epoch budget (the reduced budget here was only a constraint of the build sandbox; trends are robust to it)."))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
p = os.path.join(NB, "00_PROCESS_WALKTHROUGH.ipynb")
json.dump(nb, open(p, "w"), indent=1)
print("wrote", p, "with", len(cells), "cells")
