# Resolving the Accuracy–Explainability–Fairness Trilemma via Homoscedastic Knowledge Distillation

Code, experiments, and paper for the WSC 2026 submission *Resolving the
Accuracy–Explainability–Fairness Trilemma in Financial Risk Scoring via
Homoscedastic Knowledge Distillation*.

An XGBoost **teacher** + TreeSHAP is distilled into a dual-head **student** MLP
that emits a risk score (label-supervised, BCE) and SHAP-like attributions
(distilled, MSE) in one forward pass, balanced by homoscedastic task uncertainty,
with an L1 demographic-parity penalty (safety-net masked) for active bias
mitigation. This repo adds a **scenario-based simulation study**, equal-opportunity
/ equalized-odds metrics, a batched-throughput benchmark, and a **discrete-event
deployment simulation**.

## Repository layout

```
rtxfair_sim/            reusable engine
  config.py             dataset configs, baselines, scenario grid
  data.py               notebook-faithful preprocessing (+ reads ./csvs), resampling, synthetic surrogate
  model.py              FairStudentXAI + teacher/SHAP helpers
  metrics.py            AUC, SHAP R2, DP / EO / EOdds gaps, latency, batched throughput
  train.py              student training loop (grad-norm tracking; DP or EO penalty)
  sim.py                scenario builders, runner, mean +/- 95% CI aggregation
  figures.py            high-res scenario-response figures
  dessim.py             discrete-event M/G/c queue (deployment sim)

run_chunk.py            resumable scenario runner  -> results/<dataset>_runs.csv
run_p2a.py              teacher tuning + AUC-anomaly ablation
run_p2c.py              batched-throughput benchmark
run_des.py              deployment simulation under Poisson load
measure_service_times.py  service-time distributions for run_des
run_smoke.py            offline end-to-end smoke test (synthetic surrogate)
export_data.py          fetch the three UCI datasets (needs internet)

notebooks/
  00_PROCESS_WALKTHROUGH.ipynb   narrated, step-by-step explanation (start here)
  {german,creditcard,bank}_REVISION.ipynb   per-dataset executable pipelines

results/                computed CSVs (means, CIs, sweeps, throughput)
figures/                generated figures (PNG + PDF, 300 dpi)
draft_text/             ready-to-paste paper sections (markdown)
paper/                  wsc26paper.tex, demobib.bib, fig_wsc_eloi/ (figures for the paper)
REVIEWER_CHECKLIST.md   reviewer-point -> action mapping
results_log.md          chronological log incl. flagged contradictions
csvs/                   raw UCI datasets (git-ignored; see below)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> NumPy is pinned `<2` for PyTorch-2.2 interop; XGBoost is pinned to avoid pulling
> GPU/NCCL wheels. CPU-only is sufficient.

## Data (not committed)

The UCI datasets are **git-ignored** (`csvs/`, `data/`). Two ways to provide them:

- **Fetch:** `pip install ucimlrepo && python export_data.py` (writes `data/*.csv`), or
- **Place raw files** in `csvs/`: `german.data`, `bank-full.csv`,
  `default of credit card clients.xls`. The loader reconstructs the exact
  `ucimlrepo` schema (German `Attribute1..20`, Credit Card `X1..X23`, Bank standard
  names), so results stay comparable to the paper's Table 1.

## Reproduce

```bash
# (0) offline sanity check — no data needed
python run_smoke.py

# (1) scenario study (resumable; repeat until it prints REMAINING 0)
python run_chunk.py german 40                 # full grid, 5 seeds, full budget
python run_chunk.py creditcard 40 70 core 3   # focused grid, reduced epochs
python run_chunk.py bank 40 60 core 3

# (2) AUC-anomaly: teacher tuning + ablation
python run_p2a.py german          # or creditcard / bank / all

# (3) throughput + deployment simulation
python run_p2c.py all
python measure_service_times.py creditcard
python run_des.py creditcard 50 1
```

On a normal multi-core / GPU machine, Credit Card and Bank can be run at the full
1200-epoch budget; the reduced budget here was only a constraint of the build
environment (trends are robust to it). `run_chunk.py` freezes-safe chunking exists
because the original environment paused background jobs between steps; on a normal
machine you can also use `run_study.py <dataset>` for a single long run.

## Key findings

- **Fairness does not improve accuracy.** Across seeds the AUC differences over the
  fairness weight λ are within 95% CI (German, Credit Card) or *negative* (Bank).
  The submitted "fair beats baseline" was a single-run artifact.
- **The DP penalty's effect depends on protected-group prevalence:** clean bias
  removal at no accuracy cost when the group is well represented (Credit Card,
  47%); a real accuracy trade-off when scarce (Bank, 2.7%); unreliable when tiny
  (German, 31 instances, heavy safety-net masking).
- **Demographic parity can conflict with equal opportunity** (Bank: DP gap falls
  while the EO gap rises) — motivating the EO-adapted penalty.
- **Deployment:** the distilled student sustains ~6,600 req/s within a 50 ms SLA
  vs ~250 req/s for an accurate teacher+TreeSHAP path (~27×); the *submitted* small
  teacher is itself fast, so the win is largest for accurate ensembles.

See `notebooks/00_PROCESS_WALKTHROUGH.ipynb` for the full story and
`REVIEWER_CHECKLIST.md` for the point-by-point response.

## Paper

`paper/wsc26paper.tex` (with `demobib.bib` and `fig_wsc_eloi/`). Compiles on
Overleaf with the WSC class file (`wscpaperproc.cls`, supplied by the WSC author kit).
```
