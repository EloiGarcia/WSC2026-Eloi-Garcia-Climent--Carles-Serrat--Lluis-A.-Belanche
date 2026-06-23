# Running results log — WSC revision

Chronological log of what was run, what was found, and anything that contradicts
or qualifies the submitted paper. Findings that need honest reframing are tagged
**⚠ CONTRADICTION** or **⚠ CAVEAT**.

## Environment
- Sandbox: Linux aarch64, 4 CPU, CPU-only (no GPU/MPS). torch 2.2.2, xgboost 2.0.3,
  shap 0.49.1, numpy 1.26.4, scikit-learn 1.7.2.
- UCI/OpenML/GitHub blocked in sandbox → datasets supplied locally via
  `export_data.py` / `data/*.csv`.

## Code map (revision_outputs/rtxfair_sim/)
- `config.py` — dataset configs, baselines, scenario grid.
- `data.py` — notebook-faithful preprocessing + minority resampling + synthetic surrogate.
- `model.py` — FairStudentXAI (unchanged arch) + teacher/SHAP helpers.
- `metrics.py` — AUC, SHAP R², DP / mean-score / EO / EOdds gaps, latency, throughput.
- `train.py` — student loop with grad-norm/divergence tracking + DP/EO penalty.
- `sim.py` — scenario builders, runner, mean±95%CI aggregation.
- `figures.py` — high-res scenario-response figures.

## Findings flagged from STEP 0 code read (before running)
- **⚠ CONTRADICTION (P2a):** the student's RISK head is trained on ground-truth
  labels (BCE), not on the teacher's predictions; only the explanation head
  distills SHAP. So "fair student beats baseline AUC" compares a fresh
  label-trained MLP against the XGBoost teacher — and the teacher uses FIXED,
  untuned hyperparameters (`n_estimators=100, max_depth=4, lr=0.1`) in all three
  notebooks. This is the leading explanation to verify.
- **⚠ CAVEAT:** per-dataset training budgets differ (German 450 ep / batch 256 /
  λ=1.5; Credit Card & Bank 1200 / 512 / λ=2.5). Baseline anchored per dataset to
  preserve Table-1 comparability.
- **⚠ CAVEAT:** "homoscedastic" weighting covers only risk+SHAP; the fairness term
  uses a fixed λ, not learned uncertainty. Title slightly overclaims.
- Only Demographic Parity was implemented; EO/EOdds added in this revision (P2b).
- Latency test was sequential single-sample CPU only; batched throughput added (P2c).
- Minor bug: German notebook cell 8 references `history["w_fair"]` (never created) → crashes.

## Smoke test (synthetic surrogate) — PASSED
- Full pipeline runs: data→teacher→SHAP→student(DP & EO)→metrics→resample→
  throughput→sweep→aggregate→figure.
- Sanity: DP penalty minimised DP gap; EO penalty minimised EO gap (directionally correct).
- Minority resampling hit target fractions exactly (1/3/5/10%).

## Data
- Raw UCI files supplied in `csvs/` (german.data; bank-full.csv = 45,211;
  "default of credit card clients.xls" = 30,000). `data.py` reconstructs the
  ucimlrepo schema (Attribute1..20 / X1..X23 / standard bank names).
- Fixed-teacher test AUC reproduces the paper baselines: Credit Card 0.7778,
  Bank 0.9286 (German 0.8088). Sensitive groups: German native=31 (tiny),
  Credit Card University=46.9%, Bank senior=2.7%.

## Sandbox compute constraint (important)
- Background processes are FROZEN between tool calls; only ~40 s of compute runs
  per call. German (3.3 s/run) was completed here in resumable chunks
  (`run_chunk.py`). Credit Card / Bank are ~180 s per full 1200-epoch run, so a
  180-run study cannot complete in-sandbox — those are delivered as runnable code
  (`run_chunk.py <name>`, minutes on a normal multi-core / GPU machine).

## P1 scenario study — GERMAN CREDIT (COMPLETE, 180 runs, 5 seeds)
- λ sweep (mean±95%CI AUC): λ0=0.7997±0.016, λ0.5=0.788, λ1=0.788, λ2.5=0.792±0.007,
  λ5=0.786. **All overlap within CI → fairness pressure does NOT change AUC
  significantly.**
- **⚠ CONTRADICTION:** the DP penalty does NOT reliably shrink the test DP gap on
  German — DP gap rises from 0.114 (λ0) to 0.188 (λ5). Cause: protected group=31,
  so 25.6% of batches are safety-net-masked and the in-batch parity signal is weak.
  Mild λ (≤1) gives a tiny reduction (0.114→0.103); strong λ destabilises it.
  Needs honest framing: the penalty's effectiveness is group-size dependent.
- EO gap ≈ EOdds gap on German (small native group → FPR diff not binding); both
  track DP behaviour (rise at high λ).
- **Stability (clean, supports the paper's stability claim):**
  minority 10%→1%: masked-batch frac 1.1%→93.6%, max grad-norm 11.8→47.2.
  batch 1024→64: masked-batch frac 0%→91.5%, max grad-norm 1.8→154.8.
  No run ever diverged (no NaN) — safety net holds.
- Teacher capacity n_est 50→200: SHAP R² 0.421→0.482 (better targets → better
  distillation), AUC flat (0.784→0.793) → label-trained risk head ~teacher-independent.

## P2a — GERMAN (teacher tuning + ablation)
- Teacher AUC fixed=0.8088 → tuned=0.8117 (negligible; German teacher already near
  optimal). Best params n_est=500, depth=6, lr=0.01, subsample/colsample=0.6.
- Student λ=0 AUC=0.7997 < teacher 0.8088 → **no AUC anomaly on German.**
- The submitted anomaly (fair>baseline) is Bank/Credit-Card-specific. Note: the
  FIXED teacher already beats BOTH students there (CC teacher 0.778 vs paper
  students 0.728/0.749; Bank teacher 0.929 vs 0.896/0.907), so the anomaly is a
  within-noise λ0-vs-λ>0 student difference, not a fairness-driven accuracy gain.
  → run `run_p2a.py creditcard|bank` to confirm with multi-seed CIs.

## P2c — batched throughput, ALL THREE (device=CPU, 4-core aarch64)
- Sequential single-sample P99: German 0.277 ms, CC 0.227 ms, Bank 0.242 ms.
- Batched: ~6.5k samp/s at bs=1 → 1.0–3.0 M samp/s at bs=4096 (≈300–400×
  per-sample speedup). Table: `results/p2c_throughput.csv`.

## P1 — CREDIT CARD & BANK (COMPLETE, focused core grid: λ+batch+minority, 3 seeds)
Reduced budget vs German due to sandbox limit: **3 seeds, 60–70 epochs** (German
has 5 seeds / full budget / full grid + factorial). Trends are the deliverable;
absolute AUC is epoch-budget-dependent and not directly comparable to the
1200-epoch Table 1 — re-run `run_chunk.py` at full budget for headline cells.

CREDIT CARD (University 46.9% — well balanced):
- λ sweep: AUC flat 0.767→0.768 (CI ~0.003); **DP gap falls 0.037→0.013**, EO gap
  0.076→0.022. → the DP penalty works cleanly here, bias removed at ~zero AUC cost.
  The IDEAL trilemma result (contrast with German where the penalty failed).
- batch: masked-frac 0% everywhere (group never <5); small batch lowers DP gap but
  grad-norm 98 (bs64) vs 0.7 (bs1024) — noise vs stability trade-off.
- minority 10→1%: masked-frac 0→0.44, AUC 0.764→0.743 — degrades under scarcity.

BANK (senior 2.7% — imbalanced):
- λ sweep: **a real trade-off** — DP gap collapses 0.325→0.035 but **AUC declines
  significantly 0.926→0.897** (CI ~0.004). 
- **⚠ KEY:** optimising DP **worsens EO**: EO gap 0.118 (λ0) → 0.49 (λ0.5) → 0.22
  (λ5); EOdds similar. Motivates the P2b EO penalty directly.
- **⚠ CONTRADICTION confirmed:** the paper's Bank "fair beats baseline" AUC
  (0.8959→0.9066) does NOT reproduce — multi-seed shows fairness *reduces* AUC.
  The original was a single-run artefact (epoch budget differs, but the direction
  is the point).
- batch/minority stability mirrors German (bs64 masks 97%, grad-norm 68).

## P2a — teacher tuning, ALL THREE
- German 0.8088→0.8117 · Credit Card 0.7778→0.7800 · Bank 0.9286→0.9343.
- In every case the (tuned) teacher BEATS the student, and λ does not raise student
  AUC → the submitted anomaly is not a fairness-driven accuracy gain. Files:
  `results/{creditcard,bank}_p2a_teacher.json`, `results/german_p2a_summary.csv`.

## DES deployment simulation (optional P1 add-on) — built + run (Credit Card)
- M/G/1 queue, Poisson arrivals, measured per-request service times, 5 reps.
- Single-server capacity: student 6,634 req/s; teacher_fixed(100,d4) 5,746;
  teacher_large(500,d8)+SHAP 248. Large-teacher route breaches the 50 ms SLA at
  ~250 req/s (violation 0→1.0 by 300); student serves ~6,600 req/s, 0 violations.
- **⚠ CAVEAT:** the *submitted* teacher (100,d4)+TreeSHAP is only ~1.2× slower than
  the student per request — distillation's serving advantage is large only for a
  big/accurate teacher (500,d8 → 27×). Corrects any "huge speedup for any teacher"
  impression. Files: `run_des.py`, `dessim.py`, `figures/des_creditcard_*`.

## Cross-dataset synthesis (honest)
The DP penalty's behaviour is governed by protected-group resolution:
- well-balanced group (Credit Card) → clean DP reduction, no AUC cost;
- small group (Bank 2.7%) → strong DP reduction but real AUC cost AND EO worsening;
- tiny group (German 31) → penalty fails / destabilises (gap rises).
Fairness never *improves* AUC (neutral or costly). This replaces the paper's claim.
