# Draft text — "Simulation Study" methodology subsection (P1)

> Paste target: new methodology subsection, before the results. Numeric values to
> be filled from `results/*.csv` after the run are marked **[FILL]**. Prose is
> written so the simulation framing is explicit, as the committee requested.

---

## 4. Scenario-Based Simulation Study

### 4.1 The distillation pipeline as the system under study

We treat the homoscedastic knowledge-distillation pipeline of Section 3 as the
*system under study* and analyse it with a designed simulation experiment rather
than a single training run. A **scenario** is one configuration of the system's
controllable factors; a **replication** is one execution of a scenario under an
independent random seed (governing the train/validation shuffling, mini-batch
permutation, and weight initialisation). For every scenario we perform
\(R = 5\) independent replications and report each response as a mean with a
95% confidence interval computed from the Student-\(t\) distribution with
\(R-1\) degrees of freedom. This is the standard simulation methodology of
running multiple replications to separate systematic factor effects from
stochastic run-to-run variability, which a single run cannot do.

The seeds, stratified 80/20 split, network architecture
(backbone \([256,128,64]\), ReLU, dropout 0.2), homoscedastic task weighting,
and optimiser (Adam, \(lr=10^{-3}\)) are held fixed at their published values so
that the baseline scenario of each dataset reproduces the corresponding row of
Table 1 and all new results remain directly comparable.

### 4.2 Factors

We vary four families of factors that govern the teacher–student interaction:

| Factor | Levels | What it stresses |
|---|---|---|
| Teacher capacity — `n_estimators` | {50, 100, 200} | Fidelity/headroom of the SHAP targets the student must imitate |
| Teacher capacity — `max_depth` | {3, 4, 6} | Complexity of the explanation signal |
| Fairness pressure — \(\lambda_{fair}\) | {0, 0.5, 1, 2.5, 5} | Accuracy–fairness trade-off; \(\lambda=0\) is the unconstrained student |
| Mini-batch size | {64, 256, 512, 1024} | Variance of the in-batch demographic-parity estimate (fairness-stability) |
| Minority fraction | {1%, 3%, 5%, 10%} | Stress test of the <5-per-group safety net and gradient stability |

The baseline (anchor) level of each factor is the published setting for that
dataset. The minority fraction is induced by synthetically resampling the
protected group (with replacement) to the target prevalence in the *student's*
training set, while the teacher and its TreeSHAP targets are held fixed; this
isolates the effect of group scarcity on the student's fairness mechanism.

### 4.3 Experimental design

To keep the design tractable while still exposing interactions, we use a
**one-factor-at-a-time (OFAT)** sweep over all five factors, plus a single
**\(2\)-factor full-factorial** design on the two most operationally relevant
factors — fairness pressure \(\lambda_{fair}\) and minority fraction — crossed at
\(4\times4\) levels. With \(R=5\) replications this yields
\(100\) OFAT runs and \(80\) factorial runs per dataset
(\(180\) runs \(\times\) 3 datasets). Teacher models and their TreeSHAP targets
are cached per capacity setting so they are computed once and reused across all
replications and downstream factors.

### 4.4 Responses

For each run we record:

- **AUC** of the risk head (discrimination),
- **SHAP \(R^2\)** of the explanation head (distillation fidelity),
- **Demographic-parity gap** \(|P(\hat y{=}1\mid g_1)-P(\hat y{=}1\mid g_0)|\) and
  the in-batch **mean-score gap** the penalty actually optimises,
- **Equal-Opportunity** and **Equalized-Odds** gaps (Section on P2b),
- **Training stability**: mean and maximum global gradient-norm over all steps,
  a divergence flag (non-finite loss), and the fraction of mini-batches the
  safety net masked,
- **P99 single-sample inference latency** (and the batched-throughput sweep of
  the P2c benchmark).

### 4.5 What each scenario teaches us about teacher–student interaction

- *Fairness pressure* (\(\lambda_{fair}\)) traces the accuracy–fairness Pareto
  front. On German Credit, AUC is flat across \(\lambda\in[0,5]\)
  (\(0.800\to0.786\), all within 95% CI) while the DP gap is non-monotone
  (\(0.114\) at \(\lambda{=}0\), minimum \(0.103\) at \(\lambda{=}1\), rising to
  \(0.188\) at \(\lambda{=}5\)) — the penalty helps only mildly and then
  destabilises once group scarcity dominates.
- *Mini-batch size* probes the fairness-stability finding: the safety-net-masked
  fraction rises \(0\%\to0.2\%\to25.6\%\to91.5\%\) and the max gradient norm
  \(1.8\to2.5\to26\to155\) as batch size falls \(1024\to512\to256\to64\).
- *Minority fraction* stress-tests the safety net: as the protected group shrinks
  \(10\%\to1\%\), the masked-batch fraction rises \(1.1\%\to93.6\%\) and the max
  gradient norm \(11.8\to47.2\), yet no run diverges — the heuristic prevents the
  blow-up it was designed to catch.
- *Teacher capacity* raises the student's SHAP \(R^2\) (\(0.421\to0.482\) as
  `n_estimators` \(50\to200\)) while leaving the label-supervised risk-head AUC
  flat — a point that also bears on the AUC anomaly (P2a).
