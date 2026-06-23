# Draft text — "Simulation Study: Results" subsection (P1)

> German Credit values are final (mean ± 95% CI over 5 seeds, `results/german_agg_*.csv`).
> Credit Card / Bank cells marked **[RUN]** are produced by `run_chunk.py <name>`
> on a multi-core/GPU machine and slot directly into the same tables/figures.

---

## 5. Simulation Results

### 5.1 Fairness pressure (\(\lambda_{fair}\))

Sweeping \(\lambda_{fair}\in\{0,0.5,1,2.5,5\}\) over five replications, the
risk-head AUC on German Credit is statistically flat: \(0.800\pm0.016\) at
\(\lambda=0\), \(0.792\pm0.007\) at \(\lambda=2.5\), and \(0.786\pm0.012\) at
\(\lambda=5\) — every interval overlaps. Fairness pressure therefore does **not**
change discrimination performance to within replication noise; it neither buys
nor costs accuracy on this dataset. This is the multi-seed result that replaces
the earlier single-run comparison and motivates the reframing in Section 6 (P2a).

The demographic-parity response is more nuanced and, importantly, *not*
monotone-improving on German: the test DP gap is \(0.114\) at \(\lambda=0\),
dips slightly to \(0.103\) at \(\lambda=1\), then **rises** to \(0.157\)
(\(\lambda=2.5\)) and \(0.188\) (\(\lambda=5\)). The cause is structural: German's
protected group contains only 31 training instances, so 25.6% of mini-batches
trip the <5-per-group safety net and contribute no fairness gradient, while the
remaining in-batch parity estimates are high-variance. We report this honestly —
**the penalty's effectiveness is contingent on protected-group size**, and on
severely imbalanced data a large \(\lambda\) can destabilise rather than improve
parity.

The other two datasets confirm that protected-group resolution is the governing
variable (3 seeds, reduced epoch budget; trends, not absolute AUC):

- **UCI Credit Card** (University group, 46.9% — well balanced): the penalty
  behaves ideally. As \(\lambda\) increases the DP gap falls monotonically
  \(0.037\to0.013\) and the EO gap \(0.076\to0.022\), while AUC is flat
  (\(0.767\pm0.004\to0.768\)). This is bias removal at no measurable accuracy
  cost — the trilemma "resolved" in the favourable regime.
- **Bank Marketing** (senior group, 2.7% — imbalanced): a genuine trade-off. The
  DP gap collapses \(0.325\to0.035\) but AUC declines significantly
  \(0.926\to0.897\) (CI \(\approx0.004\)), and — importantly — the *Equal-Opportunity*
  gap is **worsened** by the DP-targeted penalty (\(0.118\) at \(\lambda{=}0\),
  rising to \(0.49\) at \(\lambda{=}0.5\)); see Section 7. Bank is therefore the
  case where the trilemma genuinely binds and where DP and EO objectives conflict.

In short: the penalty cleanly resolves the trilemma when the protected group is
well represented, costs accuracy when it is small, and fails/destabilises when it
is tiny.

### 5.2 Mini-batch size and the fairness-stability mechanism

Batch size is the dominant control on training stability, exactly as the
fairness-stability discussion predicted. As batch size falls
\(1024\to512\to256\to64\), the fraction of safety-net-masked batches rises
\(0\%\to0.2\%\to25.6\%\to91.5\%\) and the maximum global gradient norm rises
\(1.8\to2.5\to26\to155\). Small batches make the in-batch demographic-parity
estimate so noisy that the safety net is almost always active and the surviving
fairness gradients are large and erratic; large batches yield a stable estimate
and a quiescent penalty. No configuration diverged (no non-finite loss in any of
the 180 runs), which supports the claim that the safety net prevents the
instability it was designed to catch.

### 5.3 Minority fraction (stress test of the safety net)

Synthetically resampling the protected group to \(\{1,3,5,10\}\%\) confirms the
mechanism end-to-end. At 10% prevalence only 1.1% of batches are masked and the
max gradient norm is 11.8; at 1% prevalence 93.6% of batches are masked and the
max gradient norm reaches 47.2 — yet training still completes without divergence.
Figure (german_stability_minority) plots both responses with CI bands. This is
the direct evidence that the heuristic degrades gracefully under group scarcity
rather than failing.

### 5.4 Teacher capacity

Increasing teacher `n_estimators` \(50\to100\to200\) raises the student's SHAP
\(R^2\) (\(0.421\to0.477\to0.482\)) — a stronger teacher yields explanation
targets the student imitates more faithfully — while the risk-head AUC stays flat
(\(0.784\to0.787\to0.793\), within CI). This cleanly separates the two heads: the
distilled explanation head tracks teacher quality, the label-supervised risk head
does not. (`max_depth` shows the same pattern: `results/german_agg_teacher_depth.csv`.)

### 5.5 Factorial (\(\lambda_{fair}\times\) minority fraction)

The \(4\times4\) factorial confirms the OFAT reading: at every minority level the
DP gap is lowest for small \(\lambda\) and inflated at \(\lambda\ge2.5\), and the
inflation is worst when the minority fraction is small — i.e. fairness pressure
and group scarcity interact, both pushing toward the safety-net-masked,
high-variance regime (`results/german_agg_factorial.csv`).

> **Takeaway for the committee.** Treated as a designed simulation, the pipeline
> reveals that the teacher–student interaction is governed less by \(\lambda\)
> alone than by the *statistical resolution of the in-batch fairness signal*
> (batch size × protected-group size). The headline contribution is robust where
> that signal is well-resolved and degrades predictably where it is not.
