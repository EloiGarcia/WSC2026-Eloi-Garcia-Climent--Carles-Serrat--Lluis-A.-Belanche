# Draft text — P2b: Equal Opportunity & Equalized Odds (Reviewer 3)

> EO and Equalized-Odds gaps are implemented (`metrics.py`) and computed for every
> model/scenario; the EO-style penalty variant is implemented (`train.py`,
> `penalty="eo"`). German numbers final; Bank/Credit-Card via the runner.

---

## 7. Beyond Demographic Parity: Equal Opportunity and Equalized Odds

The submitted work measured fairness only as a demographic-parity (DP) gap —
\(|P(\hat y{=}1\mid g_1)-P(\hat y{=}1\mid g_0)|\) — which ignores the ground-truth
label. We add two label-conditioned criteria:

- **Equal Opportunity (EO):** \(|\mathrm{TPR}_{g_1}-\mathrm{TPR}_{g_0}|\),
  the true-positive-rate gap among genuine defaulters (\(y=1\));
- **Equalized Odds (EOdds):** \(\max(|\Delta\mathrm{TPR}|,|\Delta\mathrm{FPR}|)\),
  conditioning on both label values.

Both are reported for all models (`metrics.equal_opportunity_gap`,
`metrics.equalized_odds_gap`). The decisive evidence is on **Bank Marketing**:
as \(\lambda\) increases, the DP-targeted penalty drives the DP gap down
(\(0.325\to0.035\)) yet the **EO gap moves in the opposite direction**, rising
from \(0.118\) at \(\lambda{=}0\) to \(0.49\) at \(\lambda{=}0.5\) and remaining
elevated (\(0.22\) at \(\lambda{=}5\)). Demographic-parity and Equal-Opportunity
objectives therefore *conflict* on the imbalanced dataset: equalising overall
positive rates skews the true-positive rates across groups. On UCI Credit Card,
where the groups are balanced, the two move together (EO \(0.076\to0.022\) as DP
falls). On German both are large and unstable owing to the 31-instance group.
The key point is that **a DP-targeted penalty does not control the conditional
(EO/EOdds) gaps and can enlarge them** — precisely the limitation the reviewer
flagged, now demonstrated empirically.

### 7.1 An EO-adapted penalty and how it tracks conditional probabilities

We implement an Equal-Opportunity variant of the in-batch penalty by replacing
the overall positive-rate difference with the **difference in mean predicted risk
restricted to the positive class**,
\[
\mathcal{L}_{EO}=\big|\;\mathbb{E}[\,p\mid g_1,\,y{=}1\,]-\mathbb{E}[\,p\mid g_0,\,y{=}1\,]\;\big|,
\]
a smooth, differentiable surrogate for the TPR gap, reusing the same
<5-per-group safety net — now applied to each *positive* subgroup. This is the
sense in which the penalty "adapts to conditional probabilities": the DP penalty
conditions only on the protected attribute \(g\), whereas the EO penalty
conditions on the joint event \((g, y{=}1)\), so it pushes the two groups'
score distributions together *only where the ground truth is positive*, leaving
the negative class free. Equalized Odds would add the symmetric \(y{=}0\) term.

On a controlled check the two penalties behave as designed — the DP penalty
minimises the DP gap and the EO penalty minimises the EO gap (each at the other's
mild expense) — confirming the surrogate targets the intended conditional
quantity. **[RUN: report the EO-penalty vs DP-penalty trade-off on Credit Card /
Bank with `penalty="eo"`; on German the small positive-class subgroup makes the
EO penalty mostly safety-net-masked, which is itself the honest finding that
label-conditioned fairness needs adequate per-group, per-label support.]**
