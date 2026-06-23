# Draft text — P2a clarification: the AUC anomaly (Reviewer 3)

> Implements the agreed fix: (a) reword the architecture, (b) replace single-run
> numbers with multi-seed mean±CI, (c) reframe the accuracy claim, (d) back it
> with the tuned-teacher ablation. German numbers final; Bank/Credit-Card
> confirmation via `run_p2a.py` slots in where marked **[RUN]**.

---

## 6. Clarification: Why the Fair Student Appeared to Beat the Baseline on AUC

The submitted Table 1 reports the fairness-constrained student exceeding the
unconstrained model on AUC for Bank Marketing (\(0.8959\to0.9066\)) and UCI
Credit Card (\(0.7276\to0.7486\)). Because a fairness constraint should not, in
general, *improve* discrimination, we investigated this and clarify it as
follows.

**(a) What is actually being distilled.** Only the explanation head is distilled
from the teacher; the **risk head is trained directly on the ground-truth labels**
via binary cross-entropy. The student is therefore a multi-task model
(label-supervised risk + TreeSHAP-distilled explanation), not a pure soft-label
distillation of the teacher's predictions. We have corrected the architecture
description accordingly. A direct consequence is that "student vs baseline" in
Table 1 is a comparison between two runs of the *same* label-supervised network
that differ only in \(\lambda_{fair}\) — not student-vs-teacher.

**(b) The effect is within replication noise — or even reverses.** Repeating the
comparison over multiple seeds with 95% CIs: on German Credit the risk-head AUC is
\(0.800\pm0.016\) (\(\lambda{=}0\)) vs \(0.792\pm0.007\) (\(\lambda{=}2.5\)) —
overlapping. On UCI Credit Card it is flat (\(0.767\pm0.004\) across all \(\lambda\)).
On **Bank Marketing the effect reverses**: AUC *declines* with fairness pressure,
\(0.926\to0.897\) (\(\lambda{=}0\to5\), CI \(\approx0.004\)) — i.e. the
multi-seed evidence shows fairness *costing* accuracy, not improving it. The
submitted single-run \(0.8959\to0.9066\) (Bank) and \(0.7276\to0.7486\)
(Credit Card) gains do not reproduce and are best explained as run-to-run noise.

**(c) Reframed claim.** We no longer claim that fairness regularisation improves
accuracy. The correct, and still strong, statement is that **the demographic-parity
penalty achieves its fairness effect at no statistically significant cost to AUC**
— the mild \(L_1\) parity term acts at most as a weak regulariser. This is the
appropriate "trilemma" result: bias mitigation that is accuracy-neutral.

**(d) Tuned-teacher ablation.** We confirmed the teacher was left at fixed,
untuned hyperparameters (`n_estimators=100, max_depth=4, lr=0.1`) in all three
notebooks, and ran a proper randomised 3-fold CV search. Tuning moves the teacher only marginally and, in every case, the tuned teacher
**beats the student**:

| Dataset | Teacher AUC (fixed) | Teacher AUC (tuned) | Student AUC (range over λ) |
|---|---|---|---|
| German Credit | 0.8088 | 0.8117 | 0.786–0.800 |
| UCI Credit Card | 0.7778 | 0.7800 | 0.767–0.768 |
| Bank Marketing | 0.9286 | 0.9343 | 0.897–0.926 |

Since the (properly tuned) teacher outperforms the student on all three datasets,
the student is not a state-of-the-art accuracy model and the original
fair-vs-baseline ordering is a noise-level artefact — not evidence that fairness
improves accuracy. On Bank the multi-seed result in fact shows fairness *reducing*
AUC. (Best tuned configs in `results/*_p2a_*`.)

Together, (a)–(d) replace the surprising headline with a defensible one and make
the teacher a properly tuned reference baseline.
