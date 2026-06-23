# Draft text — Discrete-event deployment simulation (optional P1 add-on)

> Strengthens the WSC fit: a queueing simulation of the serving system under load.
> Service times measured on Credit Card (`results/des_service_times_creditcard.npz`);
> sweep + figures from `run_des.py`. Device: CPU, single server (c=1), SLA = 50 ms.

---

## 9. Deployment Simulation Under Load

The scenario study (Sections 4–5) treats *training* as the system under study; we
complement it with a discrete-event simulation of *serving*. Applicant requests
arrive as a Poisson process of rate \(\lambda\) and are routed to one of three
explanation back-ends, modelled as an M/G/1 FIFO queue whose service-time
distribution is sampled (with replacement) from measured per-request latencies:

| Route | Mean service (ms) | Single-server capacity (req/s) |
|---|---|---|
| Student (distilled MLP) | 0.151 | 6,634 |
| Teacher + TreeSHAP (100 trees, depth 4 — submitted) | 0.174 | 5,746 |
| Teacher + TreeSHAP (500 trees, depth 8 — accurate/tuned) | 4.03 | 248 |

For each route × arrival-rate × seed (5 replications) we record mean queue wait,
P99 sojourn (end-to-end) latency, throughput, utilisation, and the SLA-violation
rate against a 50 ms target.

**Result.** Under load the routes separate sharply (Figures `des_creditcard_sla`,
`des_creditcard_p99`). The accurate teacher+TreeSHAP path saturates early: its P99
sojourn crosses the 50 ms SLA at ≈250 req/s and its violation rate jumps from 0 to
1.0 between 200 and 300 req/s. The distilled student sustains ≈6,600 req/s on a
single core with no SLA violations and sub-millisecond P99 — a ~27× higher
sustainable throughput at equal hardware, which is the operational case for
distillation.

**Honest caveat.** The *submitted* teacher (100 trees, depth 4) is itself fast
(0.174 ms) and queues almost identically to the student, so on small tree
ensembles distillation buys little serving headroom; the deployment advantage
materialises specifically when the teacher is large/accurate enough to be worth
distilling. This is consistent with, and quantifies, the latency motivation of the
paper, while correcting the impression that the speed-up is large for any teacher.

The simulation is parameterised (`run_des.py [dataset] [sla_ms] [n_servers]`), so
reviewers can explore multi-server scaling and alternative SLAs; with \(c\)
servers each route's capacity scales ~linearly until the arrival process itself
becomes the bottleneck.
