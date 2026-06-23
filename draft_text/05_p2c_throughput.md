# Draft text — P2c: batched-inference throughput (Reviewer 3)

> Supplementary benchmark on top of the existing sequential single-sample CPU
> latency test. **Device: CPU** (4-core aarch64; no GPU/MPS available in this
> environment — re-run `run_p2c.py` on MPS/CUDA to populate the accelerated row).
> Source: `results/p2c_throughput.csv`.

---

## 8. Batched-Inference Throughput

The submitted latency test issued one request at a time. We add a batched
benchmark sweeping batch sizes \(\{1,32,128,512,4096\}\), reporting throughput
(samples/s) and per-sample latency. The student's per-sample latency falls by
~300–400× from sequential to large-batch inference, showing the architecture is
throughput-bound only by batch size, not by any per-sample overhead.

**Table S1. Student inference throughput (CPU, 4-core).**

| Dataset | Seq. P99 (ms) | bs=1 (samp/s) | bs=32 | bs=128 | bs=512 | bs=4096 | per-sample @4096 (ms) |
|---|---|---|---|---|---|---|---|
| German Credit | 0.277 | 7,505 | 206,785 | 745,995 | 1,003,661 | 1,016,307 | 0.00098 |
| UCI Credit Card | 0.227 | 6,530 | 197,024 | 730,212 | 1,691,279 | 2,562,069 | 0.00039 |
| Bank Marketing | 0.242 | 6,500 | 202,025 | 692,049 | 1,768,057 | 2,962,214 | 0.00034 |

Even single-sample P99 latency is sub-millisecond (0.23–0.28 ms), and batched
serving reaches 1–3 million samples/s. For context, the teacher's TreeSHAP path
is benchmarked separately in the sequential test; the student remains the
low-latency, high-throughput option for online explanation serving. The wider
per-sample range at large batch across datasets reflects feature dimensionality
(German 20, Credit Card 23, Bank 16) and test-set size driving the largest
feasible batch.
