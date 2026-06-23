# revision_outputs/rtxfair_sim/metrics.py
"""
Metrics for the WSC revision.

Fairness:
  - demographic_parity_gap : |P(yhat=1|g1) - P(yhat=1|g0)|        (original, threshold 0.5)
  - mean_score_gap         : |E[p|g1] - E[p|g0]|                   (what the DP penalty optimises)
  - equal_opportunity_gap  : |TPR_g1 - TPR_g0|   conditioned on y=1 (P2b)
  - equalized_odds_gap     : max(|dTPR|, |dFPR|) conditioned on y   (P2b)

Accuracy / fidelity:
  - AUC (risk head), SHAP R^2 (explanation head)

Performance:
  - sequential P99 latency (single-sample, CPU)  -> matches current paper test
  - batched throughput sweep (samples/sec + per-sample latency)  (P2c)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, r2_score


# ---------------------------------------------------------------- fairness ----
def _rate(mask_group, positive):
    if mask_group.sum() == 0:
        return np.nan
    return float(positive[mask_group].mean())


def demographic_parity_gap(prob, sensitive, thr=0.5):
    yhat = (prob >= thr).astype(float)
    g1, g0 = (sensitive == 1), (sensitive == 0)
    return abs(_rate(g1, yhat) - _rate(g0, yhat))


def mean_score_gap(prob, sensitive):
    g1, g0 = (sensitive == 1), (sensitive == 0)
    if g1.sum() == 0 or g0.sum() == 0:
        return np.nan
    return abs(float(prob[g1].mean()) - float(prob[g0].mean()))


def equal_opportunity_gap(prob, y, sensitive, thr=0.5):
    """|TPR_g1 - TPR_g0|, TPR = P(yhat=1 | y=1, group)."""
    yhat = (prob >= thr).astype(float)
    out = {}
    tprs = []
    for g, m in [("g1", sensitive == 1), ("g0", sensitive == 0)]:
        pos = m & (y == 1)
        tpr = float(yhat[pos].mean()) if pos.sum() > 0 else np.nan
        out[f"TPR_{g}"] = tpr
        tprs.append(tpr)
    gap = abs(tprs[0] - tprs[1]) if not any(np.isnan(tprs)) else np.nan
    out["EO_gap"] = gap
    return out


def equalized_odds_gap(prob, y, sensitive, thr=0.5):
    """max(|dTPR|, |dFPR|), each conditioned on the true label."""
    yhat = (prob >= thr).astype(float)
    def rates(m):
        pos, neg = m & (y == 1), m & (y == 0)
        tpr = float(yhat[pos].mean()) if pos.sum() > 0 else np.nan
        fpr = float(yhat[neg].mean()) if neg.sum() > 0 else np.nan
        return tpr, fpr
    tpr1, fpr1 = rates(sensitive == 1)
    tpr0, fpr0 = rates(sensitive == 0)
    dtpr = abs(tpr1 - tpr0) if not (np.isnan(tpr1) or np.isnan(tpr0)) else np.nan
    dfpr = abs(fpr1 - fpr0) if not (np.isnan(fpr1) or np.isnan(fpr0)) else np.nan
    vals = [v for v in (dtpr, dfpr) if not np.isnan(v)]
    return {"dTPR": dtpr, "dFPR": dfpr,
            "EOdds_gap": (max(vals) if vals else np.nan)}


# ----------------------------------------------------- accuracy / fidelity ----
def auc(prob, y):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, prob))


def shap_r2(shap_true, shap_pred):
    return float(r2_score(shap_true, shap_pred))


# ------------------------------------------------------------- performance ----
@torch.no_grad()
def sequential_p99_latency(model, X_scaled, n=1000, device="cpu"):
    """Single-sample sequential inference latency (ms). Matches the paper test."""
    model = model.to(device).eval()
    lat = []
    limit = min(n, len(X_scaled))
    _ = model(torch.tensor(X_scaled[:1]).float().to(device))  # warmup
    for i in range(limit):
        bx = torch.tensor(X_scaled[i:i + 1]).float().to(device)
        t0 = time.perf_counter()
        _ = model(bx)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return {"mean_ms": float(lat.mean()),
            "p90_ms": float(np.percentile(lat, 90)),
            "p99_ms": float(np.percentile(lat, 99))}


@torch.no_grad()
def batched_throughput(model, X_scaled, batch_sizes=(1, 32, 128, 512, 4096),
                       repeats=20, device="cpu"):
    """Throughput (samples/sec) and per-sample latency for several batch sizes.
    Returns a list of dicts (one per batch size). Device is reported by caller."""
    model = model.to(device).eval()
    X = torch.tensor(X_scaled).float().to(device)
    N = X.shape[0]
    _ = model(X[:1])  # warmup
    rows = []
    for bs in batch_sizes:
        bs_eff = min(bs, N)
        batch = X[:bs_eff]
        # time several forward passes
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(batch)
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        rows.append({
            "batch_size": bs,
            "per_sample_ms": t / bs_eff * 1000.0,
            "throughput_sps": bs_eff / t,
            "batch_latency_ms": t * 1000.0,
            "device": device,
        })
    return rows


def grad_global_norm(model):
    """L2 norm of all parameter gradients (for stability tracking)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5
