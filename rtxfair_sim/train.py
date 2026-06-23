# revision_outputs/rtxfair_sim/train.py
"""
Student training loop, faithful to the notebooks, with two additions for the
revision:
  (1) gradient-norm / divergence tracking (training-stability response),
  (2) a selectable fairness penalty: 'dp' (original demographic parity) or
      'eo' (equal-opportunity surrogate: difference of mean score among y=1),
      both reusing the <5-per-group safety-net heuristic.
Seeds, architecture and the homoscedastic weighting are preserved unchanged.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import FairStudentXAI
from . import metrics as M
from .config import PENALTY_DP, PENALTY_EO

SAFETY_MIN = 5  # safety-net threshold (per group)


def _dp_penalty(p_risk, sens):
    m1 = (sens == 1).squeeze()
    m0 = (sens == 0).squeeze()
    if m1.sum() < SAFETY_MIN or m0.sum() < SAFETY_MIN:
        return torch.tensor(0.0, device=p_risk.device)
    return torch.abs(p_risk[m1].mean() - p_risk[m0].mean())


def _eo_penalty(p_risk, sens, y):
    """Equal-opportunity surrogate: |E[p | g1,y=1] - E[p | g0,y=1]|.
    Soft, differentiable TPR difference; safety net on each positive subset."""
    s = sens.squeeze(); yy = y.squeeze()
    pos1 = (s == 1) & (yy == 1)
    pos0 = (s == 0) & (yy == 1)
    if pos1.sum() < SAFETY_MIN or pos0.sum() < SAFETY_MIN:
        return torch.tensor(0.0, device=p_risk.device)
    return torch.abs(p_risk.squeeze()[pos1].mean() - p_risk.squeeze()[pos0].mean())


def train_student(X_train, y_train, shap_train, sensitive_train,
                  X_test, y_test, shap_test, sensitive_test,
                  scenario, log_every=0):
    """Train a FairStudentXAI under `scenario` and return a record dict with all
    responses. `scenario` is a ScenarioConfig."""
    torch.manual_seed(scenario.seed)
    np.random.seed(scenario.seed)

    Xtr = X_train.astype(np.float32)
    ytr = y_train.astype(np.float32)
    Str = shap_train.astype(np.float32)
    sens = sensitive_train.astype(np.float32)

    model = FairStudentXAI(input_dim=Xtr.shape[1], output_dim=Str.shape[1])
    crit_r = nn.BCELoss()
    crit_e = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=scenario.lr)

    n = Xtr.shape[0]
    bs = scenario.batch_size
    nb = int(np.ceil(n / bs))

    grad_norms = []
    diverged = False
    masked_batches = 0
    total_batches = 0

    t0 = time.time()
    model.train()
    for epoch in range(scenario.epochs):
        idx = np.random.permutation(n)
        for i in range(nb):
            b = idx[i * bs: min((i + 1) * bs, n)]
            bx = torch.tensor(Xtr[b])
            by = torch.tensor(ytr[b]).unsqueeze(1)
            bsh = torch.tensor(Str[b])
            bse = torch.tensor(sens[b]).unsqueeze(1)

            opt.zero_grad()
            p_risk, p_expl = model(bx)
            loss_r = crit_r(p_risk, by)
            loss_e = crit_e(p_expl, bsh)
            if scenario.penalty == PENALTY_EO:
                loss_f = _eo_penalty(p_risk, bse, by)
            else:
                loss_f = _dp_penalty(p_risk, bse)

            total_batches += 1
            if float(loss_f) == 0.0:
                masked_batches += 1

            w_loss_r = torch.exp(-model.w_risk) * loss_r + 0.5 * model.w_risk
            w_loss_e = torch.exp(-model.w_expl) * loss_e + 0.5 * model.w_expl
            loss = w_loss_r + w_loss_e + scenario.lambda_fair * loss_f

            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            grad_norms.append(M.grad_global_norm(model))
            opt.step()
        if diverged:
            break
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1}/{scenario.epochs} loss={float(loss):.4f}")
    train_seconds = time.time() - t0

    # ---- evaluation on test ----
    model.eval()
    with torch.no_grad():
        risk, expl = model(torch.tensor(X_test.astype(np.float32)))
        prob = risk.numpy().ravel()
        shp = expl.numpy()

    eo = M.equal_opportunity_gap(prob, y_test, sensitive_test)
    eod = M.equalized_odds_gap(prob, y_test, sensitive_test)
    lat = M.sequential_p99_latency(model, X_test, n=min(1000, len(X_test)))

    gn = np.array(grad_norms) if grad_norms else np.array([np.nan])
    record = dict(
        **scenario.asdict(),
        AUC=M.auc(prob, y_test),
        SHAP_R2=M.shap_r2(shap_test, shp),
        DP_gap=M.demographic_parity_gap(prob, sensitive_test),
        mean_score_gap=M.mean_score_gap(prob, sensitive_test),
        EO_gap=eo["EO_gap"], TPR_g1=eo["TPR_g1"], TPR_g0=eo["TPR_g0"],
        EOdds_gap=eod["EOdds_gap"], dTPR=eod["dTPR"], dFPR=eod["dFPR"],
        grad_norm_mean=float(np.nanmean(gn)), grad_norm_max=float(np.nanmax(gn)),
        diverged=bool(diverged),
        masked_batch_frac=(masked_batches / total_batches) if total_batches else np.nan,
        n_train=int(n),
        p99_latency_ms=lat["p99_ms"], mean_latency_ms=lat["mean_ms"],
        train_seconds=train_seconds,
    )
    return record, model
