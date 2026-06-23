# revision_outputs/rtxfair_sim/model.py
"""
FairStudentXAI: dual-head Deep&Wide MLP student, IDENTICAL to the notebooks.
backbone [256,128,64] ReLU dropout 0.2; risk head (Linear->Sigmoid, BCE);
explanation head (Linear -> n_features, MSE); homoscedastic task-uncertainty
parameters w_risk, w_expl. Architecture is preserved for Table-1 comparability.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn


class FairStudentXAI(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.risk_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.explanation_head = nn.Linear(64, output_dim)
        # homoscedastic uncertainty (predictive tasks only, as in the paper)
        self.w_risk = nn.Parameter(torch.zeros(1))
        self.w_expl = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = self.backbone(x)
        return self.risk_head(features), self.explanation_head(features)


def train_teacher(X_train, y_train, n_estimators=100, max_depth=4,
                  learning_rate=0.1, seed=42):
    """Fixed-or-configurable XGBoost teacher (capacity is a simulation factor)."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        random_state=seed, n_jobs=-1, eval_metric="logloss")
    clf.fit(X_train, y_train)
    return clf


def teacher_shap(teacher, X_train, X_test):
    """TreeSHAP ground-truth explanations for train and test."""
    import shap
    expl = shap.TreeExplainer(teacher)
    sv_tr = expl.shap_values(X_train)
    sv_te = expl.shap_values(X_test)
    # newer shap may return 3D for binary; collapse to positive-class 2D
    sv_tr = _as_2d(sv_tr)
    sv_te = _as_2d(sv_te)
    return sv_tr.astype("float32"), sv_te.astype("float32"), expl


def _as_2d(sv):
    import numpy as np
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n, d, classes) or (classes, n, d)
        if sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        elif sv.shape[0] == 2:
            sv = sv[1]
    return sv
