# revision_outputs/rtxfair_sim/data.py
"""
Data loading + preprocessing that reproduces the original WSC notebooks EXACTLY:
  - LabelEncode categorical columns (per-column LabelEncoder)
  - target mapping per dataset
  - stratified 80/20 split with random_state=42
  - StandardScaler fit on train
  - sensitive attribute derived from the LabelEncoded, UNSCALED training matrix

Loading order of preference:
  1) local CSVs in revision_outputs/data/{name}_X.csv,{name}_y.csv (produced by export_data.py)
  2) ucimlrepo.fetch_ucirepo(id)  (used by the notebooks; needs internet)

Also provides:
  - minority_resample(): synthetically resample the protected group to a target fraction
  - make_synthetic(): a surrogate dataset with the same interface for offline smoke tests
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import DATASETS, DatasetConfig, GLOBAL_SEED

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

# Candidate locations for the raw UCI dumps (csvs/ folder shipped by the user).
RAW_DIR_CANDIDATES = [
    os.environ.get("RTXFAIR_RAW_DIR", ""),
    os.path.normpath(os.path.join(HERE, "..", "csvs")),         # repo-root/csvs (preferred)
    os.path.normpath(os.path.join(HERE, "..", "..", "csvs")),   # parent project fallback
]


def _raw_dir():
    for d in RAW_DIR_CANDIDATES:
        if d and os.path.isdir(d):
            return d
    return None


def _read_raw(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the ucimlrepo-equivalent (features, targets) frames from the
    raw UCI files in csvs/, with the SAME column names the notebooks rely on
    (German Attribute1..20, Credit Card X1..X23, Bank standard names)."""
    rd = _raw_dir()
    if rd is None:
        raise FileNotFoundError("no csvs/ raw directory found")
    if cfg.name == "german":
        path = os.path.join(rd, "german.data")
        cols = [f"Attribute{i}" for i in range(1, 21)] + ["target"]
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
        return df[[f"Attribute{i}" for i in range(1, 21)]], df[["target"]]
    if cfg.name == "bank":
        # UCI Bank Marketing 45,211 -> bank-full.csv (';' separated, quoted)
        path = os.path.join(rd, "bank-full.csv")
        df = pd.read_csv(path, sep=";")
        return df.drop(columns=["y"]), df[["y"]]
    if cfg.name == "creditcard":
        path = os.path.join(rd, "default of credit card clients.xls")
        df = pd.read_excel(path, header=1).drop(columns=["ID"])
        target = df.columns[-1]            # 'default payment next month'
        y = df[[target]]
        X = df.drop(columns=[target]).copy()
        X.columns = [f"X{i}" for i in range(1, X.shape[1] + 1)]  # X1..X23
        return X, y
    raise ValueError(cfg.name)


@dataclass
class DataBundle:
    name: str
    feature_names: list
    X_train_raw: np.ndarray     # LabelEncoded, UNSCALED
    X_test_raw: np.ndarray
    X_train: np.ndarray         # scaled
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    sensitive_train: np.ndarray  # 0/1 float, group indicator on train
    sensitive_test: np.ndarray
    scaler: object


def _map_target(name: str, y_df: pd.DataFrame) -> np.ndarray:
    s = y_df.iloc[:, 0]
    if name == "german":
        return s.map({1: 0, 2: 1}).values.astype(np.float32)
    # generic: yes/no strings, or already 0/1, or 1/2
    if s.dtype == object:
        u = set(str(v).strip().lower() for v in s.unique())
        if u <= {"yes", "no"}:
            return s.map(lambda v: 1.0 if str(v).strip().lower() == "yes" else 0.0).values.astype(np.float32)
    vals = set(np.unique(s.values).tolist())
    if vals <= {1, 2}:
        return s.map({1: 0, 2: 1}).values.astype(np.float32)
    return s.values.astype(np.float32)


def _label_encode(X_df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    X = X_df.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X.values.astype(np.float32), list(X.columns)


def _sensitive_from_raw(cfg: DatasetConfig, X_raw: np.ndarray, feature_names: list) -> np.ndarray:
    idx = feature_names.index(cfg.sensitive_col)
    col = X_raw[:, idx]
    if cfg.sensitive_kind == "equals":
        return (col == cfg.sensitive_value).astype(np.float32)
    elif cfg.sensitive_kind == "greater_than":
        return (col > cfg.sensitive_value).astype(np.float32)
    raise ValueError(cfg.sensitive_kind)


def _load_frames(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fx = os.path.join(DATA_DIR, f"{cfg.name}_X.csv")
    fy = os.path.join(DATA_DIR, f"{cfg.name}_y.csv")
    if os.path.exists(fx) and os.path.exists(fy):
        return pd.read_csv(fx), pd.read_csv(fy)
    # raw UCI dumps in csvs/ (reconstruct ucimlrepo schema)
    if _raw_dir() is not None:
        return _read_raw(cfg)
    # fallback to ucimlrepo (needs internet; identical to notebooks)
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=cfg.uci_id)
    return ds.data.features.copy(), ds.data.targets.copy()


def load_dataset(name: str) -> DataBundle:
    cfg = DATASETS[name]
    X_df, y_df = _load_frames(cfg)
    y = _map_target(name, y_df)
    X_enc, feat = _label_encode(X_df)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_enc, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    s_tr = _sensitive_from_raw(cfg, X_tr_raw, feat)
    s_te = _sensitive_from_raw(cfg, X_te_raw, feat)

    return DataBundle(name, feat, X_tr_raw, X_te_raw, X_tr, X_te,
                      y_tr, y_te, s_tr, s_te, scaler)


def minority_resample(X, y, shap_t, sensitive, target_fraction, seed=GLOBAL_SEED):
    """Resample the *minority* sensitive group (with replacement) so it becomes
    `target_fraction` of the returned training set. Majority rows are kept as-is.
    Returns resampled (X, y, shap_t, sensitive). Used to stress-test the
    safety-net / gradient-stability claim. SHAP targets travel with their rows.
    """
    rng = np.random.RandomState(seed)
    grp1 = np.where(sensitive == 1)[0]
    grp0 = np.where(sensitive == 0)[0]
    minority, majority = (grp1, grp0) if len(grp1) <= len(grp0) else (grp0, grp1)
    n_maj = len(majority)
    # n_min / (n_min + n_maj) = f  ->  n_min = f/(1-f) * n_maj
    f = float(target_fraction)
    n_min_target = max(1, int(round(f / (1.0 - f) * n_maj)))
    min_idx = rng.choice(minority, size=n_min_target, replace=True)
    keep = np.concatenate([majority, min_idx])
    rng.shuffle(keep)
    return X[keep], y[keep], shap_t[keep], sensitive[keep]


def make_synthetic(name="synthetic", n=4000, d=20, seed=GLOBAL_SEED) -> DataBundle:
    """Offline surrogate with the same DataBundle interface for smoke testing.
    Includes a binary sensitive column with class-conditional signal so fairness
    penalties have something to act on."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    sens_full = (rng.rand(n) < 0.18).astype(np.float32)  # ~18% protected
    X[:, 0] = sens_full + 0.05 * rng.randn(n)             # encode sensitive in col 0
    w = rng.randn(d)
    logits = X @ w + 0.8 * sens_full                      # group-correlated risk
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n) < p).astype(np.float32)
    feat = [f"f{i}" for i in range(d)]
    X_tr_raw, X_te_raw, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sens_full, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)
    return DataBundle(name, feat, X_tr_raw, X_te_raw, X_tr, X_te,
                      y_tr.astype(np.float32), y_te.astype(np.float32),
                      s_tr.astype(np.float32), s_te.astype(np.float32), scaler)
