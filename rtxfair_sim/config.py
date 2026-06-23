# revision_outputs/rtxfair_sim/config.py
"""
Central configuration for the WSC revision simulation study.

Baseline (Table 1) settings are preserved EXACTLY as in the original notebooks so
new results stay comparable. The teacher is the same fixed XGBoost in all three
notebooks (this is relevant to the P2a AUC-anomaly investigation).
"""
from dataclasses import dataclass, field
from typing import Optional

# Crash-guard env (set before importing torch / xgboost / shap). Imported for side effect.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

GLOBAL_SEED = 42

# ----------------------------------------------------------------------------
# Teacher: identical fixed XGBoost across all three original notebooks.
# NOTE: never tuned in the submitted version -> see P2a.
# ----------------------------------------------------------------------------
TEACHER_BASELINE = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                        random_state=GLOBAL_SEED, n_jobs=-1)

# Student backbone is fixed by the paper: [256,128,64] ReLU, dropout 0.2.
STUDENT_BACKBONE = (256, 128, 64)
STUDENT_DROPOUT = 0.2


@dataclass
class DatasetConfig:
    name: str               # short key
    pretty: str             # human-readable
    uci_id: int             # ucimlrepo id
    # sensitive attribute, defined on the RAW (pre-scaling) feature matrix:
    sensitive_col: str      # column name in the ucimlrepo features frame
    sensitive_kind: str     # "equals" | "greater_than"
    sensitive_value: float  # threshold / category code
    sensitive_desc: str
    # baseline training budget (as in the corresponding notebook):
    epochs: int
    batch_size: int
    lambda_fair: float
    lr: float = 1e-3
    # target mapping note (documented per dataset in data.py)
    target_note: str = ""


DATASETS = {
    "german": DatasetConfig(
        name="german", pretty="German Credit", uci_id=144,
        sensitive_col="Attribute20", sensitive_kind="equals", sensitive_value=1.0,
        sensitive_desc="LabelEncoded Attribute20 (mirrors notebook mask_1==1); groups foreign vs native",
        epochs=450, batch_size=256, lambda_fair=1.5,
        target_note="target 1=Good->0, 2=Bad->1 (y=1 is 'Bad'/default).",
    ),
    "creditcard": DatasetConfig(
        name="creditcard", pretty="UCI Credit Card", uci_id=350,
        sensitive_col="X3", sensitive_kind="equals", sensitive_value=2.0,
        sensitive_desc="EDUCATION == 2 (University) vs rest",
        epochs=1200, batch_size=512, lambda_fair=2.5,
        target_note="target Y = default payment next month (1=default).",
    ),
    "bank": DatasetConfig(
        name="bank", pretty="Bank Marketing", uci_id=222,
        sensitive_col="age", sensitive_kind="greater_than", sensitive_value=60.0,
        sensitive_desc="age > 60 (Senior) vs rest",
        epochs=1200, batch_size=512, lambda_fair=2.5,
        target_note="target y = term-deposit subscription (yes->1).",
    ),
}

# ----------------------------------------------------------------------------
# Simulation scenario grid (P1). Reduced OFAT + one 2-factor factorial, >=5 seeds.
# Each factor's first value is the baseline anchor.
# ----------------------------------------------------------------------------
N_SEEDS = 5
SEEDS = [42, 43, 44, 45, 46]

GRID = dict(
    teacher_n_estimators=[100, 50, 200],
    teacher_max_depth=[4, 3, 6],
    lambda_fair=[0.0, 0.5, 1.0, 2.5, 5.0],
    batch_size=[512, 64, 256, 1024],
    minority_fraction=[None, 0.01, 0.03, 0.05, 0.10],  # None = native distribution
)

# The 2-factor factorial runs the full cross product of these two (most influential):
FACTORIAL = dict(
    lambda_fair=[0.0, 1.0, 2.5, 5.0],
    minority_fraction=[0.01, 0.03, 0.05, 0.10],
)

# Penalty variants supported by the runner.
PENALTY_DP = "dp"     # |mean P(group1) - mean P(group0)|  (demographic parity, original)
PENALTY_EO = "eo"     # |TPR_group1 - TPR_group0|          (equal opportunity, P2b)


@dataclass
class ScenarioConfig:
    dataset: str
    seed: int = GLOBAL_SEED
    teacher_n_estimators: int = 100
    teacher_max_depth: int = 4
    teacher_learning_rate: float = 0.1
    lambda_fair: float = 2.5
    batch_size: int = 512
    epochs: int = 1200
    lr: float = 1e-3
    minority_fraction: Optional[float] = None  # synthetic resample of protected group
    penalty: str = PENALTY_DP
    label: str = ""  # free-text scenario / arm label

    def asdict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
