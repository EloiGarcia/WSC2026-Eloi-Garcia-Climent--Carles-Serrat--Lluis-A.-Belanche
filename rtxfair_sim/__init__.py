# revision_outputs/rtxfair_sim/__init__.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from . import config, data, model, metrics, train, sim, figures  # noqa: F401

__all__ = ["config", "data", "model", "metrics", "train", "sim", "figures"]
