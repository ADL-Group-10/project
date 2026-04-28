"""
src/common_utils/seed.py

Global reproducibility seed. Must be called before any model init, dataloader creation, or augmentation setup.

Who calls this:
    Achyuth  — top of get_dataloader() and snow_pipeline.py
    Dharmini  — top of trainer.py (once per training run)
    Sathish  — top of objective.py (once per Optuna trial via set_trial_seed)

Usage:
    from src.common_utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.project.seed)   # call once, early, before anything else
"""

import os
import random
import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """
    Set all random seeds for full reproducibility across Python, NumPy,
    PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: Integer seed. Use cfg.project.seed (42 by default).

    Note:
        Sets cudnn.deterministic=True and cudnn.benchmark=False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


def set_seed_from_config(cfg: DictConfig) -> None:
    """
    Convenience wrapper — reads seed from config.project.seed.

    Dharmini calls this at the top of trainer.py:
        set_seed_from_config(cfg)
    """
    set_seed(cfg.project.seed)


def set_trial_seed(cfg: DictConfig, trial_number: int) -> None:
    """
    Per-trial seed for Optuna (Sathish).

    Keeps each trial deterministic but distinct from other trials.
    Call at the top of objective.py before any model or dataloader init.

    Args:
        cfg:          Merged config from load_config().
        trial_number: trial.number from the Optuna Trial object.

    """
    set_seed(cfg.project.seed + trial_number)
