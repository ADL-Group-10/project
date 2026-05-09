"""
Convert cfg.tuning.search_space (config.yaml) into Optuna trial.suggest_*
calls, and overlay the result onto cfg via cfg.tuning.targets.
"""

from __future__ import annotations
import copy
from typing import Any

import optuna
from omegaconf import DictConfig, OmegaConf


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> dict[str, Any]:
    """Ask Optuna for one hyperparameter set for this trial."""
    s = cfg.tuning.search_space
    return {
        "lr":            trial.suggest_float("lr", float(s.lr[0]), float(s.lr[1]), log=True),
        "lrf":           trial.suggest_float("lrf", float(s.lrf[0]), float(s.lrf[1])),
        "batch_size":    trial.suggest_categorical("batch_size", [int(b) for b in s.batch_size]),
        "box_weight":    trial.suggest_float("box_weight", float(s.box_weight[0]), float(s.box_weight[1])),
        "focal_gamma":   trial.suggest_float("focal_gamma", float(s.focal_gamma[0]), float(s.focal_gamma[1])),
        "warmup_epochs": trial.suggest_int("warmup_epochs", int(s.warmup_epochs[0]), int(s.warmup_epochs[1])),
    }


def apply_hp_to_config(cfg: DictConfig, hp: dict[str, Any]) -> DictConfig:
    """Overlay suggested hyperparameters onto a deepcopy of cfg, using cfg.tuning.targets."""
    cfg = copy.deepcopy(cfg)
    targets = cfg.tuning.targets
    for hp_name, value in hp.items():
        OmegaConf.update(cfg, str(targets[hp_name]), value, merge=False)
    return cfg
