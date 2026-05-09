"""Optuna objective: one trial → one short trainer run."""

from __future__ import annotations

import optuna
from omegaconf import DictConfig

from src.common_utils import get_logger, set_trial_seed
from src.model import build_trainer

from .search_space import suggest_hyperparameters, apply_hp_to_config


def build_objective(base_cfg: DictConfig):
    """Return a closure compatible with study.optimize()."""
    logger = get_logger("tuning.objective", base_cfg)
    metric_name = str(base_cfg.tuning.metric)

    def objective(trial: optuna.Trial) -> float:
        set_trial_seed(base_cfg, trial.number)

        hp = suggest_hyperparameters(trial, base_cfg)
        logger.info(
            f"[trial {trial.number}] suggested: "
            + ", ".join(f"{k}={v}" for k, v in hp.items())
        )

        trial_cfg = apply_hp_to_config(base_cfg, hp)
        metric = train_one_trial(trial_cfg, trial)
        logger.info(f"[trial {trial.number}] {metric_name} = {metric:.4f}")
        return metric

    return objective


def train_one_trial(cfg: DictConfig, trial: optuna.Trial) -> float:
    """Build the configured trainer and drive one trial."""
    return build_trainer(cfg).train_one_trial(trial_number=trial.number)
