"""
src/tuning/search_space.py

Turns `cfg.tuning.search_space` (declared in config.yaml) into Optuna
trial.suggest_* calls. Keeping this in one place means the search space is
data-driven — edit config.yaml, not Python, to change ranges.

Contract with config.yaml (see config.yaml → tuning.search_space):
    lr:            [low, high]          -> suggest_float(log=True)
    batch_size:    [8, 16, 32]          -> suggest_categorical
    box_weight:    [low, high]          -> suggest_float (uniform)
    focal_gamma:   [low, high]          -> suggest_float (uniform)
    warmup_epochs: [low, high]          -> suggest_int

If you add a new key to the yaml, add one branch here — ordered so the
most-frequently-tuned params appear first.
"""

from __future__ import annotations
import copy
from typing import Any

import optuna
from omegaconf import DictConfig


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> dict[str, Any]:
    """
    Ask Optuna for one concrete hyperparameter set for this trial.

    Args:
        trial: Optuna Trial object passed into the objective.
        cfg:   Merged config (load_config()). Reads cfg.tuning.search_space.

    Returns:
        Flat dict of suggested hyperparameters keyed by name. The objective
        then overlays these into cfg before training.

    Example:
        {'lr': 0.00043, 'batch_size': 16, 'box_weight': 7.8,
         'focal_gamma': 1.2, 'warmup_epochs': 3}
    """
    space = cfg.tuning.search_space
    hp: dict[str, Any] = {}

    # learning rate — log-uniform, most sensitive parameter
    lr_low, lr_high = space.lr
    hp["lr"] = trial.suggest_float("lr", float(lr_low), float(lr_high), log=True)

    # batch size — categorical (memory-bound on LTU cluster)
    hp["batch_size"] = trial.suggest_categorical(
        "batch_size", [int(b) for b in space.batch_size]
    )

    # loss weights — uniform
    bw_low, bw_high = space.box_weight
    hp["box_weight"] = trial.suggest_float("box_weight", float(bw_low), float(bw_high))

    fg_low, fg_high = space.focal_gamma
    hp["focal_gamma"] = trial.suggest_float("focal_gamma", float(fg_low), float(fg_high))

    # warmup epochs — integer
    wu_low, wu_high = space.warmup_epochs
    hp["warmup_epochs"] = trial.suggest_int("warmup_epochs", int(wu_low), int(wu_high))

    return hp


def apply_hp_to_config(cfg: DictConfig, hp: dict[str, Any]) -> DictConfig:
    """
    Overlay suggested hyperparameters onto a copy of the config.
    The objective uses this to build the per-trial cfg that trainer.py sees.

    Mapping:
        lr, batch_size, warmup_epochs  → cfg.training.*
        box_weight, focal_gamma        → cfg.loss.*

    Also caps cfg.training.epochs at cfg.tuning.trial_epochs so a per-trial
    training run fits inside the tuning budget. Dharmini's final v1/v2 runs
    load config.yaml fresh, so they still see the full training.epochs.
    """
    # deepcopy preserves struct flags, interpolations, and resolver state —
    # safer than an OmegaConf.to_container round-trip.
    cfg = copy.deepcopy(cfg)

    # tuned hyperparameters
    cfg.training.lr             = hp["lr"]
    cfg.training.batch_size     = hp["batch_size"]
    cfg.training.warmup_epochs  = hp["warmup_epochs"]
    cfg.loss.box_weight         = hp["box_weight"]
    cfg.loss.focal_gamma        = hp["focal_gamma"]

    # per-trial epoch budget — keeps tuning trials short so the 2h timeout
    # isn't blown after a handful of trials. Only applied when the key exists
    # in the yaml; falls back to cfg.training.epochs if not set.
    trial_epochs = getattr(cfg.tuning, "trial_epochs", None)
    if trial_epochs is not None:
        cfg.training.epochs = int(trial_epochs)

    return cfg
