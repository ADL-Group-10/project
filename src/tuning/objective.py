"""
src/tuning/objective.py

The Optuna `objective(trial)` function — one trial = one short v1 training run.

Skeleton contract:
    - Suggest HPs from the declared search space
    - Seed deterministically per trial (set_trial_seed)
    - Overlay HPs onto cfg → hand to trainer
    - Report intermediate val_mAP50 so Hyperband can prune
    - Return final val_mAP50 (direction=maximize)

Swap point — `train_one_trial(cfg, trial)`:
    This is the seam between tuning and training. Right now it's a stub that
    returns a plausible mAP50 so the study runs end-to-end without blocking
    on Dharmini's trainer. When the trainer lands, replace the body with:

        from src.model.baseline_yolov9.trainer import YOLOv9Trainer
        #                      ^ confirm the actual module path with Dharmini;
        #                        src/model/baseline_yolov9/trainer.py is the
        #                        placeholder in the repo today.

        trainer = YOLOv9Trainer(cfg)
        best = 0.0
        for epoch in range(cfg.training.epochs):
            trainer.train_one_epoch()
            if epoch % cfg.training.val_every_n_epochs == 0:
                map50 = trainer.validate()["mAP50"]
                best  = max(best, map50)
                trial.report(map50, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return best

    Note: cfg.training.epochs is already capped at cfg.tuning.trial_epochs
    inside apply_hp_to_config, so the loop above honours the tuning budget
    without any extra logic here.

The rest of the tuning machinery doesn't care how train_one_trial is
implemented, as long as it returns a float.
"""

from __future__ import annotations
import math
import random
import time

import optuna
from omegaconf import DictConfig

from src.common_utils import (
    get_logger,
    set_trial_seed,
)

from .search_space import suggest_hyperparameters, apply_hp_to_config


# ── Objective factory ──────────────────────────────────────────────────────

def build_objective(base_cfg: DictConfig):
    """
    Return a closure compatible with study.optimize(...).

    Using a closure lets us inject a cfg (for tests / CLI overrides) without
    touching module-level globals.
    """
    logger = get_logger("tuning.objective", base_cfg)

    def objective(trial: optuna.Trial) -> float:
        # 1. deterministic per-trial seed
        set_trial_seed(base_cfg, trial.number)

        # 2. ask Optuna for this trial's HPs
        hp = suggest_hyperparameters(trial, base_cfg)
        logger.info(
            f"[trial {trial.number}] suggested: "
            + ", ".join(f"{k}={v}" for k, v in hp.items())
        )

        # 3. overlay onto a per-trial cfg copy (also caps training.epochs
        #    at tuning.trial_epochs if that key is present)
        trial_cfg = apply_hp_to_config(base_cfg, hp)

        # 4. force v1 — tuning always runs on the clean baseline.
        #    After load_config(variant="v1") + merge, the v1 variant keys
        #    live at the ROOT of cfg (cfg.experiment, cfg.use_snow_aug),
        #    not under cfg.variants.v1. These assignments target the merged
        #    paths the trainer actually reads.
        trial_cfg.experiment                = "v1_baseline"
        trial_cfg.use_snow_aug              = False
        trial_cfg.augmentation.use_snow_aug = False

        # 5. train one trial, report intermediate metrics for pruning
        try:
            final_map50 = train_one_trial(trial_cfg, trial)
        except optuna.TrialPruned:
            logger.info(f"[trial {trial.number}] pruned")
            raise
        except Exception as e:
            logger.exception(f"[trial {trial.number}] failed: {e}")
            # Re-raise so Optuna marks it FAIL rather than silently succeeding.
            raise

        logger.info(f"[trial {trial.number}] final val_mAP50 = {final_map50:.4f}")
        return final_map50

    return objective


# ── Training seam ──────────────────────────────────────────────────────────

def train_one_trial(cfg: DictConfig, trial: optuna.Trial) -> float:
    from src.model.trainer import YOLOv9Trainer
    trainer = YOLOv9Trainer(cfg)
    trainer.dataset_yaml = "/project/outputs/yolo/dataset.yaml"
    map50 = trainer.train_one_trial({
        "lr":            cfg.training.lr,
        "batch_size":    cfg.training.batch_size,
        "warmup_epochs": cfg.training.warmup_epochs,
        "box_weight":    cfg.loss.box_weight,
        "trial_number":  trial.number,
    })
    trial.report(map50, step=cfg.training.epochs)
    return map50  
