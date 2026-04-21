"""
src/tuning/study.py

Creates an Optuna study, wires in TPE + Hyperband + WandB, and runs it.

Contract with config.yaml → tuning:
    n_trials        → study.optimize(n_trials=...)
    timeout_seconds → study.optimize(timeout=...)
    direction       → create_study(direction=...)
    metric          → used as metric_name on the WandB callback
    pruner          → "hyperband" | "median" | "none"
    sampler         → "tpe" | "random"
    study_name      → create_study(study_name=...)
    trial_epochs    → Hyperband max_resource AND per-trial epoch cap
                      (honoured inside apply_hp_to_config; falls back to
                      cfg.training.epochs if not set)

Storage:
    Uses SQLite at cfg.paths.optuna_db so the study resumes cleanly between
    sessions. Call run_study() again with the same study_name and it picks
    up where it left off.

    NOTE: load_if_exists=True means if you edit cfg.tuning.search_space and
    rerun with the same study_name, the old param distributions are still
    stored. Either change study_name or delete the sqlite file when you
    materially change the search space.

WandB:
    If cfg.logging.wandb_project is set, each trial is logged as its own
    W&B run (as_multirun=True) so parallel-coordinates plots work out of
    the box. If the optuna W&B integration isn't installed, or if
    cfg.logging.wandb_project is missing/None, the study still runs — just
    without per-trial W&B runs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import warnings

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners  import HyperbandPruner, MedianPruner, NopPruner
from optuna.exceptions import ExperimentalWarning
from omegaconf import DictConfig

from src.common_utils import load_config, get_paths, get_logger

from .objective import build_objective

warnings.filterwarnings("ignore", category=ExperimentalWarning)

# ── Sampler / pruner factories ─────────────────────────────────────────────

def _build_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    name = str(cfg.tuning.sampler).lower()
    seed = int(cfg.project.seed)
    if name == "tpe":
        return TPESampler(seed=seed, multivariate=True, group=True)
    if name == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler '{name}' — use 'tpe' or 'random'")


def _build_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    """
    Build the configured pruner.

    Hyperband's `max_resource` should match the number of steps a trial
    actually reports — which is cfg.tuning.trial_epochs when set (so the
    pruner brackets line up with the stub's and the real trainer's report
    cadence), falling back to cfg.training.epochs otherwise.
    """
    name = str(cfg.tuning.pruner).lower()
    if name == "hyperband":
        budget = getattr(cfg.tuning, "trial_epochs", None)
        if budget is None:
            budget = cfg.training.epochs
        max_resource = max(4, int(budget))
        return HyperbandPruner(
            min_resource=1,
            max_resource=max_resource,
            reduction_factor=3,
        )
    if name == "median":
        return MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    if name in ("none", "nop"):
        return NopPruner()
    raise ValueError(f"Unknown pruner '{name}' — use 'hyperband' | 'median' | 'none'")


# ── Public API ─────────────────────────────────────────────────────────────

def create_study(cfg: DictConfig) -> optuna.Study:
    """
    Create (or resume) the Optuna study declared in cfg.tuning.

    Uses SQLite storage so repeated calls with the same study_name resume.
    The storage directory is already created by get_paths(cfg).
    """
    paths = get_paths(cfg)
    storage_url = f"sqlite:///{paths.optuna_db.as_posix()}"

    study = optuna.create_study(
        study_name     = str(cfg.tuning.study_name),
        direction      = str(cfg.tuning.direction),
        storage        = storage_url,
        sampler        = _build_sampler(cfg),
        pruner         = _build_pruner(cfg),
        load_if_exists = True,
    )
    return study


def run_study(
    cfg: Optional[DictConfig] = None,
    n_trials: Optional[int] = None,
    timeout:  Optional[int] = None,
) -> optuna.Study:
    """
    One-shot: load config → create study → optimize → return study.

    Args:
        cfg:      Merged config. If None, load_config() is called.
        n_trials: Override cfg.tuning.n_trials (handy for quick smoke runs).
        timeout:  Override cfg.tuning.timeout_seconds.

    Returns:
        The completed optuna.Study (trials available via study.trials).

    Example:
        from src.tuning import run_study, write_best_to_config
        study = run_study(n_trials=5)   # smoke run
        write_best_to_config(study)
    """
    if cfg is None:
        cfg = load_config(variant="v1")   # tuning always runs on v1

    logger = get_logger("tuning.study", cfg)
    study  = create_study(cfg)

    n_trials = int(n_trials if n_trials is not None else cfg.tuning.n_trials)
    timeout  = int(timeout  if timeout  is not None else cfg.tuning.timeout_seconds)

    callbacks = _build_callbacks(cfg)

    logger.info(
        f"Starting study '{cfg.tuning.study_name}' — "
        f"{n_trials} trials, {timeout}s cap, sampler={cfg.tuning.sampler}, "
        f"pruner={cfg.tuning.pruner}"
    )

    objective = build_objective(cfg)
    study.optimize(
        objective,
        n_trials       = n_trials,
        timeout        = timeout,
        callbacks      = callbacks,
        gc_after_trial = True,
    )

    _log_summary(study, logger)
    return study


# ── WandB callback ─────────────────────────────────────────────────────────

def _build_callbacks(cfg: DictConfig) -> list:
    """Build the callbacks list. WandB is optional — degrades cleanly."""
    callbacks: list = []
    logger = get_logger("tuning.wandb", cfg)

    if not getattr(cfg.logging, "wandb_project", None):
        logger.info("wandb_project not set — skipping W&B integration.")
        return callbacks

    try:
        from optuna.integration.wandb import WeightsAndBiasesCallback
    except ImportError:
        try:
            # Newer optuna-integration package split-out
            from optuna_integration.wandb import WeightsAndBiasesCallback
        except ImportError:
            logger.warning(
                "optuna W&B integration unavailable — install `optuna-integration[wandb]`."
            )
            return callbacks

    # Entity is optional — W&B will use the default entity if not specified,
    # which is useful for contributors who aren't part of ltu-group10 yet.
    wandb_kwargs: dict = {
        "project": str(cfg.logging.wandb_project),
        "group":   f"optuna-{cfg.tuning.study_name}",
        "dir":     str(cfg.paths.wandb_dir),
        "config":  {"sampler": str(cfg.tuning.sampler),
                    "pruner":  str(cfg.tuning.pruner)},
    }
    entity = getattr(cfg.logging, "wandb_entity", None)
    if entity:
        wandb_kwargs["entity"] = str(entity)

    wandb_cb = WeightsAndBiasesCallback(
        metric_name  = str(cfg.tuning.metric),
        wandb_kwargs = wandb_kwargs,
        as_multirun  = True,   # one W&B run per trial
    )
    callbacks.append(wandb_cb)
    logger.info(
        f"W&B callback active → project={wandb_kwargs['project']}"
        + (f", entity={entity}" if entity else " (default entity)")
    )
    return callbacks


# ── Summary ────────────────────────────────────────────────────────────────

def _log_summary(study: optuna.Study, logger) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    total     = len(study.trials)

    logger.info(
        f"Study done — {total} total in storage "
        f"({len(completed)} complete, {len(pruned)} pruned, {len(failed)} failed)"
    )

    if not completed:
        logger.warning("No completed trials — nothing to report as best.")
        return

    best = study.best_trial
    logger.info(f"Best trial #{best.number} — {study.best_value:.4f}")
    for k, v in best.params.items():
        logger.info(f"   {k}: {v}")
