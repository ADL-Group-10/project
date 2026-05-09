"""
Build the Optuna study (sampler + pruner + SQLite storage + optional W&B
callback) and run it. Resumes prior runs via study_name.
"""

from __future__ import annotations
from typing import Optional
import warnings

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners  import HyperbandPruner, MedianPruner, NopPruner
from optuna.exceptions import ExperimentalWarning
from omegaconf import DictConfig, OmegaConf

from src.common_utils import load_config, get_paths, get_logger

from .objective import build_objective

warnings.filterwarnings("ignore", category=ExperimentalWarning)


def _build_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    name = str(cfg.tuning.sampler).lower()
    seed = int(cfg.project.seed)
    if name == "tpe":
        return TPESampler(seed=seed, multivariate=True, group=True)
    if name == "random":
        return RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler '{name}' — use 'tpe' or 'random'")


def _build_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    """Hyperband max_resource matches the per-trial reporting cadence."""
    name = str(cfg.tuning.pruner).lower()
    if name == "hyperband":
        budget = getattr(cfg.tuning, "trial_epochs", None) or cfg.training.epochs
        return HyperbandPruner(
            min_resource=1,
            max_resource=max(4, int(budget)),
            reduction_factor=3,
        )
    if name == "median":
        return MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    if name in ("none", "nop"):
        return NopPruner()
    raise ValueError(f"Unknown pruner '{name}' — use 'hyperband' | 'median' | 'none'")


def create_study(cfg: DictConfig) -> optuna.Study:
    """Create or resume the Optuna study declared in cfg.tuning."""
    paths = get_paths(cfg)
    storage_url = f"sqlite:///{paths.optuna_db.as_posix()}"

    return optuna.create_study(
        study_name     = str(cfg.tuning.study_name),
        direction      = str(cfg.tuning.direction),
        storage        = storage_url,
        sampler        = _build_sampler(cfg),
        pruner         = _build_pruner(cfg),
        load_if_exists = True,
    )


def run_study(
    cfg: Optional[DictConfig] = None,
    n_trials: Optional[int] = None,
    timeout:  Optional[int] = None,
) -> optuna.Study:
    """Load config → create study → optimize → return study."""
    if cfg is None:
        cfg = load_config(variant="v1")

    logger = get_logger("tuning.study", cfg)
    study  = create_study(cfg)

    n_trials = int(n_trials if n_trials is not None else cfg.tuning.n_trials)
    timeout  = int(timeout  if timeout  is not None else cfg.tuning.timeout_seconds)

    logger.info(
        f"Starting study '{cfg.tuning.study_name}' — "
        f"{n_trials} trials, {timeout}s cap, sampler={cfg.tuning.sampler}, "
        f"pruner={cfg.tuning.pruner}"
    )

    study.optimize(
        build_objective(cfg),
        n_trials       = n_trials,
        timeout        = timeout,
        callbacks      = _build_callbacks(cfg),
        gc_after_trial = True,
    )

    _log_summary(study, logger)
    _log_study_summary_to_wandb(study, cfg, logger)
    return study


def _build_callbacks(cfg: DictConfig) -> list:
    """W&B callback — optional, degrades cleanly if unavailable."""
    callbacks: list = []
    logger = get_logger("tuning.wandb", cfg)

    if not getattr(cfg.logging, "wandb_project", None):
        logger.info("wandb_project not set — skipping W&B integration.")
        return callbacks

    try:
        from optuna.integration.wandb import WeightsAndBiasesCallback
    except ImportError:
        try:
            from optuna_integration.wandb import WeightsAndBiasesCallback
        except ImportError:
            logger.warning(
                "optuna W&B integration unavailable — install `optuna-integration[wandb]`."
            )
            return callbacks

    wandb_kwargs: dict = {
        "project": str(cfg.logging.wandb_project),
        "group":   f"optuna-{cfg.tuning.study_name}",
        "dir":     str(cfg.paths.wandb_dir),
        "config":  OmegaConf.to_container(cfg.tuning, resolve=True),
    }
    entity = getattr(cfg.logging, "wandb_entity", None)
    if entity:
        wandb_kwargs["entity"] = str(entity)

    callbacks.append(WeightsAndBiasesCallback(
        metric_name  = str(cfg.tuning.metric),
        wandb_kwargs = wandb_kwargs,
        as_multirun  = True,
    ))
    logger.info(
        f"W&B callback active → project={wandb_kwargs['project']}"
        + (f", entity={entity}" if entity else " (default entity)")
    )
    return callbacks


def _log_study_summary_to_wandb(study: optuna.Study, cfg: DictConfig, logger) -> None:
    """Log Optuna visualization plots and best-trial info to a single W&B run."""
    if not getattr(cfg.logging, "wandb_project", None):
        return
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    try:
        import wandb
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as ovm
    except ImportError:
        logger.info("wandb or optuna matplotlib backend missing — skipping study summary.")
        return

    run = wandb.init(
        project  = str(cfg.logging.wandb_project),
        entity   = getattr(cfg.logging, "wandb_entity", None),
        group    = f"optuna-{cfg.tuning.study_name}",
        name     = f"{cfg.tuning.study_name}-summary",
        job_type = "study-summary",
        dir      = str(cfg.paths.wandb_dir),
        reinit   = True,
    )

    plots = [
        ("optimization_history", ovm.plot_optimization_history),
        ("param_importances",    ovm.plot_param_importances),
        ("parallel_coordinate",  ovm.plot_parallel_coordinate),
        ("slice",                ovm.plot_slice),
    ]
    for name, fn in plots:
        try:
            obj = fn(study)
            fig = obj.figure if hasattr(obj, "figure") else obj.flatten()[0].figure
            wandb.log({f"optuna/{name}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            logger.warning(f"failed to log {name}: {e}")

    wandb.log({
        "study/best_value":  float(study.best_value),
        "study/best_trial":  int(study.best_trial.number),
        "study/n_complete":  len(completed),
        "study/n_total":     len(study.trials),
    })
    wandb.config.update({"best_params": dict(study.best_trial.params)}, allow_val_change=True)
    wandb.finish()


def _log_summary(study: optuna.Study, logger) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    logger.info(
        f"Study done — {len(study.trials)} total in storage "
        f"({len(completed)} complete, {len(pruned)} pruned, {len(failed)} failed)"
    )

    if not completed:
        logger.warning("No completed trials — nothing to report as best.")
        return

    best = study.best_trial
    logger.info(f"Best trial #{best.number} — {study.best_value:.4f}")
    for k, v in best.params.items():
        logger.info(f"   {k}: {v}")
