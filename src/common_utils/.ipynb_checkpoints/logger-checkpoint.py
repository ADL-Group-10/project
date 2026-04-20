"""
src/common_utils/logger.py

Standardised logger for all modules. Consistent timestamps and format.

Who calls this:
    Achyuth  — data loading warnings (missing annotations, split issues)
    Dharmini  — training progress, loss values, checkpoint saves
    Sathish  — Optuna trial progress, pruned trial notifications
    TBD  — evaluation progress, metric results

Usage:
    from src.common_utils import load_config, get_logger

    cfg    = load_config()
    logger = get_logger(__name__, cfg)

    logger.info("Starting training run...")
    logger.warning("Missing annotation: frame_042.jpg")
    logger.debug("Batch 12/200 — loss: 0.432")   # only shown at DEBUG level
"""

import logging
import sys
from pathlib import Path
from omegaconf import DictConfig

_loggers: dict[str, logging.Logger] = {}   # registry — prevents duplicate handlers

FMT      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%H:%M:%S"


def get_logger(
    name:     str,
    cfg:      DictConfig | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Safe to call at module level — calling twice with the same name returns
    the same logger (no duplicate handlers).

    Args:
        name:     Use __name__ so log lines show the calling module.
                  e.g. "src.models.trainer", "src.data.nvd_dataset"
        cfg:      Merged config. Reads cfg.project.log_level if provided.
                  Defaults to INFO if None.
        log_file: Optional path to also write logs to disk.
                  Useful for long cluster training runs (Dharmini).

    Returns:
        logging.Logger

    Example output:
        14:32:01 | INFO     | src.models.trainer   | Epoch 12/100 — loss: 0.341
        14:32:05 | WARNING  | src.data.nvd_dataset | Missing label: frame_042.jpg
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.propagate = False

    level_str = "INFO"
    if cfg is not None:
        level_str = getattr(cfg.project, "log_level", "INFO")
    logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    formatter = logging.Formatter(FMT, datefmt=DATE_FMT)

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setFormatter(formatter)
    logger.addHandler(stdout_h)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)

    _loggers[name] = logger
    return logger


def get_training_logger(cfg: DictConfig) -> logging.Logger:
    """
    Logger that also writes to outputs/logs/{experiment}.log.

    Example (in trainer.py):
        logger = get_training_logger(cfg)
        logger.info(f"Epoch {epoch} — train_loss: {loss:.4f}")
    """
    log_path = Path(cfg.paths.log_dir) / f"{cfg.experiment}.log"
    return get_logger("training", cfg, log_file=log_path)
