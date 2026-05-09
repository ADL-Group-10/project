"""
Patch the best trial's hyperparameters back into config.yaml.

The HP → yaml-path mapping is read from cfg.tuning.targets, so this module
is framework-agnostic. A timestamped backup is saved before overwriting.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import shutil

import optuna
from omegaconf import OmegaConf

from src.common_utils import get_logger


def write_best_to_config(
    study: optuna.Study,
    config_path: str | Path = "config.yaml",
    backup: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Write study.best_trial.params back into config.yaml.

    Args:
        study:       Completed Optuna study.
        config_path: Path to config.yaml (default: project-root config.yaml).
        backup:      If True, copies the existing yaml to
                     config.yaml.bak-YYYYMMDD-HHMMSS before overwriting.
        dry_run:     If True, logs what *would* change and returns the diff
                     without writing. Useful when reviewing study output.

    Returns:
        Dict of {yaml_path_str: (old_value, new_value)} describing what was
        (or would be) changed.

    Raises:
        ValueError: study has no completed trials.
    """
    logger = get_logger("tuning.write_best")
    path = Path(config_path)

    if not study.trials or study.best_trial is None:
        raise ValueError("Study has no best trial — nothing to write.")

    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found: {path}")

    cfg = OmegaConf.load(path)
    best_params = study.best_trial.params
    targets = cfg.tuning.targets
    diff: dict = {}

    for hp_name, key_path in targets.items():
        if hp_name not in best_params:
            continue
        key_path = str(key_path)
        new_val  = best_params[hp_name]
        old_val  = OmegaConf.select(cfg, key_path, default=None)
        if old_val == new_val:
            continue
        diff[key_path] = (old_val, new_val)
        OmegaConf.update(cfg, key_path, new_val, merge=False)

    if dry_run:
        logger.info("[dry-run] would change:")
        for k, (old, new) in diff.items():
            logger.info(f"   {k}: {old} → {new}")
        return diff

    if backup:
        stamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".bak-{stamp}")
        shutil.copy2(path, backup_path)
        logger.info(f"Backup saved to {backup_path}")

    OmegaConf.save(cfg, path)
    logger.info(f"Patched {path} with best trial #{study.best_trial.number}")
    for k, (old, new) in diff.items():
        logger.info(f"   {k}: {old} → {new}")

    return diff
