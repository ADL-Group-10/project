"""
src/tuning/run_tuning.py

CLI entry point.

    python -m src.tuning.run_tuning                                       # full study
    python -m src.tuning.run_tuning --n-trials 1 --trial-epochs 2 --no-write   # smoke
    python -m src.tuning.run_tuning --no-write                            # skip yaml patch
    python -m src.tuning.run_tuning --dry-run                             # show best HPs only

Also importable as a function (run()) for notebook use.
"""

from __future__ import annotations
import argparse

from src.common_utils import load_config, get_logger
from .study      import run_study
from .write_best import write_best_to_config


def run(
    n_trials:     int | None = None,
    timeout:      int | None = None,
    trial_epochs: int | None = None,
    write:        bool = True,
    dry_run:      bool = False,
) -> None:
    cfg    = load_config(variant="v1")
    if trial_epochs is not None:
        cfg.tuning.trial_epochs = int(trial_epochs)
    logger = get_logger("tuning.cli", cfg)

    study = run_study(cfg=cfg, n_trials=n_trials, timeout=timeout)

    if not study.trials:
        logger.warning("No trials ran — aborting write_best.")
        return

    if not write:
        logger.info("--no-write passed — skipping config.yaml patch.")
        return

    write_best_to_config(study, dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Optuna HP tuning for YOLOv9 v1 baseline.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override cfg.tuning.n_trials")
    p.add_argument("--timeout", type=int, default=None,
                   help="Override cfg.tuning.timeout_seconds")
    p.add_argument("--trial-epochs", type=int, default=None,
                   help="Override cfg.tuning.trial_epochs (useful for sanity runs)")
    p.add_argument("--no-write", action="store_true",
                   help="Do not patch config.yaml with best trial")
    p.add_argument("--dry-run", action="store_true",
                   help="Log the proposed config.yaml diff instead of writing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        n_trials     = args.n_trials,
        timeout      = args.timeout,
        trial_epochs = args.trial_epochs,
        write        = not args.no_write,
        dry_run      = args.dry_run,
    )
