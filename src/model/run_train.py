"""
CLI entry point for full training.

    python -m src.model.run_train --variant v1
    python -m src.model.run_train --variant v2 --epochs 1   # smoke run
    python -m src.model.run_train --variant v3_ds
"""

from __future__ import annotations
import argparse

from src.common_utils import load_config

from . import build_trainer


def run(variant: str = "v1", epochs: int | None = None, resume: bool = False) -> None:
    cfg = load_config(variant=variant)
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    build_trainer(cfg).train(resume=resume)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full training for the given variant.")
    p.add_argument("--variant", default="v1", choices=["v0", "v1", "v2", "v3_ds"])
    p.add_argument("--epochs", type=int, default=None,
                   help="Override cfg.training.epochs (useful for sanity runs)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume training from last.pt checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(variant=args.variant, epochs=args.epochs, resume=args.resume)
