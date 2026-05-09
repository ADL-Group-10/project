"""
Framework-agnostic Trainer contract.

Implement this Protocol and point cfg.model.trainer_class at the dotted path
to plug a new framework (Ultralytics, MMDetection, Detectron2, custom Torch
loop, etc.) into both full training and the Optuna tuning seam.
"""

from __future__ import annotations
from typing import Protocol

from omegaconf import DictConfig


class Trainer(Protocol):
    def __init__(self, cfg: DictConfig) -> None: ...

    def train(self, name: str | None = None) -> None:
        """Full training run. If name is None, defaults to cfg.experiment."""
        ...

    def validate(self) -> dict:
        """Run validation; return a dict of metrics (mAP50, mAP50_95, precision, recall)."""
        ...

    def train_one_trial(self, trial_number: int = 0) -> float:
        """Optuna trial: short training, return the metric to optimize per cfg.tuning.direction."""
        ...
