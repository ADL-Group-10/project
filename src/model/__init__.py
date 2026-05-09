"""Model layer: trainer dispatch + Protocol."""

from __future__ import annotations
import importlib

from omegaconf import DictConfig

from .protocol import Trainer


def build_trainer(cfg: DictConfig) -> Trainer:
    """Instantiate the trainer at cfg.model.trainer_class."""
    module_path, class_name = str(cfg.model.trainer_class).rsplit(".", 1)
    TrainerCls = getattr(importlib.import_module(module_path), class_name)
    return TrainerCls(cfg)


__all__ = ["Trainer", "build_trainer"]
