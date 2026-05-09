"""Ultralytics trainer. Implements src.model.protocol.Trainer."""

import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import torch
from ultralytics import YOLO, settings as ul_settings

from src.common_utils import get_device_str
from src.data import DataPipeline


def _configure_wandb(cfg) -> None:
    """Point Ultralytics' built-in WandB logger at the configured project."""
    ul_settings.update({"wandb": True})
    os.environ["WANDB_PROJECT"] = str(cfg.logging.wandb_project)
    os.environ["WANDB_ENTITY"]  = str(cfg.logging.wandb_entity)
    os.environ["WANDB_DIR"]     = str(cfg.paths.wandb_dir)
    print(f"[trainer] WandB : project={cfg.logging.wandb_project}, entity={cfg.logging.wandb_entity}")


def _cos_lr_from_cfg(scheduler: str) -> bool:
    """Map cfg.training.lr_scheduler to Ultralytics' cos_lr boolean."""
    s = str(scheduler).lower()
    if s == "cosine":
        return True
    if s == "linear":
        return False
    raise ValueError(f"Unsupported lr_scheduler '{scheduler}' for UltralyticsTrainer — use 'cosine' or 'linear'.")


class UltralyticsTrainer:
    """One trainer; variant behavior comes from cfg."""

    def __init__(self, cfg) -> None:
        self.cfg    = cfg
        self.device = get_device_str(cfg)
        self.model  = YOLO(cfg.model.weights)
        print(f"[trainer] {cfg.model.weights} loaded on {self.device}")

        augment      = "snow" if cfg.augmentation.use_snow_aug else "base"
        domain_shift = bool(getattr(cfg.domain_shift, "enabled", False))
        pipeline     = DataPipeline("config.yaml")
        dataset_path, self._aug = pipeline.run(augment=augment, domain_shift=domain_shift)
        self.dataset_yaml = str((dataset_path / "dataset.yaml").resolve())

        self.results_dir = Path(cfg.paths.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train(self, name: str | None = None) -> None:
        """Full training run."""
        name = name or str(self.cfg.experiment)
        t  = self.cfg.training
        lo = self.cfg.loss

        os.environ["WANDB_NAME"] = name
        _configure_wandb(self.cfg)
        self._init_wandb_run(name)
        self._attach_albumentations()
        self._attach_val_cadence()
        self._attach_watch_model()
        self._attach_cfg_snapshot()
        self._attach_cache_clear()

        self.model.train(
            # --- DATA & HARDWARE ---
            data          = self.dataset_yaml,
            imgsz         = self.cfg.model.img_size,
            device        = self.device,
            workers       = t.num_workers,

            # --- TRAINING PARAMS ---
            epochs        = t.epochs,
            batch         = t.batch_size,
            optimizer     = t.optimizer.capitalize(),
            weight_decay  = t.weight_decay,
            warmup_epochs = t.warmup_epochs,
            patience      = t.early_stopping_patience,
            seed          = self.cfg.project.seed,
            freeze        = 10 if self.cfg.model.freeze_backbone else 0,
            save_period   = int(t.save_every_n_epochs),

            # --- LR SCHEDULER ---
            lr0           = t.lr,
            lrf           = t.lrf,
            cos_lr        = _cos_lr_from_cfg(t.lr_scheduler),

            # --- REGULARIZATION ---
            dropout       = t.dropout,

            # --- LOSS ---
            box           = lo.box_weight,
            cls           = lo.cls_weight,
            dfl           = lo.dfl_weight,

            # --- OUTPUT ---
            project       = str(self.results_dir.resolve()),
            name          = name,
            verbose       = True,
            exist_ok      = True,
            plots         = True,
        )
        print(f"[trainer] Done. Best model: {self.results_dir.resolve()}/{name}/weights/best.pt")

    def validate(self) -> dict:
        """Run validation, return metrics dict."""
        metrics = self.model.val(
            data    = self.dataset_yaml,
            imgsz   = self.cfg.model.img_size,
            device  = self.device,
            verbose = False,
        )
        return {
            "mAP50":     metrics.box.map50,
            "mAP50_95":  metrics.box.map,
            "precision": metrics.box.mp,
            "recall":    metrics.box.mr,
        }

    def train_one_trial(self, trial_number: int = 0) -> float:
        """
        Optuna trial. HPs are read from self.cfg — the tuner overlays
        suggested values via apply_hp_to_config before instantiating us.
        WandB is disabled here to avoid run spam during tuning.
        """
        ul_settings.update({"wandb": False})

        self.model = YOLO(self.cfg.model.weights)  # fresh weights per trial
        self._attach_albumentations()
        self._attach_val_cadence()
        self._attach_cache_clear()
        self.model.train(
            # --- DATA & HARDWARE ---
            data          = self.dataset_yaml,
            imgsz         = self.cfg.tuning.trial_img_size,
            device        = self.device,
            workers       = self.cfg.training.num_workers,

            # --- TRAINING PARAMS ---
            epochs        = self.cfg.tuning.trial_epochs,
            batch         = self.cfg.training.batch_size,
            lr0           = self.cfg.training.lr,
            warmup_epochs = self.cfg.training.warmup_epochs,
            box           = self.cfg.loss.box_weight,
            freeze        = 10 if self.cfg.model.freeze_backbone else 0,

            # --- LR SCHEDULER ---
            lrf           = self.cfg.training.lrf,
            cos_lr        = _cos_lr_from_cfg(self.cfg.training.lr_scheduler),

            # --- REGULARIZATION ---
            dropout       = self.cfg.training.dropout,
            weight_decay  = self.cfg.training.weight_decay,

            # --- LOSS ---
            optimizer     = self.cfg.training.optimizer.capitalize(),
            cls           = self.cfg.loss.cls_weight,
            dfl           = self.cfg.loss.dfl_weight,

            # --- OUTPUT ---
            project       = str((self.results_dir.parent / "optuna_trials").resolve()),
            name          = f"trial_{trial_number}",
            verbose       = False,
            exist_ok      = True,
            plots         = True,
        )

        metrics = self.validate()
        torch.cuda.empty_cache()
        return metrics["mAP50"]

    # ── Private ───────────────────────────────────────────────────

    def _compose_for(self, split: str):
        """Return the Albumentations Compose for a given split, or None."""
        if isinstance(self._aug, dict):
            return self._aug.get(split)
        return self._aug if split == "train" else None

    def _attach_albumentations(self) -> None:
        """Inject the data-layer Compose into Ultralytics' built-in Albumentations slot."""
        from ultralytics.data.augment import Albumentations as ULAlb

        def _patch(trainer):
            loaders = [("train", getattr(trainer, "train_loader", None))]
            v = getattr(getattr(trainer, "validator", None), "dataloader", None)
            if v is not None:
                loaders.append(("val", v))
            for split, loader in loaders:
                compose = self._compose_for(split)
                if not (loader and compose):
                    continue
                for t in loader.dataset.transforms.transforms:
                    if isinstance(t, ULAlb):
                        t.transform = compose
                        t.p = 1.0

        self.model.add_callback("on_pretrain_routine_start", _patch)

    def _attach_val_cadence(self) -> None:
        """Run validation every cfg.training.val_every_n_epochs epochs (plus the final epoch)."""
        cadence = int(self.cfg.training.val_every_n_epochs)
        if cadence <= 1:
            return  # default: validate every epoch

        def _patch(trainer):
            epoch = int(getattr(trainer, "epoch", 0))
            last  = epoch >= int(getattr(trainer, "epochs", 0)) - 1
            trainer.args.val = ((epoch + 1) % cadence == 0) or last

        self.model.add_callback("on_train_epoch_start", _patch)

    def _attach_watch_model(self) -> None:
        """Call wandb.watch on the underlying torch model when cfg.logging.watch_model is true."""
        if not bool(getattr(self.cfg.logging, "watch_model", False)):
            return

        def _patch(trainer):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.watch(trainer.model)
            except ImportError:
                pass

        self.model.add_callback("on_train_start", _patch)

    def _init_wandb_run(self, name: str) -> None:
        """
        Pre-init wandb.run with our project name so Ultralytics' WandB callback
        inherits it. Without this, Ultralytics derives the wandb project name
        from `trainer.args.project` (the output dir), giving a wrong project.
        """
        if not getattr(self.cfg.logging, "wandb_project", None):
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            return
        wandb.init(
            project = str(self.cfg.logging.wandb_project),
            entity  = getattr(self.cfg.logging, "wandb_entity", None),
            name    = name,
            dir     = str(self.cfg.paths.wandb_dir),
            reinit  = True,
        )

    def _attach_cache_clear(self) -> None:
        """Release CUDA cache and Python objects between train and val each epoch."""
        def _patch(_trainer):
            torch.cuda.empty_cache()
            gc.collect()
        self.model.add_callback("on_train_epoch_end", _patch)

    def _attach_cfg_snapshot(self) -> None:
        """Push the merged cfg to wandb.config once the run is active."""
        def _patch(trainer):
            try:
                import wandb
                from omegaconf import OmegaConf
                if wandb.run is not None:
                    wandb.config.update(
                        OmegaConf.to_container(self.cfg, resolve=True),
                        allow_val_change=True,
                    )
            except ImportError:
                pass

        self.model.add_callback("on_train_start", _patch)
