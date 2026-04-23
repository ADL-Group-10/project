"""
src/model/baseline_yolov9/trainer.py

V1 Baseline Trainer — standard augmentation, no snow transforms.

Sathish calls train_one_epoch() + validate() from objective.py.
He passes in a cfg with hyperparams already set . just use cfg.training.* directly.

Usage (notebook):
    from src.model.baseline_yolov9.trainer import YOLOv9Trainer
    from src.common_utils import load_config

    cfg     = load_config(variant="v1")
    trainer = YOLOv9Trainer(cfg)
    trainer.train()                  # full training loop
"""

import sys
import subprocess
from pathlib import Path

import torch
import torch.optim as optim

from src.common_utils import load_config, save_checkpoint
from src.common_utils.paths import get_paths
from src.data import DataPipeline


def _ensure_yolov9(repo_path: Path) -> None:
    """Clone YOLOv9 into third_party/ if not already there."""
    if not repo_path.exists():
        print(f"[trainer] Cloning YOLOv9 into {repo_path} ...")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/WongKinYiu/yolov9.git", str(repo_path)],
            check=True,
        )
        print("[trainer] YOLOv9 cloned.")
    # Make sure yolov9 is importable
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


class YOLOv9Trainer:
    """
    Wraps YOLOv9 for training on the Nordic Vehicle Dataset.

    (tuning) calls:
        trainer.train_one_epoch()       → runs one epoch, returns train loss
        trainer.validate()              → returns {"mAP50": float, "mAP50_95": float}

    model training (notebook) calls:
        trainer.train()                 → full loop with checkpointing

    Args:
        cfg: Merged OmegaConf config from load_config(variant="v1").
             Sathish will pass in a cfg with his trial hyperparams already
             written into cfg.training.* — no changes needed here.
    """

    def __init__(self, cfg) -> None:
        self.cfg    = cfg
        self.paths  = get_paths(cfg)
        self.device = self._get_device()

        # ── YOLOv9 setup ──────────────────────────────────────────
        yolov9_repo = Path(cfg.model.yolov9_repo)
        _ensure_yolov9(yolov9_repo)

        # Import YOLOv9 internals (only available after clone + sys.path)
        from models.yolo import Model as YOLOv9Model          # noqa: E402
        from utils.general import check_img_size              # noqa: E402

        # Load pretrained weights
        weights_path = yolov9_repo / cfg.model.weights
        if not weights_path.exists():
            # Download via YOLOv9's own helper if weights file missing
            print(f"[trainer] Weights not found at {weights_path}, attempting download...")
            from utils.downloads import attempt_download
            attempt_download(weights_path)
        
       #ckpt        = torch.load(weights_path, map_location=self.device)
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model  = ckpt["model"].float().to(self.device)

        # Freeze backbone if config says so
        if cfg.model.freeze_backbone:
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False
            print("[trainer] Backbone frozen.")

        # ── Data ──────────────────────────────────────────────────
        # V1 uses "base" augmentation (no snow)
        pipeline            = DataPipeline(str(Path("config.yaml")))
        dataset_path, augs  = pipeline.run(augment="base")
        self.dataset_yaml   = str(dataset_path / "dataset.yaml")
        self.aug_pipeline   = augs

        # ── Optimizer ─────────────────────────────────────────────
        # tuning writes best lr + weight_decay into cfg.training before
        # model runs the final training — this just reads them.
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # ── State ─────────────────────────────────────────────────
        self.current_epoch  = 0
        self.best_map50     = 0.0

    # ── Public API (called by tuning objective.py) ─────────────

    def train_one_epoch(self) -> float:
        """
        Run one training epoch. Returns average training loss.
        Sathish calls this inside his Optuna trial loop.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # YOLOv9's own dataloader
        from utils.dataloaders import create_dataloader
        train_loader, _ = create_dataloader(
            path        = self.dataset_yaml,
            imgsz       = self.cfg.model.img_size,
            batch_size  = self.cfg.training.batch_size,
            stride      = 32,
            hyp         = self._build_hyp(),
            augment     = True,
            workers     = self.cfg.training.num_workers,
            prefix      = "train: ",
        )

        for batch in train_loader:
            imgs, targets, *_ = batch
            imgs    = imgs.to(self.device).float() / 255.0
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss, _ = self.model.compute_loss(preds, targets)
            loss.backward()
            self.optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        self.current_epoch += 1
        avg_loss = total_loss / max(num_batches, 1)
        print(f"[trainer] Epoch {self.current_epoch} — train_loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self) -> dict:
        """
        Run validation. Returns {"mAP50": float, "mAP50_95": float, "val_loss": float}.
        tuning calls this to get the metric he reports to Optuna.
        """
        from val import run as yolo_val   # YOLOv9 built-in val script

        results = yolo_val(
            data        = self.dataset_yaml,
            weights     = None,            # model already in memory
            model       = self.model,
            imgsz       = self.cfg.model.img_size,
            batch_size  = self.cfg.training.batch_size,
            conf_thres  = self.cfg.evaluation.conf_threshold,
            iou_thres   = self.cfg.evaluation.iou_threshold,
            device      = str(self.device),
            workers     = self.cfg.training.num_workers,
            verbose     = False,
        )

        # results tuple: (mp, mr, map50, map, losses, ...)
        map50    = float(results[0][2])
        map50_95 = float(results[0][3])
        val_loss = float(results[1][0]) if len(results) > 1 else 0.0

        metrics = {
            "mAP50":    map50,
            "mAP50_95": map50_95,
            "val_loss": val_loss,
        }
        print(f"[trainer] Val — mAP50: {map50:.4f}  mAP50-95: {map50_95:.4f}")
        return metrics

    # ── Full training loop (called from notebook) ──────────────────

    def train(self) -> None:
        """
        Full training loop with checkpointing and early stopping.
        Call this from the notebook — not from Sathish's tuning code.
        """
        epochs           = self.cfg.training.epochs
        patience         = self.cfg.training.early_stopping_patience
        val_every        = self.cfg.training.val_every_n_epochs
        save_every       = self.cfg.training.save_every_n_epochs
        no_improve_count = 0

        print(f"\n[trainer] Starting V1 training — {epochs} epochs on {self.device}")
        print(f"[trainer] Experiment: {self.cfg.experiment}\n")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            self.scheduler.step()

            # Validate every N epochs
            if (epoch + 1) % val_every == 0:
                metrics  = self.validate()
                map50    = metrics["mAP50"]
                is_best  = map50 > self.best_map50

                if is_best:
                    self.best_map50  = map50
                    no_improve_count = 0
                else:
                    no_improve_count += val_every

                save_checkpoint(
                    model     = self.model,
                    optimizer = self.optimizer,
                    epoch     = epoch,
                    metrics   = metrics,
                    cfg       = self.cfg,
                    is_best   = is_best,
                )

            # Periodic checkpoint (even without validation)
            elif (epoch + 1) % save_every == 0:
                save_checkpoint(
                    model     = self.model,
                    optimizer = self.optimizer,
                    epoch     = epoch,
                    metrics   = {"train_loss": train_loss},
                    cfg       = self.cfg,
                    is_best   = False,
                )

            # Early stopping
            if no_improve_count >= patience:
                print(f"[trainer] Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                break

        print(f"\n[trainer] Training complete. Best mAP50: {self.best_map50:.4f}")
        print(f"[trainer] Best checkpoint saved to: {self.paths.best_ckpt_v1}")

    # ── Private helpers ────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        requested = self.cfg.project.device   # "cuda:0" from config
        if "cuda" in requested and torch.cuda.is_available():
            return torch.device(requested)
        print(f"[trainer] {requested} not available, falling back to CPU.")
        return torch.device("cpu")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config. Sathish may tune lr + weight_decay."""
        lr           = self.cfg.training.lr
        weight_decay = self.cfg.training.weight_decay
        name         = self.cfg.training.optimizer.lower()

        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.937, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer '{name}' in config. Use 'adam' or 'sgd'.")

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine annealing LR scheduler with warmup epochs."""
        warmup  = self.cfg.training.warmup_epochs
        epochs  = self.cfg.training.epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return epoch / max(warmup, 1)          # linear warmup
            # cosine decay after warmup
            import math
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_hyp(self) -> dict:
        """
        Build YOLOv9 hyp dict from config.
        tunes box_weight, focal_gamma — they come through cfg.loss.*
        """
        return {
            "lr0":          self.cfg.training.lr,
            "weight_decay": self.cfg.training.weight_decay,
            "box":          self.cfg.loss.box_weight,
            "cls":          self.cfg.loss.cls_weight,
            "dfl":          self.cfg.loss.dfl_weight,
            "fl_gamma":     self.cfg.loss.focal_gamma,
            "hsv_h":        self.cfg.augmentation.standard.hsv_h,
            "hsv_s":        self.cfg.augmentation.standard.hsv_s,
            "hsv_v":        self.cfg.augmentation.standard.hsv_v,
            "mosaic":       self.cfg.augmentation.standard.mosaic_p,
            "fliplr":       self.cfg.augmentation.standard.horizontal_flip_p,
        }
