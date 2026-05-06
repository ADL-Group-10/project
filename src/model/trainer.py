"""
src/model/trainer.py  (FIXED)

Changes from original:
  - All values read from cfg — no hardcoding
  - cos_lr=True  → cosine LR scheduler (the drop curve TA wants)
  - lrf           → read from cfg.training.lrf         (add lrf: 0.01 to yaml)
  - dropout       → read from cfg.training.dropout     (add dropout: 0.1 to yaml)
  - trial_img_size→ read from cfg.tuning.trial_img_size (add trial_img_size: 320 to yaml)
  - results_dir   → read from cfg.paths.results_dir    (no hardcoded paths)
  - wandb         → removed _setup_wandb(), Ultralytics logs automatically
                    set env vars in notebook before training instead
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO, settings as ul_settings

from src.common_utils import load_config
from src.data import DataPipeline


def _configure_wandb(cfg) -> None:
    """
    Tell Ultralytics to log to the correct WandB project.
    No manual wandb.init() — Ultralytics handles it automatically.
    Call this once before model.train().
    """
    ul_settings.update({"wandb": True})
    os.environ["WANDB_PROJECT"] = str(cfg.logging.wandb_project)
    os.environ["WANDB_ENTITY"]  = str(cfg.logging.wandb_entity)
    print(f"[trainer] WandB → project={cfg.logging.wandb_project}, entity={cfg.logging.wandb_entity}")


class YOLOv9Trainer:

    def __init__(self, cfg) -> None:
        self.cfg    = cfg
        self.device = self._get_device()

        self.model = YOLO("yolov9c.pt")
        print(f"[trainer] YOLOv9 loaded on {self.device}")

        # Read from cfg — no hardcoded paths
        self.base_project = str(cfg.paths.results_dir)

        if Path(cfg.paths.yolo_output + "/dataset.yaml").exists():
            self.dataset_yaml = cfg.paths.yolo_output + "/dataset.yaml"
            print("[trainer] Using existing dataset.")
        else:
            pipeline = DataPipeline("config.yaml")
            dataset_path, _ = pipeline.run(augment="base")
            self.dataset_yaml = str((dataset_path / "dataset.yaml").resolve())

        self.out_dir = Path(self.base_project)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────

    def train(self, name="v1") -> None:
        """Full training run — call from notebook."""
        t  = self.cfg.training
        lo = self.cfg.loss

        # WandB — Ultralytics auto-logs, just set the destination
        os.environ["WANDB_NAME"] = name
        _configure_wandb(self.cfg)

        self.results = self.model.train(
            # --- DATA & HARDWARE ---
            data          = self.dataset_yaml,
            imgsz         = self.cfg.model.img_size,
            device        = self.device,
            workers       = t.num_workers,

            # --- TRAINING PARAMS ---
            epochs        = t.epochs,
            batch         = t.batch_size,
            optimizer     = t.optimizer.upper(),
            weight_decay  = t.weight_decay,
            warmup_epochs = t.warmup_epochs,
            patience      = t.early_stopping_patience,
            seed          = self.cfg.project.seed,

            # --- LR SCHEDULER (what TA wants to see) ---
            lr0           = t.lr,       # set to 0.004 in yaml
            lrf           = t.lrf,      # set to 0.01  in yaml → drops to lr0*lrf
            cos_lr        = True,       # cosine curve always on

            # --- REGULARIZATION (stops val loss going up) ---
            dropout       = t.dropout,  # set to 0.1 in yaml

            # --- LOSS WEIGHTS ---
            box           = lo.box_weight,
            cls           = lo.cls_weight,
            dfl           = lo.dfl_weight,

            # --- OUTPUT ---
            project       = self.base_project,
            name          = name,
            verbose       = True,
            exist_ok      = True,
        )
        print(f"[trainer] Done. Best model: {self.base_project}/{name}/weights/best.pt")

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

    def train_one_trial(self, trial_cfg: dict) -> float:
        """
        For Optuna tuning.
        All values from cfg or trial_cfg — nothing hardcoded.
        Uses cfg.tuning.trial_img_size (320) for fast trials.
        WandB disabled during trials to avoid run spam.
        """
        
        # Disable WandB during Optuna trials — too many runs
        ul_settings.update({"wandb": False})

        self.model = YOLO("yolov9c.pt")  # fresh model per trial
        self.model.train(
            # --- DATA & HARDWARE ---
            data          = self.dataset_yaml,
            imgsz         = self.cfg.tuning.trial_img_size,  # 320 from yaml
            device        = self.device,
            workers       = self.cfg.training.num_workers,

            # --- TUNED BY OPTUNA ---
            epochs        = self.cfg.tuning.trial_epochs,
            batch         = trial_cfg.get("batch_size",    self.cfg.training.batch_size),
            lr0           = trial_cfg.get("lr",            self.cfg.training.lr),
            warmup_epochs = trial_cfg.get("warmup_epochs", self.cfg.training.warmup_epochs),
            box           = trial_cfg.get("box_weight",    self.cfg.loss.box_weight),

            # --- LR SCHEDULER ---
            lrf           = self.cfg.training.lrf,
            cos_lr        = True,

            # --- REGULARIZATION ---
            dropout       = self.cfg.training.dropout,
            weight_decay  = self.cfg.training.weight_decay,

            # --- FIXED ---
            optimizer     = self.cfg.training.optimizer.upper(),
            cls           = self.cfg.loss.cls_weight,
            dfl           = self.cfg.loss.dfl_weight,

            # --- OUTPUT ---
            project       = str(Path(self.cfg.paths.results_dir).parent / "optuna_trials"),
            name          = f"trial_{trial_cfg.get('trial_number', 0)}",
            verbose       = False,
            exist_ok      = True,
        )
        
        metrics = self.validate()
        torch.cuda.empty_cache()
        del self.model
        return metrics["mAP50"]

    # ── Private ───────────────────────────────────────────────────

    def _get_device(self) -> str:
        import torch
        requested = self.cfg.project.device
        if "cuda" in requested and torch.cuda.is_available():
            return "0"
        return "cpu"


class YOLOv9TrainerSA(YOLOv9Trainer):
    """V2 — snow augmentation."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        pipeline = DataPipeline("config.yaml")
        dataset_path, _ = pipeline.run(augment="snow")
        self.dataset_yaml = str((dataset_path / "dataset.yaml").resolve())
        self.out_dir = Path(self.base_project) / "v2"
        print("[trainer_sa] V2 Snow-Augmented Trainer ready.")

    def train(self) -> None:
        super().train(name="v2")
        print(f"[trainer_sa] Done. Best model: {self.out_dir}/weights/best.pt")


class YOLOv9TrainerDS(YOLOv9Trainer):
    """V3 — Domain Shift Experiment."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        pipeline = DataPipeline("config.yaml")
        dataset_path, _ = pipeline.run(augment="snow", domain_shift=True)
        self.dataset_yaml = str((dataset_path / "dataset.yaml").resolve())
        self.out_dir = Path(self.base_project) / "v3_ds"
        print("[trainer_ds] V3 Domain-Shift Trainer Initialized.")

    def train(self) -> None:
        super().train(name="v3_ds")
        print(f"[trainer_ds] Done. Best model: {self.out_dir}/weights/best.pt")