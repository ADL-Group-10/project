"""
src/model/snow_augmented_yolov9/trainer_sa.py

V2 Snow-Augmented Trainer — identical to V1 trainer but uses snow augmentation.


The ONLY difference from trainer.py:
  - augment="snow"  instead of augment="base"
  - variant="v2"    instead of variant="v1"
  - saves to outputs/checkpoints/v2/

Usage (notebook):
    from src.model.snow_augmented_yolov9.trainer_sa import YOLOv9TrainerSA
    from src.common_utils import load_config

    cfg     = load_config(variant="v2")
    trainer = YOLOv9TrainerSA(cfg)
    trainer.train()
"""

from pathlib import Path
from src.data import DataPipeline

# Import the base trainer and subclass it — only override what changes
from src.model.baseline_yolov9.trainer import YOLOv9Trainer


class YOLOv9TrainerSA(YOLOv9Trainer):
    """
    V2 trainer — snow-aware augmentation on top of V1 baseline.

    Inherits everything from YOLOv9Trainer.
    The only difference: data is loaded with augment="snow".

    Args:
        cfg: load_config(variant="v2") — epochs=120, lr=0.0008, use_snow_aug=true
    """

    def __init__(self, cfg) -> None:
        # Call parent __init__ which sets up model, optimizer, scheduler
        # Parent will call DataPipeline with augment="base" — we then override it
        super().__init__(cfg)

        # Override: replace base augmentation with snow augmentation
        pipeline            = DataPipeline(str(Path("config.yaml")))
        dataset_path, augs  = pipeline.run(augment="snow")   # ← only change
        self.dataset_yaml   = str(dataset_path / "dataset.yaml")
        self.aug_pipeline   = augs

        print("[trainer_sa] V2 Snow-Augmented Trainer ready.")
        print("[trainer_sa] Snow transforms: desaturation, blur, brightness jitter, snow overlay.")
