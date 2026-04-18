"""Data Pipeline — NVD to YOLO preparation."""

from pathlib import Path

import cv2
import numpy as np
import yaml
import albumentations as A

from .snow_augmentation import SnowAugmentation


class DataPipeline:
    """Single entry point for NVD dataset preparation.

    Usage:
        pipeline = DataPipeline("config.yaml")
        path, aug = pipeline.run(augment="none")    # resize + normalize only
        path, aug = pipeline.run(augment="base")    # base augmentation
        path, aug = pipeline.run(augment="snow")    # base + snow augmentation
    """

    def __init__(self, config_path: str) -> None:
        """Load config and prepare dataset if YOLO files don't exist yet."""
        self.config_path = Path(config_path)
        self.config: dict = {}
        self.raw_dir: Path = Path()
        self.output_dir: Path = Path()
        self.snow_aug: SnowAugmentation = None

        # TODO: Load config, resolve paths, init SnowAugmentation
        # TODO: if not self._exists(): self._setup()

    def run(self, augment: str = "none") -> tuple[Path, A.Compose]:
        """Return (dataset_path, augmentation_pipeline).

        Args:
            augment:
                "none" → resize + normalize only
                "base" → geometric, color jitter, resize, normalize
                "snow" → base + snow augmentation (via SnowAugmentation)

        Returns:
            dataset_path: dir with images/, labels/, dataset.yaml
            aug_pipeline: A.Compose with bbox_params for YOLO format
        """
        pass

    def summary(self) -> dict:
        """Print and return per-split stats: image counts, bbox counts, image sizes."""
        pass

    def show_samples(self, n: int = 5, split: str = "train", augment: str = "none") -> None:
        """Display n random images with bboxes. For base/snow, shows
        original vs augmented side-by-side. For notebook use."""
        pass

    def _setup(self) -> None:
        """Full preparation chain (runs once, skipped if data exists):
        1. Parse NVD annotations → YOLO .txt format
        2. Split images by video sequence per config
        3. Organize into images/{train,val,test}/ and labels/{train,val,test}/
        4. Write dataset.yaml (nc=1, names=['car'])
        5. Validate image↔label pairing and label format
        6. Log per-split stats
        """
        pass

    def _exists(self) -> bool:
        """True if output_dir has valid YOLO structure with files in each split."""
        pass

if __name__ == "__main__":
    pipeline = DataPipeline("config.yaml")
    path, aug = pipeline.run(augment="snow")
    pipeline.summary()
    pipeline.show_samples(n=3, split="train", augment="snow")
    print(f"Dataset ready at: {path}")