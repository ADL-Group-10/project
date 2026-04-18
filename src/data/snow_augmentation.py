"""Snow-aware augmentation pipeline using Albumentations."""

import albumentations as A
import numpy as np


class SnowAugmentation:
    """Builds snow-specific transforms for V2 experiment.

    Usage:
        snow_aug = SnowAugmentation(config["snow_augmentation"])
        snow_transforms = snow_aug.get()
    """

    def __init__(self, snow_config: dict) -> None:
        """Init with snow_augmentation section from config.yaml."""
        self.config = snow_config

    def get(self) -> list:
        """Return list of snow-specific A.* transforms.

        Transforms (each with its own probability from config):
            - Desaturation (overcast winter lighting)
            - Gaussian blur (lens-snow / motion blur)
            - Brightness jitter (short Nordic daylight)
            - RandomSnow, RandomFog (optional overlays)

        Returns raw list — caller wraps in A.Compose with bbox_params.
        """
        pass

    def preview(self, image: np.ndarray, n: int = 6) -> list[np.ndarray]:
        """Apply snow transforms n times to same image. Returns list of augmented images."""
        pass