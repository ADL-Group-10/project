"""Snow-aware augmentation pipeline using Albumentations."""

import albumentations as A
import numpy as np


class SnowAugmentation:
    """Builds snow-specific transforms for V2 experiment.

    Usage:
        snow_aug = SnowAugmentation(config["augmentation"]["snow"])
        transforms = snow_aug.get()
    """

    def __init__(self, snow_config: dict) -> None:
        """Init with augmentation.snow section from config.yaml."""
        self.config = snow_config

    def get(self) -> list:
        # Build list of snow-specific transforms based on config, each with its own probability
        transforms = []
        cfg = self.config

        # Desaturation — simulates overcast winter lighting by reducing color saturation
        if cfg.get("desaturation", {}).get("enabled", False):
            ds = cfg["desaturation"]
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=[int(ds["saturation_limit"][0] * 255),
                                     int(ds["saturation_limit"][1] * 255)],
                    val_shift_limit=0,
                    p=ds["p"],
                )
            )

        # Gaussian blur — simulates lens-snow and motion blur from snow particles
        if cfg.get("blur", {}).get("enabled", False):
            bl = cfg["blur"]
            transforms.append(
                A.GaussianBlur(
                    blur_limit=tuple(bl["blur_limit"]),
                    p=bl["p"],
                )
            )

        # Brightness jitter — simulates short Nordic daylight with dimmer scenes
        if cfg.get("brightness_jitter", {}).get("enabled", False):
            bj = cfg["brightness_jitter"]
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=tuple(bj["brightness_limit"]),
                    contrast_limit=0,
                    p=bj["p"],
                )
            )

        # Snow overlay — adds white speckle patterns resembling falling/settled snow
        if cfg.get("snow_overlay", {}).get("enabled", False):
            so = cfg["snow_overlay"]
            transforms.append(
                A.RandomSnow(
                    snow_point_range=(so["snow_point_lower"], so["snow_point_upper"]),
                    p=so["p"],
                )
            )

        return transforms

    def preview(self, image: np.ndarray, n: int = 6) -> list[np.ndarray]:
        # Apply snow transforms n times to same image for visual comparison
        pipeline = A.Compose(self.get())
        return [pipeline(image=image)["image"] for _ in range(n)]