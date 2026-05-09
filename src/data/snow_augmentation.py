"""Snow-aware augmentation transforms using Albumentations."""

import albumentations as A


class SnowAugmentation:
    """Builds snow-specific transforms from config.

    Usage:
        transforms = SnowAugmentation(config["augmentation"]["snow"]).get()
    """

    def __init__(self, snow_config) -> None:
        self.config = snow_config

    def get(self) -> list:
        transforms = []
        cfg = self.config

        # Geometric first — must run before color transforms so bboxes are
        # remapped against the original pixel grid.
        if cfg.get("perspective", {}).get("enabled", False):
            ps = cfg["perspective"]
            transforms.append(A.Perspective(
                scale=tuple(ps["scale"]),
                keep_size=True,
                p=ps["p"],
            ))

        if cfg.get("desaturation", {}).get("enabled", False):
            ds = cfg["desaturation"]
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=[int(ds["saturation_limit"][0] * 255),
                                 int(ds["saturation_limit"][1] * 255)],
                val_shift_limit=0,
                p=ds["p"],
            ))

        if cfg.get("blur", {}).get("enabled", False):
            bl = cfg["blur"]
            transforms.append(A.GaussianBlur(
                blur_limit=tuple(bl["blur_limit"]),
                p=bl["p"],
            ))

        if cfg.get("brightness_jitter", {}).get("enabled", False):
            bj = cfg["brightness_jitter"]
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=tuple(bj["brightness_limit"]),
                contrast_limit=0,
                p=bj["p"],
            ))

        if cfg.get("snow_overlay", {}).get("enabled", False):
            so = cfg["snow_overlay"]
            transforms.append(A.RandomSnow(
                snow_point_range=(so["snow_point_lower"], so["snow_point_upper"]),
                p=so["p"],
            ))

        # Runs AFTER snow_overlay so the synthetic flakes themselves streak —
        # simulates long-exposure capture during active snowfall.
        if cfg.get("motion_blur", {}).get("enabled", False):
            mb = cfg["motion_blur"]
            transforms.append(A.MotionBlur(
                blur_limit=tuple(mb["blur_limit"]),
                p=mb["p"],
            ))

        # Threshold-based partial inversion via A.Solarize: only pixels with
        # value > threshold get flipped to (255 - value). Lower threshold ->
        # more of the image inverts. High threshold (~200+) flips only the
        # brightest pixels (snow), leaving mid-tone structures (cars) intact.
        # threshold may be (low, high); Albumentations samples uniformly per
        # image. Off by default — see README for guidance.
        if cfg.get("invert", {}).get("enabled", False):
            inv = cfg["invert"]
            transforms.append(A.Solarize(
                threshold_range=tuple(t / 255.0 for t in inv["threshold"]),
                p=inv["p"],
            ))

        return transforms
