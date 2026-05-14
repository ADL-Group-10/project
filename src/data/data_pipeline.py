"""Data Pipeline — NVD to YOLO preparation with decord frame extraction."""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml
import albumentations as A

from src.common_utils import load_config, get_logger, set_seed_from_config, get_paths
from .snow_augmentation import SnowAugmentation


class DataPipeline:
    """Single entry point for NVD dataset preparation.

    Usage:
        pipeline = DataPipeline()                                # variant from config.yaml
        pipeline = DataPipeline(variant="v0")                    # explicit variant
        path, aug = pipeline.run()                               # augment derived from variant
        path, aug = pipeline.run(augment="snow")                 # explicit override
        path, augs = pipeline.run(domain_shift=True)             # per-split pipelines
    """

    def __init__(self, config_path: str = "config.yaml", variant: str | None = None) -> None:
        self.config = load_config(config_path, variant=variant)
        self.logger = get_logger(__name__, self.config)
        set_seed_from_config(self.config)

        self.raw_dir = get_paths(self.config, create_dirs=False).nvd_root
        self.output_dir = Path(self.config.paths.yolo_output)
        self.splits = self.config.paths.splits
        self.img_size = self.config.model.img_size
        self.num_classes = self.config.model.num_classes
        self.aug_config = self.config.augmentation

        if not self._exists():
            self._setup()

    # ── Public ─────────────────────────────────────────────────

    def run(self, augment: str | None = None, domain_shift: bool | None = None):
        """Build augmentation pipeline(s).

        Args:
            augment:      "none" | "base" | "snow". If None, derived from the active variant
                          in config (v0 → "none", v1 → "base", v2/v3_ds → "snow").
            domain_shift: If None, derived from cfg.domain_shift.enabled.

        Returns:
            (output_dir, pipeline) or (output_dir, {"train": ..., "val": ..., "test": ...})
        """
        if augment is None:
            augment = self._augment_from_cfg()
        if domain_shift is None:
            domain_shift = bool(getattr(self.config.domain_shift, "enabled", False))

        if augment not in ("none", "base", "snow"):
            raise ValueError(f"augment must be 'none', 'base', or 'snow', got '{augment}'")

        if domain_shift:
            ds_cfg = self.config.domain_shift
            return self.output_dir, {
                "train": self._build_pipeline(augment, snow_config=ds_cfg.light_snow),
                "val":   self._build_pipeline(augment, snow_config=ds_cfg.light_snow),
                "test":  self._build_pipeline(augment, snow_config=ds_cfg.heavy_snow),
            }

        return self.output_dir, self._build_pipeline(augment)

    def summary(self) -> dict:
        stats = {}
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            num_images = len(list(img_dir.glob("*.png")))
            total_bboxes = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    total_bboxes += sum(1 for line in f if line.strip())

            avg_bboxes = total_bboxes / num_images if num_images > 0 else 0
            stats[split] = {"images": num_images, "annotations": total_bboxes, "avg_bboxes": round(avg_bboxes, 2)}
            self.logger.info(f"{split:>5s}: {num_images:>6d} images | {total_bboxes:>6d} bboxes | {avg_bboxes:.2f} avg/img")

        return stats

    # ── Private ────────────────────────────────────────────────

    def _augment_from_cfg(self) -> str:
        """Map the active variant's augmentation block to a mode string."""
        if bool(getattr(self.aug_config, "disable", False)):
            return "none"
        if bool(self.aug_config.use_snow_aug):
            return "snow"
        return "base"

    def _build_pipeline(self, augment: str, snow_config=None) -> A.Compose:
        transforms = []
        std = self.aug_config["standard"]

        if augment in ("base", "snow"):
            transforms += [
                A.HorizontalFlip(p=std["horizontal_flip_p"]),
                A.HueSaturationValue(
                    hue_shift_limit=int(std["hsv_h"] * 180),
                    sat_shift_limit=int(std["hsv_s"] * 50),
                    val_shift_limit=int(std["hsv_v"] * 50),
                    p=0.5,
                ),
            ]

        if augment == "snow":
            cfg = snow_config if snow_config else self.aug_config["snow"]
            transforms += SnowAugmentation(cfg).get()

        transforms += [
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )

    def _setup(self) -> None:
        self.logger.info("Setting up YOLO dataset from NVD...")

        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for split, sequences in self.splits.items():
            for seq_name in sequences:
                self.logger.info(f"Processing {seq_name} -> {split}")
                self._process_sequence(seq_name, split)

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({
                "path": str(self.output_dir.resolve()),
                "train": "images/train", "val": "images/val", "test": "images/test",
                "nc": self.num_classes, "names": ["car"],
            }, f, default_flow_style=False)
        self.logger.info(f"Wrote {yaml_path}")

        self._validate()
        self.summary()
        self.logger.info("Setup complete.")

    def _process_sequence(self, seq_name: str, split: str) -> None:
        seq_dir = self._find_dir(seq_name)
        xml_path = self._find_xml(seq_dir)
        frame_annotations = self._parse_cvat_xml(ET.parse(xml_path).getroot())

        mp4_files = [f for f in seq_dir.iterdir() if f.suffix.lower() == ".mp4"]
        png_files = sorted(f for f in seq_dir.iterdir() if f.suffix.lower() == ".png")

        if not mp4_files and not png_files:
            for child in seq_dir.iterdir():
                if child.is_dir():
                    png_files = sorted(f for f in child.iterdir() if f.suffix.lower() == ".png")
                    if png_files:
                        break

        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split

        if mp4_files:
            self._extract_frames(mp4_files[0], frame_annotations, img_dir, lbl_dir, seq_name)
        elif png_files:
            self._copy_png_frames(png_files, frame_annotations, img_dir, lbl_dir, seq_name)
        else:
            raise FileNotFoundError(f"No .mp4 or .png files found in {seq_dir}")

    def _parse_cvat_xml(self, root: ET.Element) -> dict[int, list[list[float]]]:
        """Parse CVAT 1.1 XML, handling rotated bounding boxes.

        Rotates all 4 corners and takes the axis-aligned envelope —
        matching the official NVD repo's AnnotationBox.GetStraightBoundingBox().
        """
        frame_annotations: dict[int, list[list[float]]] = {}

        width_el = root.find("meta//original_size/width")
        height_el = root.find("meta//original_size/height")
        if width_el is None or height_el is None:
            raise ValueError("XML missing <meta><original_size>")
        frame_w = int(width_el.text)
        frame_h = int(height_el.text)

        for track in root.findall(".//track"):
            if track.get("label") != "car":
                continue
            for box in track.findall("box"):
                if box.get("outside") == "1":
                    continue

                frame_num = int(box.get("frame"))
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                rotation = float(box.get("rotation", "0"))

                cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
                corners = np.array([[xtl, ytl], [xbr, ytl],
                                    [xbr, ybr], [xtl, ybr]])

                if abs(rotation) > 0.01:
                    rad = rotation * np.pi / 180
                    cos_r, sin_r = np.cos(rad), np.sin(rad)
                    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                    corners = (rot_mat @ (corners - [cx, cy]).T).T + [cx, cy]

                x_min = max(0.0, min(float(corners[:, 0].min()), frame_w))
                y_min = max(0.0, min(float(corners[:, 1].min()), frame_h))
                x_max = max(0.0, min(float(corners[:, 0].max()), frame_w))
                y_max = max(0.0, min(float(corners[:, 1].max()), frame_h))

                w, h = x_max - x_min, y_max - y_min
                if w <= 0 or h <= 0:
                    continue

                x_c = (x_min + w / 2) / frame_w
                y_c = (y_min + h / 2) / frame_h

                if frame_num not in frame_annotations:
                    frame_annotations[frame_num] = []
                frame_annotations[frame_num].append(
                    [0, x_c, y_c, w / frame_w, h / frame_h]
                )

        return frame_annotations

    def _extract_frames(self, mp4_path, frame_annotations, img_dir, lbl_dir, seq_name) -> None:
        import decord
        decord.bridge.set_bridge("native")

        annotated_indices = sorted(frame_annotations.keys())
        vr = decord.VideoReader(str(mp4_path), ctx=decord.cpu(0))
        self.logger.info(f"decord: extracting {len(annotated_indices)} frames from {mp4_path.name}")

        for frame_idx in annotated_indices:
            if frame_idx >= len(vr):
                self.logger.warning(f"Frame {frame_idx} out of range ({len(vr)} total), skipping")
                continue
            frame = vr[frame_idx].asnumpy()
            name = f"{seq_name}_frame_{frame_idx:06d}"
            cv2.imwrite(str(img_dir / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._write_yolo_label(lbl_dir / f"{name}.txt", frame_annotations[frame_idx])

        self.logger.info(f"decord: done -- {len(annotated_indices)} frames from {mp4_path.name}")

    def _copy_png_frames(self, png_files, frame_annotations, img_dir, lbl_dir, seq_name) -> None:
        import shutil
        saved = 0
        for frame_idx, png_path in enumerate(png_files):
            if frame_idx in frame_annotations:
                name = f"{seq_name}_frame_{frame_idx:06d}"
                shutil.copy2(png_path, img_dir / f"{name}.png")
                self._write_yolo_label(lbl_dir / f"{name}.txt", frame_annotations[frame_idx])
                saved += 1
        self.logger.info(f"Copied {saved} annotated frames from {seq_name}")

    def _find_dir(self, seq_name: str) -> Path:
        for name in [seq_name, seq_name.replace(" ", "_")]:
            path = self.raw_dir / name
            if path.exists():
                return path
        norm = seq_name.lower().replace(" ", "_")
        for item in self.raw_dir.iterdir():
            if item.is_dir() and item.stem.lower().replace(" ", "_") == norm:
                return item
        raise FileNotFoundError(f"No directory found for '{seq_name}' in {self.raw_dir}")

    def _find_xml(self, seq_dir: Path) -> Path:
        xml_files = list(seq_dir.glob("*.xml"))
        if xml_files:
            return xml_files[0]
        raise FileNotFoundError(f"No .xml file found in {seq_dir}")

    def _write_yolo_label(self, path: Path, annotations: list[list[float]]) -> None:
        with open(path, "w") as f:
            for ann in annotations:
                cls_id = int(ann[0])
                coords = " ".join(f"{v:.6f}" for v in ann[1:])
                f.write(f"{cls_id} {coords}\n")

    def _validate(self) -> None:
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            images = {p.stem for p in img_dir.glob("*.png")}
            labels = {p.stem for p in lbl_dir.glob("*.txt")}

            if images - labels:
                raise ValueError(f"{split}: {len(images - labels)} images have no label")
            if labels - images:
                raise ValueError(f"{split}: {len(labels - images)} labels have no image")
            if not images:
                raise ValueError(f"{split}: split is empty")

            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    for ln, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            raise ValueError(f"{lbl_file.name}:{ln}: expected 5 values, got {len(parts)}")
                        if int(parts[0]) >= self.num_classes:
                            raise ValueError(f"{lbl_file.name}:{ln}: invalid class_id {parts[0]}")
                        if not all(0.0 <= float(v) <= 1.0 for v in parts[1:]):
                            raise ValueError(f"{lbl_file.name}:{ln}: values out of [0,1]")
        self.logger.info("Validation passed.")

    def _exists(self) -> bool:
        for sub in ("images/train", "images/val", "images/test",
                     "labels/train", "labels/val", "labels/test"):
            d = self.output_dir / sub
            if not d.exists() or not any(d.iterdir()):
                return False
        return (self.output_dir / "dataset.yaml").exists()

    @staticmethod
    def _read_yolo_label(path: Path) -> tuple[list, list]:
        bboxes, class_labels = [], []
        if path.exists():
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(v) for v in parts[1:]])
        return bboxes, class_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .visualizer import AnnotationVisualizer as AV

    pipeline = DataPipeline()
    pipeline.summary()

    img_dir = pipeline.output_dir / "images" / "train"
    lbl_dir = pipeline.output_dir / "labels" / "train"
    sample_path = list(img_dir.glob("*.png"))[0]

    samples_dir = pipeline.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Try to get CVAT annotations for the raw image (dual boxes)
    raw_annotations = None
    stem = sample_path.stem
    stem_parts = stem.rsplit("_frame_", 1)
    if len(stem_parts) == 2:
        try:
            seq_dir = pipeline._find_dir(stem_parts[0])
            xml_path = pipeline._find_xml(seq_dir)
            raw_annotations, _, _ = AV.parse_cvat_frame(
                str(xml_path), int(stem_parts[1]))
        except FileNotFoundError:
            pass

    image = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    bboxes, class_labels = DataPipeline._read_yolo_label(
        lbl_dir / (sample_path.stem + ".txt"))

    def apply_and_visualize(image, bboxes, class_labels, aug):
        result = aug(image=image, bboxes=bboxes, class_labels=class_labels)
        out_img = ((result["image"] * std + mean) * 255).clip(0, 255).astype(np.uint8)
        return AV.visualize(out_img, result["bboxes"])

    def wrap_single(transform):
        return A.Compose(
            [transform,
             A.Resize(height=pipeline.img_size, width=pipeline.img_size),
             A.Normalize(mean=mean.tolist(), std=std.tolist())],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                     min_visibility=0.3),
        )

    std_cfg  = pipeline.aug_config["standard"]
    snow_cfg = pipeline.aug_config["snow"]

    singles = {
        "flip":         A.HorizontalFlip(p=1.0),
        "hsv":          A.HueSaturationValue(
                            hue_shift_limit=int(std_cfg["hsv_h"] * 180),
                            sat_shift_limit=int(std_cfg["hsv_s"] * 255),
                            val_shift_limit=int(std_cfg["hsv_v"] * 255), p=1.0),
        "perspective":  A.Perspective(scale=tuple(snow_cfg["perspective"]["scale"]),
                                       keep_size=True, p=1.0),
        "desaturate":   A.HueSaturationValue(
                            hue_shift_limit=0,
                            sat_shift_limit=[int(snow_cfg["desaturation"]["saturation_limit"][0] * 255),
                                             int(snow_cfg["desaturation"]["saturation_limit"][1] * 255)],
                            val_shift_limit=0, p=1.0),
        "blur":         A.GaussianBlur(blur_limit=tuple(snow_cfg["blur"]["blur_limit"]), p=1.0),
        "brightness":   A.RandomBrightnessContrast(
                            brightness_limit=tuple(snow_cfg["brightness_jitter"]["brightness_limit"]),
                            contrast_limit=0, p=1.0),
        "snow_overlay": A.RandomSnow(
                            snow_point_range=(snow_cfg["snow_overlay"]["snow_point_lower"],
                                              snow_cfg["snow_overlay"]["snow_point_upper"]), p=1.0),
        "motion_blur":  A.MotionBlur(blur_limit=tuple(snow_cfg["motion_blur"]["blur_limit"]), p=1.0),
        "invert":       A.Solarize(threshold_range=tuple(t / 255.0 for t in snow_cfg["invert"]["threshold"]), p=1.0),
    }
    single_pipelines = {name: wrap_single(t) for name, t in singles.items()}

    _, base_aug = pipeline.run(augment="base")
    _, snow_aug = pipeline.run(augment="snow")
    _, ds_augs  = pipeline.run(augment="snow", domain_shift=True)

    ds_cfg = pipeline.config.domain_shift

    # Raw cell: use CVAT annotations (dual boxes) if available, else YOLO boxes
    if raw_annotations:
        raw_vis = AV.visualize(image.copy(), raw_annotations)
    else:
        raw_vis = AV.visualize(image.copy(), bboxes)

    variant_cells = {
        "raw": raw_vis,
        "V1 base": apply_and_visualize(image, bboxes, class_labels, base_aug),
        "V2 full snow": apply_and_visualize(image, bboxes, class_labels, snow_aug),
        "V3 train (light)": apply_and_visualize(image, bboxes, class_labels, ds_augs["train"]),
        "V3 test (heavy)": apply_and_visualize(image, bboxes, class_labels, ds_augs["test"]),
    }

    # Image 1: variants — raw on top, V1/V2 in middle, V3 at bottom
    fig1 = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = fig1.add_gridspec(3, 2, hspace=0.3)

    ax_raw = fig1.add_subplot(gs[0, :])
    ax_raw.imshow(variant_cells["raw"])
    ax_raw.set_title("raw", fontsize=10, fontweight="bold")
    ax_raw.set_xticks([]); ax_raw.set_yticks([])

    for col, (label, key) in enumerate([("V1 base", "V1 base"), ("V2 full snow", "V2 full snow")]):
        ax = fig1.add_subplot(gs[1, col])
        ax.imshow(variant_cells[key])
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    for col, (label, key) in enumerate([("V3 train (light)", "V3 train (light)"), ("V3 test (heavy)", "V3 test (heavy)")]):
        ax = fig1.add_subplot(gs[2, col])
        ax.imshow(variant_cells[key])
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig1.suptitle(f"Variants: raw, V1, V2, V3\nsample: {sample_path.stem}", fontsize=12, fontweight="bold")
    AV.save_figure(fig1, samples_dir / "comparison_variants.png")
    print(f"Saved -> {samples_dir / 'comparison_variants.png'}")

    # Image 2: individual transforms (each at p=1.0)
    individual_cells = [
        (name, apply_and_visualize(image, bboxes, class_labels, p))
        for name, p in single_pipelines.items()]

    fig2 = AV.comparison_grid(
        individual_cells,
        suptitle=f"Individual transforms (p=1.0)\nsample: {sample_path.stem}",
        cols=5, cell_size=(4, 3),
    )
    AV.save_figure(fig2, samples_dir / "comparison_transforms.png")
    print(f"Saved -> {samples_dir / 'comparison_transforms.png'}")
