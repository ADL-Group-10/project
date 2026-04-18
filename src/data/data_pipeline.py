"""Data Pipeline — NVD to YOLO preparation with DALI acceleration."""

from pathlib import Path
import xml.etree.ElementTree as ET

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

    _FRAME_WIDTH = 1920
    _FRAME_HEIGHT = 1080

    def __init__(self, config_path: str) -> None:
        """Load config and prepare dataset if YOLO files don't exist yet."""
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["paths"]["nvd_root"])
        self.output_dir = Path(self.config["paths"]["yolo_output"])
        self.splits = self.config["paths"]["splits"]
        self.img_size = self.config["model"]["img_size"]
        self.num_classes = self.config["model"]["num_classes"]
        self.aug_config = self.config["augmentation"]

        if not self._exists():
            self._setup()

    # ── Public ─────────────────────────────────────────────────

    def run(self, augment: str = "none") -> tuple[Path, A.Compose]:
        # Build augmentation pipeline based on variant: none / base / snow
        if augment not in ("none", "base", "snow"):
            raise ValueError(f"augment must be 'none', 'base', or 'snow', got '{augment}'")

        transforms = []
        std = self.aug_config["standard"]

        # Base transforms (geometric + color) for base and snow variants
        if augment in ("base", "snow"):
            transforms += [
                A.HorizontalFlip(p=std["horizontal_flip_p"]),
                A.HueSaturationValue(
                    hue_shift_limit=int(std["hsv_h"] * 180),
                    sat_shift_limit=int(std["hsv_s"] * 255),
                    val_shift_limit=int(std["hsv_v"] * 255),
                    p=0.5,
                ),
            ]

        # Snow transforms on top of base
        if augment == "snow":
            snow_aug = SnowAugmentation(self.aug_config["snow"])
            transforms += snow_aug.get()

        # Always applied: resize + normalize
        transforms += [
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"], min_visibility=0.3,
            ),
        )

        return self.output_dir, pipeline

    def summary(self) -> dict:
        # Print and return per-split image count, bbox count, avg bboxes per image
        stats = {}
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            num_images = len(images)

            total_bboxes = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    total_bboxes += sum(1 for line in f if line.strip())

            avg_bboxes = total_bboxes / num_images if num_images > 0 else 0
            stats[split] = {
                "images": num_images,
                "annotations": total_bboxes,
                "avg_bboxes": round(avg_bboxes, 2),
            }
            print(f"{split:>5s}: {num_images:>6d} images | {total_bboxes:>6d} bboxes | {avg_bboxes:.2f} avg/img")

        return stats

    def show_samples(self, n: int = 5, split: str = "train", augment: str = "none") -> None:
        # Display n random images with bboxes, side-by-side with augmented version if augment != "none"
        import matplotlib.pyplot as plt

        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split

        images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        chosen = np.random.choice(images, size=min(n, len(images)), replace=False)

        _, aug_pipeline = self.run(augment=augment)
        cols = 2 if augment != "none" else 1
        fig, axes = plt.subplots(n, cols, figsize=(6 * cols, 4 * n))
        if n == 1:
            axes = np.array([axes])
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for i, img_path in enumerate(chosen):
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            bboxes, class_labels = self._read_yolo_label(lbl_path)

            axes[i, 0].imshow(self._draw_bboxes(image.copy(), bboxes))
            axes[i, 0].set_title(f"Original — {img_path.name}")
            axes[i, 0].axis("off")

            if cols == 2:
                result = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = (result["image"] * 255).astype(np.uint8)
                axes[i, 1].imshow(self._draw_bboxes(aug_img, result["bboxes"]))
                axes[i, 1].set_title(f"Augmented ({augment})")
                axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

    # ── Private ────────────────────────────────────────────────

    def _setup(self) -> None:
        # One-time setup: parse XML, extract frames, organize splits, validate
        print("Setting up YOLO dataset from NVD...")

        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for split, sequences in self.splits.items():
            for seq_name in sequences:
                print(f"  Processing {seq_name} → {split}")
                self._process_sequence(seq_name, split)

        # Write dataset.yaml inline (only used here)
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({
                "path": str(self.output_dir.resolve()),
                "train": "images/train", "val": "images/val", "test": "images/test",
                "nc": self.num_classes, "names": ["car"],
            }, f, default_flow_style=False)
        print(f"  Wrote {yaml_path}")

        self._validate()
        self.summary()
        print("Setup complete.")

    def _process_sequence(self, seq_name: str, split: str) -> None:
        # Parse annotations for one sequence, then extract frames (.mp4) or copy (.png)
        seq_dir = self._find_path(seq_name, is_dir=True)
        xml_path = self._find_path(seq_name, is_dir=False, suffix=".xml")
        frame_annotations = self._parse_cvat_xml(ET.parse(xml_path).getroot())

        mp4_files = list(seq_dir.glob("*.mp4"))
        png_files = sorted(seq_dir.glob("*.png"))

        if mp4_files:
            self._extract_frames(mp4_files[0], frame_annotations, split, seq_name)
        elif png_files:
            self._copy_png_frames(png_files, frame_annotations, split, seq_name)
        else:
            raise FileNotFoundError(f"No .mp4 or .png files found in {seq_dir}")

    def _parse_cvat_xml(self, root: ET.Element) -> dict[int, list[list[float]]]:
        # Convert CVAT tracks (xtl/ytl/xbr/ybr pixels) to per-frame YOLO labels (normalized center+size)
        frame_annotations: dict[int, list[list[float]]] = {}

        for track in root.findall(".//track"):
            if track.get("label") != "car":
                continue
            for box in track.findall("box"):
                if box.get("outside") == "1":
                    continue

                frame_num = int(box.get("frame"))
                xtl = max(0.0, min(float(box.get("xtl")), self._FRAME_WIDTH))
                ytl = max(0.0, min(float(box.get("ytl")), self._FRAME_HEIGHT))
                xbr = max(0.0, min(float(box.get("xbr")), self._FRAME_WIDTH))
                ybr = max(0.0, min(float(box.get("ybr")), self._FRAME_HEIGHT))

                w, h = xbr - xtl, ybr - ytl
                if w <= 0 or h <= 0:
                    continue

                x_c = (xtl + w / 2) / self._FRAME_WIDTH
                y_c = (ytl + h / 2) / self._FRAME_HEIGHT

                if frame_num not in frame_annotations:
                    frame_annotations[frame_num] = []
                frame_annotations[frame_num].append(
                    [0, x_c, y_c, w / self._FRAME_WIDTH, h / self._FRAME_HEIGHT]
                )

        return frame_annotations

    def _extract_frames(self, mp4_path, frame_annotations, split, seq_name) -> None:
        # Extract annotated frames from .mp4. Try DALI (GPU), fall back to OpenCV (CPU)
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        annotated_set = set(frame_annotations.keys())

        try:
            from nvidia.dali import pipeline_def, fn

            @pipeline_def(batch_size=1, num_threads=4, device_id=0)
            def video_pipe():
                return fn.readers.video(
                    filenames=[str(mp4_path)], sequence_length=1,
                    device="gpu", name="reader",
                )

            pipe = video_pipe()
            pipe.build()
            frame_idx = 0
            try:
                while True:
                    (output,) = pipe.run()
                    if frame_idx in annotated_set:
                        frame = output.as_cpu().at(0)[0]
                        name = f"{seq_name}_frame_{frame_idx:06d}"
                        cv2.imwrite(str(img_dir / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        self._write_yolo_label(lbl_dir / f"{name}.txt", frame_annotations[frame_idx])
                    frame_idx += 1
            except StopIteration:
                pass
            print(f"    DALI: extracted {len(annotated_set)} frames from {mp4_path.name}")

        except (ImportError, RuntimeError) as e:
            print(f"    DALI unavailable ({e}), falling back to OpenCV")
            cap = cv2.VideoCapture(str(mp4_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {mp4_path}")

            frame_idx, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in annotated_set:
                    name = f"{seq_name}_frame_{frame_idx:06d}"
                    cv2.imwrite(str(img_dir / f"{name}.png"), frame)
                    self._write_yolo_label(lbl_dir / f"{name}.txt", frame_annotations[frame_idx])
                    saved += 1
                frame_idx += 1
            cap.release()
            print(f"    OpenCV: extracted {saved} frames from {mp4_path.name}")

    def _copy_png_frames(self, png_files, frame_annotations, split, seq_name) -> None:
        # Copy pre-extracted .png frames that have annotations, write YOLO labels
        import shutil
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        saved = 0

        for frame_idx, png_path in enumerate(png_files):
            if frame_idx in frame_annotations:
                name = f"{seq_name}_frame_{frame_idx:06d}"
                shutil.copy2(png_path, img_dir / f"{name}.png")
                self._write_yolo_label(lbl_dir / f"{name}.txt", frame_annotations[frame_idx])
                saved += 1

        print(f"    Copied {saved} annotated frames from {seq_name}")

    def _find_path(self, seq_name: str, is_dir: bool = True, suffix: str = "") -> Path:
        # Locate a file or directory by name, handling space/underscore differences
        candidates = [
            seq_name,
            seq_name.replace(" ", "_"),
        ]

        for name in candidates:
            path = self.raw_dir / (name + suffix) if not is_dir else self.raw_dir / name
            if path.exists():
                return path

        # Fuzzy fallback — normalize both sides
        norm_seq = seq_name.lower().replace(" ", "_")
        for item in self.raw_dir.iterdir():
            if is_dir and not item.is_dir():
                continue
            if not is_dir and item.suffix != suffix:
                continue
            if item.stem.lower().replace(" ", "_") == norm_seq:
                return item

        kind = "directory" if is_dir else f"'{suffix}' file"
        raise FileNotFoundError(f"No {kind} found for '{seq_name}' in {self.raw_dir}")

    def _write_yolo_label(self, path: Path, annotations: list[list[float]]) -> None:
        # Write one YOLO label file: class_id x_center y_center width height per line
        with open(path, "w") as f:
            for ann in annotations:
                f.write(" ".join(f"{v:.6f}" if i > 0 else str(int(v))
                                for i, v in enumerate(ann)) + "\n")

    def _validate(self) -> None:
        # Check every image has a matching label, every label has valid YOLO format
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            images = {p.stem for p in img_dir.glob("*.png")} | {p.stem for p in img_dir.glob("*.jpg")}
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

        print("  Validation passed.")

    def _exists(self) -> bool:
        # Check if processed YOLO data already exists with all splits populated
        for sub in ("images/train", "images/val", "images/test",
                     "labels/train", "labels/val", "labels/test"):
            d = self.output_dir / sub
            if not d.exists() or not any(d.iterdir()):
                return False
        return (self.output_dir / "dataset.yaml").exists()

    @staticmethod
    def _read_yolo_label(path: Path) -> tuple[list, list]:
        # Read YOLO label file, return (bboxes, class_labels)
        bboxes, class_labels = [], []
        if path.exists():
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(v) for v in parts[1:]])
        return bboxes, class_labels

    @staticmethod
    def _draw_bboxes(image: np.ndarray, bboxes: list) -> np.ndarray:
        # Draw YOLO-format bboxes on image for visualization
        h, w = image.shape[:2]
        for bbox in bboxes:
            x_c, y_c, bw, bh = bbox
            x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
            x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image