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

    # ── Public API ─────────────────────────────────────────────

    def run(self, augment: str = "none") -> tuple[Path, A.Compose]:
        """Return (dataset_path, augmentation_pipeline).

        Args:
            augment:
                "none" → resize + normalize only
                "base" → standard augmentation from config
                "snow" → base + snow augmentation from config
        """
        if augment not in ("none", "base", "snow"):
            raise ValueError(f"augment must be 'none', 'base', or 'snow', got '{augment}'")

        transforms = []
        std = self.aug_config["standard"]

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

        if augment == "snow":
            snow_aug = SnowAugmentation(self.aug_config["snow"])
            transforms += snow_aug.get()

        transforms += [
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,
            ),
        )

        return self.output_dir, pipeline

    def summary(self) -> dict:
        """Print and return per-split stats: image counts, bbox counts."""
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
        """Display n random images with bboxes. For base/snow, shows
        original vs augmented side-by-side. For notebook use."""
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

            orig_display = self._draw_bboxes(image.copy(), bboxes)
            axes[i, 0].imshow(orig_display)
            axes[i, 0].set_title(f"Original — {img_path.name}")
            axes[i, 0].axis("off")

            if cols == 2:
                result = aug_pipeline(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
                aug_display = self._draw_bboxes(
                    (result["image"] * 255).astype(np.uint8),
                    result["bboxes"],
                )
                axes[i, 1].imshow(aug_display)
                axes[i, 1].set_title(f"Augmented ({augment})")
                axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

    # ── Private — Setup Chain ──────────────────────────────────

    def _setup(self) -> None:
        """Full preparation chain (runs once, skipped if data exists):
        1. Extract frames from .mp4 videos (skip for .png sequences)
        2. Parse CVAT XML annotations → YOLO .txt labels
        3. Organize into images/{train,val,test}/ and labels/{train,val,test}/
        4. Write dataset.yaml
        5. Validate image↔label pairing
        6. Print stats
        """
        print("Setting up YOLO dataset from NVD...")

        for split in ("train", "val", "test"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for split, sequences in self.splits.items():
            for seq_name in sequences:
                print(f"  Processing {seq_name} → {split}")
                self._process_sequence(seq_name, split)

        self._write_dataset_yaml()
        self._validate()
        self.summary()
        print("Setup complete.")

    def _process_sequence(self, seq_name: str, split: str) -> None:
        """Extract frames + parse annotations for one video sequence."""
        seq_dir = self._find_seq_dir(seq_name)
        xml_path = self._find_xml(seq_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        frame_annotations = self._parse_cvat_xml(root)

        mp4_files = list(seq_dir.glob("*.mp4"))
        png_files = sorted(seq_dir.glob("*.png"))

        if mp4_files:
            self._extract_and_save_frames(
                mp4_files[0], frame_annotations, split, seq_name
            )
        elif png_files:
            self._copy_png_frames(
                png_files, frame_annotations, split, seq_name
            )
        else:
            raise FileNotFoundError(
                f"No .mp4 or .png files found in {seq_dir}"
            )

    def _parse_cvat_xml(self, root: ET.Element) -> dict[int, list[list[float]]]:
        """Parse CVAT 1.1 XML into per-frame YOLO bounding boxes.

        Converts xtl/ytl/xbr/ybr (absolute pixels) to
        class_id x_center y_center width height (normalized 0-1).
        Skips boxes marked outside="1".
        """
        frame_annotations: dict[int, list[list[float]]] = {}

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

                # Clamp to frame boundaries
                xtl = max(0.0, min(xtl, self._FRAME_WIDTH))
                ytl = max(0.0, min(ytl, self._FRAME_HEIGHT))
                xbr = max(0.0, min(xbr, self._FRAME_WIDTH))
                ybr = max(0.0, min(ybr, self._FRAME_HEIGHT))

                w = xbr - xtl
                h = ybr - ytl
                if w <= 0 or h <= 0:
                    continue

                x_center = (xtl + w / 2) / self._FRAME_WIDTH
                y_center = (ytl + h / 2) / self._FRAME_HEIGHT
                norm_w = w / self._FRAME_WIDTH
                norm_h = h / self._FRAME_HEIGHT

                if frame_num not in frame_annotations:
                    frame_annotations[frame_num] = []

                frame_annotations[frame_num].append(
                    [0, x_center, y_center, norm_w, norm_h]
                )

        return frame_annotations

    def _extract_and_save_frames(
        self,
        mp4_path: Path,
        frame_annotations: dict[int, list[list[float]]],
        split: str,
        seq_name: str,
    ) -> None:
        """Extract annotated frames from .mp4. Tries DALI first, falls back to OpenCV."""
        try:
            self._extract_with_dali(mp4_path, frame_annotations, split, seq_name)
        except (ImportError, RuntimeError) as e:
            print(f"    DALI unavailable ({e}), falling back to OpenCV")
            self._extract_with_opencv(mp4_path, frame_annotations, split, seq_name)

    def _extract_with_dali(
        self,
        mp4_path: Path,
        frame_annotations: dict[int, list[list[float]]],
        split: str,
        seq_name: str,
    ) -> None:
        """GPU-accelerated frame extraction using NVIDIA DALI."""
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali as dali

        annotated_frames = sorted(frame_annotations.keys())
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split

        @pipeline_def(batch_size=1, num_threads=4, device_id=0)
        def video_pipeline():
            video = fn.readers.video(
                filenames=[str(mp4_path)],
                sequence_length=1,
                device="gpu",
                name="reader",
            )
            return video

        pipe = video_pipeline()
        pipe.build()

        frame_idx = 0
        annotated_set = set(annotated_frames)

        try:
            while True:
                (output,) = pipe.run()
                if frame_idx in annotated_set:
                    frame = output.as_cpu().at(0)[0]
                    frame_name = f"{seq_name}_frame_{frame_idx:06d}"

                    cv2.imwrite(
                        str(img_dir / f"{frame_name}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )
                    self._write_yolo_label(
                        lbl_dir / f"{frame_name}.txt",
                        frame_annotations[frame_idx],
                    )

                frame_idx += 1
        except StopIteration:
            pass

        print(f"    DALI: extracted {len(annotated_frames)} frames from {mp4_path.name}")

    def _extract_with_opencv(
        self,
        mp4_path: Path,
        frame_annotations: dict[int, list[list[float]]],
        split: str,
        seq_name: str,
    ) -> None:
        """CPU fallback for frame extraction using OpenCV."""
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        annotated_set = set(frame_annotations.keys())
        saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in annotated_set:
                frame_name = f"{seq_name}_frame_{frame_idx:06d}"

                cv2.imwrite(str(img_dir / f"{frame_name}.png"), frame)
                self._write_yolo_label(
                    lbl_dir / f"{frame_name}.txt",
                    frame_annotations[frame_idx],
                )
                saved += 1

            frame_idx += 1

        cap.release()
        print(f"    OpenCV: extracted {saved} frames from {mp4_path.name}")

    def _copy_png_frames(
        self,
        png_files: list[Path],
        frame_annotations: dict[int, list[list[float]]],
        split: str,
        seq_name: str,
    ) -> None:
        """Copy pre-extracted .png frames and create YOLO labels."""
        import shutil

        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        saved = 0

        for frame_idx, png_path in enumerate(png_files):
            if frame_idx in frame_annotations:
                frame_name = f"{seq_name}_frame_{frame_idx:06d}"

                shutil.copy2(png_path, img_dir / f"{frame_name}.png")
                self._write_yolo_label(
                    lbl_dir / f"{frame_name}.txt",
                    frame_annotations[frame_idx],
                )
                saved += 1

        print(f"    Copied {saved} annotated frames from {seq_name}")

    def _find_seq_dir(self, seq_name: str) -> Path:
        """Locate the sequence directory, handling space/underscore differences."""
        seq_dir = self.raw_dir / seq_name
        if seq_dir.exists():
            return seq_dir

        alt_dir = self.raw_dir / seq_name.replace(" ", "_")
        if alt_dir.exists():
            return alt_dir

        norm_seq = seq_name.lower().replace(" ", "_")
        for d in self.raw_dir.iterdir():
            if d.is_dir() and d.name.lower().replace(" ", "_") == norm_seq:
                return d

        raise FileNotFoundError(
            f"No directory found for sequence '{seq_name}' in {self.raw_dir}"
        )

    def _find_xml(self, seq_name: str) -> Path:
        """Locate the CVAT XML annotation file for a sequence."""
        xml_path = self.raw_dir / f"{seq_name}.xml"
        if xml_path.exists():
            return xml_path

        alt_name = seq_name.replace(" ", "_")
        xml_path = self.raw_dir / f"{alt_name}.xml"
        if xml_path.exists():
            return xml_path

        norm_seq = seq_name.lower().replace(" ", "_")
        for xml_file in self.raw_dir.glob("*.xml"):
            norm_xml = xml_file.stem.lower().replace(" ", "_")
            if norm_seq == norm_xml:
                return xml_file

        raise FileNotFoundError(
            f"No XML annotation found for sequence '{seq_name}' in {self.raw_dir}"
        )

    def _write_yolo_label(self, path: Path, annotations: list[list[float]]) -> None:
        """Write one YOLO label file. Each line: class_id x_c y_c w h"""
        with open(path, "w") as f:
            for ann in annotations:
                f.write(" ".join(f"{v:.6f}" if i > 0 else str(int(v))
                                for i, v in enumerate(ann)) + "\n")

    def _write_dataset_yaml(self) -> None:
        """Write dataset.yaml for YOLOv9."""
        dataset_yaml = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": self.num_classes,
            "names": ["car"],
        }
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        print(f"  Wrote {yaml_path}")

    def _validate(self) -> None:
        """Check that every image has a label and every label is valid YOLO format."""
        for split in ("train", "val", "test"):
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            images = {p.stem for p in img_dir.glob("*.png")} | \
                     {p.stem for p in img_dir.glob("*.jpg")}
            labels = {p.stem for p in lbl_dir.glob("*.txt")}

            missing_labels = images - labels
            missing_images = labels - images

            if missing_labels:
                raise ValueError(
                    f"{split}: {len(missing_labels)} images have no label file"
                )
            if missing_images:
                raise ValueError(
                    f"{split}: {len(missing_images)} labels have no image file"
                )
            if not images:
                raise ValueError(f"{split}: split is empty")

            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            raise ValueError(
                                f"{lbl_file.name}:{line_num}: expected 5 values, got {len(parts)}"
                            )
                        cls_id = int(parts[0])
                        values = [float(v) for v in parts[1:]]
                        if cls_id >= self.num_classes:
                            raise ValueError(
                                f"{lbl_file.name}:{line_num}: invalid class_id {cls_id}"
                            )
                        if not all(0.0 <= v <= 1.0 for v in values):
                            raise ValueError(
                                f"{lbl_file.name}:{line_num}: values out of [0, 1] range"
                            )

        print("  Validation passed.")

    def _exists(self) -> bool:
        """True if output_dir has valid YOLO structure with files in each split."""
        required = ["images/train", "images/val", "images/test",
                     "labels/train", "labels/val", "labels/test"]

        for subdir in required:
            d = self.output_dir / subdir
            if not d.exists():
                return False
            if not any(d.iterdir()):
                return False

        if not (self.output_dir / "dataset.yaml").exists():
            return False

        return True

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _read_yolo_label(path: Path) -> tuple[list, list]:
        """Read a YOLO label file. Returns (bboxes, class_labels)."""
        bboxes = []
        class_labels = []
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
        """Draw YOLO-format bboxes on an image for visualization."""
        h, w = image.shape[:2]
        for bbox in bboxes:
            x_c, y_c, bw, bh = bbox
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image