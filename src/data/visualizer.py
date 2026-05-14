"""Standalone bbox visualization with dual boxes and zoom insets.

Usage:
    from src.data.visualizer import AnnotationVisualizer as AV

    # From CVAT XML (rotated polygon + axis-aligned box + zoom)
    anns, w, h = AV.parse_cvat_frame("annotation.xml", frame_num=1721)
    vis = AV.visualize(image, anns)

    # From YOLO boxes (axis-aligned box + zoom)
    vis = AV.visualize(image, [[0.48, 0.28, 0.05, 0.04], ...])

    # Comparison grid
    fig = AV.comparison_grid([("raw", img1), ("snow", img2)])
    AV.save_figure(fig, "out.png")
"""

import xml.etree.ElementTree as ET

import cv2
import numpy as np


class AnnotationVisualizer:

    # ── Core: draw boxes + zoom insets on any image ────────────

    @staticmethod
    def visualize(image: np.ndarray, boxes,
                  color: tuple = (0, 255, 0), thickness: int = 2,
                  zoom_factor: float = 2.5, max_insets: int = 4) -> np.ndarray:
        """Draw bounding boxes with zoom insets.

        Args:
            image:  RGB numpy array.
            boxes:  either a list of YOLO boxes [x_c, y_c, w, h]
                    or a list of annotation dicts from parse_cvat_frame()
                    (dicts with 'polygon', 'straight', 'yolo_box' keys).
            zoom_factor: how much to enlarge each inset crop.
            max_insets:  max number of zoom insets to draw.

        Returns:
            Composited RGB image with boxes and zoom insets.
        """
        if not boxes:
            return image.copy()

        img_h, img_w = image.shape[:2]
        is_annotation = isinstance(boxes[0], dict) and "polygon" in boxes[0]

        if is_annotation:
            regions = []
            for ann in boxes:
                regions.append({
                    "straight": ann["straight"],
                    "polygon": ann["polygon"],
                })
        else:
            regions = []
            for bbox in boxes:
                x_c, y_c, bw, bh = bbox[:4]
                x1 = int((x_c - bw / 2) * img_w)
                y1 = int((y_c - bh / 2) * img_h)
                x2 = int((x_c + bw / 2) * img_w)
                y2 = int((y_c + bh / 2) * img_h)
                regions.append({"straight": [x1, y1, x2, y2]})

        vis = image.copy()

        # Draw all boxes on full image
        for r in regions:
            x1, y1, x2, y2 = r["straight"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            if "polygon" in r:
                pts = r["polygon"].reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, color, thickness)

        # Sort by area (largest first) and pick insets
        by_area = sorted(regions,
                         key=lambda r: (r["straight"][2] - r["straight"][0])
                                     * (r["straight"][3] - r["straight"][1]),
                         reverse=True)
        insets = by_area[:max_insets]

        inset_h = int(img_h * 0.35)
        inset_w = inset_h
        margin = 10
        inset_y = img_h - inset_h - margin

        for i, r in enumerate(insets):
            x1, y1, x2, y2 = r["straight"]
            cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
            box_size = max(x2 - x1, y2 - y1)
            pad = box_size * zoom_factor / 2

            crop_x1 = max(0, int(cx_px - pad))
            crop_y1 = max(0, int(cy_px - pad))
            crop_x2 = min(img_w, int(cx_px + pad))
            crop_y2 = min(img_h, int(cy_px + pad))

            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if crop.size == 0:
                continue

            # Draw boxes on crop (shifted coords)
            sx1, sy1 = x1 - crop_x1, y1 - crop_y1
            sx2, sy2 = x2 - crop_x1, y2 - crop_y1
            cv2.rectangle(crop, (sx1, sy1), (sx2, sy2), color, thickness)
            if "polygon" in r:
                shifted = r["polygon"] - [crop_x1, crop_y1]
                cv2.polylines(crop, [shifted.reshape(-1, 1, 2)],
                              True, color, thickness)

            inset = cv2.resize(crop, (inset_w, inset_h))
            inset_x = margin + i * (inset_w + margin)
            if inset_x + inset_w > img_w:
                break

            # Border, place, connect
            cv2.rectangle(vis,
                          (inset_x - 2, inset_y - 2),
                          (inset_x + inset_w + 2, inset_y + inset_h + 2),
                          (255, 255, 255), 2)
            vis[inset_y:inset_y + inset_h,
                inset_x:inset_x + inset_w] = inset
            cv2.line(vis, (cx_px, cy_px),
                     (inset_x + inset_w // 2, inset_y),
                     (255, 255, 255), 1, cv2.LINE_AA)

            # Highlight source box
            pad_vis = 5
            cv2.rectangle(vis,
                          (max(0, x1 - pad_vis), max(0, y1 - pad_vis)),
                          (min(img_w, x2 + pad_vis), min(img_h, y2 + pad_vis)),
                          (255, 255, 255), 2)

        return vis

    # ── CVAT parsing ───────────────────────────────────────────

    @staticmethod
    def parse_cvat_frame(xml_path: str, frame_num: int) -> tuple[list[dict], int, int]:
        """Parse one frame from CVAT XML -> annotation dicts.

        Returns (annotations, frame_width, frame_height).
        Each dict: polygon (4,2 int), straight [x1,y1,x2,y2],
                   yolo_box [xc,yc,w,h], rotation (float).
        """
        root = ET.parse(xml_path).getroot()
        frame_w = int(root.find("meta//original_size/width").text)
        frame_h = int(root.find("meta//original_size/height").text)

        results = []
        for track in root.findall(".//track"):
            if track.get("label") != "car":
                continue
            for box in track.findall("box"):
                if box.get("outside") == "1" or int(box.get("frame")) != frame_num:
                    continue

                xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
                xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
                rotation = float(box.get("rotation", "0"))

                corners = AnnotationVisualizer._rotate_corners(
                    xtl, ytl, xbr, ybr, rotation)

                x_min = max(0, corners[:, 0].min())
                y_min = max(0, corners[:, 1].min())
                x_max = min(frame_w, corners[:, 0].max())
                y_max = min(frame_h, corners[:, 1].max())
                w, h = x_max - x_min, y_max - y_min
                if w <= 0 or h <= 0:
                    continue

                results.append({
                    "polygon": corners.astype(int),
                    "straight": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "yolo_box": [(x_min + w / 2) / frame_w,
                                 (y_min + h / 2) / frame_h,
                                 w / frame_w, h / frame_h],
                    "rotation": rotation,
                })

        return results, frame_w, frame_h

    # ── Grid layout ────────────────────────────────────────────

    @staticmethod
    def comparison_grid(cells: list[tuple[str, np.ndarray]],
                        suptitle: str = "", cols: int = 5,
                        cell_size: tuple = (4, 3)):
        """Arrange (label, image) pairs into a grid. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        n = len(cells)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cell_size[0] * cols,
                                          cell_size[1] * rows))
        axes_flat = np.array(axes).flatten()

        for ax, (label, img) in zip(axes_flat, cells):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=9, fontweight="bold")

        for ax in axes_flat[n:]:
            ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    @staticmethod
    def save_figure(fig, path, dpi: int = 110):
        """Save and close a matplotlib figure."""
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    # ── Internal ───────────────────────────────────────────────

    @staticmethod
    def _rotate_corners(xtl, ytl, xbr, ybr, rotation_deg):
        cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
        corners = np.array([[xtl, ytl], [xbr, ytl],
                            [xbr, ybr], [xtl, ybr]], dtype=float)
        if abs(rotation_deg) > 0.01:
            rad = rotation_deg * np.pi / 180
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            corners = (rot_mat @ (corners - [cx, cy]).T).T + [cx, cy]
        return corners
