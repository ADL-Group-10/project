import time
from pathlib import Path

import torch
from ultralytics import YOLO

from src.common_utils import load_config, get_device_str


class Evaluator:
    def __init__(self, variant: str) -> None:
        self.variant = variant
        print(f"[evaluator] '{variant}'")
        self.cfg = load_config(variant=variant)

        results_dir = Path(self.cfg.paths.results_dir)
        weights = results_dir / self.variant / "weights" / "best.pt"
        if not weights.exists():
            available = [d.name for d in results_dir.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                f"Weights for {self.variant} missing at {weights}.\n"
                f"Found these folders instead: {available}"
            )

        self.model = YOLO(str(weights))
        self.dataset_yaml = str(Path(self.cfg.paths.yolo_output) / "dataset.yaml")

    def run(self) -> dict:
        """Evaluate on test split."""
        ev = self.cfg.evaluation
        metrics = self.model.val(
            data    = self.dataset_yaml,
            split   = "test",
            imgsz   = self.cfg.model.img_size,
            device  = get_device_str(self.cfg),
            iou     = float(ev.iou_threshold),
            conf    = float(ev.conf_threshold),
            verbose = True,
            plots   = True,
        )

        mp, mr = metrics.box.mp, metrics.box.mr
        inference_ms = self._measure_speed() if bool(ev.measure_inference_speed) else None
        result = {
            "mAP50":        metrics.box.map50,
            "mAP75":        metrics.box.map75,
            "mAP50_95":     metrics.box.map,
            "precision":    mp,
            "recall":       mr,
            "f1":           2 * mp * mr / max(mp + mr, 1e-6),
            "inference_ms": inference_ms,
        }
        self._log_to_wandb(result, getattr(metrics, "save_dir", None))
        return result

    def _log_to_wandb(self, metrics: dict, save_dir) -> None:
        """Log test metrics and Ultralytics plot artifacts to a W&B run, if configured."""
        if not getattr(self.cfg.logging, "wandb_project", None):
            return
        try:
            import wandb
        except ImportError:
            return

        wandb.init(
            project  = str(self.cfg.logging.wandb_project),
            entity   = getattr(self.cfg.logging, "wandb_entity", None),
            name     = f"eval-{self.variant}",
            job_type = "eval",
            dir      = str(self.cfg.paths.wandb_dir),
            reinit   = True,
        )
        wandb.log({f"test/{k}": v for k, v in metrics.items() if v is not None})
        if save_dir:
            for pattern in ("*.png", "*.jpg"):
                for img in Path(save_dir).glob(pattern):
                    wandb.log({f"plots/{img.stem}": wandb.Image(str(img))})
        wandb.finish()

    def visualize_predictions(self,
                              num_samples: int = 4,
                              conf: float = 0.25,
                              save_dir: str | None = None,
                              seed: int = 42,
                              show_gt: bool = True) -> Path:
        """Run inference on N random test images and save a (GT, Pred) grid.

        Reuses AnnotationVisualizer.visualize() for box drawing + zoom insets,
        and AV.comparison_grid() / AV.save_figure() for layout and output.
        Predictions drawn in red, ground truth in green.

        Returns:
            Path to the saved PNG.
        """
        import random
        import cv2
        from src.data.visualizer import AnnotationVisualizer as AV

        img_dir = Path(self.cfg.paths.yolo_output) / "images" / "test"
        lbl_dir = Path(self.cfg.paths.yolo_output) / "labels" / "test"
        images = sorted(img_dir.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No test images in {img_dir}")

        samples = random.Random(seed).sample(images, min(num_samples, len(images)))

        out_dir = Path(save_dir) if save_dir else Path(self.cfg.paths.results_dir) / self.variant
        out_dir.mkdir(parents=True, exist_ok=True)

        GREEN, RED = (0, 255, 0), (255, 0, 0)
        cells = []
        for img_path in samples:
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

            result = self.model.predict(str(img_path), conf=conf, verbose=False)[0]
            pred_boxes = result.boxes.xywhn.cpu().numpy().tolist()

            gt_boxes: list = []
            if show_gt:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    for line in lbl_path.read_text().splitlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes.append([float(v) for v in parts[1:]])
                cells.append((f"{img_path.stem}\nGT ({len(gt_boxes)})",
                              AV.visualize(img, gt_boxes, color=GREEN)))
            cells.append((f"{img_path.stem}\nPred ({len(pred_boxes)})",
                          AV.visualize(img, pred_boxes, color=RED)))

        fig = AV.comparison_grid(
            cells,
            suptitle=f"{self.variant} predictions on test (conf ≥ {conf})",
            cols=2 if show_gt else 1,
            cell_size=(7, 4),
        )
        out_path = out_dir / f"predictions_{self.variant}.png"
        AV.save_figure(fig, out_path)
        print(f"[evaluator] Saved {out_path}")
        return out_path

    def _measure_speed(self) -> float:
        ev     = self.cfg.evaluation
        device = torch.device(get_device_str(self.cfg))
        size   = int(self.cfg.model.img_size)
        dummy  = torch.zeros(1, 3, size, size).to(device)
        model  = self.model.model.to(device).eval()

        warmup = int(ev.speed_warmup_iters)
        n_iter = int(ev.speed_eval_iters)
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)
            start = time.perf_counter()
            for _ in range(n_iter):
                model(dummy)
            end = time.perf_counter()
        return (end - start) / n_iter * 1000

    @staticmethod
    def compare_all(variants: list[str] | None = None) -> None:
        """Compare all variants side-by-side. Skips missing checkpoints gracefully."""
        if variants is None:
            variants = ["v0", "v1", "v2", "v3_ds"]

        evaluators = {}
        for v in variants:
            try:
                evaluators[v] = Evaluator(v)
            except FileNotFoundError as e:
                print(f"[compare_all] skipping {v}: {e}")

        if not evaluators:
            print("[compare_all] No variants available to compare.")
            return

        results = {v: e.run() for v, e in evaluators.items()}
        keys    = ["mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1", "inference_ms"]
        col_w   = 13
        width   = 20 + col_w * len(evaluators) + 2

        print(f"\n{'=' * width}")
        print(f"  {'Metric':<20}" + "".join(f"{v.upper():>{col_w}}" for v in evaluators))
        print(f"{'=' * width}")
        for key in keys:
            row = f"  {key:<20}"
            for v in evaluators:
                val = results[v].get(key)
                row += f"{('N/A' if val is None else f'{val:.4f}'):>{col_w}}"
            print(row)
        print(f"{'=' * width}")

        cfg = next(iter(evaluators.values())).cfg
        if not getattr(cfg.logging, "wandb_project", None):
            return
        try:
            import wandb
        except ImportError:
            return
        wandb.init(
            project  = str(cfg.logging.wandb_project),
            entity   = getattr(cfg.logging, "wandb_entity", None),
            name     = "compare-all",
            job_type = "compare",
            dir      = str(cfg.paths.wandb_dir),
            reinit   = True,
        )
        columns = ["metric"] + list(evaluators.keys())
        rows    = [[k] + [results[v].get(k) for v in evaluators] for k in keys]
        wandb.log({"comparison": wandb.Table(columns=columns, data=rows)})
        wandb.finish()
