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
    def compare_all() -> None:
        """Compare V1, V2, and V3 side-by-side in one table."""
        e1 = Evaluator("v1")
        e2 = Evaluator("v2")
        e3 = Evaluator("v3_ds")

        m1, m2, m3 = e1.run(), e2.run(), e3.run()

        print(f"\n{'='*85}")
        print(f"  {'Metric':<20} {'V1 BASE':>12} {'V2 AUG':>12} {'V3 DS':>12} {'V2 vs V3 Δ':>12}")
        print(f"{'='*85}")

        keys = ["mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1", "inference_ms"]
        for key in keys:
            v1, v2, v3 = m1[key], m2[key], m3[key]
            if v1 is None or v2 is None or v3 is None:
                print(f"  {key:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
                continue
            diff = v3 - v2
            sign = "+" if diff >= 0 else ""
            print(f"  {key:<20} {v1:>12.4f} {v2:>12.4f} {v3:>12.4f} {sign+f'{diff:.4f}':>12}")

        print(f"{'='*85}")

        # WandB comparison table
        cfg = e1.cfg
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
        rows = [[k, m1.get(k), m2.get(k), m3.get(k)] for k in keys]
        wandb.log({"comparison": wandb.Table(columns=["metric", "v1", "v2", "v3_ds"], data=rows)})
        wandb.finish()
