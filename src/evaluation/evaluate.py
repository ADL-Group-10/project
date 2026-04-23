"""
src/model/evaluate.py

Evaluation script — runs the trained V1 and V2 models on the test set
and produces mAP, Precision, Recall, F1, and inference speed metrics.

Usage (notebook):
    from src.model.evaluate import Evaluator
    from src.common_utils import load_config

    cfg       = load_config(variant="v1")
    evaluator = Evaluator(cfg)
    results   = evaluator.run()          # runs on test set
    evaluator.print_report(results)      # prints clean table
    evaluator.compare("v1", "v2")        # loads both best.pt and compares
"""

import sys
import time
from pathlib import Path

import torch

from src.common_utils import load_config
from src.common_utils.checkpoint import load_checkpoint, get_best_metric
from src.common_utils.paths import get_paths
from src.model.baseline_yolov9.trainer import _ensure_yolov9


class Evaluator:
    """
    Loads a trained checkpoint and evaluates it on the NVD test set.

    Args:
        cfg: Merged config from load_config(variant="v1" or "v2").
             The variant tells it which checkpoint to load (best.pt).
    """

    def __init__(self, cfg) -> None:
        self.cfg    = cfg
        self.paths  = get_paths(cfg, create_dirs=False)
        self.device = self._get_device()

        # Make sure YOLOv9 is available
        yolov9_repo = Path(cfg.model.yolov9_repo)
        _ensure_yolov9(yolov9_repo)

        # Load the best checkpoint for this variant
        variant      = cfg.variants.active
        ckpt_path    = self.paths.best_ckpt_v1 if variant == "v1" else self.paths.best_ckpt_v2

        print(f"[evaluator] Loading {variant} checkpoint from {ckpt_path}")

        # Build a model shell then load weights into it
        ckpt         = torch.load(ckpt_path, map_location=self.device)
        self.model   = ckpt["model"].float().to(self.device)
        self.model.eval()

        # Dataset yaml (same for both variants — test set doesn't change)
        from src.data import DataPipeline
        pipeline          = DataPipeline("config.yaml")
        dataset_path, _   = pipeline.run(augment="none")   # no augmentation for eval
        self.dataset_yaml = str(dataset_path / "dataset.yaml")

    # ── Public ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Evaluate on the test set. Returns all metrics as a dict.

        Returns:
            {
                "mAP50":          float,
                "mAP50_95":       float,
                "precision":      float,
                "recall":         float,
                "f1":             float,
                "inference_ms":   float,   # ms per image
                "num_params_M":   float,   # millions of parameters
            }
        """
        from val import run as yolo_val

        print(f"[evaluator] Running test set evaluation...")

        results = yolo_val(
            data       = self.dataset_yaml,
            weights    = None,
            model      = self.model,
            imgsz      = self.cfg.model.img_size,
            batch_size = self.cfg.training.batch_size,
            conf_thres = self.cfg.evaluation.conf_threshold,
            iou_thres  = self.cfg.evaluation.iou_threshold,
            task       = "test",          # use test split
            device     = str(self.device),
            verbose    = True,
            save_json  = False,
        )

        # results tuple from yolo_val: (mp, mr, map50, map50_95, ...)
        precision = float(results[0][0])
        recall    = float(results[0][1])
        map50     = float(results[0][2])
        map50_95  = float(results[0][3])
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)

        # Inference speed
        inference_ms = self._measure_speed()

        # Parameter count
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        metrics = {
            "mAP50":        map50,
            "mAP50_95":     map50_95,
            "precision":    precision,
            "recall":       recall,
            "f1":           f1,
            "inference_ms": inference_ms,
            "num_params_M": num_params,
        }

        return metrics

    def print_report(self, metrics: dict) -> None:
        """Print a clean results table."""
        variant = self.cfg.variants.active
        print(f"\n{'='*45}")
        print(f"  Evaluation Report — {variant.upper()}")
        print(f"{'='*45}")
        print(f"  mAP@0.5          : {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95     : {metrics['mAP50_95']:.4f}")
        print(f"  Precision        : {metrics['precision']:.4f}")
        print(f"  Recall           : {metrics['recall']:.4f}")
        print(f"  F1 Score         : {metrics['f1']:.4f}")
        print(f"  Inference speed  : {metrics['inference_ms']:.2f} ms/image")
        print(f"  Parameters       : {metrics['num_params_M']:.1f}M")
        print(f"{'='*45}\n")

    @staticmethod
    def compare(variant_a: str = "v1", variant_b: str = "v2") -> None:
        """
        Load best.pt from both variants and print a side-by-side comparison.
        Call this after both models are trained.

        Usage:
            Evaluator.compare("v1", "v2")
        """
        cfg_a = load_config(variant=variant_a)
        cfg_b = load_config(variant=variant_b)

        eval_a = Evaluator(cfg_a)
        eval_b = Evaluator(cfg_b)

        m_a = eval_a.run()
        m_b = eval_b.run()

        print(f"\n{'='*60}")
        print(f"  {'Metric':<20} {variant_a.upper():>12} {variant_b.upper():>12} {'Δ':>10}")
        print(f"{'='*60}")

        for key in ["mAP50", "mAP50_95", "precision", "recall", "f1", "inference_ms"]:
            a_val = m_a.get(key, 0.0)
            b_val = m_b.get(key, 0.0)
            diff  = b_val - a_val
            sign  = "+" if diff >= 0 else ""
            # For inference speed, lower is better — flag it
            label = key if key != "inference_ms" else "inference_ms ↓"
            print(f"  {label:<20} {a_val:>12.4f} {b_val:>12.4f} {sign+f'{diff:.4f}':>10}")

        print(f"{'='*60}")
        winner = variant_b if m_b["mAP50"] > m_a["mAP50"] else variant_a
        print(f"  Best mAP50: {winner.upper()}\n")

    # ── Private ───────────────────────────────────────────────────

    def _measure_speed(self) -> float:
        """
        Measure average inference time in ms per image.
        Runs warmup iterations first, then timed iterations.
        """
        warmup_iters = self.cfg.evaluation.speed_warmup_iters
        eval_iters   = self.cfg.evaluation.speed_eval_iters
        img_size     = self.cfg.model.img_size

        dummy = torch.zeros(1, 3, img_size, img_size).to(self.device)

        # Warmup — GPU needs a few runs to reach stable speed
        with torch.no_grad():
            for _ in range(warmup_iters):
                self.model(dummy)

        # Timed run
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(eval_iters):
                self.model(dummy)
        end = time.perf_counter()

        ms_per_image = (end - start) / eval_iters * 1000
        print(f"[evaluator] Inference speed: {ms_per_image:.2f} ms/image")
        return ms_per_image

    def _get_device(self) -> torch.device:
        requested = self.cfg.project.device
        if "cuda" in requested and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
