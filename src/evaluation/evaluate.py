import time
from pathlib import Path
import torch
from ultralytics import YOLO
from src.common_utils import load_config

class Evaluator:
    def __init__(self, variant: str) -> None:
            self.variant = variant
            print(f"[evaluator] '{variant}'")
            try:
                self.cfg = load_config(variant=variant)
            except KeyError:
                # Fallback to 'v2' or 'active' config so we at least have paths/settings
                print(f"[evaluator] '{variant}' not found in config.yaml. Using 'v2' as a template.")
                self.cfg = load_config(variant="v2") 
            
            # 2. Resolve weights using the REAL variant name (e.g., 'v3_ds')
            # Even if we used v2's config, we look in the v3_ds folder for weights
            results_dir = Path(self.cfg.paths.results_dir)
            weights = results_dir / self.variant / "weights" / "best.pt"
            
            if not weights.exists():
                # List available folders to help you debug
                available = [d.name for d in results_dir.iterdir() if d.is_dir()]
                raise FileNotFoundError(
                    f"Weights for {self.variant} missing at {weights}.\n"
                    f"Found these folders instead: {available}"
                )
    
            self.model = YOLO(str(weights))
            self.dataset_yaml = "/project/outputs/yolo/dataset.yaml"

    def run(self) -> dict:
        """Evaluate on test split."""
        metrics = self.model.val(
            data    = self.dataset_yaml,
            split   = "test", # Crucial for Domain Shift: tests on 'Heavy Snow' for V3
            imgsz   = self.cfg.model.img_size,
            device  = "0" if torch.cuda.is_available() else "cpu",
            verbose = True,
        )
        
        # Metric extraction
        mp, mr = metrics.box.mp, metrics.box.mr
        result = {
            "mAP50":      metrics.box.map50,
            "mAP50_95":   metrics.box.map,
            "precision":  mp,
            "recall":     mr,
            "f1":         2 * mp * mr / max(mp + mr, 1e-6),
            "inference_ms": self._measure_speed(),
        }
        return result

    @staticmethod
    def compare(variant_a: str, variant_b: str) -> None:
        """Compare any two variants side by side."""
        ea = Evaluator(variant_a)
        eb = Evaluator(variant_b)
        ma = ea.run()
        mb = eb.run()

        print(f"\n{'='*60}")
        print(f"  {'Metric':<20} {variant_a.upper():>12} {variant_b.upper():>12} {'Δ':>10}")
        print(f"{'='*60}")
        for key in ["mAP50", "mAP50_95", "precision", "recall", "f1", "inference_ms"]:
            a, b  = ma.get(key, 0.0), mb.get(key, 0.0)
            diff  = b - a
            sign  = "+" if diff >= 0 else ""
            print(f"  {key:<20} {a:>12.4f} {b:>12.4f} {sign+f'{diff:.4f}':>10}")
        print(f"{'='*60}")
        winner = variant_b if mb["mAP50"] > ma["mAP50"] else variant_a
        print(f"  Best mAP50: {winner.upper()}\n")

    def _measure_speed(self) -> float:
        device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dummy   = torch.zeros(1, 3, 640, 640).to(device)
        model   = self.model.model.to(device).eval()

        with torch.no_grad():
            for _ in range(50):  # warmup
                model(dummy)
            start = time.perf_counter()
            for _ in range(200):
                model(dummy)
        return (time.perf_counter() - start) / 200 * 1000

    @staticmethod
    def compare_all() -> None:
        """Compare V1, V2, and V3 side-by-side in one table."""
        # 1. Initialize V1 and V2 normally
        e1 = Evaluator("v1")
        e2 = Evaluator("v2")
        e3 = Evaluator("v3_ds")

        # Run evaluations
        m1, m2, m3 = e1.run(), e2.run(), e3.run()

        print(f"\n{'='*85}")
        print(f"  {'Metric':<20} {'V1 BASE':>12} {'V2 AUG':>12} {'V3 DS':>12} {'V2 vs V3 Δ':>12}")
        print(f"{'='*85}")
        
        for key in ["mAP50", "mAP50_95", "precision", "recall", "f1", "inference_ms"]:
            v1, v2, v3 = m1[key], m2[key], m3[key]
            # Calculate the Domain Shift impact (Gap between Augmentation and Shift)
            diff = v3 - v2
            sign = "+" if diff >= 0 else ""
            
            print(f"  {key:<20} {v1:>12.4f} {v2:>12.4f} {v3:>12.4f} {sign+f'{diff:.4f}':>12}")
            
        print(f"{'='*85}")
        # Note: In a domain shift experiment, V2 is usually the 'best' performer, 
        # while V3 proves robustness or shows the performance gap.