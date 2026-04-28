"""
src/common_utils/checkpoint.py

Unified checkpoint save / load with identical serialisation keys.

Who calls this:
    Dharmini  — save_checkpoint() in trainer.py after each epoch
    TBD  — load_checkpoint() in evaluation notebooks
    Sathish  — get_best_metric() to compare trial results during Optuna study

Save example (Dharmini — trainer.py):
    from src.common_utils import save_checkpoint

    save_checkpoint(
        model     = model,
        optimizer = optimizer,
        epoch     = epoch,
        metrics   = {"val_map50": 0.72, "val_map50_95": 0.41, "val_loss": 0.28},
        cfg       = cfg,
        is_best   = val_map50 > best_so_far,
    )

Load example (TBD — evaluation notebook):
    from src.common_utils import load_checkpoint, get_checkpoint_path

    ckpt_path    = get_checkpoint_path(cfg, variant="v1")
    model, meta  = load_checkpoint(ckpt_path, model, device=device)
    print(meta["metrics"])    # {"val_map50": 0.72, ...}
    print(meta["epoch"])      # 87
"""

import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Any


def save_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict[str, float],
    cfg:       DictConfig,
    is_best:   bool = False,
    tag:       str | None = None,
) -> Path:
    """
    Save a training checkpoint.

    Saves to:  outputs/checkpoints/{variant}/epoch_{epoch:03d}.pt
    If is_best: also saves to outputs/checkpoints/{variant}/best.pt

    Args:
        model:     YOLOv9 model (nn.Module).
        optimizer: Optimizer state (needed for training resumption).
        epoch:     Current epoch (0-indexed).
        metrics:   Dict of metric values, e.g.
                   {"val_map50": 0.72, "val_map50_95": 0.41, "val_loss": 0.28}
        cfg:       Merged config — provides variant name and checkpoint dir.
        is_best:   Also save as best.pt if True.
        tag:       Optional label appended to filename, e.g. "pruned".

    Returns:
        Path to the saved epoch checkpoint.
    """
    variant  = cfg.variants.active
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fname = f"epoch_{epoch:03d}"
    if tag:
        fname += f"_{tag}"
    fname += ".pt"

    payload = {
        "epoch":      epoch,
        "variant":    variant,
        "experiment": cfg.experiment,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
        "cfg":        OmegaConf.to_container(cfg, resolve=True),
    }

    save_path = ckpt_dir / fname
    torch.save(payload, save_path)

    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(payload, best_path)
        print(f"[checkpoint] New best saved → {best_path}  metrics={metrics}")

    return save_path


def load_checkpoint(
    path:      str | Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device:    torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load a checkpoint into a model (and optionally an optimizer).

    Args:
        path:      Path to .pt file. Use get_checkpoint_path(cfg) for best.pt.
        model:     Model instance — must match the saved architecture.
        optimizer: Pass None for evaluation-only loading (no optimizer state).
        device:    Target device. Defaults to CPU for safe cross-device loading;
                   call model.to(device) after this function.

    Returns:
        (model, meta) where meta contains:
            "epoch"      : int   — epoch at which this checkpoint was saved
            "variant"    : str   — "v1" or "v2"
            "experiment" : str   — experiment name from config
            "metrics"    : dict  — metrics recorded at save time
            "cfg"        : dict  — original config as plain Python dict

    Raises:
        FileNotFoundError: checkpoint file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run the training notebook first, or check outputs/checkpoints/."
        )

    map_loc = device if device is not None else torch.device("cpu")
    ckpt    = torch.load(path, map_location=map_loc)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    meta = {
        "epoch":      ckpt.get("epoch", 0),
        "variant":    ckpt.get("variant", "unknown"),
        "experiment": ckpt.get("experiment", "unknown"),
        "metrics":    ckpt.get("metrics", {}),
        "cfg":        ckpt.get("cfg", {}),
    }

    print(
        f"[checkpoint] Loaded {path.name} — "
        f"variant={meta['variant']}, epoch={meta['epoch']}, "
        f"metrics={meta['metrics']}"
    )

    return model, meta


def get_best_metric(checkpoint_path: str | Path) -> dict[str, float]:
    """
    Load only the metrics dict from a checkpoint — no model weights.

    Used by:
        TBD  — quick comparison of V1 vs V2 in evaluation notebooks
        Sathish  — inspect trial results during Optuna study

    Example:
        v1 = get_best_metric("outputs/checkpoints/v1/best.pt")
        v2 = get_best_metric("outputs/checkpoints/v2/best.pt")
        print(f"V1 mAP50: {v1['val_map50']:.3f}")
        print(f"V2 mAP50: {v2['val_map50']:.3f}")
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("metrics", {})
