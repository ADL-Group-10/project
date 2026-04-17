"""
src/common_utils/device.py

Resolves compute device from config with graceful CPU fallback.

Usage:
    from src.common_utils import load_config, get_device

    cfg    = load_config()
    device = get_device(cfg)
    model  = model.to(device)
"""

import torch
from omegaconf import DictConfig


def get_device(cfg: DictConfig) -> torch.device:
    """
    Resolve compute device from cfg.project.device.
    Falls back to CPU with a warning if CUDA is requested but unavailable.
    
    Args:
        cfg: Merged config from load_config().

    Returns:
        torch.device — requested device, or cpu if CUDA unavailable.
    """
    requested: str = cfg.project.device

    if "cuda" in requested:
        if not torch.cuda.is_available():
            print(
                f"[device] WARNING: '{requested}' requested but CUDA unavailable. "
                f"Falling back to CPU. On LTU cluster, ensure you are on a GPU node."
            )
            return torch.device("cpu")

        if ":" in requested:
            idx   = int(requested.split(":")[1])
            n_gpu = torch.cuda.device_count()
            if idx >= n_gpu:
                fallback = "cuda:0"
                print(
                    f"[device] WARNING: '{requested}' requested but only "
                    f"{n_gpu} GPU(s) available. Falling back to '{fallback}'."
                )
                return torch.device(fallback)

    device = torch.device(requested)
    _log_device_info(device)
    return device


def get_device_str(cfg: DictConfig) -> str:
    """
    Same as get_device() but returns a string.
    Useful for libraries that take a device string instead of torch.device.
    """
    return str(get_device(cfg))


def _log_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        idx  = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"[device] Using GPU {idx}: {name} ({mem:.1f} GB)")
    else:
        print("[device] Using CPU")
