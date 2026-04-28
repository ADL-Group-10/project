"""
src/common_utils/config_loader.py

Loads config.yaml and merges the active variant on top of base settings.
First import in every notebook and every src/ module.

Who calls this:
    Achyuth  — get_dataloader(), snow_pipeline.py
    Dharmini  — trainer.py, yolov9_wrapper.py
    Sathish  — objective.py (Optuna), every tuning notebook
    TBD  — evaluation notebooks

Usage:
    from src.common_utils import load_config

    cfg = load_config()               # reads variants.active from yaml
    cfg = load_config(variant="v2")   # override without editing the file
    cfg = load_config("path/to/config.yaml")  # custom path
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str = "config.yaml",
    variant: str | None = None,
) -> DictConfig:
    """
    Load config.yaml and merge the active variant into base config.

    Args:
        config_path:  Path to config.yaml. Resolved relative to cwd, or
                      walks up to project root if not found in cwd.
        variant:      If provided, overrides variants.active in the yaml.
                      Pass "v1" or "v2" to switch variant at runtime.

    Returns:
        OmegaConf DictConfig — base settings merged with the active variant.
        Variant keys (lr, epochs, use_snow_aug, experiment) override base.

    Raises:
        FileNotFoundError: config.yaml not found.
        KeyError: requested variant does not exist in yaml.
    """
    path = _resolve_config_path(config_path)
    base = OmegaConf.load(path)

    if variant is not None:
        OmegaConf.update(base, "variants.active", variant)

    active = base.variants.active
    if active not in base.variants:
        available = [k for k in base.variants if k != "active"]
        raise KeyError(
            f"Variant '{active}' not found in config.yaml. "
            f"Available: {available}"
        )

    variant_cfg = base.variants[active]
    cfg = OmegaConf.merge(base, variant_cfg)
    return cfg


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the merged config. Useful at the top of any notebook."""
    print(OmegaConf.to_yaml(cfg))


def _resolve_config_path(config_path: str) -> Path:
    """
    Find config.yaml. Tries:
      1. The path as given (absolute or relative to cwd)
      2. Walking up from cwd until a project root is found (has setup.py)
    """
    p = Path(config_path)
    if p.exists():
        return p

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / config_path
        if candidate.exists():
            return candidate
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            break

    raise FileNotFoundError(
        f"config.yaml not found at '{config_path}' or any parent directory. "
        f"Run from the project root or pass an absolute path."
    )
