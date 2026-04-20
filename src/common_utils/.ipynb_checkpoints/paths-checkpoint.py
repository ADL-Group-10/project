"""
src/common_utils/paths.py

Resolves all filesystem paths from config so no module hardcodes strings.
Creates output directories on first call.

Who calls this:
    Achyuth  — sequence dirs for data pipeline + domain-shift split lists
    Dharmini  — checkpoint save dirs in trainer.py
    TBD  — checkpoint load paths in evaluation notebooks

Usage:
    from src.common_utils import load_config, get_paths

    cfg   = load_config()
    paths = get_paths(cfg)

    paths.nvd_root            # Path("/data/NVD")
    paths.train_sequences     # [Path("/data/NVD/2022-12-04_Hjenberg-02"), ...]
    paths.checkpoint_dir_v1   # Path("outputs/checkpoints/v1")
    paths.best_ckpt_v1        # Path("outputs/checkpoints/v1/best.pt")
"""

from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import DictConfig


@dataclass
class ProjectPaths:
    """All resolved project paths. Access as attributes."""

    # raw data
    nvd_root:             Path       = field(default_factory=Path)
    train_sequences:      list[Path] = field(default_factory=list)
    val_sequences:        list[Path] = field(default_factory=list)
    test_sequences:       list[Path] = field(default_factory=list)

    # domain-shift sub-experiment
    light_snow_sequences: list[Path] = field(default_factory=list)
    heavy_snow_sequences: list[Path] = field(default_factory=list)

    # outputs
    checkpoint_dir_v1:    Path       = field(default_factory=Path)
    checkpoint_dir_v2:    Path       = field(default_factory=Path)
    best_ckpt_v1:         Path       = field(default_factory=Path)
    best_ckpt_v2:         Path       = field(default_factory=Path)
    results_dir:          Path       = field(default_factory=Path)

    # Optuna
    optuna_db:            Path       = field(default_factory=Path)

    # logging
    wandb_dir:            Path       = field(default_factory=Path)
    log_dir:              Path       = field(default_factory=Path)


def get_paths(cfg: DictConfig, create_dirs: bool = True) -> ProjectPaths:
    """
    Resolve all paths from config and optionally create output directories.

    Args:
        cfg:         Merged OmegaConf config from load_config().
        create_dirs: If True (default), creates output dirs that don't exist.
                     Pass False in read-only environments or unit tests.

    Returns:
        ProjectPaths dataclass with all paths as pathlib.Path objects.
    """
    nvd_root  = Path(cfg.paths.nvd_root)
    ckpt_base = Path(cfg.paths.checkpoints_dir)
    ckpt_v1   = ckpt_base / "v1"
    ckpt_v2   = ckpt_base / "v2"

    paths = ProjectPaths(
        nvd_root             = nvd_root,
        train_sequences      = [nvd_root / s for s in cfg.paths.splits.train],
        val_sequences        = [nvd_root / s for s in cfg.paths.splits.val],
        test_sequences       = [nvd_root / s for s in cfg.paths.splits.test],
        light_snow_sequences = [nvd_root / s for s in cfg.paths.domain_shift.light_snow],
        heavy_snow_sequences = [nvd_root / s for s in cfg.paths.domain_shift.heavy_snow],
        checkpoint_dir_v1    = ckpt_v1,
        checkpoint_dir_v2    = ckpt_v2,
        best_ckpt_v1         = ckpt_v1 / "best.pt",
        best_ckpt_v2         = ckpt_v2 / "best.pt",
        results_dir          = Path(cfg.paths.results_dir),
        optuna_db            = Path(cfg.paths.optuna_db),
        wandb_dir            = Path(cfg.paths.wandb_dir),
        log_dir              = Path(cfg.paths.log_dir),
    )

    if create_dirs:
        _make_output_dirs(paths)

    return paths


def get_checkpoint_path(cfg: DictConfig, variant: str | None = None) -> Path:
    """
    Return best.pt path for the active (or specified) variant.

    Args:
        cfg:     Merged config from load_config().
        variant: "v1" or "v2". Defaults to cfg.variants.active.

    Example:
        ckpt_path = get_checkpoint_path(cfg, variant="v2")
        model, meta = load_checkpoint(ckpt_path, model)
    """
    v = variant or cfg.variants.active
    return Path(cfg.paths.checkpoints_dir) / v / "best.pt"


def _make_output_dirs(paths: ProjectPaths) -> None:
    """Create all output directories that don't yet exist."""
    for d in [
        paths.checkpoint_dir_v1,
        paths.checkpoint_dir_v2,
        paths.results_dir,
        paths.optuna_db.parent,
        paths.wandb_dir,
        paths.log_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
