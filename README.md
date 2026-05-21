# NVD Car Detection in Snow — YOLOv9

Car detection in snowy conditions on the Nordic Vehicle Dataset (NVD).
Course project for D7047E Advanced Deep Learning, LTU Group 10.

## Variants

| Variant | Augmentation | Purpose |
|---|---|---|
| `v0` | none (raw data, all built-ins off) | Lower-bound reference |
| `v1` | base — Ultralytics mosaic + horizontal flip + HSV jitter | Standard YOLO baseline |
| `v2` | base + snow stack (desaturation, brightness, snow overlay, optional blur/motion/perspective) | Snow-augmented |
| `v3_ds` | v2 stack split per-split: `light_snow` on train+val, `heavy_snow` on test | Domain-shift test |

Active variant is set in `config.yaml -> variants.active`.

## Quick start

```bash
pip install -e .
wandb login                                    # one-time

# All variants, full pipeline (cluster, detached tmux)
tmux new -d -s adl ./scripts/run_pipeline.sh

# Smoke test (3-epoch sanity over every code path)
tmux new -d -s smoke ./scripts/smoke.sh

# V1-only pipeline (data -> tune -> train v1)
tmux new -d -s adl ./scripts/v1.sh
```

Logs land in `outputs/<name>_<timestamp>.log`. WandB groups all runs from one launch under `WANDB_RUN_GROUP`.

## Repo layout

```
src/
  common_utils/      config, paths, seed, device, logger, checkpoint helpers
  data/              DataPipeline (NVD -> YOLO), SnowAugmentation, AnnotationVisualizer
  model/             UltralyticsTrainer + Trainer protocol + CLI
  tuning/            Optuna study, search space, write_best, CLI
  evaluation/        Evaluator (test eval, compare_all, visualize_predictions)
scripts/             run_pipeline.sh, smoke.sh, v1.sh
config.yaml          single source of truth — all hyperparams live here
outputs/             checkpoints, results, optuna DB, wandb, logs (gitignored)
```

## Module CLIs

```bash
# Data prep (idempotent — skips if YOLO output already present)
python -m src.data.data_pipeline

# Train one variant
python -m src.model.run_train --variant {v0,v1,v2,v3_ds} [--epochs N] [--resume]

# Optuna tuning (always tunes on v1, writes best HPs back into config.yaml)
python -m src.tuning.run_tuning [--n-trials N] [--trial-epochs N] [--no-write] [--dry-run]

# Evaluation
python -c "from src.evaluation.evaluate import Evaluator; Evaluator.compare_all()"
```

## Data pipeline (`src/data/`)

`DataPipeline` parses CVAT 1.1 XML, handles rotated bounding boxes (rotates 4 corners, takes axis-aligned envelope — matches the official NVD parser), extracts frames via decord (0-based frame indexing), writes YOLO `images/{train,val,test}` + `labels/{train,val,test}` + `dataset.yaml`. Re-runs are no-ops once output exists.

Variant -> augmentation mapping is automatic: `v0 -> "none"`, `v1 -> "base"`, `v2`/`v3_ds -> "snow"`. Override with `pipeline.run(augment=..., domain_shift=...)`. Albumentations owns flip + HSV + the snow stack; Ultralytics' built-ins are zeroed in the trainer so augmentation is single-sourced.

Snow transforms (in order, geometric first so bboxes are remapped against the original grid): `perspective`, `desaturation`, `blur`, `brightness_jitter`, `snow_overlay`, `motion_blur`. Each block under `augmentation.snow.*` has `enabled`, `p`, and transform-specific params — flip `enabled` to disable without deleting config.

Splits and snow parameters live entirely in `config.yaml` (`paths.splits`, `augmentation.snow.*`, `domain_shift.{light,heavy}_snow.*`).

## Model (`src/model/`)

`UltralyticsTrainer` (default, set by `config.yaml -> model.trainer_class`) wraps Ultralytics YOLO:
- Loads weights from `model.weights`, builds dataset via `DataPipeline`, runs `model.train(...)`.
- Monkey-patches Ultralytics' `Albumentations` so the data-layer Compose survives mosaic-close and OOM-retry dataloader rebuilds.
- Adds callbacks for validation cadence, cache clearing, cfg snapshot to WandB, and (optional) `wandb.watch`.
- Same class exposes `train_one_trial(trial_number)` for Optuna (WandB disabled, fresh weights, `tuning.trial_*` params).

To swap frameworks: implement the `Trainer` protocol (`train`, `validate`, `train_one_trial`) and point `model.trainer_class` at it — no changes needed in `src/tuning/`.

## Tuning (`src/tuning/`)

Optuna study with TPE sampler + Hyperband pruner + WandB callback. All settings in `config.yaml -> tuning`:

| Key | What it controls |
|---|---|
| `targets` | HP name -> cfg path it overwrites (`lr -> training.lr`, etc.) |
| `search_space.*` | Bounds for `trial.suggest_*` |
| `n_trials`, `timeout_seconds`, `trial_epochs`, `trial_fraction`, `trial_img_size` | Trial budget |
| `direction`, `metric`, `pruner`, `sampler`, `study_name` | Study setup |
| `paths.optuna_db` | SQLite storage |

`write_best_to_config` patches `config.yaml` with the best trial's HPs (backed up to `config.yaml.bak-<timestamp>`). Tuning always runs on `v1` (clean baseline); all variants then retrain using the tuned values.

## Scripts (`scripts/`)

| Script | Workload |
|---|---|
| `run_pipeline.sh` | data -> v0 train -> tune -> v1 train -> v2 train -> v3_ds train -> compare_all |
| `v1.sh` | data -> tune -> v1 train |
| `smoke.sh` | 1-trial tuning + 3-epoch train of v0/v1/v2/v3_ds |

All three: lock GPU 0, set a shared `WANDB_RUN_GROUP`, log to `outputs/*.log`, fail fast (`set -euo pipefail`).

`run_pipeline.sh` and `v1.sh` use `_done` markers under `outputs/.pipeline_markers[_v1]/` — a crash leaves them in place so the next launch resumes; a clean finish wipes them. Force a fresh run: `rm -rf outputs/.pipeline_markers*`.

## Where to look for results

```
outputs/results/<variant>/weights/best.pt    # best checkpoint
outputs/results/<variant>/                   # Ultralytics plots, val/test curves
outputs/yolo/samples/                        # comparison_variants.png, comparison_transforms.png
outputs/optuna/study.db                      # Optuna storage (open with optuna-dashboard)
outputs/wandb/                               # WandB run cache
outputs/logs/                                # structured logger output
```

Inspect tuning live: `optuna-dashboard sqlite:///outputs/optuna/study.db --port 8080`.