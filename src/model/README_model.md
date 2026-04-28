# src/model — YOLOv9 Training & Evaluation



This module trains and evaluates YOLOv9 on the Nordic Vehicle Dataset (NVD)
for car detection under snowy conditions.

---

## Files

```
src/model/
    baseline_yolov9/
        trainer.py          ← V1: standard augmentation
    snow_augmented_yolov9/
        trainer_sa.py       ← V2: snow-aware augmentation (inherits V1)
    evaluate.py             ← evaluation on test set, V1 vs V2 comparison

notebooks/
    train.ipynb             ← run everything from here (start here!)
```

---

## What each file does

### `trainer.py` — V1 Baseline
- Clones YOLOv9 into `third_party/yolov9` automatically if not present
- Loads pretrained `yolov9-c.pt` weights
- Trains on NVD with standard augmentation (`augment="base"`)
- Saves checkpoints to `outputs/checkpoints/v1/`
- Best model saved as `outputs/checkpoints/v1/best.pt`

### `trainer_sa.py` — V2 Snow Augmented
- Inherits everything from `trainer.py`
- Only difference: uses `augment="snow"` (adds desaturation, blur, snow overlay)
- Saves to `outputs/checkpoints/v2/best.pt`

### `evaluate.py` — Evaluation
- Loads `best.pt` for a given variant
- Runs on the **test set** (never seen during training)
- Reports: mAP50, mAP50-95, Precision, Recall, F1, inference speed
- `Evaluator.compare("v1", "v2")` produces a side-by-side table

---

## How to run (use the notebook)

Open `notebooks/train.ipynb` and run top to bottom:

```
Cell 1: Install dependencies
Cell 2: Check GPU
Cell 3: Load config
Cell 4: Check dataset (DataPipeline — slow on first run)
Cell 5: Preview augmentation visually
Cell 6: Train V1
Cell 7: Train V2
Cell 8: Compare results
```

---

## How hyperparameter tuning plugs in 

Optuna code calls `YOLOv9Trainer` directly from `objective.py`:

```python
from src.model.baseline_yolov9.trainer import YOLOv9Trainer

trainer = YOLOv9Trainer(cfg)         # cfg has trial hyperparams already set
for epoch in range(cfg.training.epochs):
    trainer.train_one_epoch()
    if epoch % cfg.training.val_every_n_epochs == 0:
        metrics = trainer.validate()
        map50   = metrics["mAP50"]
        trial.report(map50, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
```

After tuning finishes, Sathish writes best hyperparams back to `config.yaml`.
then re-runs `train.ipynb` — no code changes needed, just re-run.

---

## Outputs

```
outputs/
    checkpoints/
        v1/
            best.pt              ← best V1 model (hand to evaluation person)
            epoch_010.pt
            epoch_020.pt
            ...
        v2/
            best.pt              ← best V2 model
            ...
    results/                     ← evaluation outputs
    wandb/                       ← training graphs (auto-logged)
    logs/                        ← text logs
```

---

## Config keys used

All settings come from `config.yaml` — never hardcoded here.

| Key | Used by | What it controls |
|-----|---------|-----------------|
| `training.epochs` | trainer | how long to train |
| `training.lr` | trainer | learning rate (Sathish tunes this) |
| `training.batch_size` | trainer | batch size (Sathish tunes this) |
| `training.optimizer` | trainer | "adam" or "sgd" |
| `loss.box_weight` | trainer | localization loss weight (Sathish tunes) |
| `loss.focal_gamma` | trainer | focal loss gamma (Sathish tunes) |
| `model.freeze_backbone` | trainer | freeze backbone layers |
| `evaluation.conf_threshold` | evaluator | confidence cutoff |
| `evaluation.iou_threshold` | evaluator | IoU cutoff for mAP |
| `evaluation.speed_eval_iters` | evaluator | inference speed measurement |

---

## Dependencies

These are installed via `setup.py` or `pip install -e .`:

- `torch` — core deep learning
- `albumentations` — augmentation
- `omegaconf` — config loading
- `wandb` — experiment tracking (optional, skipped if not logged in)
- YOLOv9 — auto-cloned into `third_party/yolov9` on first run
