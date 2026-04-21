# src/tuning — Optuna hyperparameter tuning

Owned by **Sathish**. Tunes the YOLOv9 v1 baseline (no snow aug), writes the
best trial back into `config.yaml`, then hands off to Dharmini for the
final v1 and v2 training runs.

## Files

```
src/tuning/
    __init__.py         # public API
    search_space.py     # yaml → trial.suggest_*
    objective.py        # build_objective() + train_one_trial() seam
    study.py            # create/run study, sampler+pruner, W&B callback
    write_best.py       # patch config.yaml with best trial
    run_tuning.py       # CLI entry point
```

## Wire-up

All config flows from `config.yaml → tuning`:

| yaml key                 | Used by                 |
|--------------------------|-------------------------|
| `n_trials`               | `study.optimize`        |
| `timeout_seconds`        | `study.optimize`        |
| `direction`              | `optuna.create_study`   |
| `metric`                 | W&B callback metric_name|
| `pruner`                 | `_build_pruner` (hyperband/median/none) |
| `sampler`                | `_build_sampler` (tpe/random) |
| `study_name`             | storage + W&B group     |
| `search_space.*`         | `suggest_hyperparameters` |
| `paths.optuna_db`        | SQLite storage          |
| `logging.wandb_project`  | W&B callback (optional) |

## Usage

```python
from src.tuning import run_study, write_best_to_config

study = run_study()                # reads cfg.tuning.*
write_best_to_config(study)        # patch config.yaml
```

CLI:

```bash
python -m src.tuning.run_tuning --n-trials 5       # smoke
python -m src.tuning.run_tuning                    # full
python -m src.tuning.run_tuning --dry-run          # inspect diff only
```

## Current status — SKELETON

`train_one_trial()` in `objective.py` is a **stub** that returns a plausible
dummy mAP50 curve. It exists so the Optuna + W&B + pruning pipeline runs
end-to-end today, before Dharmini's `trainer.py` is ready.

### Swapping in the real trainer

In `objective.py`, replace the body of `train_one_trial(cfg, trial)` with:

```python
from src.models import YOLOv9Trainer   # Dharmini's module

def train_one_trial(cfg, trial):
    trainer = YOLOv9Trainer(cfg)
    best = 0.0
    for epoch in range(cfg.training.epochs):
        trainer.train_one_epoch()
        if epoch % cfg.training.val_every_n_epochs == 0:
            map50 = trainer.validate()["mAP50"]
            best = max(best, map50)
            trial.report(map50, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best
```

Everything else — search space, study, W&B, write-back — stays the same.
