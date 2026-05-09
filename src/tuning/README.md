# src/tuning — Optuna hyperparameter tuning

## Files

```
src/tuning/
    __init__.py         # public API
    protocol.py         # TrialTrainer Protocol (framework contract)
    search_space.py     # yaml → trial.suggest_*; overlay onto cfg
    objective.py        # build_objective() + train_one_trial() seam
    study.py            # create/run study, sampler+pruner, W&B callback
    write_best.py       # patch config.yaml with best trial
    run_tuning.py       # CLI entry point
```

## Wire-up

All config flows from `config.yaml → tuning`:

| yaml key                 | Used by                                   |
|--------------------------|-------------------------------------------|
| `trainer_class`          | dotted path to a `TrialTrainer`           |
| `targets`                | HP name → cfg path it overwrites          |
| `search_space.*`         | `suggest_hyperparameters`                 |
| `n_trials`               | `study.optimize`                          |
| `timeout_seconds`        | `study.optimize`                          |
| `direction`              | `optuna.create_study`                     |
| `metric`                 | W&B callback `metric_name`, log labels    |
| `pruner`                 | `_build_pruner` (hyperband/median/none)   |
| `sampler`                | `_build_sampler` (tpe/random)             |
| `study_name`             | storage key + W&B group                   |
| `paths.optuna_db`        | SQLite storage                            |
| `logging.wandb_project`  | W&B callback (optional)                   |

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

## Plugging in a new framework

Implement a class matching `TrialTrainer` (see `protocol.py`):

```python
class MyTrainer:
    def __init__(self, cfg): ...
    def train_one_trial(self, trial_number: int = 0) -> float: ...
```

Then in `config.yaml`:

```yaml
tuning:
  trainer_class: my.module.MyTrainer
  targets:
    lr: optimizer.learning_rate   # whichever cfg paths your framework reads
    ...
```

No edits to `src/tuning/` are required.
