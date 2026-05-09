# src/model — Trainer (framework-agnostic seam)

## Files

```
src/model/
    protocol.py     # Trainer Protocol — contract any framework's trainer satisfies
    trainer.py      # UltralyticsTrainer — concrete implementation
    run_train.py    # CLI: run a full training for a chosen variant
    __init__.py     # build_trainer(cfg) — dispatches via cfg.model.trainer_class
```

## Public API

```python
from src.model import build_trainer, Trainer

cfg     = load_config(variant="v1")    # or "v2", "v3_ds"
trainer = build_trainer(cfg)            # instantiates cfg.model.trainer_class
trainer.train()                         # full run
trainer.train_one_trial(0)              # one short Optuna trial
trainer.validate()                      # dict of metrics
```

## CLI

```bash
python -m src.model.run_train --variant v1
python -m src.model.run_train --variant v2
python -m src.model.run_train --variant v3_ds
```

## Plugging in a new framework

Implement a class matching `Trainer` in `protocol.py`:

```python
class MyTrainer:
    def __init__(self, cfg): ...
    def train(self, name=None): ...
    def validate(self) -> dict: ...
    def train_one_trial(self, trial_number=0) -> float: ...
```

Then in `config.yaml`:

```yaml
model:
  trainer_class: my.module.MyTrainer
```

No edits to `src/tuning/` or `src/model/protocol.py` are required. Framework-specific augmentation wiring (e.g. Ultralytics' `Albumentations` callback in `trainer.py`) lives inside the concrete trainer class.
