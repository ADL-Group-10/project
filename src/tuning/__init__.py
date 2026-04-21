"""
src/tuning/__init__.py

Public API for the Optuna hyperparameter-tuning module.

Owned by Sathish. Tunes ONLY on v1 (no snow aug, clean baseline) per the
contract in config.yaml. After the study finishes, call write_best_to_config()
to patch the tuned values back into config.yaml — then Dharmini re-runs the
final v1 and v2 training with those locked hyperparameters.

Usage:
    from src.tuning import run_study, write_best_to_config

    study = run_study()                       # reads cfg.tuning.*
    write_best_to_config(study)               # patch config.yaml in place
"""

from .search_space import suggest_hyperparameters
from .objective    import build_objective, train_one_trial
from .study        import run_study, create_study
from .write_best   import write_best_to_config

__all__ = [
    "suggest_hyperparameters",
    "build_objective",
    "train_one_trial",
    "run_study",
    "create_study",
    "write_best_to_config",
]
