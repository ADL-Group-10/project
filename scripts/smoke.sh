#!/bin/bash
# 3-epoch smoke test for all five modules: tuning, train v0, train v1, train v2, train v3_ds.
# Verifies the augmentation Compose patch lands and every code path executes.
# Run on cluster after pulling latest code.
#
# Usage:
#   ./scripts/smoke.sh                       # in foreground
#   tmux new -d -s smoke ./scripts/smoke.sh  # detached

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP="smoke-$(date +%Y%m%d_%H%M%S)"

LOG="outputs/smoke_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs

exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "=== $(date) Smoke test started"
echo "================================================================"

echo "=== Environment ==="
echo "Host:       $(hostname)"
echo "User:       $(whoami)"
echo "PWD:        $(pwd)"
echo "Python:     $(python --version 2>&1)"
echo "CUDA dev:   ${CUDA_VISIBLE_DEVICES}"
echo "WandB grp:  ${WANDB_RUN_GROUP}"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv || true
echo
echo "--- Key packages ---"
pip show ultralytics torch optuna wandb albumentations 2>/dev/null \
    | grep -E "^(Name|Version)" \
    | paste - -

echo
echo "=== $(date) Cleanup smoke artifacts ==="
rm -rf runs/ \
       outputs/results/v0 outputs/results/v1 outputs/results/v2 outputs/results/v3_ds \
       outputs/optuna/study.db \
       outputs/optuna_trials/

echo
echo "=== $(date) Tuning smoke (1 trial × 3 epochs, --no-write) ==="
python -m src.tuning.run_tuning --n-trials 1 --trial-epochs 3 --no-write

echo
echo "=== $(date) Train v0 smoke (3 epochs, raw data, no aug) ==="
python -m src.model.run_train --variant v0 --epochs 3

echo
echo "=== $(date) Train v1 smoke (3 epochs) ==="
python -m src.model.run_train --variant v1 --epochs 3

echo
echo "=== $(date) Train v2 smoke (3 epochs) ==="
python -m src.model.run_train --variant v2 --epochs 3

echo
echo "=== $(date) Train v3_ds smoke (3 epochs) ==="
python -m src.model.run_train --variant v3_ds --epochs 3

echo
echo "================================================================"
echo "=== $(date) Smoke DONE — review WandB curves + per-variant logs"
echo "================================================================"
echo "WandB group: $WANDB_RUN_GROUP"
