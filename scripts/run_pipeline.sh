#!/bin/bash
# Full production pipeline: data prep → tune → train V1/V2/V3 → evaluate.
#
# Behaviour:
#   - Fresh run        → auto-cleans previous artifacts, runs every step.
#   - Resume after crash → detects the in_progress marker, skips completed
#                          steps (per-step *_done markers), re-runs the rest.
#   - Successful run   → all markers wiped, next launch starts fresh again.
#
# To force a fresh run after a crash:
#   rm -rf outputs/.pipeline_markers
#
# Usage (always in detached tmux so SSH disconnect doesn't kill it):
#   tmux new -d -s adl ./scripts/run_pipeline.sh
#   tmux attach -t adl       # to view live; Ctrl-b d to detach
#   tail -f outputs/pipeline_*.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP="full-run-$(date +%Y%m%d_%H%M%S)"

LOG="outputs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs

exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "=== $(date) Pipeline started"
echo "================================================================"

# ── Resume-or-fresh detection ────────────────────────────────────
MARKER_DIR="outputs/.pipeline_markers"
RUNNING="$MARKER_DIR/in_progress"
mkdir -p "$MARKER_DIR"

if [[ -f "$RUNNING" ]]; then
    echo "=== Previous run did not complete (in_progress marker present)."
    echo "=== RESUMING — skipping any step whose *_done marker exists."
else
    echo "=== Fresh run — cleaning previous artifacts."
    rm -rf runs/ \
           outputs/results/v1 outputs/results/v2 outputs/results/v3_ds \
           outputs/optuna/study.db \
           outputs/optuna_trials/
    rm -f "$MARKER_DIR"/*_done
fi

touch "$RUNNING"

# Helper: run a step unless its _done marker exists.
step() {
    local name="$1"; shift
    local marker="$MARKER_DIR/${name}_done"
    if [[ -f "$marker" ]]; then
        echo
        echo "=== $(date) [skip] $name — already done."
        return
    fi
    echo
    echo "=== $(date) $name ==="
    "$@"
    touch "$marker"
}

# ── Environment header ───────────────────────────────────────────
echo
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

# ── Pipeline steps (each idempotent via _done marker) ────────────
step data   python -m src.data.data_pipeline
step tune   python -m src.tuning.run_tuning
step v1     python -m src.model.run_train --variant v1
step v2     python -m src.model.run_train --variant v2
step v3_ds  python -m src.model.run_train --variant v3_ds
step eval   python -c "from src.evaluation.evaluate import Evaluator; Evaluator.compare_all()"

# ── Success: wipe all markers so next launch is fresh ────────────
rm -rf "$MARKER_DIR"

echo
echo "================================================================"
echo "=== $(date) Pipeline DONE"
echo "================================================================"
echo "Best models: outputs/results/{v1,v2,v3_ds}/weights/best.pt"
echo "WandB project: nvd-snow-yolov9-optuna  (group: $WANDB_RUN_GROUP)"
