#!/bin/bash
# V1-only pipeline: data prep → tune → train V1.
#
# Usage (from project root, in detached tmux):
#   tmux new -d -s adl ./scripts/v1.sh
#   tmux attach -t adl
#   tail -f outputs/v1_*.log

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP="v1-run-$(date +%Y%m%d_%H%M%S)"

LOG="outputs/v1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs

# Log everything to file + stdout
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "=== $(date) V1 pipeline started (scripts/v1.sh)"
echo "================================================================"

# ── Resume-or-fresh detection ────────────────────────────────────
MARKER_DIR="outputs/.pipeline_markers_v1"
RUNNING="$MARKER_DIR/in_progress"
mkdir -p "$MARKER_DIR"

if [[ -f "$RUNNING" ]]; then
    echo "=== Previous V1 run did not complete (in_progress present)."
    echo "=== RESUMING — skipping any step whose *_done marker exists."
else
    echo "=== Fresh V1 run — cleaning previous V1 artifacts."
    rm -rf runs/ \
           outputs/results/v1 \
           outputs/optuna/study.db \
           outputs/optuna_trials/
    rm -f "$MARKER_DIR"/*_done
fi

touch "$RUNNING"

# Helper: run a step unless its _done marker exists
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

# ── V1-only steps ────────────────────────────────────────────────
step data  python -m src.data.data_pipeline
step tune  python -m src.tuning.run_tuning
step v1    python -m src.model.run_train --variant v1


# ── Success: wipe markers so next launch is fresh ────────────────
rm -rf "$MARKER_DIR"

echo
echo "================================================================"
echo "=== $(date) V1 pipeline DONE"
echo "================================================================"
echo "Best model: outputs/results/v1/weights/best.pt"
echo "WandB project: nvd-snow-yolov9-optuna  (group: $WANDB_RUN_GROUP)"