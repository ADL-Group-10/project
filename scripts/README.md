# scripts/

Two shell scripts for running the project end-to-end on the cluster.

| Script | Purpose | Workload |
|--------|---------|----------|
| `smoke.sh` | Sanity test of every code path | 1 tuning trial × 3 epochs, then V1/V2/V3 × 3 epochs each |
| `run_pipeline.sh` | Full production: data → tune → train V1/V2/V3 → evaluate | as defined by `cfg.tuning` and `cfg.training` |

Both auto-log to `outputs/<name>_<timestamp>.log`, lock the GPU via `CUDA_VISIBLE_DEVICES=0`, set a `WANDB_RUN_GROUP` so all runs from one launch cluster together in the W&B UI, and bail on the first failure (`set -euo pipefail`).

`run_pipeline.sh` also has resume-on-crash markers — a successful run wipes them, a crash leaves them behind so the next launch skips completed steps automatically.

---

## Before running

Do these once on the cluster (most are one-time setup; a few are per-launch).

### One-time setup
1. **Clone + install**:
   ```bash
   git clone <repo>
   cd <repo>
   pip install -e .
   ```
2. **WandB auth**: `wandb login` (paste API key when prompted).
3. **Enable WandB run-finished emails**: WandB UI → avatar → User settings → Alerts → enable "Run finished" + "Run crashed".
4. *(Optional)* `pip install optuna-dashboard` for a live web view of tuning trials.

### Per-launch checks
1. **Pull latest code**:
   ```bash
   git pull
   ```
2. **Confirm GPU is free**:
   ```bash
   nvidia-smi
   ```
3. **Confirm dataset exists**:
   ```bash
   python -c "from pathlib import Path; from src.common_utils import load_config; \
              p = Path(load_config().paths.nvd_root); print(p, p.exists())"
   ```
   Must print `True`.
4. **Disk space**: `df -h outputs/`.

---

## Running the scripts

Always launch in detached `tmux` so an SSH disconnect doesn't kill the run.

### Smoke
```bash
tmux new -d -s smoke ./scripts/smoke.sh
```
Auto-cleans previous artifacts on every launch (no resume — disposable by design).

### Full pipeline
```bash
tmux new -d -s adl ./scripts/run_pipeline.sh
```

---

## While the script is running

You can do all of these from any SSH session — no need to attach to the tmux that's running the pipeline.

### Health peek (no attach needed)
```bash
tmux ls                                        # confirm session is alive
tmux capture-pane -t adl -pS -200              # last 200 log lines
tail -f outputs/pipeline_*.log                 # live log; Ctrl-C to stop watching
```

### Resource checks
```bash
nvidia-smi                                     # GPU util / memory / temp
df -h outputs/                                 # disk usage
du -sh outputs/results/* 2>/dev/null           # per-variant artifact size
```

### Progress signals
- WandB email arrives when each variant finishes.
- `ls -lt outputs/results/*/weights/best.pt` shows which variants have written a best checkpoint.
- WandB dashboard: filter by group `full-run-YYYYMMDD_HHMMSS`; watch `metrics/mAP50` and `train/box_loss` curves. A static epoch counter for an extended period indicates a stall.

### Live tuning view (optional, only useful during Step 1)
On the cluster, in a second tmux:
```bash
tmux new -d -s dash 'optuna-dashboard sqlite:///outputs/optuna/study.db --port 8080 --host 127.0.0.1'
```
Then SSH-tunnel from your laptop and open `http://localhost:8080`:
```bash
ssh -L 8080:127.0.0.1:8080 <cluster>
```

### Re-attaching to the pipeline tmux (optional)
```bash
tmux attach -t adl    # interactive view; Ctrl-b d to detach without killing
```

---

## After the script completes

### 1. Verify completion
```bash
tail -20 outputs/pipeline_*.log
```
Look for `Pipeline DONE` and the printed comparison table.

```bash
ls outputs/results/v1/weights/best.pt \
   outputs/results/v2/weights/best.pt \
   outputs/results/v3_ds/weights/best.pt
```
All three must exist.

### 2. Pull the headline result
```bash
grep -A 20 "V1 BASE" outputs/pipeline_*.log
```
Or open the `compare-all` run in WandB.

### 3. Archive off the cluster
```bash
tar czf nvd-snow-yolov9-results-$(date +%Y%m%d).tar.gz \
    outputs/results/*/weights/best.pt \
    outputs/pipeline_*.log \
    config.yaml \
    config.yaml.bak-*

# from laptop
scp <cluster>:/project/nvd-snow-yolov9-results-*.tar.gz .
```

### 4. Clean up tmux
```bash
tmux kill-session -t adl
tmux kill-session -t dash    # if you launched the dashboard
```

---

## Resume after a crash

If `run_pipeline.sh` exits with a failure, relaunch as normal:
```bash
tmux new -d -s adl ./scripts/run_pipeline.sh
```
The script detects the leftover `outputs/.pipeline_markers/in_progress` marker, **does not clean** previous artifacts, and **skips any step whose `_done` marker exists**.

To force a fresh run instead of resuming:
```bash
rm -rf outputs/.pipeline_markers
tmux new -d -s adl ./scripts/run_pipeline.sh
```

---

## Troubleshooting

| Symptom | First check |
|---------|-------------|
| Script exited immediately | `cat outputs/pipeline_*.log` — usually a missing dependency or wrong cwd. Run from project root. |
| `tmux ls` shows no session | Pipeline finished or died. Check the log. |
| WandB project is empty | `wandb login` may have expired. Re-auth and relaunch. |
| Disk filling up | `du -sh outputs/wandb/` — local WandB cache can grow. Safe to delete after sync. |
| GPU eviction / OOM | Check `nvidia-smi`. Relaunching uses resume markers; failed step retries from scratch. |
| Tuning study has too many trials | If `tune_done` marker is missing but `study.db` was preserved, Optuna keeps adding trials. Delete `outputs/optuna/study.db` and the marker dir, relaunch. |

---

## What's NOT in these scripts

These are deliberately external — your concerns, not the script's:

- `pip install -e .`
- `wandb login`
- WandB notification settings (configured in the WandB web UI)
- GPU node booking / SLURM submission (wrap the `tmux new -d -s adl …` line in your scheduler's submission script if needed)
