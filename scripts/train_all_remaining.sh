#!/bin/bash
# Sequential driver for the remaining experiments after E0_s42 completes.
# Run inside a fresh tmux session:
#   tmux new -d -s train_queue 'bash scripts/train_all_remaining.sh 2>&1 | tee ~/cmdrill-yolov12/logs/queue.log'
#
# Order: E1 -> E2 -> E3 -> E4_s42 -> E0_s123 -> E4_s123 -> C2 -> C3 -> C4 -> C1

set -e
source /root/anaconda3/etc/profile.d/conda.sh
cd ~/cmdrill-yolov12/yolov12_ours
LOG=~/cmdrill-yolov12/logs
RUNS=~/cmdrill-yolov12/runs/cmdrill
BASELINE_RUN="${RUNS}/E0_yolov12s_seed42"
# mAP@0.5 regression tolerance — if an ablation is this many points below baseline
# (default 0.005 = 0.5pp, within typical seed-noise), treat as regression and stop.
# Override with: TOLERANCE=0.01 bash scripts/train_all_remaining.sh
TOLERANCE="${TOLERANCE:-0.005}"

log() { echo "=== $(date +%H:%M:%S) :: $* ==="; }
check_regression() {
    local run_name="$1"
    log "check vs E0 baseline (tolerance=${TOLERANCE})"
    python scripts/check_improvement.py \
        --exp "${RUNS}/${run_name}" \
        --baseline "${BASELINE_RUN}" \
        --tolerance "${TOLERANCE}"
}

log "START queue"
log "git pull (pick up any last-minute fixes)"
git stash 2>/dev/null || true
git pull origin cmdrill-dev 2>&1 | tail -5
chmod +x scripts/*.sh

# Ablations (yolov12_ours env) — each gated by a *best-epoch* regression check against E0.
# Order rationale (Session 5 v3, after M3 PASSED + M1 contamination kill):
#   M2 SKIPPED — paused at e82, BiFPN+P2 alone underperformed; preserved for optional resume
#                via   RESUME=yes bash scripts/train_ablation.sh M2 42
#   M3 SKIPPED — already PASSED on Session 5 first run: best=0.6560 @ e228 (Δ=+0.0008 vs
#                E0=0.6552), patience-stopped e278. E_M3_seed42 dir kept.
#
#   1. M1 first  — DS-A2C2f only with zero-init ds_fuse fix. Fresh start (default RESUME=no).
#                  Old E_M1_seed42 archived as _E_M1_seed42_KILLED_*_pre_session5_resumed.
#   2. M4        — full combined method (M1 fixes + M2 BiFPN+P2 + M3 Inner-MPDIoU).
#                  Old E_M4_seed42 archived before resuming this queue.
log "E_M1_seed42"          && bash scripts/train_ablation.sh  M1 42  2>&1 | tee "$LOG/E_M1_seed42.log"
check_regression  "E_M1_seed42"

log "E_M4_seed42"          && bash scripts/train_ablation.sh  M4 42  2>&1 | tee "$LOG/E_M4_seed42.log"
check_regression  "E_M4_seed42"

# === Session 6 paper pivot (self-rescuer compliance detection) ===
# M5 = M1 architecture (DS-A2C2f) + M3 loss (Inner-MPDIoU), no BiFPN+P2.
# Per per-class analysis: M1 alone +0.61pp on self_rescuer, M3 alone +0.51pp on self_rescuer,
# but M4 (combined with BiFPN+P2) drops both due to P2 head gradient dilution.
# M5 keeps the two improvements that compound and drops the one that interferes.
# Expected: self_rescuer +0.7-1.0pp, overall ±0.2pp. This is the paper's recommended config.
log "E_M5_seed42"          && bash scripts/train_ablation.sh  M5 42  2>&1 | tee "$LOG/E_M5_seed42.log"
check_regression  "E_M5_seed42"

# Stability seeds
log "E0_yolov12s_seed123"  && bash scripts/train_baseline.sh      123 2>&1 | tee "$LOG/E0_s123.log"
# E_M4_seed123 SKIPPED (Session 6 pivot): M4 is not the paper's main result, it's used in
# the ablation as the negative-interaction case. Stability seed instead validates M5 below.
log "E_M5_seed123"         && bash scripts/train_ablation.sh  M5 123 2>&1 | tee "$LOG/E_M5_seed123.log"

# YOLO-family comparisons (base env, ultralytics native).
# Note: ultralytics ships yolo11s.yaml (not yolov11s.yaml) and rtdetr-l.yaml
# (no R18 variant; we use the closest `l` variant, naming the run to reflect that).
log "C2_yolov8s"           && bash scripts/train_comparison.sh C2 42 2>&1 | tee "$LOG/C2.log"
log "C3_yolo11s"           && bash scripts/train_comparison.sh C3 42 2>&1 | tee "$LOG/C3.log"
log "C4_rtdetr_l"          && bash scripts/train_comparison.sh C4 42 2>&1 | tee "$LOG/C4.log"

# Faster R-CNN (separate stack)
log "C1_fasterrcnn"        && conda activate yolov12_base && python scripts/train_fasterrcnn.py \
    --data ~/cmdrill-yolov12/datasets/dsdpm66.yaml \
    --out  ~/cmdrill-yolov12/runs/cmdrill/C1_fasterrcnn \
    --epochs 36 --batch 8 --seed 42 2>&1 | tee "$LOG/C1.log"

log "queue done"
