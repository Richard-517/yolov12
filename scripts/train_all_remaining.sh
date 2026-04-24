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

log() { echo "=== $(date +%H:%M:%S) :: $* ==="; }

log "START queue"

# Ablations (yolov12_ours env)
log "E_M1_seed42"          && bash scripts/train_ablation.sh  M1 42  2>&1 | tee "$LOG/E_M1_seed42.log"
log "E_M2_seed42"          && bash scripts/train_ablation.sh  M2 42  2>&1 | tee "$LOG/E_M2_seed42.log"
log "E_M3_seed42"          && bash scripts/train_ablation.sh  M3 42  2>&1 | tee "$LOG/E_M3_seed42.log"
log "E_M4_seed42"          && bash scripts/train_ablation.sh  M4 42  2>&1 | tee "$LOG/E_M4_seed42.log"

# Stability seeds
log "E0_yolov12s_seed123"  && bash scripts/train_baseline.sh      123 2>&1 | tee "$LOG/E0_s123.log"
log "E_M4_seed123"         && bash scripts/train_ablation.sh  M4 123 2>&1 | tee "$LOG/E_M4_seed123.log"

# YOLO-family comparisons (base env, ultralytics native)
log "C2_yolov8s"           && bash scripts/train_comparison.sh C2 42 2>&1 | tee "$LOG/C2.log"
log "C3_yolov11s"          && bash scripts/train_comparison.sh C3 42 2>&1 | tee "$LOG/C3.log"
log "C4_rtdetr_r18"        && bash scripts/train_comparison.sh C4 42 2>&1 | tee "$LOG/C4.log"

# Faster R-CNN (separate stack)
log "C1_fasterrcnn"        && conda activate yolov12_base && python scripts/train_fasterrcnn.py \
    --data ~/cmdrill-yolov12/datasets/dsdpm66.yaml \
    --out  ~/cmdrill-yolov12/runs/cmdrill/C1_fasterrcnn \
    --epochs 36 --batch 8 --seed 42 2>&1 | tee "$LOG/C1.log"

log "queue done"
