#!/bin/bash
# Poll for train_E0 tmux; when it exits, check results.csv for completion,
# and if successful, kick off the remaining queue.
#
# Deploy in a separate tmux session:
#   tmux new -d -s watchdog 'bash ~/cmdrill-yolov12/yolov12_ours/scripts/watchdog.sh 2>&1 | tee ~/cmdrill-yolov12/logs/watchdog.log'

set -u
LOG_DIR=~/cmdrill-yolov12/logs
mkdir -p "$LOG_DIR"

E0_CSV=~/cmdrill-yolov12/runs/cmdrill/E0_yolov12s_seed42/results.csv

echo "[$(date +%H:%M:%S)] watchdog START, polling for E0 completion"

while tmux has-session -t train_E0 2>/dev/null; do
    sleep 600   # poll every 10 min; E0 is 20+ hours so no rush
done

echo "[$(date +%H:%M:%S)] train_E0 tmux gone; inspecting E0 outcome"

if [ ! -f "$E0_CSV" ]; then
    echo "[$(date +%H:%M:%S)] FATAL: $E0_CSV missing. Aborting queue."
    exit 1
fi

LAST_EPOCH=$(tail -1 "$E0_CSV" | cut -d, -f1)
echo "[$(date +%H:%M:%S)] E0 last recorded epoch = ${LAST_EPOCH}"

# Accept either fully completed (300) OR early-stop via patience (>=50, shows convergence)
if [ -z "$LAST_EPOCH" ] || [ "$LAST_EPOCH" = "epoch" ]; then
    echo "[$(date +%H:%M:%S)] FATAL: couldn't parse epoch count. Aborting queue."
    exit 1
fi

if [ "$LAST_EPOCH" -lt 50 ]; then
    echo "[$(date +%H:%M:%S)] WARN: E0 exited after only ${LAST_EPOCH} epochs (looks like crash, not convergence)."
    echo "[$(date +%H:%M:%S)]       NOT kicking off queue; user should investigate."
    exit 1
fi

echo "[$(date +%H:%M:%S)] E0 completed ${LAST_EPOCH} epochs. Starting remaining queue."
tmux new -d -s train_queue \
    "bash ~/cmdrill-yolov12/yolov12_ours/scripts/train_all_remaining.sh 2>&1 | tee $LOG_DIR/queue.log"

echo "[$(date +%H:%M:%S)] watchdog DONE (train_queue tmux launched)"
