#!/bin/bash
# Fetch best.pt weights from the H100 training server to local machine.
# Run inside G:\YOLO\ (git-bash). Requires the same SSH identity as prior sessions.
#
# Usage: bash scripts/fetch_weights.sh [runs_local_dir]
#   runs_local_dir defaults to ../runs_local relative to yolov12_ours

set -e

SERVER="root@20.62.104.255"
PORT=6168
REMOTE_BASE="~/cmdrill-yolov12/runs/cmdrill"
LOCAL_BASE=${1:-"/g/YOLO/runs_local"}

# Map: run name on server -> local subdir name (kept identical)
RUNS=(
  "E0_yolov12s_seed42"
  "E0_yolov12s_seed123"
  "E_M1_seed42"
  "E_M2_seed42"
  "E_M3_seed42"
  "E_M4_seed42"
  "E_M4_seed123"
  "C2_yolov8s_seed42"
  "C3_yolo11s_seed42"
  "C4_rtdetr_l_seed42"
)

mkdir -p "$LOCAL_BASE"

for name in "${RUNS[@]}"; do
  echo "=== $name ==="
  # Check if remote weights/best.pt exists (only for finished runs)
  if ! ssh -p $PORT $SERVER "test -f $REMOTE_BASE/$name/weights/best.pt"; then
    echo "  [skip] $name not finished yet (no best.pt)"
    continue
  fi
  mkdir -p "$LOCAL_BASE/$name/weights"
  scp -P $PORT "$SERVER:$REMOTE_BASE/$name/weights/best.pt"   "$LOCAL_BASE/$name/weights/" 2>&1 | tail -2
  # Also grab results.csv + args.yaml for reference
  scp -P $PORT "$SERVER:$REMOTE_BASE/$name/results.csv"       "$LOCAL_BASE/$name/" 2>&1 | tail -1
  scp -P $PORT "$SERVER:$REMOTE_BASE/$name/args.yaml"         "$LOCAL_BASE/$name/" 2>&1 | tail -1
done

echo "=== done ==="
echo "Local runs: $LOCAL_BASE"
ls "$LOCAL_BASE"
