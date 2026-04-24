#!/bin/bash
# Train YOLOv12s baseline (experiment E0) on DsDPM66.
# Usage: bash scripts/train_baseline.sh [seed]   (seed defaults to 42)

set -e

source /root/anaconda3/etc/profile.d/conda.sh
conda activate yolov12_base
cd ~/cmdrill-yolov12/yolov12_baseline

SEED=${1:-42}
DATA=${DATA:-"../datasets/dsdpm66.yaml"}
PROJECT=${PROJECT:-"../runs/cmdrill"}
NAME="E0_yolov12s_seed${SEED}"

echo "[E0] seed=${SEED}  data=${DATA}  name=${NAME}"
nvidia-smi --query-gpu=memory.free --format=csv

python - <<PY
from ultralytics import YOLO
model = YOLO("yolov12s.yaml")
model.train(
    data="${DATA}",
    epochs=300, batch=64, imgsz=640,
    optimizer="SGD", lr0=0.01, lrf=0.01,
    momentum=0.937, weight_decay=5e-4,
    warmup_epochs=3.0, mosaic=1.0, close_mosaic=10,
    device="0", workers=8, patience=50,
    amp=True, seed=${SEED},
    save=True, save_period=10,
    project="${PROJECT}", name="${NAME}",
    plots=True,
)
PY
