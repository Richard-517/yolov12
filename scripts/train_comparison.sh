#!/bin/bash
# Train comparison baselines C1-C4 on DsDPM66.
# Usage: bash scripts/train_comparison.sh <C2|C3|C4>
#   C1 Faster R-CNN: use scripts/train_fasterrcnn.py (torchvision-based, separate)
#   C2 YOLOv8s:      ultralytics direct
#   C3 YOLOv11s:     ultralytics direct
#   C4 RT-DETR-R18:  ultralytics native (rtdetr-r18.yaml)

set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate yolov12_base   # use base env (ultralytics ships v8/v11/rtdetr YAMLs)
cd ~/cmdrill-yolov12/yolov12_baseline

VARIANT=${1:?variant required: C2, C3, or C4}
SEED=${2:-42}
DATA=${DATA:-"../datasets/dsdpm66.yaml"}
PROJECT=${PROJECT:-"../runs/cmdrill"}

case "${VARIANT}" in
    C2) CFG="yolov8s.yaml";    NAME="C2_yolov8s" ;;
    C3) CFG="yolov11s.yaml";   NAME="C3_yolov11s" ;;
    C4) CFG="rtdetr-r18.yaml"; NAME="C4_rtdetr_r18" ;;
    *)  echo "unknown variant: ${VARIANT}"; exit 1 ;;
esac

echo "[${VARIANT}] cfg=${CFG}  name=${NAME}_seed${SEED}"

if [ "${VARIANT}" = "C4" ]; then
    # RT-DETR's official recipe differs; use ultralytics defaults that ship with rtdetr-r18.yaml
    python - <<PY
from ultralytics import RTDETR
model = RTDETR("${CFG}")
model.train(
    data="${DATA}",
    epochs=300, batch=32, imgsz=640,
    device="0", workers=8, patience=50, amp=True, seed=${SEED},
    project="${PROJECT}", name="${NAME}_seed${SEED}",
    plots=True,
)
PY
else
    python - <<PY
from ultralytics import YOLO
model = YOLO("${CFG}")
model.train(
    data="${DATA}",
    epochs=300, batch=64, imgsz=640,
    optimizer="SGD", lr0=0.01, lrf=0.01,
    momentum=0.937, weight_decay=5e-4,
    warmup_epochs=3.0, mosaic=1.0, close_mosaic=10,
    device="0", workers=8, patience=50, amp=True, seed=${SEED},
    project="${PROJECT}", name="${NAME}_seed${SEED}",
    plots=True,
)
PY
fi
