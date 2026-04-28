#!/bin/bash
# Train an ablation variant (M1/M2/M3/M4) on DsDPM66.
# Usage: bash scripts/train_ablation.sh <M1|M2|M3|M4> [seed]
#
# M1: DS-A2C2f (configs/yolov12s_M1_dsa2c2f.yaml)
# M2: BiFPN+P2 (configs/yolov12s_M2_bifpn.yaml)
# M3: Inner-MPDIoU only, baseline arch (yolov12s.yaml + iou_type=inner_mpdiou)
# M4: full CMDrill-YOLOv12s (configs/yolov12s_M4_ours.yaml + iou_type=inner_mpdiou)

set -e

source /root/anaconda3/etc/profile.d/conda.sh
conda activate yolov12_ours
cd ~/cmdrill-yolov12/yolov12_ours

VARIANT=${1:?variant required: M1, M2, M3, or M4}
SEED=${2:-42}
DATA=${DATA:-"../datasets/dsdpm66.yaml"}
PROJECT=${PROJECT:-"../runs/cmdrill"}

case "${VARIANT}" in
    M1)  CFG="configs/yolov12s_M1_dsa2c2f.yaml"; IOU="ciou";          RATIO=0.7; WARMUP=0 ;;
    M2)  CFG="configs/yolov12s_M2_bifpn.yaml";   IOU="ciou";          RATIO=0.7; WARMUP=0 ;;
    M3)  CFG="yolov12s.yaml";                    IOU="inner_mpdiou";  RATIO=0.7; WARMUP=10 ;;
    M4)  CFG="configs/yolov12s_M4_ours.yaml";    IOU="inner_mpdiou";  RATIO=0.7; WARMUP=10 ;;
    # WIoU v3 backup variants (CMDrill-YOLOv12 Session 5).
    # If M3/M4 with Inner-MPDIoU don't improve, swap iou_type='wiou' (same arch).
    M3W) CFG="yolov12s.yaml";                    IOU="wiou";          RATIO=0.7; WARMUP=0 ;;
    M4W) CFG="configs/yolov12s_M4_ours.yaml";    IOU="wiou";          RATIO=0.7; WARMUP=0 ;;
    *)   echo "unknown variant: ${VARIANT}"; exit 1 ;;
esac

NAME="E_${VARIANT}_seed${SEED}"
LAST_PT="${PROJECT}/${NAME}/weights/last.pt"
RESUME=${RESUME:-no}   # yes | no   (default: no — fresh start unless explicitly resumed)

# Resume only when RESUME=yes is explicitly set AND last.pt exists.
# History note: previously default was "auto" (resume if last.pt exists), but this caused
# M1 to silently continue from Session 4's pre-fix weights (random-init bug DS-A2C2f) instead
# of fresh-training the Session 5 zero-init fix. Default-no protects validation of fixes.
# Caller asks for resume explicitly:  RESUME=yes bash scripts/train_ablation.sh M2 42
if [ "${RESUME}" = "yes" ] && [ -f "${LAST_PT}" ]; then
    echo "[${VARIANT}] RESUMING (RESUME=yes) from ${LAST_PT}"
    python - <<PY
from ultralytics import YOLO
YOLO("${LAST_PT}").train(resume=True)
PY
else
    echo "[${VARIANT}] FRESH start  cfg=${CFG}  iou=${IOU}  ratio=${RATIO}  warmup=${WARMUP}  seed=${SEED}"
    python - <<PY
from ultralytics import YOLO
model = YOLO("${CFG}")
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
    # CMDrill-YOLOv12 Inner-MPDIoU hyperparameters (ignored when iou_type=ciou):
    iou_type="${IOU}",
    inner_ratio=${RATIO},
    ciou_warmup_epochs=${WARMUP},
)
PY
fi
