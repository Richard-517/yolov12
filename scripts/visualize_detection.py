"""Generate qualitative comparison figures for CMDrill-YOLOv12 paper.

Runs inference with both the baseline and CMDrill-YOLOv12s weights on the same
images and saves side-by-side annotated comparisons.

Usage:
    python scripts/visualize_detection.py \
        --baseline runs/cmdrill/E0_yolov12s_seed42/weights/best.pt \
        --ours     runs/cmdrill/E_M4_seed42/weights/best.pt \
        --images   datasets/DsDPM66/images/val \
        --out      runs/cmdrill/figures \
        --per-category 2
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SCENARIO_CLASSES = {
    "small_object": {1, 2},      # compressed_oxygen_self_rescuer, mining_helmet
    "elongated":    {3, 4},      # drill_pipe, drill_rig
    "interaction":  {5},         # miner_drillpipe_interaction
    "multi_class":  None,        # any image with >= 2 distinct labels
}


def pick_representative(images_dir: Path, labels_dir: Path,
                         per_category: int = 2, seed: int = 42) -> dict[str, list[Path]]:
    random.seed(seed)
    by_scenario: dict[str, list[Path]] = defaultdict(list)
    for lbl in labels_dir.iterdir():
        if lbl.suffix != ".txt":
            continue
        with lbl.open() as f:
            cids = {int(line.split()[0]) for line in f if line.split()}
        if not cids:
            continue
        img = images_dir / (lbl.stem + ".jpg")
        if not img.exists():
            continue
        if cids <= SCENARIO_CLASSES["small_object"]:
            by_scenario["small_object"].append(img)
        if cids <= SCENARIO_CLASSES["elongated"]:
            by_scenario["elongated"].append(img)
        if cids <= SCENARIO_CLASSES["interaction"]:
            by_scenario["interaction"].append(img)
        if len(cids) >= 2:
            by_scenario["multi_class"].append(img)

    picks = {}
    for k, candidates in by_scenario.items():
        random.shuffle(candidates)
        picks[k] = candidates[:per_category]
    return picks


def annotate(model: YOLO, img_path: Path, conf: float = 0.25) -> np.ndarray:
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    return res.plot()  # BGR np.ndarray with boxes drawn


def side_by_side(base_img: np.ndarray, ours_img: np.ndarray, labels=("Baseline", "CMDrill-YOLOv12s")) -> np.ndarray:
    h = max(base_img.shape[0], ours_img.shape[0])
    base_img = cv2.copyMakeBorder(base_img, 0, h - base_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ours_img = cv2.copyMakeBorder(ours_img, 0, h - ours_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    combined = np.hstack([base_img, ours_img])
    for i, lbl in enumerate(labels):
        x = 20 + i * base_img.shape[1]
        cv2.rectangle(combined, (x - 10, 10), (x + 260, 50), (255, 255, 255), -1)
        cv2.putText(combined, lbl, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return combined


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None,
                   help="Labels directory matching --images (default: <images>/../labels/<split>)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-category", type=int, default=2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    labels_dir = args.labels
    if labels_dir is None:
        # images/val/ → labels/val/
        split = args.images.name
        labels_dir = args.images.parent.parent / "labels" / split

    args.out.mkdir(parents=True, exist_ok=True)
    m_base = YOLO(str(args.baseline))
    m_ours = YOLO(str(args.ours))

    picks = pick_representative(args.images, labels_dir, args.per_category, args.seed)
    for scenario, imgs in picks.items():
        for i, img in enumerate(imgs):
            a = annotate(m_base, img, args.conf)
            b = annotate(m_ours, img, args.conf)
            fig = side_by_side(a, b)
            out_path = args.out / f"{scenario}_{i}_{img.stem}.jpg"
            cv2.imwrite(str(out_path), fig)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
