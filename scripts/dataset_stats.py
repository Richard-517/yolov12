"""Emit per-class statistics for DsDPM66 that feed paper Table 1.

Usage:
    python scripts/dataset_stats.py --yaml /root/cmdrill-yolov12/datasets/dsdpm66.yaml

Outputs a Markdown table with columns:
  class | #images | #instances | instances/image | median-bbox-area%
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def compute_stats(yaml_path: Path) -> None:
    cfg = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg["path"])
    names = cfg["names"]
    nc = cfg["nc"]

    per_class_images: dict[int, set[str]] = defaultdict(set)
    per_class_instances: dict[int, int] = defaultdict(int)
    per_class_areas: dict[int, list[float]] = defaultdict(list)
    total_images = 0

    for split in ("train", "val"):
        lbl_dir = root / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.iterdir():
            if lbl.suffix != ".txt":
                continue
            total_images += 1
            with lbl.open() as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])  # normalized w,h
                    per_class_images[cid].add(lbl.stem)
                    per_class_instances[cid] += 1
                    per_class_areas[cid].append(w * h * 100)  # as percentage

    print("| class | #images | #instances | avg inst/img | median bbox area (% of img) |")
    print("|:---|---:|---:|---:|---:|")
    for cid in range(nc):
        name = names[cid] if isinstance(names, dict) else names[cid]
        n_img = len(per_class_images[cid])
        n_inst = per_class_instances[cid]
        avg_inst = n_inst / n_img if n_img else 0.0
        areas = per_class_areas[cid]
        med_area = statistics.median(areas) if areas else 0.0
        print(f"| {name} | {n_img} | {n_inst} | {avg_inst:.2f} | {med_area:.3f}% |")
    print(f"\nTotal images (train+val): {total_images}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", type=Path, required=True, help="Path to dsdpm66.yaml")
    args = p.parse_args()
    compute_stats(args.yaml)


if __name__ == "__main__":
    main()
