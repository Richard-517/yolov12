"""Prepare the DsDPM 66 dataset for YOLOv12 training.

Each per-class upload (2026-04-24 verified) lays out:

    {class_name}/
    +-- images/{train,val}/*.jpg
    +-- YOLO_labels/{train,val}/*.txt
    +-- COCO_annotations/*.json

YOLO labels use class_id=0 (per-class single-class labeling) for 5 of the 6 classes.
The interaction class uses *two* internal ids (0 and 1), both of which we collapse
into the canonical "miner_drillpipe_interaction" id=5.

This script:
  1. Validates every class folder under --raw has the expected layout.
  2. For each image, rewrites the paired label with the canonical class id.
  3. Moves the image to {out}/images/{split}/ with a `{class}_` filename prefix,
     and the rewritten label to {out}/labels/{split}/.
  4. Emits {out}/../dsdpm66.yaml.

Using `move` (not copy) keeps peak disk usage flat.

Usage:
    python scripts/prepare_dataset.py --raw /root/cmdrill-yolov12/datasets/raw --out /root/cmdrill-yolov12/datasets/DsDPM66
    python scripts/prepare_dataset.py --raw ... --out ... --dry-run    # plan only, no moves
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CANONICAL_CLASSES = [
    "coal_miner",
    "compressed_oxygen_self_rescuer",
    "mining_helmet",
    "drill_pipe",
    "drill_rig",
    "miner_drillpipe_interaction",
]

# Map the upstream per-class folder name to this project's canonical id.
# The interaction folder uses 2 internal ids (0 and 1), both -> 5.
FOLDER_TO_CANONICAL_ID = {
    "coal_miner": 0,
    "compressed_oxygen_self_rescuer": 1,
    "mining_helmet": 2,
    "drill_pipe": 3,
    "drill_rig": 4,
    "interaction_between_miner_and_drill_pipe": 5,
}


def rewrite_label_text(label_text: str, target_id: int) -> str:
    """Replace every line's leading class_id with `target_id`, keep bbox fields intact."""
    out_lines: list[str] = []
    for raw_line in label_text.splitlines():
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        parts[0] = str(target_id)
        out_lines.append(" ".join(parts))
    if not out_lines:
        return ""
    return "\n".join(out_lines) + "\n"


def process(raw: Path, out: Path, dry_run: bool) -> None:
    assert raw.is_dir(), f"{raw} not a directory"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    per_class_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "inst": 0})
    skipped: list[str] = []

    for folder_name, target_id in FOLDER_TO_CANONICAL_ID.items():
        folder = raw / folder_name
        if not folder.is_dir():
            print(f"[warn] missing class folder: {folder_name} (skipping)")
            continue
        for split in ("train", "val"):
            img_dir = folder / "images" / split
            lbl_dir = folder / "YOLO_labels" / split
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                print(f"[warn] missing {img_dir} or {lbl_dir}, skipping split")
                continue

            imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            print(f"[info] {folder_name}/{split}: {len(imgs)} images -> canonical id {target_id}")

            for img in imgs:
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    skipped.append(f"no-label: {img}")
                    continue

                new_name = f"{folder_name}_{img.stem}"
                dst_img = out / "images" / split / f"{new_name}{img.suffix}"
                dst_lbl = out / "labels" / split / f"{new_name}.txt"

                label_text = lbl.read_text()
                n_lines = sum(1 for line in label_text.splitlines() if len(line.split()) >= 5)
                per_class_totals[folder_name]["inst"] += n_lines

                if dry_run:
                    continue

                if not dst_img.exists():
                    shutil.move(str(img), str(dst_img))
                if not dst_lbl.exists():
                    dst_lbl.write_text(rewrite_label_text(label_text, target_id))

                per_class_totals[folder_name][split] += 1

    print("\n=== summary ===")
    for folder, counts in per_class_totals.items():
        print(f"{folder:45s}  train={counts['train']:>6d}  val={counts['val']:>6d}  inst={counts['inst']:>6d}")
    if skipped:
        print(f"\n[warn] skipped {len(skipped)} files, first 5: {skipped[:5]}")

    if dry_run:
        print("\n[dry-run] no files actually moved.")
        return

    # dsdpm66.yaml sits next to the DsDPM66 data directory.
    yaml_path = out.parent / "dsdpm66.yaml"
    yaml_body = [
        f"path: {out}",
        "train: images/train",
        "val: images/val",
        "nc: 6",
        "names:",
    ]
    for i, name in enumerate(CANONICAL_CLASSES):
        yaml_body.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(yaml_body) + "\n")
    print(f"\n[done] wrote {yaml_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True,
                   help="Directory containing the 6 per-class subfolders.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output canonical DsDPM66 layout.")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and report but do not move files.")
    args = p.parse_args()
    process(args.raw.resolve(), args.out.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
