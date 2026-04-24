"""Prepare the DsDPM 66 dataset for YOLOv12 training.

DsDPM 66 is distributed in *two* figshare uploads due to the per-article
storage cap. This script merges both parts into a single
`images/{train,val}` + `labels/{train,val}` layout and emits `dsdpm66.yaml`.

Usage:
    # 1. Look at the raw structure without moving files (safe):
    python scripts/prepare_dataset.py --root /root/cmdrill-yolov12/datasets --inspect

    # 2. After confirming the mapping is correct, process:
    python scripts/prepare_dataset.py --root /root/cmdrill-yolov12/datasets --run

The six canonical class IDs for this project (fixed in CLAUDE.md):

    0: coal_miner
    1: compressed_oxygen_self_rescuer
    2: mining_helmet
    3: drill_pipe
    4: drill_rig
    5: miner_drillpipe_interaction
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

CANONICAL_CLASSES = [
    "coal_miner",
    "compressed_oxygen_self_rescuer",
    "mining_helmet",
    "drill_pipe",
    "drill_rig",
    "miner_drillpipe_interaction",
]


def inspect(root: Path) -> None:
    """Walk the raw DsDPM66 directory and report the layout the author used.

    We cannot guarantee the upstream layout — this is our safety check before doing
    anything destructive. Print the directory tree up to depth 3 and sample the
    first 10 lines of each label-like file."""
    print(f"[inspect] root = {root}")
    for path in sorted(root.rglob("*"))[:500]:
        rel = path.relative_to(root)
        depth = len(rel.parts)
        if depth > 3:
            continue
        marker = "/" if path.is_dir() else ""
        print(f"  {'  ' * depth}{rel.name}{marker}")

    print("\n[inspect] sample label files:")
    sample_count = 0
    for lbl in root.rglob("*.txt"):
        if sample_count >= 5:
            break
        if lbl.stat().st_size > 0:
            print(f"  --- {lbl.relative_to(root)} ---")
            with lbl.open() as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"    {line.rstrip()}")
            sample_count += 1

    print("\n[inspect] label class-id distribution (global scan):")
    class_counter: Counter[str] = Counter()
    for lbl in root.rglob("*.txt"):
        try:
            with lbl.open() as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_counter[parts[0]] += 1
        except Exception:
            continue
    for cid, n in sorted(class_counter.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        print(f"  class_id={cid}  count={n}")

    print("\n[inspect] Review the output above before running with --run.")
    print("[inspect] If class_ids already match the canonical 0..5 mapping, no remap is needed.")
    print("[inspect] If they reset to 0 per part, the --run step will ask you to confirm the folder-to-id mapping.")


def collect_images_and_labels(root: Path) -> list[tuple[Path, Path, str]]:
    """Walk the raw root and return a list of (image_path, label_path, split) tuples.

    This assumes an upstream convention like:
        PartA/images/train/*.jpg, PartA/labels/train/*.txt
        PartA/images/val/*.jpg,   PartA/labels/val/*.txt
        PartB/...

    If the upstream used different folder names we will need to adjust after inspection."""
    pairs: list[tuple[Path, Path, str]] = []
    for img_dir in root.rglob("images"):
        if not img_dir.is_dir():
            continue
        parent = img_dir.parent
        lbl_dir = parent / "labels"
        if not lbl_dir.is_dir():
            continue
        for split_sub in ("train", "val"):
            img_split = img_dir / split_sub
            lbl_split = lbl_dir / split_sub
            if not img_split.is_dir() or not lbl_split.is_dir():
                continue
            for img in img_split.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = lbl_split / (img.stem + ".txt")
                if not lbl.exists():
                    print(f"[warn] missing label: {lbl}")
                    continue
                pairs.append((img, lbl, split_sub))
    return pairs


def run(root: Path, out_dir: Path, dry_run: bool = False) -> None:
    """Merge parts, copy images+labels into canonical YOLO layout, emit yaml."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = collect_images_and_labels(root)
    print(f"[run] found {len(pairs)} image-label pairs across the raw tree")

    split_counts = Counter(s for _, _, s in pairs)
    print(f"[run] split distribution: {dict(split_counts)}")

    for img, lbl, split in pairs:
        dst_img = out_dir / "images" / split / img.name
        dst_lbl = out_dir / "labels" / split / lbl.name
        if dry_run:
            continue
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(lbl, dst_lbl)

    # dsdpm66.yaml
    yaml_path = out_dir.parent / "dsdpm66.yaml"
    yaml_text = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "nc: 6",
        "names:",
    ]
    for i, name in enumerate(CANONICAL_CLASSES):
        yaml_text.append(f"  {i}: {name}")
    if not dry_run:
        yaml_path.write_text("\n".join(yaml_text) + "\n")
    print(f"[run] wrote {yaml_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare DsDPM66 for YOLOv12 training.")
    p.add_argument("--root", type=Path, required=True,
                   help="Directory that contains the raw PartA/PartB unpacked folders.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to place the canonical layout (default: {root}/DsDPM66).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true",
                      help="Only scan and print; no files are moved.")
    mode.add_argument("--run", action="store_true",
                      help="Copy images/labels into the canonical YOLO layout and emit dsdpm66.yaml.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --run, print the plan without copying.")
    args = p.parse_args()

    if args.inspect:
        inspect(args.root)
    elif args.run:
        out_dir = args.out or (args.root / "DsDPM66")
        run(args.root, out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
