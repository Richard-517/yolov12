"""Train torchvision Faster R-CNN (ResNet-50 FPN) on DsDPM66 (paper comparison C1).

torchvision expects boxes in (x1, y1, x2, y2) pixel format and labels where
0 is reserved for background. We read the YOLO-format labels (class 0-5,
normalized xywh) and convert.

Usage:
    python scripts/train_fasterrcnn.py \
        --data /root/cmdrill-yolov12/datasets/dsdpm66.yaml \
        --out  /root/cmdrill-yolov12/runs/cmdrill/C1_fasterrcnn \
        --epochs 12 --batch 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # fail-fast: ensure available before training


class DsDPM66Dataset(torch.utils.data.Dataset):
    """Reads DsDPM66 YOLO-format labels as torchvision-format tensors."""

    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted(p for p in self.images_dir.iterdir()
                                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tensor = F.to_tensor(img)

        lbl_path = self.labels_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # torchvision reserves 0 for background

        if not boxes:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, loader, device, epoch, print_every=50):
    model.train()
    t0 = time.time()
    running = {"loss_total": 0.0, "n": 0}
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running["loss_total"] += float(losses.item())
        running["n"] += 1

        if (i + 1) % print_every == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch}  batch {i+1}/{len(loader)}  loss={running['loss_total']/running['n']:.4f}  {elapsed:.1f}s")
    return running["loss_total"] / max(running["n"], 1)


@torch.no_grad()
def evaluate_map(model, loader, device, nc: int) -> dict:
    """Lightweight mAP50 proxy using torchmetrics-style per-class accumulation.

    For the paper we will re-evaluate C1 with the standard pycocotools flow
    (script to follow); this function is a sanity metric for the training loop.
    """
    model.eval()
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        metric.update(
            [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in preds],
            [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets],
        )
    result = metric.compute()
    return {
        "map_50": float(result["map_50"]),
        "map_50_95": float(result["map"]),
        "per_class_ap_50": [float(x) for x in result.get("map_per_class", [])],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="dsdpm66.yaml")
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.data.read_text())
    root = Path(cfg["path"])
    nc = cfg["nc"]

    train_ds = DsDPM66Dataset(root / "images" / "train", root / "labels" / "train")
    val_ds = DsDPM66Dataset(root / "images" / "val",   root / "labels" / "val")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=nc + 1).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # standard 1x schedule
    milestones = [int(args.epochs * 2 / 3), int(args.epochs * 8 / 9)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()
        metrics = evaluate_map(model, val_loader, device, nc)
        print(f"[epoch {epoch}] loss={loss:.4f}  mAP50={metrics['map_50']:.4f}  mAP50-95={metrics['map_50_95']:.4f}")
        history.append({"epoch": epoch, "loss": loss, **metrics})
        torch.save(model.state_dict(), args.out / f"epoch_{epoch}.pt")
        (args.out / "results.json").write_text(json.dumps(history, indent=2))

    torch.save(model.state_dict(), args.out / "last.pt")
    print(f"[done] weights + results.json under {args.out}")


if __name__ == "__main__":
    main()
