"""Measure inference FPS on the current GPU (H100 NVL for CMDrill-YOLOv12).

Usage:
    python scripts/benchmark_fps.py --weights runs/cmdrill/E0_yolov12s_seed42/weights/best.pt
    python scripts/benchmark_fps.py --cfg yolov12s.yaml    # structure-only, random weights

Reports batch=1 FP16 latency (excluding NMS) averaged over 1000 runs after 50 warmups.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ultralytics import YOLO


def bench(model, warmup: int = 50, iters: int = 1000, imgsz: int = 640) -> float:
    m = model.model.cuda().half().eval()
    x = torch.randn(1, 3, imgsz, imgsz, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return iters / elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=None, help="Trained .pt checkpoint")
    p.add_argument("--cfg", type=str, default=None, help="YAML config (random init, structure-only FPS)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()

    if args.weights:
        model = YOLO(str(args.weights))
        label = args.weights.name
    elif args.cfg:
        model = YOLO(args.cfg)
        label = args.cfg
    else:
        raise SystemExit("one of --weights or --cfg is required")

    fps = bench(model, args.warmup, args.iters, args.imgsz)
    device_name = torch.cuda.get_device_name(0)
    print(f"{label}\t{device_name}\tbatch=1 fp16\t{fps:.2f} FPS  ({1000/fps:.3f} ms/img)  excluding NMS")


if __name__ == "__main__":
    main()
