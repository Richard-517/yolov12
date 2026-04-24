"""Batch FPS benchmark on local RTX 4090.

Iterates a `runs_local/` directory of completed training outputs, loads each
best.pt, measures batch=1 fp16 latency (1000 iters, 50 warmup, no NMS), and
emits a CSV. This is what gets dropped into paper Table 3's FPS column.

Usage (from yolov12_ours root, in the local venv):
    .venv/Scripts/python.exe scripts/bench_all_local.py \
        --runs /g/YOLO/runs_local \
        --out  /g/YOLO/results_local/fps_4090.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from ultralytics import YOLO


def bench_one(weights: Path, imgsz: int, warmup: int, iters: int) -> float:
    model = YOLO(str(weights))
    net = model.model.cuda().half().eval()
    x = torch.randn(1, 3, imgsz, imgsz, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return iters / elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True,
                   help="Directory containing <run_name>/weights/best.pt trees (fetched from server).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.get_device_name(0)
    rows = []
    for run_dir in sorted(args.runs.iterdir()):
        w = run_dir / "weights" / "best.pt"
        if not w.exists():
            print(f"[skip] {run_dir.name} (no best.pt)")
            continue
        print(f"[bench] {run_dir.name}")
        try:
            fps = bench_one(w, args.imgsz, args.warmup, args.iters)
            rows.append({
                "run": run_dir.name, "device": device,
                "imgsz": args.imgsz, "precision": "fp16", "batch": 1,
                "fps": round(fps, 2), "ms_per_img": round(1000 / fps, 3),
            })
            print(f"   {fps:.2f} FPS   {1000/fps:.3f} ms/img")
        except Exception as e:
            print(f"   [error] {e}")

    # Write CSV
    if rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"[done] wrote {args.out} ({len(rows)} rows)")
    else:
        print("[warn] no successful benches")


if __name__ == "__main__":
    main()
