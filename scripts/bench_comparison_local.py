"""Bench C2/C3/C4 best.pt locally (4090 fp16 batch=1) + record params/FLOPs.

C4 (RT-DETR-L) requires the RTDETR wrapper class, not YOLO.
"""
from __future__ import annotations
import csv, time
from pathlib import Path
import torch
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_num_params, get_flops

RUNS = [
    ("C2_yolov8s_seed42",  "G:/YOLO/runs_local/C2_yolov8s_seed42/weights/best.pt",  YOLO),
    ("C3_yolo11s_seed42",  "G:/YOLO/runs_local/C3_yolo11s_seed42/weights/best.pt",  YOLO),
    ("C4_rtdetr_l_seed42", "G:/YOLO/runs_local/C4_rtdetr_l_seed42/weights/best.pt", RTDETR),
]
IMGSZ, WARMUP, ITERS = 640, 50, 1000
DEVICE = torch.cuda.get_device_name(0)

fps_rows, pf_rows = [], []
for name, w, cls in RUNS:
    print(f"\n[bench] {name}")
    m = cls(w)
    net = m.model.cuda().half().eval()
    n_layers = len(list(net.modules()))
    n_params = get_num_params(net) / 1e6
    try:
        gflops = get_flops(net, IMGSZ)
    except Exception as e:
        gflops = float("nan"); print(f"  FLOPs n/a: {e}")
    pf_rows.append({"run": name, "layers": n_layers, "params_M": round(n_params, 3), "GFLOPs": round(gflops, 2)})
    print(f"  layers={n_layers} params={n_params:.3f}M GFLOPs={gflops:.2f}")
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(WARMUP): net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS): net(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    fps = ITERS / dt
    fps_rows.append({"run": name, "device": DEVICE, "imgsz": IMGSZ, "precision": "fp16", "batch": 1,
                     "fps": round(fps, 2), "ms_per_img": round(1000/fps, 3)})
    print(f"  {fps:.2f} FPS  {1000/fps:.3f} ms/img")

# Append to existing CSVs
fps_csv = Path("G:/YOLO/results_local/fps_4090.csv")
existing_fps = list(csv.DictReader(fps_csv.open())) if fps_csv.exists() else []
existing_fps_runs = {r["run"] for r in existing_fps}
all_fps = existing_fps + [r for r in fps_rows if r["run"] not in existing_fps_runs]
with fps_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_fps[0].keys())
    w.writeheader(); w.writerows(all_fps)

pf_csv = Path("G:/YOLO/results_local/params_flops.csv")
existing_pf = list(csv.DictReader(pf_csv.open())) if pf_csv.exists() else []
existing_pf_runs = {r["run"] for r in existing_pf}
all_pf = existing_pf + [r for r in pf_rows if r["run"] not in existing_pf_runs]
with pf_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_pf[0].keys())
    w.writeheader(); w.writerows(all_pf)

print(f"\n[done] appended {len(fps_rows)} fps rows + {len(pf_rows)} params rows")
