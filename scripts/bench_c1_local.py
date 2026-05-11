"""Bench C1 Faster R-CNN R50 best epoch locally (4090 fp16 batch=1).

Distinct from bench_comparison_local.py because Faster R-CNN is a torchvision model,
not ultralytics. Loads the state_dict from epoch_25.pt into fasterrcnn_resnet50_fpn.
"""
from __future__ import annotations
import csv, time
from pathlib import Path
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

CKPT = "G:/YOLO/runs_local/C1_fasterrcnn/epoch_25.pt"
NC = 6
IMGSZ, WARMUP, ITERS = 640, 30, 300
DEVICE = torch.cuda.get_device_name(0)

print("[bench] C1_fasterrcnn_seed42 (epoch 25)")
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=NC + 1)
sd = torch.load(CKPT, map_location="cpu")
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
elif isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
model.load_state_dict(sd)
model = model.cuda().eval()

n_layers = len(list(model.modules()))
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  layers={n_layers} params={n_params:.3f}M")

try:
    from thop import profile
    x_thop = torch.randn(1, 3, IMGSZ, IMGSZ).cuda()
    flops, _ = profile(model, inputs=(x_thop,), verbose=False)
    gflops = flops / 1e9
    print(f"  GFLOPs={gflops:.2f} (via thop)")
except Exception as e:
    gflops = float("nan")
    print(f"  GFLOPs n/a: {e}")

x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda")
with torch.no_grad():
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
fps = ITERS / dt
print(f"  fp32: {fps:.2f} FPS  {1000/fps:.3f} ms/img")

fps_csv = Path("G:/YOLO/results_local/fps_4090.csv")
rows = list(csv.DictReader(fps_csv.open()))
new = {"run": "C1_fasterrcnn_seed42", "device": DEVICE, "imgsz": IMGSZ,
       "precision": "fp32", "batch": 1, "fps": round(fps, 2), "ms_per_img": round(1000/fps, 3)}
if not any(r["run"] == "C1_fasterrcnn_seed42" for r in rows):
    rows.append(new)
    with fps_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

pf_csv = Path("G:/YOLO/results_local/params_flops.csv")
pf_rows = list(csv.DictReader(pf_csv.open()))
new_pf = {"run": "C1_fasterrcnn_seed42", "layers": n_layers,
          "params_M": round(n_params, 3), "GFLOPs": round(gflops, 2) if gflops == gflops else "nan"}
if not any(r["run"] == "C1_fasterrcnn_seed42" for r in pf_rows):
    pf_rows.append(new_pf)
    with pf_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pf_rows[0].keys())
        w.writeheader(); w.writerows(pf_rows)

print("[done] appended C1 row")
