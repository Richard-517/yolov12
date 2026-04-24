"""Aggregate results from all CMDrill-YOLOv12 runs into paper tables.

Reads runs/cmdrill/*/results.csv and each run's weights/best.pt → val.json and
emits the CSVs referenced in paper tables 1-4.

Usage:
    python scripts/collect_results.py --runs /root/cmdrill-yolov12/runs/cmdrill \
        --data /root/cmdrill-yolov12/datasets/dsdpm66.yaml \
        --out /root/cmdrill-yolov12/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from ultralytics import YOLO

EXPERIMENT_ORDER = [
    "E0_yolov12s_seed42",
    "E0_yolov12s_seed123",
    "E_M1_seed42",
    "E_M2_seed42",
    "E_M3_seed42",
    "E_M4_seed42",
    "E_M4_seed123",
    "C1_fasterrcnn",
    "C2_yolov8s",
    "C3_yolov11s",
    "C4_rtdetr_r18",
]

HUMAN_LABEL = {
    "E0_yolov12s_seed42": "M0 (YOLOv12s baseline, s42)",
    "E0_yolov12s_seed123": "M0 (YOLOv12s baseline, s123)",
    "E_M1_seed42": "M1 (DS-A2C2f)",
    "E_M2_seed42": "M2 (BiFPN+P2)",
    "E_M3_seed42": "M3 (Inner-MPDIoU)",
    "E_M4_seed42": "M4 (CMDrill-YOLOv12s, s42)",
    "E_M4_seed123": "M4 (CMDrill-YOLOv12s, s123)",
    "C1_fasterrcnn": "Faster R-CNN (R50)",
    "C2_yolov8s": "YOLOv8s",
    "C3_yolov11s": "YOLOv11s",
    "C4_rtdetr_r18": "RT-DETR-R18",
}


def load_last_epoch(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    return df.iloc[-1].to_dict()


def model_params_flops(run_dir: Path) -> tuple[float, float] | None:
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        return None
    try:
        model = YOLO(str(best))
        nparam = sum(p.numel() for p in model.model.parameters()) / 1e6
        # FLOPs via model.info
        info = model.info(detailed=False, verbose=False)
        flops_g = info[-1] if isinstance(info, (list, tuple)) else 0.0
        return nparam, float(flops_g)
    except Exception as exc:
        print(f"[warn] model info failed for {run_dir.name}: {exc}")
        return None


def collect(runs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in EXPERIMENT_ORDER:
        rd = runs_dir / name
        if not rd.is_dir():
            continue
        last = load_last_epoch(rd / "results.csv")
        if last is None:
            print(f"[skip] no results.csv in {rd}")
            continue
        pf = model_params_flops(rd)
        row = {
            "exp": name,
            "label": HUMAN_LABEL.get(name, name),
            "mAP50": last.get("metrics/mAP50(B)"),
            "mAP50_95": last.get("metrics/mAP50-95(B)"),
            "precision": last.get("metrics/precision(B)"),
            "recall": last.get("metrics/recall(B)"),
            "params_M": pf[0] if pf else None,
            "flops_G": pf[1] if pf else None,
        }
        rows.append(row)

    # Ablation table (M0-M4 at seed=42)
    ablation_keys = ["E0_yolov12s_seed42", "E_M1_seed42", "E_M2_seed42", "E_M3_seed42", "E_M4_seed42"]
    ablation = [r for r in rows if r["exp"] in ablation_keys]
    pd.DataFrame(ablation).to_csv(out_dir / "table2_ablation.csv", index=False)
    print(f"[done] table2_ablation.csv ({len(ablation)} rows)")

    # Comparison table (M0, M4, C1-C4)
    comparison_keys = ["E0_yolov12s_seed42", "E_M4_seed42", "C1_fasterrcnn", "C2_yolov8s", "C3_yolov11s", "C4_rtdetr_r18"]
    comparison = [r for r in rows if r["exp"] in comparison_keys]
    pd.DataFrame(comparison).to_csv(out_dir / "table3_comparison.csv", index=False)
    print(f"[done] table3_comparison.csv ({len(comparison)} rows)")

    # Full dump
    pd.DataFrame(rows).to_csv(out_dir / "all_runs.csv", index=False)
    print(f"[done] all_runs.csv ({len(rows)} rows)")


def per_category_ap(runs_dir: Path, data_yaml: Path, target_exps: list[str], out_csv: Path) -> None:
    """Run val with --plots=False on best.pt, read per-class AP50 from Ultralytics's verbose output."""
    entries = []
    for exp in target_exps:
        rd = runs_dir / exp
        best = rd / "weights" / "best.pt"
        if not best.exists():
            print(f"[skip] no best.pt in {rd}")
            continue
        print(f"[val] {exp}")
        m = YOLO(str(best))
        results = m.val(data=str(data_yaml), split="val", plots=False, verbose=False)
        names = results.names
        ap50 = results.box.ap50   # per-class tensor
        ap = results.box.ap       # per-class tensor (mAP@0.5:0.95)
        for cid, n in names.items():
            entries.append({
                "exp": exp,
                "class_id": cid,
                "class_name": n,
                "AP50": float(ap50[cid]) if cid < len(ap50) else None,
                "AP50_95": float(ap[cid]) if cid < len(ap) else None,
            })
    pd.DataFrame(entries).to_csv(out_csv, index=False)
    print(f"[done] {out_csv} ({len(entries)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    collect(args.runs, args.out)

    # Per-category improvement (Table 4): M0 vs M4
    per_category_ap(
        runs_dir=args.runs,
        data_yaml=args.data,
        target_exps=["E0_yolov12s_seed42", "E_M4_seed42"],
        out_csv=args.out / "table4_per_category_improvement.csv",
    )


if __name__ == "__main__":
    main()
