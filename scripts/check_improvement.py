"""Guardrail check for CMDrill-YOLOv12 ablation experiments.

Reads the BEST epoch mAP@0.5 from an experiment's results.csv and compares it
to a baseline (E0). Intended as a gate between queue stages: if an ablation
variant (M1/M2/M3/M4) regresses the baseline beyond a tolerance, we stop the
queue for investigation instead of burning GPU time on the remaining experiments.

Note: best mAP, not last-epoch mAP — patience-based early stop lets curves
plateau-then-drift, so iloc[-1] would falsely accept regressions whose peak
was below baseline (this is what bit M4 in the first round).

Exit codes:
    0  — experiment does not regress the baseline (delta >= -tolerance)
    1  — regression detected (stop the queue)
    2  — results missing or malformed (also stop the queue; safer)

Usage (inside train_all_remaining.sh after each ablation train call):
    python scripts/check_improvement.py \
        --exp      ~/cmdrill-yolov12/runs/cmdrill/E_M1_seed42 \
        --baseline ~/cmdrill-yolov12/runs/cmdrill/E0_yolov12s_seed42 \
        --tolerance 0.005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

METRIC = "metrics/mAP50(B)"


def read_best_map(csv_path: Path) -> tuple[float, int] | None:
    """Return (best_mAP50, epoch_of_best). Best is what ultralytics' best.pt is
    saved at and what the paper reports."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if len(df) == 0 or METRIC not in df.columns:
        return None
    bi = df[METRIC].idxmax()
    return float(df[METRIC].iloc[bi]), int(df["epoch"].iloc[bi])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path, required=True, help="Path to experiment run dir (containing results.csv)")
    p.add_argument("--baseline", type=Path, required=True, help="Path to baseline run dir")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="Minimum allowed (exp - baseline) mAP@0.5. "
                        "Default 0.005 = half a percent, within typical seed noise.")
    args = p.parse_args()

    exp = read_best_map(args.exp / "results.csv")
    base = read_best_map(args.baseline / "results.csv")

    if exp is None:
        print(f"[check] FATAL: no results at {args.exp}/results.csv", file=sys.stderr)
        sys.exit(2)
    if base is None:
        print(f"[check] FATAL: no baseline results at {args.baseline}/results.csv", file=sys.stderr)
        sys.exit(2)

    exp_map, exp_epoch = exp
    base_map, base_epoch = base
    delta = exp_map - base_map
    print(f"[check] {args.exp.name}: best mAP@0.5 = {exp_map:.4f} @ epoch {exp_epoch}")
    print(f"[check] {args.baseline.name} (baseline): best mAP@0.5 = {base_map:.4f} @ epoch {base_epoch}")
    print(f"[check] delta = {delta:+.4f}   tolerance = -{args.tolerance:.4f}")

    if delta >= -args.tolerance:
        print(f"[check] OK — {args.exp.name} does not regress the baseline beyond tolerance")
        sys.exit(0)

    print(f"[check] REGRESSION — {args.exp.name} is {-delta:.4f} below baseline, exceeding tolerance")
    print(f"[check] Stopping queue so the next experiments do not waste GPU time.")
    print(f"[check] Inspect:")
    print(f"[check]   tail -20 {args.exp}/results.csv")
    print(f"[check]   cat {args.exp}/args.yaml")
    sys.exit(1)


if __name__ == "__main__":
    main()
