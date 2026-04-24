"""Guardrail check for CMDrill-YOLOv12 ablation experiments.

Reads the final epoch mAP@0.5 from an experiment's results.csv and compares it
to a baseline (E0). Intended as a gate between queue stages: if an ablation
variant (M1/M2/M3/M4) regresses the baseline beyond a tolerance, we stop the
queue for investigation instead of burning GPU time on the remaining experiments.

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


def read_final_map(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0 or METRIC not in df.columns:
        return None
    return float(df[METRIC].iloc[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path, required=True, help="Path to experiment run dir (containing results.csv)")
    p.add_argument("--baseline", type=Path, required=True, help="Path to baseline run dir")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="Minimum allowed (exp - baseline) mAP@0.5. "
                        "Default 0.005 = half a percent, within typical seed noise.")
    args = p.parse_args()

    exp_map = read_final_map(args.exp / "results.csv")
    base_map = read_final_map(args.baseline / "results.csv")

    if exp_map is None:
        print(f"[check] FATAL: no results at {args.exp}/results.csv", file=sys.stderr)
        sys.exit(2)
    if base_map is None:
        print(f"[check] FATAL: no baseline results at {args.baseline}/results.csv", file=sys.stderr)
        sys.exit(2)

    delta = exp_map - base_map
    print(f"[check] {args.exp.name}: mAP@0.5 = {exp_map:.4f}")
    print(f"[check] {args.baseline.name} (baseline): mAP@0.5 = {base_map:.4f}")
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
