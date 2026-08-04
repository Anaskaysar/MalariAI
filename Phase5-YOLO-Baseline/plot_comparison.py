"""
Phase5-YOLO-Baseline/plot_comparison.py
=========================================
Generates the grouped bar chart comparing per-class AP@0.5 between
Faster R-CNN (Baseline A) and YOLOv8 (Baseline B) -- the one figure
ultralytics doesn't produce automatically. Uses the same class colours
as the rest of the manuscript's figures (shared/label_map.py).

Reads both metrics.json files so the figure always reflects whatever
numbers are actually on disk -- rerun this after replacing the
placeholder YOLOv8 metrics.json with the real one downloaded from Kaggle.

Usage
-----
    python Phase5-YOLO-Baseline/plot_comparison.py \
        --frcnn Phase2-BaselineA/checkpoints-kaggle-80epoch/metrics.json \
        --yolo  Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json \
        --out   Phase5-YOLO-Baseline/fig_yolo_vs_frcnn_comparison.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.label_map import FOREGROUND_NAMES, SHORT_NAME, PARASITE_CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frcnn", default="Phase2-BaselineA/checkpoints-kaggle-80epoch/metrics.json")
    ap.add_argument("--yolo",  default="Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json")
    ap.add_argument("--out",   default="Phase5-YOLO-Baseline/fig_yolo_vs_frcnn_comparison.png")
    args = ap.parse_args()

    frcnn = json.load(open(args.frcnn))
    yolo  = json.load(open(args.yolo))

    classes = FOREGROUND_NAMES
    a_vals  = [100 * frcnn["per_class_ap"][c] for c in classes]
    b_vals  = [100 * yolo["per_class_ap"][c]  for c in classes]
    labels  = [SHORT_NAME[c] for c in classes]

    x = np.arange(len(classes))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars_a = ax.bar(x - w/2, a_vals, w, label=f"Faster R-CNN (mAP@0.5={100*frcnn['map_50']:.1f}%)",
                     color="#8c8c8c", edgecolor="black", linewidth=0.6)
    bars_b = ax.bar(x + w/2, b_vals, w, label=f"YOLOv8 (mAP@0.5={100*yolo['map_50']:.1f}%)",
                     color="#4472c4", edgecolor="black", linewidth=0.6)

    # Highlight rare parasite-stage classes on the x-axis (bold + asterisk)
    tick_labels = [f"{lab}*" if cls in PARASITE_CLASSES else lab
                   for cls, lab in zip(classes, labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)

    for bars in (bars_a, bars_b):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AP@0.5 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class detection AP@0.5: Faster R-CNN vs. YOLOv8")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.01, -0.02, "* parasite life-cycle stage (infected classes)", fontsize=7.5, style="italic")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
