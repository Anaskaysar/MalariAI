"""
Phase5-YOLO-Baseline/compare_baselines.py
==========================================
Merges Phase2-BaselineA's metrics.json (Faster R-CNN) and this folder's
YOLOv8 metrics.json into one markdown comparison table, ready to paste
into the manuscript's results section / table.

Usage
-----
    python Phase5-YOLO-Baseline/compare_baselines.py \
        --frcnn Phase2-BaselineA/checkpoints-kaggle-80epoch/metrics.json \
        --yolo  Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frcnn", default="Phase2-BaselineA/checkpoints-kaggle-80epoch/metrics.json")
    ap.add_argument("--yolo",  required=True)
    args = ap.parse_args()

    frcnn = json.load(open(args.frcnn))
    yolo  = json.load(open(args.yolo))

    classes = list(frcnn["per_class_ap"].keys())

    print("| Class | Faster R-CNN (Baseline A) | YOLOv8 |")
    print("|---|---|---|")
    for c in classes:
        a = frcnn["per_class_ap"].get(c, float("nan"))
        b = yolo["per_class_ap"].get(c, float("nan"))
        print(f"| {c} | {100*a:.2f}% | {100*b:.2f}% |")
    print(f"| **mAP@0.5** | **{100*frcnn['map_50']:.2f}%** | **{100*yolo['map_50']:.2f}%** |")


if __name__ == "__main__":
    main()
