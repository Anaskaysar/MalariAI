"""
Phase5-YOLO-Baseline/eval_yolo.py
==================================
Evaluates a trained YOLOv8 checkpoint on the val split and writes
metrics.json in the SAME schema as Phase2-BaselineA/checkpoints-*/metrics.json
(map_50, per_class_ap keyed by class name) so the two baselines can be
compared directly with compare_baselines.py.

Usage
-----
    python Phase5-YOLO-Baseline/eval_yolo.py \
        --weights Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/weights/best.pt \
        --data    Phase5-YOLO-Baseline/yolo_data/data.yaml \
        --out     Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.label_map import FOREGROUND_NAMES   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data",    required=True)
    ap.add_argument("--imgsz",   type=int, default=1024)
    ap.add_argument("--out",     default="Phase5-YOLO-Baseline/metrics.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    model   = YOLO(args.weights)
    results = model.val(data=args.data, imgsz=args.imgsz, split="val", iou=0.5)

    map50          = float(results.box.map50)
    per_class_ap50 = results.box.ap50   # array indexed by YOLO class id (0..5)

    metrics = {
        "map_50": round(map50, 4),
        "per_class_ap": {
            FOREGROUND_NAMES[i]: round(float(ap), 4)
            for i, ap in enumerate(per_class_ap50)
        },
        "model":   "YOLOv8",
        "weights": str(args.weights),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
