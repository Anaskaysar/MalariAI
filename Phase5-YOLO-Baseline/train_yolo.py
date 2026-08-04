"""
Phase5-YOLO-Baseline/train_yolo.py
===================================
Trains a YOLOv8 detector on the BBBC041 malaria dataset as a modern
state-of-the-art baseline, for direct comparison against Baseline A
(Faster R-CNN, Phase2-BaselineA) -- added in response to the reviewer
concern about lacking comparison with real state-of-the-art methods.

Run convert_to_yolo.py FIRST to build yolo_data/.

GPU note
--------
This sandbox has no GPU (2 CPU cores / 3.8 GB RAM) -- an 80-epoch run is
not feasible here. Train this on Kaggle/Colab GPU, exactly as Baseline A
was (see Phase2-BaselineA/checkpoints-kaggle-80epoch). Only run a 1-epoch
smoke test locally to confirm the pipeline works end to end.

Usage (Kaggle/Colab, GPU)
--------------------------
    pip install ultralytics
    python Phase5-YOLO-Baseline/train_yolo.py \
        --data     Phase5-YOLO-Baseline/yolo_data/data.yaml \
        --model    yolov8s.pt \
        --epochs   80 \
        --imgsz    1024 \
        --out-dir  Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch

Local smoke test (CPU, tiny)
-----------------------------
    python Phase5-YOLO-Baseline/train_yolo.py \
        --data     Phase5-YOLO-Baseline/yolo_data/data.yaml \
        --model    yolov8n.pt --epochs 1 --imgsz 320 --batch 2 \
        --out-dir  Phase5-YOLO-Baseline/checkpoints-smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",    required=True)
    ap.add_argument("--model",   default="yolov8s.pt")
    ap.add_argument("--epochs",  type=int, default=80)
    ap.add_argument("--imgsz",   type=int, default=1024)
    ap.add_argument("--batch",   type=int, default=8)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out-dir", default="Phase5-YOLO-Baseline/checkpoints")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMalariAI -- Phase 5: YOLOv8 SOTA baseline")
    print(f"Model  : {args.model}")
    print(f"Epochs : {args.epochs}")
    print(f"Imgsz  : {args.imgsz}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        deterministic=True,
    )
    print(f"\nTraining complete. Weights + logs in: {out_dir}")


if __name__ == "__main__":
    main()
