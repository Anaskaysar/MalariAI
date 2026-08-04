# Phase 5 — YOLOv8 State-of-the-Art Baseline

Adds a modern detector baseline (YOLOv8) alongside the existing Faster
R-CNN baseline (Phase2-BaselineA), for the CBM resubmission. This folder
is fully isolated — nothing in Phase1-EDA, Phase2-BaselineA, or shared/
is modified; those files are only imported/read to guarantee the exact
same train/val split.

## Why

CMIG's desk-reject cited insufficient "computational advances." A
comparable paper CBM did publish (Zedda et al. 2025, CBM 186:109704)
benchmarks head-to-head against modern YOLO-family SOTA and quantifies
the improvement. Our manuscript currently only benchmarks against
Faster R-CNN (2015) experimentally, while citing modern methods
(YOLOv12, YOLOv4) without testing against them. This closes that gap.

## Pipeline

1. **`convert_to_yolo.py`** — converts `data/malaria/training.json` /
   `test.json` into YOLO image/label format. Reproduces Baseline A's
   exact 966/242 train/val split (seed 42) by importing
   `Phase1-EDA/dataset.py`'s `MalariaDataset` and running the identical
   `torch.utils.data.random_split` call, so both baselines are scored
   on the same held-out images.

2. **`train_yolo.py`** — trains YOLOv8 on the converted data.
   **Run the real 80-epoch training on Kaggle/Colab GPU** — this
   sandbox has no GPU (2 CPU cores, 3.8 GB RAM), same constraint that
   Baseline A worked around (see `Phase2-BaselineA/checkpoints-kaggle-80epoch`).
   Only use this locally for a 1-epoch smoke test.

3. **`eval_yolo.py`** — evaluates a trained checkpoint on the val split,
   writes `metrics.json` in the same schema as Baseline A's, so results
   are directly comparable.

4. **`compare_baselines.py`** — merges both `metrics.json` files into a
   markdown table for the manuscript.

## Step by step

```bash
# 1. Convert data (run locally or on Kaggle — CPU-only, no GPU needed)
python Phase5-YOLO-Baseline/convert_to_yolo.py

# 2. Train (Kaggle/Colab GPU)
pip install ultralytics
python Phase5-YOLO-Baseline/train_yolo.py \
    --data Phase5-YOLO-Baseline/yolo_data/data.yaml \
    --model yolov8s.pt --epochs 80 --imgsz 1024 \
    --out-dir Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch

# 3. Evaluate
python Phase5-YOLO-Baseline/eval_yolo.py \
    --weights Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/weights/best.pt \
    --data    Phase5-YOLO-Baseline/yolo_data/data.yaml \
    --out     Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json

# 4. Compare
python Phase5-YOLO-Baseline/compare_baselines.py \
    --yolo Phase5-YOLO-Baseline/checkpoints-kaggle-80epoch/metrics.json
```

## Status

- [x] Data converter written, smoke-tested on a subset locally
- [ ] Full data conversion (1,208 images)
- [ ] YOLOv8 training on Kaggle GPU
- [ ] Evaluation + comparison table
- [ ] Manuscript results section / table updated with YOLOv8 row
