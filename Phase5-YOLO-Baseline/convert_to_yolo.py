"""
Phase5-YOLO-Baseline/convert_to_yolo.py
========================================
Converts BBBC041 native JSON (data/malaria/training.json, test.json) into
YOLO-format image/label folders, reproducing the EXACT same 966/242
train/val split (seed=42, torch.utils.data.random_split) used by
Phase2-BaselineA/train_frcnn.py -- so the YOLOv8 baseline is evaluated on
the identical held-out images as Faster R-CNN (Baseline A). That makes the
two mAP@0.5 numbers directly comparable in the results table.

Does NOT modify anything outside Phase5-YOLO-Baseline/. Reads (but does
not edit) Phase1-EDA/dataset.py and shared/label_map.py to guarantee the
split is bit-identical to Baseline A.

Usage
-----
    python Phase5-YOLO-Baseline/convert_to_yolo.py \
        --train-json data/malaria/training.json \
        --test-json  data/malaria/test.json \
        --img-dir    data/malaria/images \
        --out-dir    Phase5-YOLO-Baseline/yolo_data
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import random_split
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))

from shared.label_map import FOREGROUND_NAMES     # noqa: E402
from dataset import MalariaDataset                 # noqa: E402  (read-only import)


def yolo_class_idx(label_idx: int) -> int:
    """Faster R-CNN label indices are 1..6 (0=background, never present in
    the data). YOLO wants 0-indexed foreground classes in the same order."""
    return label_idx - 1


def write_split(records: list[dict], img_dir: str, out_dir: Path, split_name: str) -> None:
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for rec in records:
        img_path = Path(img_dir) / rec["img_name"]
        if not img_path.exists():
            print(f"  ! missing image, skipping: {img_path}")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        lines = []
        for (x_min, y_min, x_max, y_max), label_idx in zip(rec["boxes"], rec["labels"]):
            cx = ((x_min + x_max) / 2) / w
            cy = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            lines.append(f"{yolo_class_idx(label_idx)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy(img_path, img_out / img_path.name)
        (lbl_out / (img_path.stem + ".txt")).write_text("\n".join(lines))
        written += 1

    print(f"  {split_name}: {written}/{len(records)} images written -> {img_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-json", default="data/malaria/training.json")
    ap.add_argument("--test-json",  default="data/malaria/test.json")
    ap.add_argument("--img-dir",    default="data/malaria/images")
    ap.add_argument("--out-dir",    default="Phase5-YOLO-Baseline/yolo_data")
    ap.add_argument("--val-split",  type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    print("Loading training.json via MalariaDataset (same class Baseline A uses)...")
    full_ds = MalariaDataset(args.train_json, args.img_dir)
    n_val   = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"Total images: {len(full_ds)}  ->  train {len(train_sub)} / val {len(val_sub)}"
          f"  (identical split to Phase2-BaselineA, seed={args.seed})")

    train_records = [full_ds._records[i] for i in train_sub.indices]
    val_records   = [full_ds._records[i] for i in val_sub.indices]

    print("Writing train split...")
    write_split(train_records, args.img_dir, out_dir, "train")
    print("Writing val split...")
    write_split(val_records, args.img_dir, out_dir, "val")

    # Also convert the held-out test.json (120-image BBBC041 test set, used
    # for Stage 1 evaluation in the paper) -- kept for reference / future use.
    if Path(args.test_json).exists():
        test_ds = MalariaDataset(args.test_json, args.img_dir)
        print("Writing test split (BBBC041 120-image holdout, reference only)...")
        write_split(test_ds._records, args.img_dir, out_dir, "test")

    names = list(FOREGROUND_NAMES)  # already 0..5 order after -1 shift
    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(names)) + "\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text)
    print(f"\ndata.yaml written -> {out_dir / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    main()
