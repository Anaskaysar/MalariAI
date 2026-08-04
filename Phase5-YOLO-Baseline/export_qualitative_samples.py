"""
Phase5-YOLO-Baseline/export_qualitative_samples.py
=====================================================
Runs the trained YOLOv8 model on a handful of validation images (the ones
with the most class diversity, to surface rare parasite stages) and
exports ground-truth + predicted boxes as a small JSON file -- NOT a
rendered image. This is used to build a clean, publication-style GT-vs-
prediction figure locally (matching fig_frcnn_predictions.png's style),
avoiding ultralytics' default val_batch mosaic, which stamps a text label
on every red-blood-cell box and becomes unreadable.

Run this LOCALLY (your malariaenv already has ultralytics + torch+cuda).
Inference-only, on ~4 images -- memory footprint is tiny compared to
training, so this should run fine even after the earlier CUDA OOM during
training.

Usage
-----
    python Phase5-YOLO-Baseline\\export_qualitative_samples.py

Defaults assume the full Kaggle output was downloaded into
Phase5-YOLO-Baseline\\Phase5-YOLO-Baseline_Results\\ (weights + yolo_data
both present there already). Override with --weights / --yolo-data-dir /
--out if your paths differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        default="Phase5-YOLO-Baseline/Phase5-YOLO-Baseline_Results/checkpoints-kaggle-80epoch/weights/best.pt",
    )
    ap.add_argument(
        "--yolo-data-dir",
        default="Phase5-YOLO-Baseline/Phase5-YOLO-Baseline_Results/yolo_data",
    )
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="Phase5-YOLO-Baseline/qualitative_samples.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_dir = Path(args.yolo_data_dir)
    img_dir  = data_dir / "images" / "val"
    lbl_dir  = data_dir / "labels" / "val"

    label_files = sorted(lbl_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {lbl_dir} -- check --yolo-data-dir")

    def n_unique_classes(txt_path: Path) -> int:
        lines = txt_path.read_text().strip().splitlines()
        return len({line.split()[0] for line in lines if line.strip()})

    label_files.sort(key=n_unique_classes, reverse=True)
    chosen = label_files[: args.n_samples]

    print(f"Loading model from {args.weights} ...")
    model = YOLO(args.weights)

    samples = []
    for txt_path in chosen:
        img_path = img_dir / (txt_path.stem + ".png")
        if not img_path.exists():
            img_path = img_dir / (txt_path.stem + ".jpg")
        if not img_path.exists():
            print(f"  ! image not found for {txt_path.stem}, skipping")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        gt = []
        for line in txt_path.read_text().strip().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            gt.append([int(cls), x1, y1, x2, y2])

        print(f"  running inference on {img_path.name} ({n_unique_classes(txt_path)} classes)...")
        r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pred = []
        for box in r.boxes:
            cls  = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred.append([cls, x1, y1, x2, y2, round(conf, 2)])

        samples.append({
            "image":  img_path.name,
            "width":  w,
            "height": h,
            "gt":     gt,
            "pred":   pred,
        })
        print(f"    gt boxes: {len(gt)}  pred boxes: {len(pred)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, indent=2))
    print(f"\nSaved {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
