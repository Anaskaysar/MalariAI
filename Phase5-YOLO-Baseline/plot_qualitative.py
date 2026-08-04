"""
Phase5-YOLO-Baseline/plot_qualitative.py
===========================================
Builds the clean GT-vs-prediction qualitative figure from
qualitative_samples.json (raw box coordinates, not a pre-rendered image),
matching the style of Phase2-BaselineA's fig_frcnn_predictions.png:
one column per sample image, GT on top / predictions on bottom, boxes
colour-coded by class, text labels shown only for parasite/rare classes
(red blood cell boxes are drawn unlabeled to avoid clutter).

Usage
-----
    python Phase5-YOLO-Baseline/plot_qualitative.py \
        --samples  Phase5-YOLO-Baseline/qualitative_samples.json \
        --img-dir  Phase5-YOLO-Baseline/Phase5-YOLO-Baseline_Results/yolo_data/images/val \
        --out      Phase5-YOLO-Baseline/fig_yolo_predictions.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.label_map import FOREGROUND_NAMES, SHORT_NAME, CLASS_COLOUR_RGB  # noqa: E402

# YOLO class idx (0..5) -> class name (FOREGROUND_NAMES already in that order)
YOLO_IDX_TO_NAME = {i: name for i, name in enumerate(FOREGROUND_NAMES)}


def rgb01(name: str):
    r, g, b = CLASS_COLOUR_RGB[name]
    return (r / 255, g / 255, b / 255)


def draw_boxes(ax, boxes, kind: str, conf_thresh: float | None = None):
    """boxes: list of [cls, x1, y1, x2, y2] (gt) or [cls, x1, y1, x2, y2, conf] (pred)."""
    n_drawn = 0
    for b in boxes:
        cls = int(b[0])
        x1, y1, x2, y2 = b[1], b[2], b[3], b[4]
        if kind == "pred" and conf_thresh is not None and b[5] < conf_thresh:
            continue
        name = YOLO_IDX_TO_NAME.get(cls, "red blood cell")
        colour = rgb01(name)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.1, edgecolor=colour, facecolor="none")
        ax.add_patch(rect)
        n_drawn += 1
        # Only label rare / parasite classes -- RBC boxes stay unlabeled to avoid clutter
        if name != "red blood cell":
            label = SHORT_NAME[name]
            if kind == "pred":
                label = f"{label} {b[5]:.2f}"
            ax.text(x1, max(0, y1 - 4), label, fontsize=6.5, color="white",
                     bbox=dict(facecolor=colour, edgecolor="none", pad=1.0))
    return n_drawn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="Phase5-YOLO-Baseline/qualitative_samples.json")
    ap.add_argument("--img-dir", default="Phase5-YOLO-Baseline/Phase5-YOLO-Baseline_Results/yolo_data/images/val")
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--out", default="Phase5-YOLO-Baseline/fig_yolo_predictions.png")
    args = ap.parse_args()

    samples = json.loads(Path(args.samples).read_text())
    n = len(samples)

    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, sample in enumerate(samples):
        img_path = Path(args.img_dir) / sample["image"]
        img = Image.open(img_path).convert("RGB")

        ax_gt, ax_pred = axes[0, col], axes[1, col]

        ax_gt.imshow(img)
        n_gt = draw_boxes(ax_gt, sample["gt"], kind="gt")
        ax_gt.set_title(f"GT ({n_gt} boxes)", fontsize=10)
        ax_gt.axis("off")

        ax_pred.imshow(img)
        n_pred = draw_boxes(ax_pred, sample["pred"], kind="pred", conf_thresh=args.conf_thresh)
        ax_pred.set_title(f"Pred ({n_pred} boxes, ≥{args.conf_thresh})", fontsize=10)
        ax_pred.axis("off")

    fig.suptitle("Ground Truth (top) vs YOLOv8 Predictions (bottom)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
