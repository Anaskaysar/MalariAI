"""
Phase3-PipelineB/stage2_inference.py
======================================
Full MalariAI end-to-end inference pipeline.

Pipeline
--------
  Input: full blood smear image (PNG / JPG)
     │
     ▼
  Stage 1 — watershed_cells()
     → N bounding boxes (annotation-agnostic, no NMS)
     │
     ▼
  Stage 2 — EfficientNet-B0 classifier
     → class label + confidence per cell
     │
     ▼
  Grad-CAM++ — per-cell heatmap
     → spatial attention map (crop detail view)
     │
     ▼
  Results saved to --out-dir:
     ├── smear_annotated.jpg      Card 1: smear with watershed outlines + class labels
     ├── crop_gallery.jpg         Card 2: grid of top-K infected cell crops
     ├── gradcam_gallery.jpg      Card 3: Grad-CAM++ overlays for top-K infected cells
     ├── fullimage_gradcam.jpg    Card 3 extra: Grad-CAM++ overlaid on full smear
     └── results.json             Per-cell: box, label, confidence, heatmap intensity

Usage
-----
  python Phase3-PipelineB/stage2_inference.py \
      --image      data/malaria/images/your_smear.png \
      --checkpoint Phase3-PipelineB/checkpoints/best.pth \
      --out-dir    Phase3-PipelineB/results/inference

  # If no checkpoint yet (smoke test with random weights):
  python Phase3-PipelineB/stage2_inference.py \
      --image      data/malaria/images/your_smear.png \
      --out-dir    Phase3-PipelineB/results/inference \
      --no-checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.label_map import (
    NUM_CLASSES, INT_TO_LABEL, LABEL_TO_INT,
    CLASS_COLOUR_RGB, PARASITE_CLASSES
)
from Phase3_PipelineB.stage1_watershed import watershed_cells, extract_crop
from Phase3_PipelineB.gradcam import GradCAMPlusPlus

# ── Avoid import errors from path quirks ──────────────────────────────────────
try:
    from Phase3_PipelineB.stage1_watershed import watershed_cells, extract_crop
    from Phase3_PipelineB.gradcam import GradCAMPlusPlus
except ImportError:
    _p3 = Path(__file__).resolve().parent
    sys.path.insert(0, str(_p3))
    from stage1_watershed import watershed_cells, extract_crop
    from gradcam import GradCAMPlusPlus


CROP_SIZE = 64

# ImageNet normalisation (EfficientNet-B0 was pretrained on ImageNet)
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
_TRANSFORM = T.Compose([
    T.Resize((CROP_SIZE, CROP_SIZE)),
    T.ToTensor(),
    _NORMALIZE,
])


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str | None, device: torch.device) -> nn.Module:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and full checkpoint dict
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print("  Checkpoint loaded.")
    else:
        print("WARNING: No checkpoint loaded — using ImageNet-only weights.")

    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def classify_crops(
    model: nn.Module,
    crops_rgb: List[np.ndarray],
    device: torch.device,
    batch_size: int = 32,
) -> List[Tuple[int, float]]:
    """
    Run Stage 2 classifier on a list of RGB uint8 crops (H×W×3).
    Returns list of (pred_class_idx, confidence) tuples.
    """
    results = []
    for start in range(0, len(crops_rgb), batch_size):
        batch_pil  = [Image.fromarray(c) for c in crops_rgb[start:start+batch_size]]
        batch_tens = torch.stack([_TRANSFORM(p) for p in batch_pil]).to(device)
        logits     = model(batch_tens)
        probs      = torch.softmax(logits, dim=1)
        preds      = logits.argmax(1)
        for pred, prob in zip(preds.cpu(), probs.cpu()):
            results.append((int(pred), float(prob[pred])))
    return results


def run_gradcam(
    cam: GradCAMPlusPlus,
    crops_rgb: List[np.ndarray],
    class_indices: List[int],
    device: torch.device,
) -> List[np.ndarray]:
    """Compute Grad-CAM++ heatmaps for each crop."""
    heatmaps = []
    for crop_rgb, cidx in zip(crops_rgb, class_indices):
        pil  = Image.fromarray(crop_rgb)
        inp  = _TRANSFORM(pil).unsqueeze(0)
        hmap, _, _ = cam(inp, class_idx=cidx)
        heatmaps.append(hmap)
    return heatmaps


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation builders
# ═══════════════════════════════════════════════════════════════════════════════

def _colour_bgr(label: str) -> Tuple[int, int, int]:
    r, g, b = CLASS_COLOUR_RGB.get(label, (200, 200, 200))
    return (b, g, r)


def build_annotated_smear(
    bgr: np.ndarray,
    boxes: List[Tuple],
    labels: List[str],
    confidences: List[float],
) -> np.ndarray:
    """Card 1: smear with per-cell outlines and class labels."""
    vis = bgr.copy()
    for box, lbl, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        colour = _colour_bgr(lbl)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)
        # Only annotate non-RBC cells to avoid clutter
        if lbl != "red blood cell":
            tag = f"{lbl[:4]} {conf:.0%}"
            cv2.putText(vis, tag, (x1, max(y1-4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
                        cv2.LINE_AA)
    return vis


def build_crop_gallery(
    crops_rgb: List[np.ndarray],
    labels: List[str],
    confidences: List[float],
    max_cells: int = 48,
    cols: int = 8,
) -> np.ndarray:
    """Card 2: grid of cell crops with labels."""
    # Sort by label priority: parasites first, then RBC, leukocyte last
    priority = {"ring": 0, "trophozoite": 1, "schizont": 2,
                "gametocyte": 3, "leukocyte": 4, "red blood cell": 5,
                "background": 6}
    order = sorted(range(len(labels)),
                   key=lambda i: (priority.get(labels[i], 9), -confidences[i]))
    order = order[:max_cells]

    CELL = 80   # display size per crop cell
    rows = (len(order) + cols - 1) // cols
    canvas = np.ones((rows * CELL + 24, cols * CELL, 3), dtype=np.uint8) * 240

    for slot, idx in enumerate(order):
        r, c     = divmod(slot, cols)
        y0, x0   = r * CELL, c * CELL
        crop_disp = cv2.resize(crops_rgb[idx], (CELL - 2, CELL - 2 - 16))
        # Colour border by class
        colour_rgb = CLASS_COLOUR_RGB.get(labels[idx], (200, 200, 200))
        colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])
        crop_bgr   = cv2.cvtColor(crop_disp, cv2.COLOR_RGB2BGR)
        cv2.rectangle(canvas,
                      (x0, y0), (x0 + CELL - 2, y0 + CELL - 18),
                      colour_bgr, 2)
        canvas[y0:y0 + CELL - 2 - 16, x0:x0 + CELL - 2] = crop_bgr
        # Label strip
        tag = f"{labels[idx][:6]} {confidences[idx]:.0%}"
        cv2.putText(canvas, tag,
                    (x0, y0 + CELL - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, colour_bgr, 1, cv2.LINE_AA)

    return canvas


def build_gradcam_gallery(
    crops_rgb: List[np.ndarray],
    heatmaps:  List[np.ndarray],
    labels:    List[str],
    confidences: List[float],
    max_cells: int = 24,
    cols: int = 6,
) -> np.ndarray:
    """Card 3: Grad-CAM++ heatmap overlays for top infected cells."""
    # Filter to parasites only, sorted by confidence
    parasite_indices = [i for i, lbl in enumerate(labels)
                        if lbl in PARASITE_CLASSES]
    parasite_indices = sorted(parasite_indices,
                              key=lambda i: -confidences[i])[:max_cells]

    if not parasite_indices:
        # Fallback: show all cells
        parasite_indices = sorted(range(len(labels)),
                                  key=lambda i: -confidences[i])[:max_cells]

    CELL = 100
    rows = (len(parasite_indices) + cols - 1) // cols
    canvas = np.ones((rows * CELL + 20, cols * CELL, 3), dtype=np.uint8) * 30

    for slot, idx in enumerate(parasite_indices):
        r, c   = divmod(slot, cols)
        y0, x0 = r * CELL, c * CELL

        crop_pil = Image.fromarray(crops_rgb[idx])
        overlay  = GradCAMPlusPlus.overlay(crop_pil, heatmaps[idx], alpha=0.5)
        overlay_arr = np.array(overlay.resize((CELL, CELL - 20)))
        overlay_bgr = cv2.cvtColor(overlay_arr, cv2.COLOR_RGB2BGR)

        canvas[y0:y0 + CELL - 20, x0:x0 + CELL] = overlay_bgr
        tag = f"{labels[idx][:6]} {confidences[idx]:.0%}"
        cv2.putText(canvas, tag, (x0 + 2, y0 + CELL - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return canvas


def build_fullimage_gradcam(
    bgr: np.ndarray,
    boxes: List[Tuple],
    heatmaps: List[np.ndarray],
    labels: List[str],
) -> np.ndarray:
    """
    Full-image Grad-CAM++ overlay.
    Splats each cell's heatmap back onto the full smear in its bounding-box
    region, accumulating intensity. Shows the spatial distribution of
    infection-stage activations across the whole slide.
    """
    H, W = bgr.shape[:2]
    full_heatmap = np.zeros((H, W), dtype=np.float32)
    count_map    = np.zeros((H, W), dtype=np.float32)

    for box, hmap, lbl in zip(boxes, heatmaps, labels):
        if lbl not in PARASITE_CLASSES:
            continue
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        resized = cv2.resize(hmap, (bw, bh), interpolation=cv2.INTER_LINEAR)
        full_heatmap[y1:y2, x1:x2] += resized
        count_map[y1:y2, x1:x2]    += 1.0

    # Average overlapping regions
    valid = count_map > 0
    full_heatmap[valid] /= count_map[valid]

    # Normalise
    fmax = full_heatmap.max()
    if fmax > 0:
        full_heatmap /= fmax

    # Apply colourmap
    import matplotlib.cm as cm
    cmap     = cm.get_cmap("jet")
    coloured = (cmap(full_heatmap)[:, :, :3] * 255).astype(np.uint8)
    coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    # Blend with original
    blended = cv2.addWeighted(bgr, 0.55, coloured_bgr, 0.45, 0)
    return blended


# ═══════════════════════════════════════════════════════════════════════════════
#  Main inference routine
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load image
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {args.image}")
    print(f"Image loaded: {args.image}  ({bgr.shape[1]}×{bgr.shape[0]})")

    # Stage 1 — watershed
    print("\n[Stage 1] Running watershed segmentation ...")
    boxes = watershed_cells(bgr)
    print(f"  Detected {len(boxes)} cells")

    # Extract crops (RGB uint8)
    crops_rgb = [extract_crop(bgr, box) for box in boxes]

    # Stage 2 — load model + classify
    ckpt = None if getattr(args, "no_checkpoint", False) else args.checkpoint
    model = load_model(ckpt, device)

    print("\n[Stage 2] Classifying crops ...")
    preds = classify_crops(model, crops_rgb, device)

    labels_list = [INT_TO_LABEL[p[0]] for p in preds]
    confs_list  = [p[1] for p in preds]
    class_idxs  = [p[0] for p in preds]

    # Count classes
    from collections import Counter
    counts = Counter(labels_list)
    n_parasites = sum(v for k, v in counts.items() if k in PARASITE_CLASSES)
    n_total     = len(labels_list)
    infection_rate = n_parasites / n_total * 100 if n_total > 0 else 0.0

    print(f"\n  Total cells detected : {n_total}")
    print(f"  Infected cells       : {n_parasites}")
    print(f"  Infection rate       : {infection_rate:.2f}%")
    print("  Class breakdown:")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<22}: {cnt}")

    # Grad-CAM++
    print("\n[Grad-CAM++] Computing heatmaps ...")
    cam = GradCAMPlusPlus(model)
    heatmaps = run_gradcam(cam, crops_rgb, class_idxs, device)
    cam.remove_hooks()
    print(f"  Heatmaps computed for {len(heatmaps)} cells")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[Output] Saving results ...")

    # Card 1: annotated smear
    smear_ann = build_annotated_smear(bgr, boxes, labels_list, confs_list)
    cv2.imwrite(str(out_dir / "smear_annotated.jpg"), smear_ann)

    # Card 2: crop gallery
    gallery = build_crop_gallery(crops_rgb, labels_list, confs_list)
    cv2.imwrite(str(out_dir / "crop_gallery.jpg"), gallery)

    # Card 3: Grad-CAM++ gallery (parasite cells)
    gcam_gallery = build_gradcam_gallery(
        crops_rgb, heatmaps, labels_list, confs_list)
    cv2.imwrite(str(out_dir / "gradcam_gallery.jpg"), gcam_gallery)

    # Full-image Grad-CAM++ overlay
    full_gcam = build_fullimage_gradcam(bgr, boxes, heatmaps, labels_list)
    cv2.imwrite(str(out_dir / "fullimage_gradcam.jpg"), full_gcam)

    # JSON results
    results = {
        "image": args.image,
        "total_cells": n_total,
        "infected_cells": n_parasites,
        "infection_rate_pct": round(infection_rate, 2),
        "class_counts": dict(counts),
        "cells": [
            {
                "idx": i,
                "box": list(boxes[i]),
                "label": labels_list[i],
                "confidence": round(confs_list[i], 4),
                "heatmap_mean": round(float(heatmaps[i].mean()), 4),
            }
            for i in range(n_total)
        ],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print("  smear_annotated.jpg  — Card 1: watershed outlines on full smear")
    print("  crop_gallery.jpg     — Card 2: cell crop grid")
    print("  gradcam_gallery.jpg  — Card 3: Grad-CAM++ crop overlays")
    print("  fullimage_gradcam.jpg— Card 3 extra: full-image heatmap overlay")
    print("  results.json         — per-cell label, confidence, heatmap stats")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MalariAI full inference: watershed → EfficientNet-B0 → Grad-CAM++")
    p.add_argument("--image",       required=True,
                   help="Path to input blood smear image (PNG/JPG)")
    p.add_argument("--checkpoint",  default="Phase3-PipelineB/checkpoints/best.pth",
                   help="Path to Stage 2 model checkpoint")
    p.add_argument("--out-dir",     default="Phase3-PipelineB/results/inference",
                   help="Directory for output images and results.json")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Run without loading a checkpoint (uses ImageNet weights only)")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
