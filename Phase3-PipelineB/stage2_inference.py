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

# Phase3-PipelineB folder has hyphens — not importable as a module.
# Add the folder directly to sys.path and import by filename.
_p3 = Path(__file__).resolve().parent
sys.path.insert(0, str(_p3))
from stage1_watershed import watershed_cells, extract_crop
from gradcam import GradCAMPlusPlus


CROP_SIZE = 64

# ── Box-size guards ────────────────────────────────────────────────────────────
# Watershed sometimes merges 3-5 touching cells into one large region.
# These merged regions are not single cells and should not be classified.
# Typical single RBC:  80-160 px wide/tall at 1600×1200 resolution.
# Typical leukocyte:  150-220 px wide/tall.
# Threshold: skip any box wider or taller than 220 px,
#            or with aspect-ratio > 2.2 (merged row/column of cells).
MAX_CELL_W       = 220   # pixels
MAX_CELL_H       = 220   # pixels
MAX_ASPECT_RATIO = 2.2   # w/h or h/w

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
    oversized_boxes: List[Tuple] | None = None,
    conf_threshold: float = 0.40,
) -> np.ndarray:
    """Card 1: smear with per-cell outlines and class labels.

    - RBC: thin green border, no text.
    - Parasites: thick coloured border + 8-char label + confidence.
    - Low-confidence parasites: same border but with '?' suffix.
    - Leukocyte: medium border + label.
    - Oversized merged regions: grey dashed outline (no classification).
    """
    vis = bgr.copy()

    # Draw oversized / merged-region boxes first (grey, thin)
    if oversized_boxes:
        for box in oversized_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (140, 140, 140), 1)
            cv2.putText(vis, "merged?", (x1, max(y1-3, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 140, 140), 1,
                        cv2.LINE_AA)

    for box, lbl, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        colour = _colour_bgr(lbl)

        if lbl == "red blood cell":
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)
        elif lbl == "leukocyte":
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 1)
            cv2.putText(vis, f"leuko {conf:.0%}", (x1, max(y1-4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)
        else:
            # Parasite — thick box + label
            thickness = 2
            suffix = "" if conf >= conf_threshold else "?"
            tag = f"{lbl[:8]}{suffix} {conf:.0%}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)
            cv2.putText(vis, tag, (x1, max(y1-5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    # Summary overlay — top-left corner
    n_parasites = sum(1 for l, c in zip(labels, confidences)
                      if l in PARASITE_CLASSES and c >= conf_threshold)
    n_total = len(labels)
    rate = n_parasites / n_total * 100 if n_total > 0 else 0.0
    summary = f"Cells: {n_total}  Infected: {n_parasites}  Rate: {rate:.1f}%"
    # Dark background strip for readability
    (tw, th), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(vis, (8, 6), (14 + tw, 14 + th + 4), (0, 0, 0), -1)
    cv2.putText(vis, summary, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

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

    CELL   = 100  # display size per crop cell (100px = clearer at normal screen size)
    LABEL_H = 18
    rows = (len(order) + cols - 1) // cols
    canvas = np.ones((rows * (CELL + LABEL_H), cols * CELL, 3), dtype=np.uint8) * 240

    for slot, idx in enumerate(order):
        r, c     = divmod(slot, cols)
        y0, x0   = r * (CELL + LABEL_H), c * CELL
        crop_disp = cv2.resize(crops_rgb[idx], (CELL - 2, CELL - 2))
        # Colour border by class
        colour_rgb = CLASS_COLOUR_RGB.get(labels[idx], (200, 200, 200))
        colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])
        crop_bgr   = cv2.cvtColor(crop_disp, cv2.COLOR_RGB2BGR)
        cv2.rectangle(canvas,
                      (x0, y0), (x0 + CELL - 2, y0 + CELL - 2),
                      colour_bgr, 2)
        canvas[y0:y0 + CELL - 2, x0:x0 + CELL - 2] = crop_bgr
        # Label strip below the crop
        tag = f"{labels[idx][:8]} {confidences[idx]:.0%}"
        cv2.putText(canvas, tag,
                    (x0 + 2, y0 + CELL + LABEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, colour_bgr, 1, cv2.LINE_AA)

    return canvas


def build_gradcam_gallery(
    crops_rgb: List[np.ndarray],
    heatmaps:  List[np.ndarray],
    labels:    List[str],
    confidences: List[float],
    max_cells: int = 24,
    cols: int = 6,
    confident_mask: List[bool] | None = None,
) -> np.ndarray:
    """Card 3: Grad-CAM++ heatmap overlays for confident infected cells.

    confident_mask: boolean list aligned to labels; True = confident parasite.
                    If None, falls back to all parasite-class entries.
    """
    if confident_mask is not None:
        parasite_indices = [i for i, m in enumerate(confident_mask) if m]
    else:
        parasite_indices = [i for i, lbl in enumerate(labels)
                            if lbl in PARASITE_CLASSES]
    parasite_indices = sorted(parasite_indices,
                              key=lambda i: -confidences[i])[:max_cells]

    if not parasite_indices:
        # Fallback: top cells by confidence
        parasite_indices = sorted(range(len(labels)),
                                  key=lambda i: -confidences[i])[:max_cells]

    CELL = 160   # larger cells — easier to see heatmap detail
    LABEL_H = 22
    rows = (len(parasite_indices) + cols - 1) // cols
    canvas = np.ones((rows * (CELL + LABEL_H), cols * CELL, 3), dtype=np.uint8) * 30

    for slot, idx in enumerate(parasite_indices):
        r, c   = divmod(slot, cols)
        y0, x0 = r * (CELL + LABEL_H), c * CELL

        # Show original crop and heatmap overlay side by side within the cell
        crop_pil    = Image.fromarray(crops_rgb[idx])
        overlay     = GradCAMPlusPlus.overlay(crop_pil, heatmaps[idx], alpha=0.5)
        overlay_arr = np.array(overlay.resize((CELL, CELL)))
        overlay_bgr = cv2.cvtColor(overlay_arr, cv2.COLOR_RGB2BGR)

        canvas[y0:y0 + CELL, x0:x0 + CELL] = overlay_bgr

        # Label strip with confidence and class name
        lbl_colour = CLASS_COLOUR_RGB.get(labels[idx], (200, 200, 200))
        lbl_bgr    = (lbl_colour[2], lbl_colour[1], lbl_colour[0])
        tag = f"{labels[idx]}  {confidences[idx]:.0%}"
        cv2.putText(canvas, tag, (x0 + 4, y0 + CELL + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, lbl_bgr, 1, cv2.LINE_AA)

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

    # Apply colourmap only at parasite locations (mask = where heatmap > 0)
    import matplotlib.cm as cm
    cmap       = cm.get_cmap("jet")
    coloured   = (cmap(full_heatmap)[:, :, :3] * 255).astype(np.uint8)
    coloured_bgr = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)

    # Build mask: only blend heatmap where parasites were detected
    mask = (full_heatmap > 0).astype(np.float32)
    # Smooth mask edges slightly for nicer blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask3 = np.stack([mask, mask, mask], axis=2)

    # Original image is unchanged everywhere except parasite regions
    blended = (bgr.astype(np.float32) * (1 - 0.55 * mask3) +
               coloured_bgr.astype(np.float32) * (0.55 * mask3)).astype(np.uint8)

    # Draw coloured bounding boxes around detected parasites
    for box, lbl in zip(boxes, labels):
        if lbl not in PARASITE_CLASSES:
            continue
        x1, y1, x2, y2 = box
        colour_rgb = CLASS_COLOUR_RGB.get(lbl, (255, 255, 0))
        colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])
        cv2.rectangle(blended, (x1, y1), (x2, y2), colour_bgr, 2)
        cv2.putText(blended, lbl[:5], (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_bgr, 1, cv2.LINE_AA)

    return blended


# ═══════════════════════════════════════════════════════════════════════════════
#  Main inference routine
# ═══════════════════════════════════════════════════════════════════════════════

def _is_oversized(box: Tuple) -> bool:
    """Return True if a watershed box is too large to be a single cell.

    Merged clusters of 3-5 touching cells produce boxes wider or taller than
    MAX_CELL_W / MAX_CELL_H, or with an extreme aspect ratio. Classifying these
    as 'leukocyte' (the closest large-cell class) produces false positives.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w > MAX_CELL_W or h > MAX_CELL_H:
        return True
    if max(w, h) / max(min(w, h), 1) > MAX_ASPECT_RATIO:
        return True
    return False


def run_inference(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_threshold = getattr(args, "conf_threshold", 0.40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load image
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {args.image}")
    print(f"Image loaded: {args.image}  ({bgr.shape[1]}×{bgr.shape[0]})")

    # Stage 1 — watershed
    print("\n[Stage 1] Running watershed segmentation ...")
    all_boxes = watershed_cells(bgr)
    print(f"  Detected {len(all_boxes)} regions total")

    # Split valid single-cell boxes from oversized merged regions
    oversized_boxes = [b for b in all_boxes if _is_oversized(b)]
    boxes           = [b for b in all_boxes if not _is_oversized(b)]
    print(f"  Valid cells   : {len(boxes)}")
    print(f"  Oversized (merged clusters, skipped): {len(oversized_boxes)}")

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

    # Count classes (only confident parasite detections count as infected)
    from collections import Counter
    counts     = Counter(labels_list)
    n_parasites = sum(
        1 for lbl, conf in zip(labels_list, confs_list)
        if lbl in PARASITE_CLASSES and conf >= conf_threshold
    )
    n_total        = len(labels_list)
    infection_rate = n_parasites / n_total * 100 if n_total > 0 else 0.0

    print(f"\n  Valid cells detected : {n_total}")
    print(f"  Infected (conf≥{conf_threshold:.0%}): {n_parasites}")
    print(f"  Infection rate       : {infection_rate:.2f}%")
    print("  Class breakdown (all predictions):")
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
    smear_ann = build_annotated_smear(
        bgr, boxes, labels_list, confs_list,
        oversized_boxes=oversized_boxes,
        conf_threshold=conf_threshold,
    )
    cv2.imwrite(str(out_dir / "smear_annotated.jpg"), smear_ann)

    # Card 2: crop gallery
    gallery = build_crop_gallery(crops_rgb, labels_list, confs_list)
    cv2.imwrite(str(out_dir / "crop_gallery.jpg"), gallery)

    # Card 3: Grad-CAM++ gallery (confident parasite cells only)
    # Filter to confident parasites for Grad-CAM display
    confident_mask = [
        lbl in PARASITE_CLASSES and conf >= conf_threshold
        for lbl, conf in zip(labels_list, confs_list)
    ]
    gcam_gallery = build_gradcam_gallery(
        crops_rgb, heatmaps, labels_list, confs_list,
        confident_mask=confident_mask,
    )
    cv2.imwrite(str(out_dir / "gradcam_gallery.jpg"), gcam_gallery)

    # Full-image Grad-CAM++ overlay (confident parasites only)
    conf_boxes   = [b for b, m in zip(boxes,       confident_mask) if m]
    conf_heatmaps = [h for h, m in zip(heatmaps,   confident_mask) if m]
    conf_labels  = [l for l, m in zip(labels_list, confident_mask) if m]
    full_gcam = build_fullimage_gradcam(bgr, conf_boxes, conf_heatmaps, conf_labels)
    cv2.imwrite(str(out_dir / "fullimage_gradcam.jpg"), full_gcam)

    # JSON results
    results = {
        "image": args.image,
        "total_cells": n_total,
        "oversized_merged_regions": len(oversized_boxes),
        "infected_cells": n_parasites,
        "infection_rate_pct": round(infection_rate, 2),
        "conf_threshold": conf_threshold,
        "class_counts": dict(counts),
        "cells": [
            {
                "idx": i,
                "box": list(boxes[i]),
                "label": labels_list[i],
                "confidence": round(confs_list[i], 4),
                "heatmap_mean": round(float(heatmaps[i].mean()), 4),
                "confident": bool(
                    confs_list[i] >= conf_threshold
                    or labels_list[i] not in PARASITE_CLASSES
                ),
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
    p.add_argument("--conf-threshold", type=float, default=0.40,
                   dest="conf_threshold",
                   help="Minimum confidence to count a parasite as a confirmed "
                        "detection (default: 0.40). Below this the cell is shown "
                        "with a '?' suffix and excluded from Grad-CAM gallery.")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
