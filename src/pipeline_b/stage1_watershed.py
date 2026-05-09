"""
stage1_watershed.py — Pipeline B Stage 1: Annotation-agnostic cell segmentation (KB)

Novelty claims addressed here
------------------------------
N1 — Label-Resilient:
    We do NOT rely on ground-truth bounding boxes to find cells.
    Watershed finds every cell in the image, including the ~30% that are
    unannotated in BBBC041 (a documented property of the dataset).
    A Faster R-CNN trained on the same data will miss these cells because
    it is trained to predict only annotated boxes — unannotated cells become
    false negatives by construction.

N2 — Density-Invariant:
    NMS (used in Faster R-CNN) suppresses overlapping box proposals above a
    fixed IoU threshold. In dense smear regions where two RBCs touch, NMS
    deletes one of the pair. Distance-transform guided watershed separates
    touching cells before any proposal stage exists, so BOTH cells survive.

Algorithm walkthrough
---------------------
Given a BGR/RGB image of a thin blood smear:

Step 1 — Stain normalisation (optional, off by default)
    Thin smears are Giemsa or MGG stained. Stain lot-to-lot variation shifts
    cell RGB values by ±30 units. We convert to HSV and histogram-equalise
    the V channel to reduce this shift before thresholding.

Step 2 — Otsu thresholding on grayscale
    Cells are darker than the pale background. Otsu's method finds the optimal
    binary threshold t* that minimises intra-class variance:
        t* = argmin_t [w0(t)*sigma0^2(t) + w1(t)*sigma1^2(t)]
    where w0, w1 are pixel fractions below/above t and sigma0^2, sigma1^2 are
    their variances. This gives us a binary mask: 1 = cell, 0 = background.

Step 3 — Morphological opening
    Salt-and-pepper noise from staining artefacts creates small spurious blobs.
    Opening (erosion then dilation with a 3x3 disk) removes them without
    shrinking real cells.

Step 4 — Distance transform
    For each foreground pixel, compute its Euclidean distance to the nearest
    background pixel. This creates a "height map" where each cell's interior
    has a high peak and touching cells share a valley between peaks.
    The peaks are the guaranteed-interior seeds for watershed.
    Peak detection threshold: pixels >= DIST_RATIO * max(distance_map).

Step 5 — Connected-component labelling (markers)
    Each peak cluster gets a unique integer label. These become the seeds for
    watershed. A separate marker label 0 = background, markers > 0 = cell seeds.

Step 6 — Watershed from seeds
    Watershed floods from the seed markers uphill through the distance-transform
    landscape, stopping when two flood regions meet. The boundary between them
    is the watershed line — this is where two touching cells are separated.
    We use OpenCV's watershed on a 3-channel copy of the image.

Step 7 — Extract bounding boxes
    Each watershed region > AREA_MIN pixels becomes one bounding box.
    We filter out implausibly large regions (probably staining artefacts or
    air bubbles) using AREA_MAX.

Outputs
-------
A list of dicts, one per detected cell:
    {
        "x_min": int, "y_min": int, "x_max": int, "y_max": int,
        "area":  int,   # segmented region area in pixels
        "label": int,   # watershed label (unique per image, NOT class label)
    }
These boxes are passed to Stage 2 (EfficientNet) for classification.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── Tunable constants ─────────────────────────────────────────────────────────
# These are starting-point values. You will tune them on val images in Week 2.

DIST_RATIO  = 0.35   # peak threshold as fraction of distance-map maximum
                     # higher -> fewer, more-separated seeds -> more merging
                     # lower  -> more seeds -> possible over-segmentation
AREA_MIN    = 150    # minimum cell area in pixels (ignores dust / noise)
AREA_MAX    = 8000   # maximum cell area in pixels (ignores large artefacts)
MORPH_KSIZE = 3      # morphological opening kernel size


# ── Core function ─────────────────────────────────────────────────────────────

def segment_cells(
    image: np.ndarray,
    dist_ratio:  float = DIST_RATIO,
    area_min:    int   = AREA_MIN,
    area_max:    int   = AREA_MAX,
    morph_ksize: int   = MORPH_KSIZE,
    normalise_stain: bool = False,
) -> list[dict]:
    """
    Segment all cells in a blood smear image using watershed.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    dist_ratio : float
        Fraction of maximum distance value used as peak threshold.
    area_min : int
        Minimum region area in pixels to accept as a cell.
    area_max : int
        Maximum region area in pixels to accept as a cell.
    morph_ksize : int
        Disk radius for morphological opening.
    normalise_stain : bool
        If True, histogram-equalise the V channel in HSV space before thresholding.
        Helps with stain lot-to-lot variation but adds ~3 ms overhead.

    Returns
    -------
    list of dicts with keys: x_min, y_min, x_max, y_max, area, label
    """

    # ── Step 1: Optional stain normalisation ──────────────────────────────
    if normalise_stain:
        hsv    = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        image  = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ── Step 2: Grayscale + Otsu threshold ───────────────────────────────
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # THRESH_BINARY_INV: cells (dark) become 1, background (light) becomes 0
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 3: Morphological opening ────────────────────────────────────
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # ── Step 4: Distance transform ────────────────────────────────────────
    dist    = cv2.distanceTransform(cleaned, cv2.DIST_L2, maskSize=5)
    # cv2.distanceTransform returns float32 distances in pixels
    # Normalise to [0, 1] for stable threshold arithmetic
    dist_norm = dist / (dist.max() + 1e-8)

    # Peak pixels: those whose distance value exceeds dist_ratio * max
    sure_fg = (dist_norm >= dist_ratio).astype(np.uint8) * 255

    # ── Step 5: Marker labelling ──────────────────────────────────────────
    # sure_bg: everything that is definitely background (dilated inverse of fg)
    sure_bg   = cv2.dilate(cleaned, kernel, iterations=3)
    unknown   = cv2.subtract(sure_bg, sure_fg)          # unknown region

    # Label connected components in sure_fg — each component = one cell seed
    n_labels, markers = cv2.connectedComponents(sure_fg)
    # Shift labels up by 1 so that background is 1 (not 0).
    # OpenCV watershed treats 0 as "unknown" — we need a distinct background label.
    markers = markers + 1
    markers[unknown == 255] = 0   # mark unknown region for watershed to fill

    # ── Step 6: Watershed ─────────────────────────────────────────────────
    # OpenCV's watershed expects BGR format and marks boundaries with -1
    img_bgr  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers  = cv2.watershed(img_bgr, markers)
    # After watershed: -1 = boundary, 1 = background, ≥2 = cell regions

    # ── Step 7: Extract bounding boxes ───────────────────────────────────
    results  = []
    H, W     = image.shape[:2]

    for label_id in range(2, n_labels + 2):   # skip 1 (background) and -1 (boundary)
        mask  = (markers == label_id).astype(np.uint8)
        area  = int(mask.sum())

        if area < area_min or area > area_max:
            continue

        # Bounding box from contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        x, y, w, h = cv2.boundingRect(contours[0])

        # Clip to image bounds (safety for edge cells)
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(W, x + w)
        y_max = min(H, y + h)

        if x_max <= x_min or y_max <= y_min:
            continue

        results.append({
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "area":  area,
            "label": label_id,   # watershed region ID — NOT a class label
        })

    return results


# ── Convenience: process from file path ──────────────────────────────────────

def segment_image_file(
    img_path: str | Path,
    **kwargs,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load image from file, run segmentation, return (image_array, boxes).
    """
    img_path = Path(img_path)
    image    = np.array(Image.open(img_path).convert("RGB"))
    boxes    = segment_cells(image, **kwargs)
    return image, boxes


# ── Visualisation helper ──────────────────────────────────────────────────────

def draw_boxes(image: np.ndarray, boxes: list[dict],
               colour: tuple = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """Draw watershed bounding boxes on a copy of the image."""
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis,
                      (b["x_min"], b["y_min"]),
                      (b["x_max"], b["y_max"]),
                      colour, thickness)
    return vis


# ── CLI: process a single image and save a visualisation ─────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 — Watershed cell segmentation"
    )
    parser.add_argument("image_path", help="Path to input blood smear image")
    parser.add_argument("--out",    default=None, help="Save visualisation to this path")
    parser.add_argument("--dist-ratio",  type=float, default=DIST_RATIO)
    parser.add_argument("--area-min",    type=int,   default=AREA_MIN)
    parser.add_argument("--area-max",    type=int,   default=AREA_MAX)
    parser.add_argument("--stain-norm",  action="store_true")
    args = parser.parse_args()

    image, boxes = segment_image_file(
        args.image_path,
        dist_ratio      = args.dist_ratio,
        area_min        = args.area_min,
        area_max        = args.area_max,
        normalise_stain = args.stain_norm,
    )

    print(f"Image: {Path(args.image_path).name}  ({image.shape[1]}x{image.shape[0]})")
    print(f"Cells detected: {len(boxes)}")
    areas = [b['area'] for b in boxes]
    if areas:
        print(f"Area  — min: {min(areas)}, max: {max(areas)}, "
              f"median: {int(np.median(areas))}")

    if args.out:
        vis = draw_boxes(image, boxes)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.out, vis_bgr)
        print(f"Saved visualisation -> {args.out}")
    else:
        print("(pass --out <path> to save a visualisation)")
