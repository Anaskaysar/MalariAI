"""
stage1_v3.py -- Stage 1 watershed v3: v2 + Gaussian blur + morphological close

Improvements over v2 (src/pipeline_b_v2/stage1_v2.py)
--------------------------------------------------------------
1. Gaussian blur (5x5) before Otsu -- smooths staining noise inside large
   infected cells, giving a cleaner binary mask.
2. Morphological CLOSE after OPEN -- seals interior holes in schizonts /
   trophozoites so they produce a single watershed seed rather than several.
3. REF_MIN_DIST raised 12 -> 16 -- seeds must be further apart, reducing
   over-segmentation of touching cells.
4. REF_AREA_MAX raised 8000 -> 14000 -- captures large schizonts that were
   previously clipped by the old ceiling.

Original v2 improvements over v1 (src/pipeline_b/stage1_watershed.py)
----------------------------------------------------------------------
1. CLAHE contrast normalisation (L channel, LAB colour space)
   Global Otsu thresholding (v1) assumes a unimodal background intensity.
   When applied across datasets with different staining (BBBC041 Giemsa vs
   MP-IDB Giemsa from a different lab), the Otsu threshold drifts and either
   clips pale cells into background or merges them with bright artefacts.
   CLAHE equalises contrast locally (8x8 tiles, clip_limit=2.0), making the
   cell/background boundary sharp regardless of absolute staining intensity.
   Applied to the L channel in LAB space so hue is not distorted.

2. Peak-based seed generation via scipy.ndimage.label + peak_local_max
   v1 seeds: binary threshold on normalised distance map (dist_norm >= 0.35).
   Problem: threshold is sensitive to global max distance -- one large merged
   blob sets the scale and causes under-seeding for small cells nearby.
   v2 seeds: scipy peak_local_max finds LOCAL maxima in the distance map with
   an explicit minimum separation (MIN_DIST). Each local peak becomes one
   watershed seed, regardless of absolute distance value.

3. Resolution-aware MIN_DIST and AREA bounds
   v1 constants (MIN_DIST implicit, AREA_MIN=150, AREA_MAX=8000) were tuned
   for BBBC041 at 1600x1200 px. Applying them unchanged to MP-IDB at
   2592x1944 px causes parameter mismatch:
     - Same AREA bounds mean cells at higher resolution are filtered differently
     - Same seed threshold merges more cells per image at higher resolution
   v2 auto-scales from a reference (1600x1200) using the geometric mean of
   the width and height scale factors:
     scale  = sqrt( (H * W) / (REF_H * REF_W) )
     MIN_DIST_px  = max(5, int(REF_MIN_DIST * scale))
     AREA_MIN_px  = int(REF_AREA_MIN * scale**2)
     AREA_MAX_px  = int(REF_AREA_MAX * scale**2)
   This keeps the effective parameter values proportional to image resolution.
   Auto-scaling can be overridden with explicit CLI flags for manual tuning.

4. Drop-in replacement
   segment_cells() has the same signature and return format as v1:
     list of dicts: {x_min, y_min, x_max, y_max, area, label}
   e2e_eval.py uses this module when --stage1-version v2 is passed.

Usage
-----
  # Run on a single image (visualisation)
  python src/pipeline_b_v2/stage1_v2.py data/MP-IDB/img/some_image.jpg --out /tmp/out.jpg

  # With explicit parameter overrides
  python src/pipeline_b_v2/stage1_v2.py image.jpg --min-dist 20 --area-min 300

  # Evaluate via e2e_eval.py
  python src/pipeline_b_v2/e2e_eval.py --dataset mpidb ... --stage1-version v2

Parameter reference
-------------------
  REF_MIN_DIST = 16   # v3: raised from 12   # minimum px between watershed seeds @ 1600x1200
  REF_AREA_MIN = 150  # minimum cell area px^2 @ 1600x1200
  REF_AREA_MAX = 14000  # v3: raised from 8000 # maximum cell area px^2 @ 1600x1200
  CLAHE clip_limit = 2.0  (higher -> more contrast boost, more noise risk)
  CLAHE tile_grid  = 8x8  (smaller tiles -> more local, noisier)
  MORPH_KSIZE = 3         (morphological opening disk radius)
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# scipy is required for peak_local_max -- installed in the project environment
try:
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    warnings.warn(
        "scipy / scikit-image not found. stage1_v2 will fall back to v1 "
        "DIST_RATIO thresholding. Install with: "
        "pip install scipy scikit-image --break-system-packages",
        RuntimeWarning,
        stacklevel=2,
    )

# Reference constants (1600 x 1200 BBBC041 baseline)
REF_W        = 1600   # reference image width
REF_H        = 1200   # reference image height
REF_MIN_DIST = 16   # v3: raised from 12     # minimum distance between watershed seeds (px)
REF_AREA_MIN = 150    # minimum cell area to accept (px^2)
REF_AREA_MAX = 14000  # v3: raised from 8000   # maximum cell area to accept (px^2)
MORPH_KSIZE  = 3      # morphological opening disk radius
DIST_RATIO   = 0.35   # fallback: v1-style threshold (used if scipy missing)

# CLAHE parameters
CLAHE_CLIP   = 2.0    # contrast clip limit (higher = stronger local contrast)
CLAHE_TILE   = 8      # NxN tile grid size

# Resolution scaling

def resolution_scale(h: int, w: int) -> float:
    """
    Geometric mean scale factor relative to the BBBC041 reference resolution.

    scale = sqrt( (H * W) / (REF_H * REF_W) )

    Examples
    --------
    BBBC041 1600x1200  -> scale = 1.00
    MP-IDB  2592x1944  -> scale = 1.62
    """
    return math.sqrt((h * w) / (REF_H * REF_W))

def auto_params(h: int, w: int) -> tuple[int, int, int]:
    """
    Return (min_dist, area_min, area_max) scaled to the given image resolution.
    """
    s = resolution_scale(h, w)
    min_dist = max(5, int(REF_MIN_DIST * s))
    area_min = max(50, int(REF_AREA_MIN * s ** 2))
    area_max = int(REF_AREA_MAX * s ** 2)
    return min_dist, area_min, area_max

# CLAHE preprocessing

def apply_clahe(image_rgb: np.ndarray,
                clip_limit: float = CLAHE_CLIP,
                tile_grid: int   = CLAHE_TILE) -> np.ndarray:
    """
    Apply CLAHE to the L channel of the LAB colour space.

    Equalising the L (luminance) channel normalises local contrast without
    shifting hue or saturation. This compensates for staining intensity
    differences between datasets and for uneven slide illumination.

    Parameters
    ----------
    image_rgb : np.ndarray  RGB, uint8, shape (H, W, 3)
    clip_limit : float       CLAHE contrast clip limit (2.0 = moderate boost)
    tile_grid  : int         NxN grid size for local histogram equalisation

    Returns
    -------
    np.ndarray  RGB, uint8, same shape -- with normalised local contrast
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(tile_grid, tile_grid)
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Core segmentation function

def segment_cells(
    image: np.ndarray,
    min_dist:  Optional[int]   = None,   # None -> auto from image resolution
    area_min:  Optional[int]   = None,   # None -> auto from image resolution
    area_max:  Optional[int]   = None,   # None -> auto from image resolution
    clahe:     bool            = True,   # apply CLAHE before thresholding
    morph_ksize: int           = MORPH_KSIZE,
    normalise_stain: bool      = False,  # legacy v1 HSV equalisation (off)
    # v1 compatibility
    dist_ratio: float          = DIST_RATIO,
) -> list[dict]:
    """
    Segment all cells in a blood smear image using distance-transform watershed.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    min_dist : int or None
        Minimum distance in pixels between watershed seed peaks.
        None = auto-compute from image resolution (recommended).
    area_min : int or None
        Minimum region area in pixels to accept as a cell.
        None = auto-compute from image resolution.
    area_max : int or None
        Maximum region area in pixels to accept as a cell.
        None = auto-compute from image resolution.
    clahe : bool
        If True (default), apply CLAHE contrast normalisation before
        thresholding. Strongly recommended for cross-dataset evaluation.
    morph_ksize : int
        Morphological opening disk radius (removes staining noise).
    normalise_stain : bool
        Legacy v1 HSV histogram equalisation. Deprecated -- CLAHE is better.
        Kept for backward compatibility with scripts that pass this kwarg.
    dist_ratio : float
        Fallback DIST_RATIO for v1-style thresholding (used only when
        scipy/skimage are unavailable).

    Returns
    -------
    list of dicts with keys: x_min, y_min, x_max, y_max, area, label
    """
    H, W = image.shape[:2]

    # Resolve auto parameters
    _min_dist, _area_min, _area_max = auto_params(H, W)
    if min_dist is None:
        min_dist = _min_dist
    if area_min is None:
        area_min = _area_min
    if area_max is None:
        area_max = _area_max

    # Step 1: Contrast normalisation
    if normalise_stain:
        # Legacy v1 path: HSV V-channel equalisation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if clahe:
        image = apply_clahe(image, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_TILE)

    # Step 2: Grayscale + Otsu thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # v3: Gaussian blur smooths staining noise inside large infected cells
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY_INV: cells (dark) -> 1, background (light) -> 0
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 3: Morphological opening then closing
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )
    # v3: OPEN removes speckle; CLOSE fills holes inside large infected cells
    opened  = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=2)
    cleaned = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Distance transform
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, maskSize=5)

    # Step 5: Seed generation
    if _SCIPY_OK:
        # v2 path: find local maxima with explicit min_distance
        # peak_local_max returns (row, col) indices of peaks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress skimage deprecation
            coords = peak_local_max(
                dist,
                min_distance=min_dist,
                labels=cleaned.astype(bool),  # only find peaks inside fg
                exclude_border=False,
            )
        # Build seed mask: one pixel per peak
        seed_mask = np.zeros(dist.shape, dtype=np.uint8)
        if len(coords) > 0:
            seed_mask[coords[:, 0], coords[:, 1]] = 255
    else:
        # v1 fallback: threshold normalised distance map
        dist_norm = dist / (dist.max() + 1e-8)
        seed_mask = (dist_norm >= dist_ratio).astype(np.uint8) * 255

    # Step 6: Marker labelling
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, seed_mask)

    n_labels, markers = cv2.connectedComponents(seed_mask)
    # Shift: background label must be > 0 for OpenCV watershed
    markers = markers + 1
    markers[unknown == 255] = 0   # unknown region for watershed to fill

    # Step 7: Watershed
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)
    # After: -1 = boundary, 1 = background, >=2 = cell regions

    # Step 8: Extract bounding boxes
    results = []

    for label_id in range(2, n_labels + 2):
        mask = (markers == label_id).astype(np.uint8)
        area = int(mask.sum())

        if area < area_min or area > area_max:
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        x, y, w, h = cv2.boundingRect(contours[0])
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
            "label": label_id,
        })

    return results

# Convenience helpers (same as v1)

def segment_image_file(
    img_path: str | Path,
    **kwargs,
) -> tuple[np.ndarray, list[dict]]:
    """Load image from file, run segmentation, return (image_array, boxes)."""
    img_path = Path(img_path)
    image    = np.array(Image.open(img_path).convert("RGB"))
    boxes    = segment_cells(image, **kwargs)
    return image, boxes

def draw_boxes(
    image: np.ndarray,
    boxes: list[dict],
    colour: tuple = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw bounding boxes on a copy of the image."""
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(
            vis,
            (b["x_min"], b["y_min"]),
            (b["x_max"], b["y_max"]),
            colour, thickness,
        )
    return vis

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 v2 -- watershed cell segmentation with CLAHE"
    )
    parser.add_argument("image_path", help="Path to input blood smear image")
    parser.add_argument("--out",        default=None,  help="Save visualisation here")
    parser.add_argument("--min-dist",   type=int,      default=None,
                        help="Minimum px between seeds (default: auto from resolution)")
    parser.add_argument("--area-min",   type=int,      default=None,
                        help="Min cell area in px^2 (default: auto)")
    parser.add_argument("--area-max",   type=int,      default=None,
                        help="Max cell area in px^2 (default: auto)")
    parser.add_argument("--no-clahe",   action="store_true",
                        help="Disable CLAHE (reproduce v1 behaviour)")
    parser.add_argument("--stain-norm", action="store_true",
                        help="Legacy HSV V-channel equalisation (v1 compat)")
    parser.add_argument("--compare-v1", action="store_true",
                        help="Also run v1 and print side-by-side box counts")
    args = parser.parse_args()

    image, boxes = segment_image_file(
        args.image_path,
        min_dist        = args.min_dist,
        area_min        = args.area_min,
        area_max        = args.area_max,
        clahe           = not args.no_clahe,
        normalise_stain = args.stain_norm,
    )

    H, W = image.shape[:2]
    s    = resolution_scale(H, W)
    md, amn, amx = auto_params(H, W)

    print(f"\nImage   : {Path(args.image_path).name}  ({W}x{H})")
    print(f"Scale   : {s:.3f}x  (ref 1600x1200)")
    print(f"Params  : min_dist={args.min_dist or md}  "
          f"area_min={args.area_min or amn}  area_max={args.area_max or amx}")
    print(f"CLAHE   : {'OFF' if args.no_clahe else 'ON'}")
    print(f"Cells   : {len(boxes)}")

    if boxes:
        areas = [b["area"] for b in boxes]
        print(f"Area    : min={min(areas)}  max={max(areas)}  "
              f"median={int(np.median(areas))}")

    if args.compare_v1:
        # Import v1 for comparison
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.pipeline_b.stage1_watershed import segment_cells as seg_v1
        img_arr = np.array(Image.open(args.image_path).convert("RGB"))
        boxes_v1 = seg_v1(img_arr)
        print(f"\nv1 cells: {len(boxes_v1)}  (dist_ratio=0.35, fixed area bounds)")
        print(f"v2 cells: {len(boxes)}  (peak_local_max + CLAHE + auto bounds)")

    if args.out:
        vis     = draw_boxes(image, boxes)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out), vis_bgr)
        print(f"\nSaved -> {args.out}")
    else:
        print("\n(pass --out <path> to save a visualisation)")
