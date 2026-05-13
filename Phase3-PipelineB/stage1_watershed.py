"""
Phase3-PipelineB/stage1_watershed.py
======================================
Stage 1: Annotation-agnostic cell segmentation via distance-transform guided watershed.

Pipeline per image
------------------
  1. BGR → Grayscale
  2. Otsu threshold  → binary foreground mask (cells = white)
  3. Morphological opening  → remove small noise blobs
  4. Euclidean distance transform  → topology map of cell interiors
  5. Local-maxima detection  → one seed marker per cell centre
  6. Marker-based watershed  → instance-level cell separation
  7. Bounding-box extraction per labelled region
  8. 64×64 crop saved (reflect-padded if cell touches edge)

Two modes
---------
  --mode crops   : save every crop to --out-dir + write manifest.csv
                   (use this before Stage 2 training if you want to audit)
  --mode eval    : match watershed boxes against GT boxes (IoU ≥ 0.5),
                   report cell-recovery rate and dense-region recall

Usage (crops mode, for Stage 2 training audit):
    python Phase3-PipelineB/stage1_watershed.py \
        --json      data/malaria/training.json \
        --img-dir   data/malaria/images \
        --out-dir   data/crops \
        --mode      crops

Usage (eval mode, for paper metrics):
    python Phase3-PipelineB/stage1_watershed.py \
        --json      data/malaria/test.json \
        --img-dir   data/malaria/images \
        --mode      eval \
        --vis-dir   Phase3-PipelineB/results/watershed_vis \
        --n-vis     20

Kaggle paths:
    --json    /kaggle/input/bbbc041-malaria/malaria/training.json
    --img-dir /kaggle/input/bbbc041-malaria/malaria/images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.label_map import LABEL_TO_INT, INT_TO_LABEL, CLASS_COLOUR_RGB

SKIP_LABELS = {"difficult"}
CROP_SIZE   = 64          # output crop dimension (pixels)
MIN_AREA    = 200         # discard watershed regions smaller than this (noise)
MIN_DIST    = 18          # peak_local_max min_distance (px) — tune to cell size
OPEN_KSIZE  = 5           # morphological opening kernel size
DIST_THRESH = 0.30        # fraction of max distance → sure-foreground threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  Core watershed routine
# ═══════════════════════════════════════════════════════════════════════════════

def watershed_cells(
    bgr: np.ndarray,
    min_area: int   = MIN_AREA,
    min_dist: int   = MIN_DIST,
    open_k: int     = OPEN_KSIZE,
    dist_thr: float = DIST_THRESH,
) -> List[Tuple[int, int, int, int]]:
    """
    Run the full watershed pipeline on one BGR image.

    Returns
    -------
    boxes : list of (x1, y1, x2, y2) in pixel coordinates — one per detected cell.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 1 — Otsu threshold
    # Cells are darker than the pinkish background in Giemsa stain.
    # THRESH_BINARY_INV makes cells = 255 (foreground), background = 0.
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 2 — Morphological opening (remove thin noise filaments)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 3 — Distance transform
    dist   = ndi.distance_transform_edt(opened)

    # Step 4 — Sure foreground via local maxima (one seed per cell)
    # peak_local_max returns coordinates of maxima separated by ≥ min_dist px.
    coords  = peak_local_max(dist, min_distance=min_dist,
                              threshold_abs=dist_thr * dist.max() if dist.max() > 0 else 1,
                              labels=opened.astype(bool))
    mask    = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # Step 5 — Label seeds, then watershed
    markers, _ = ndi.label(mask)
    labels      = watershed(-dist, markers, mask=opened.astype(bool), compactness=0.1)

    # Step 6 — Bounding boxes from labelled regions
    boxes: List[Tuple[int, int, int, int]] = []
    h, w = bgr.shape[:2]
    for region_id in range(1, labels.max() + 1):
        ys, xs = np.where(labels == region_id)
        if len(ys) < min_area:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        # sanity check — skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))

    return boxes


# ═══════════════════════════════════════════════════════════════════════════════
#  Crop extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_crop(bgr: np.ndarray, box: Tuple[int, int, int, int],
                 size: int = CROP_SIZE) -> np.ndarray:
    """
    Extract a square crop centred on the bounding box, reflect-padding if needed.
    Returns an RGB uint8 array of shape (size, size, 3).
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half   = size // 2

    # Pad the full image so centre crops never go out of bounds
    padded = cv2.copyMakeBorder(bgr, half, half, half, half, cv2.BORDER_REFLECT)
    # Shift centre by half because of padding
    px, py = cx + half, cy + half
    crop_bgr = padded[py - half: py + half, px - half: px + half]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb


def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_gt(ws_box: Tuple, gt_boxes: List, gt_labels: List,
             iou_thr: float = 0.40) -> Tuple[str, float]:
    """
    Match one watershed box to the best-IoU ground-truth box.
    Returns (label_name, best_iou).  Label is 'background' if no match.
    """
    best_iou, best_label = 0.0, "background"
    for gb, gl in zip(gt_boxes, gt_labels):
        v = iou(ws_box, gb)
        if v > best_iou:
            best_iou, best_label = v, gl
    if best_iou < iou_thr:
        return "background", best_iou
    return best_label, best_iou


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_json(json_path: Path) -> List[dict]:
    """Return list of {filename, image_id, gt_boxes, gt_labels} dicts."""
    with open(json_path) as f:
        raw = json.load(f)

    records = []
    for item in raw:
        fname = item["image"]["pathname"].lstrip("/")
        if not fname:
            continue
        gt_boxes, gt_labels = [], []
        for ann in item.get("objects", []):
            lbl = ann.get("category", "")
            if lbl in SKIP_LABELS or lbl not in LABEL_TO_INT:
                continue
            bb = ann.get("bounding_box", {})
            x1 = int(bb.get("minimum", {}).get("c", 0))
            y1 = int(bb.get("minimum", {}).get("r", 0))
            x2 = int(bb.get("maximum", {}).get("c", 0))
            y2 = int(bb.get("maximum", {}).get("r", 0))
            if x2 > x1 and y2 > y1:
                gt_boxes.append((x1, y1, x2, y2))
                gt_labels.append(lbl)
        records.append({"filename": fname, "gt_boxes": gt_boxes, "gt_labels": gt_labels})
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def draw_watershed_vis(bgr: np.ndarray, ws_boxes: List,
                       gt_boxes: List, gt_labels: List) -> np.ndarray:
    """Draw GT boxes (green) and watershed boxes (blue) on the image."""
    vis = bgr.copy()
    for (x1, y1, x2, y2), lbl in zip(gt_boxes, gt_labels):
        rgb = CLASS_COLOUR_RGB.get(lbl, (0, 200, 0))
        bgr_c = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_c, 1)
    for (x1, y1, x2, y2) in ws_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 1)
    return vis


# ═══════════════════════════════════════════════════════════════════════════════
#  Crops mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_crops(records, img_dir: Path, out_dir: Path, vis_dir: Path | None,
              n_vis: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    with open(manifest_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image_file", "cell_idx", "crop_file",
                         "x1", "y1", "x2", "y2",
                         "gt_label", "gt_iou"])

        total_ws = 0
        for rec_idx, rec in enumerate(records):
            img_path = img_dir / rec["filename"]
            if not img_path.exists():
                # try basename only
                img_path = img_dir / Path(rec["filename"]).name
            if not img_path.exists():
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            ws_boxes = watershed_cells(bgr)
            total_ws += len(ws_boxes)

            for cell_idx, box in enumerate(ws_boxes):
                crop = extract_crop(bgr, box)
                label, best_iou = match_gt(box, rec["gt_boxes"], rec["gt_labels"])

                stem      = Path(rec["filename"]).stem
                crop_name = f"{stem}_cell_{cell_idx:04d}.png"
                crop_path = out_dir / crop_name

                from PIL import Image as PILImage
                PILImage.fromarray(crop).save(str(crop_path))

                writer.writerow([rec["filename"], cell_idx, crop_name,
                                 *box, label, f"{best_iou:.3f}"])

            if (rec_idx + 1) % 50 == 0:
                print(f"  [{rec_idx+1}/{len(records)}] "
                      f"total watershed cells so far: {total_ws}")

            # optional visualisation
            if vis_dir and rec_idx < n_vis:
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_watershed_vis(bgr, ws_boxes,
                                         rec["gt_boxes"], rec["gt_labels"])
                stem = Path(rec["filename"]).stem
                cv2.imwrite(str(vis_dir / f"{stem}_watershed.jpg"), vis)

    print(f"\nDone. {total_ws} crops saved to {out_dir}")
    print(f"Manifest: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval mode  (cell-recovery + dense-region recall)
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(records, img_dir: Path, vis_dir: Path | None, n_vis: int,
             dense_thresh: int = 100):
    """
    Compute:
      - Cell recovery rate : % of GT boxes whose centre is inside a watershed region
      - Dense-region recall: same metric restricted to images with ≥ dense_thresh GT boxes
    """
    total_gt   = 0
    recovered  = 0
    dense_gt   = 0
    dense_rec  = 0

    per_image  = []

    for rec_idx, rec in enumerate(records):
        img_path = img_dir / rec["filename"]
        if not img_path.exists():
            img_path = img_dir / Path(rec["filename"]).name
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        ws_boxes = watershed_cells(bgr)
        gt_boxes = rec["gt_boxes"]
        n_gt     = len(gt_boxes)

        # For each GT box, check if any ws_box has IoU ≥ 0.3 with it
        rec_count = 0
        for gb in gt_boxes:
            for wb in ws_boxes:
                if iou(gb, wb) >= 0.30:
                    rec_count += 1
                    break

        per_image.append({
            "file": rec["filename"],
            "n_gt": n_gt,
            "n_ws": len(ws_boxes),
            "recovered": rec_count,
            "recall": rec_count / n_gt if n_gt > 0 else 1.0,
        })

        total_gt  += n_gt
        recovered += rec_count

        if n_gt >= dense_thresh:
            dense_gt  += n_gt
            dense_rec += rec_count

        if (rec_idx + 1) % 20 == 0:
            print(f"  [{rec_idx+1}/{len(records)}] "
                  f"running recall={recovered/max(total_gt,1)*100:.1f}%")

        if vis_dir and rec_idx < n_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis  = draw_watershed_vis(bgr, ws_boxes, gt_boxes, rec["gt_labels"])
            stem = Path(rec["filename"]).stem
            cv2.imwrite(str(vis_dir / f"{stem}_eval.jpg"), vis)

    overall_recall = recovered / total_gt * 100 if total_gt > 0 else 0.0
    dense_recall   = dense_rec / dense_gt  * 100 if dense_gt  > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Stage 1 Evaluation Results")
    print(f"{'='*50}")
    print(f"Images processed    : {len(per_image)}")
    print(f"GT boxes total      : {total_gt}")
    print(f"GT boxes recovered  : {recovered}")
    print(f"Cell recovery rate  : {overall_recall:.2f}%")
    print(f"Dense images (≥{dense_thresh} GT): "
          f"{sum(1 for p in per_image if p['n_gt']>=dense_thresh)}")
    print(f"Dense-region recall : {dense_recall:.2f}%")
    print(f"{'='*50}")

    # Save summary
    results = {
        "total_gt": total_gt,
        "recovered": recovered,
        "cell_recovery_pct": round(overall_recall, 2),
        "dense_threshold": dense_thresh,
        "dense_gt": dense_gt,
        "dense_recovered": dense_rec,
        "dense_recall_pct": round(dense_recall, 2),
        "per_image": per_image,
    }
    out_path = Path("Phase3-PipelineB/checkpoints/stage1_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 — Annotation-agnostic watershed cell segmentation")
    p.add_argument("--json",     required=True, help="Path to training.json or test.json")
    p.add_argument("--img-dir",  required=True, help="Path to images folder")
    p.add_argument("--mode",     default="crops",
                   choices=["crops", "eval"],
                   help="'crops' saves 64x64 crops; 'eval' reports cell-recovery rate")
    p.add_argument("--out-dir",  default="data/crops",
                   help="[crops mode] Directory to save crop images + manifest.csv")
    p.add_argument("--vis-dir",  default=None,
                   help="Directory to save watershed visualisation images (optional)")
    p.add_argument("--n-vis",    type=int, default=20,
                   help="Number of visualisation images to save")
    p.add_argument("--min-area", type=int, default=MIN_AREA,
                   help="Minimum watershed region area in pixels (default 200)")
    p.add_argument("--min-dist", type=int, default=MIN_DIST,
                   help="Minimum distance between cell-centre peaks (default 18)")
    return p.parse_args()


def main():
    args    = parse_args()
    img_dir = Path(args.img_dir)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    print(f"Parsing {args.json} ...")
    records = parse_json(Path(args.json))
    print(f"  {len(records)} images loaded")

    t0 = time.time()

    if args.mode == "crops":
        run_crops(records, img_dir, Path(args.out_dir), vis_dir, args.n_vis)
    else:
        run_eval(records, img_dir, vis_dir, args.n_vis)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
