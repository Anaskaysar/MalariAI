"""
src/pipeline_b_v2/e2e_eval.py — MalariAI v2: End-to-End Evaluation Framework
==============================================================================

What this script does
---------------------
Runs the full two-stage pipeline (Stage 1 Watershed + Stage 2 EfficientNet-B0)
over a test set and computes rigorous detection metrics:

    Stage-1 only  (annotation-agnostic cell detection)
    • Cell Recovery Rate (Recall@IoU0.5) — fraction of GT cells found
    • Detection Precision                — fraction of WS boxes that are real cells
    • F1 score
    • Per-class cell recovery rate
    • Relaxed IoU recalls (IoU@0.25, IoU@0.30) — fairer to organic WS boundaries
    • Centroid-in-GT-box recall          — boundary-independent spatial localization
    • Infected-cell sensitivity          — recall over parasitized GT cells only
    • Biological localization recall     — centroid within 0.5x GT diagonal

    End-to-End  (full pipeline: box + class)
    • Per-class AP@0.5 (Average Precision at IoU threshold 0.5)
    • mAP@0.5          (mean over parasite classes)
    • Binary Parasitized AP@0.5
    • PR curves saved as PNG

Supported datasets
------------------
    --dataset bbbc041   BBBC041 test split (test_annotations.csv)
                        Labels: red blood cell / ring / trophozoite / schizont /
                                gametocyte / leukocyte
    --dataset mpidb     MP-IDB (mpidb_annotations.csv)
                        Labels: falciparum / vivax / malariae / ovale
                        Evaluation: binary parasitized detection only
                        (species taxonomy ≠ BBBC041 stage taxonomy)

Output
------
    <out-dir>/
    ├- metrics.json          all numeric results
    ├- pr_curves.png         precision-recall curves per class
    └- stage1_stats.json     Stage 1 cell recovery breakdown

Usage
-----
    # BBBC041 test set (full e2e):
    python src/pipeline_b_v2/e2e_eval.py \\
        --dataset    bbbc041 \\
        --img-dir    data/malaria/images \\
        --ann-csv    data/processed/test_annotations.csv \\
        --checkpoint Phase3-PipelineB/checkpoints/best.pth \\
        --out-dir    results/v2/e2e_bbbc041

    # MP-IDB cross-dataset (binary parasitized detection):
    python src/pipeline_b_v2/e2e_eval.py \\
        --dataset    mpidb \\
        --img-dir    data/MP-IDB/img \\
        --ann-csv    data/processed/mpidb_annotations.csv \\
        --checkpoint Phase3-PipelineB/checkpoints/best.pth \\
        --out-dir    results/v2/e2e_mpidb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# torch / torchvision are only needed for Stage 2 — import lazily so that
# --stage1-only mode works even without a PyTorch installation.
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase3-PipelineB"))

from shared.label_map import (
    NUM_CLASSES, INT_TO_LABEL, LABEL_TO_INT, PARASITE_CLASSES
)
from stage1_watershed import watershed_cells as _watershed_cells_v1

# stage1_v2 is in the same package directory
_STAGE1_V2_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(_STAGE1_V2_PATH))
try:
    from stage1_v2 import segment_cells as _segment_cells_v2
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False

try:
    from stage1_v3 import segment_cells as _segment_cells_v3
    _V3_AVAILABLE = True
except ImportError:
    _V3_AVAILABLE = False

# Active stage1 function — set by CLI arg at startup (default: v1)
_stage1_version = "v1"

def _run_stage1(bgr: np.ndarray) -> list:
    """
    Dispatch to v1 or v2 Stage 1 depending on --stage1-version flag.
    Both return a list of (x_min, y_min, x_max, y_max) tuples.
    """
    if _stage1_version == "v2":
        if not _V2_AVAILABLE:
            raise RuntimeError(
                "stage1_v2.py not found. Check src/pipeline_b_v2/stage1_v2.py exists."
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # v2 returns dicts; convert to tuples for consistency with v1 and evaluate logic
        res = _segment_cells_v2(rgb)
        return [(d["x_min"], d["y_min"], d["x_max"], d["y_max"]) for d in res]
    elif _stage1_version == "v3":
        if not _V3_AVAILABLE:
            raise RuntimeError(
                "stage1_v3.py not found. Check src/pipeline_b_v2/stage1_v3.py exists."
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = _segment_cells_v3(rgb)
        return [(d["x_min"], d["y_min"], d["x_max"], d["y_max"]) for d in res]
    else:
        # v1 already returns tuples
        return _watershed_cells_v1(bgr)

# Dataset-specific class maps
# For MP-IDB: species labels → binary "parasitized"
MPIDB_LABEL_TO_INT = {
    "falciparum": 1, "vivax": 2, "malariae": 3, "ovale": 4,
}

# Which classes count as "parasitized" for each dataset
BBBC041_PARASITE_CLASSES = set(PARASITE_CLASSES)        # ring, trophozoite, schizont, gametocyte
MPIDB_PARASITE_CLASSES   = set(MPIDB_LABEL_TO_INT.keys())  # all 4 species

# Classes to compute AP for (per dataset)
BBBC041_EVAL_CLASSES = ["ring", "trophozoite", "schizont", "gametocyte", "leukocyte"]
MPIDB_EVAL_CLASSES   = ["parasitized"]  # binary only

# Stage 2 classifier max cell size guard (prevents classifying merged blobs)
MAX_CELL_W = 220
MAX_CELL_H = 220

# ImageNet transform for EfficientNet-B0 (only built when torch is available)
if _TORCH_AVAILABLE:
    _NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _TRANSFORM = T.Compose([T.Resize((64, 64)), T.ToTensor(), _NORMALIZE])
else:
    _NORMALIZE = None
    _TRANSFORM = None

# IoU + matching + alternative localization helpers

def iou(a: Tuple, b: Tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def box_centroid(box: Tuple) -> Tuple[float, float]:
    """Return (cx, cy) of a (x1, y1, x2, y2) box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

def centroid_in_box(wb: Tuple, gb: Tuple) -> bool:
    """True if the centroid of watershed box wb lies inside GT box gb."""
    cx, cy = box_centroid(wb)
    return gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3]

def box_diagonal(box: Tuple) -> float:
    """Euclidean diagonal length of a (x1, y1, x2, y2) box."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (w ** 2 + h ** 2) ** 0.5

def biological_localization_match(wb: Tuple, gb: Tuple, fraction: float = 0.5) -> bool:
    """
    Biological localization criterion: the centroid of the watershed box wb
    must lie within `fraction * diagonal(gb)` pixels of the centroid of GT box gb.

    This is a clinically motivated alternative to strict IoU@0.5.  A cell is
    considered "found" when the detector correctly identifies its spatial
    location to within half the cell's own diameter — sufficient for a
    pathologist to verify the detection.

    Default fraction=0.5 means: allowed distance <= 0.5 * GT box diagonal.
    For a typical BBBC041 RBC (65x65 px, diagonal ~92 px) this allows
    up to ~46 px displacement — roughly one cell radius.
    """
    cx_w, cy_w = box_centroid(wb)
    cx_g, cy_g = box_centroid(gb)
    diag = box_diagonal(gb)
    dist = ((cx_w - cx_g) ** 2 + (cy_w - cy_g) ** 2) ** 0.5
    return dist <= fraction * diag

def match_predictions_to_gt(
    pred_boxes:  List[Tuple],
    pred_labels: List[str],
    pred_scores: List[float],
    gt_boxes:    List[Tuple],
    gt_labels:   List[str],
    iou_thr:     float = 0.50,
) -> List[Dict]:
    """
    Match predicted boxes to GT boxes for one image.
    Each GT box may be matched at most once (greedy, highest-score first).

    Returns list of dicts with keys:
        pred_label, pred_score, gt_label, matched (bool), iou_val
    """
    # Sort predictions by descending confidence
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])

    gt_matched = [False] * len(gt_boxes)
    records    = []

    for i in order:
        pb = pred_boxes[i]
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou, best_j = v, j

        matched = best_iou >= iou_thr
        gt_label = gt_labels[best_j] if matched else "background"

        if matched:
            gt_matched[best_j] = True

        records.append({
            "pred_label": pred_labels[i],
            "pred_score": pred_scores[i],
            "gt_label":   gt_label,
            "matched":    matched,
            "iou_val":    best_iou,
        })

    return records

# AP computation
def compute_ap(
    all_records: List[Dict],
    class_name:  str,
    n_gt_total:  int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision for one class using the 11-point interpolation
    method (consistent with VOC 2007).

    Parameters
    ----------
    all_records  : flat list of match dicts from all images (pred_label, matched,
                   pred_score, gt_label)
    class_name   : class to evaluate
    n_gt_total   : total number of GT boxes for this class across all images

    Returns
    -------
    ap, precision_curve, recall_curve
    """
    # Keep only predictions for this class, sorted by descending confidence
    class_preds = [r for r in all_records if r["pred_label"] == class_name]
    class_preds.sort(key=lambda r: -r["pred_score"])

    if not class_preds or n_gt_total == 0:
        return 0.0, np.array([]), np.array([])

    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))

    for i, rec in enumerate(class_preds):
        if rec["matched"] and rec["gt_label"] == class_name:
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall    = cum_tp / n_gt_total
    precision = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolated AP (VOC 2007)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        ap += prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
    ap /= 11.0

    return float(ap), precision, recall

# Model loading + inference
def load_model(checkpoint_path: str, device: "torch.device") -> "nn.Module":  # type: ignore[name-defined]
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"  Checkpoint loaded: {checkpoint_path}")
    else:
        print("  WARNING: No checkpoint loaded — using ImageNet-only weights.")
    return model.to(device).eval()

def classify_batch(
    model,
    bgr:   np.ndarray,
    boxes: List[Tuple],
    device,
    batch_size: int = 64,
) -> Tuple[List[str], List[float]]:
    """
    Extract 64×64 crops from `bgr`, run Stage 2 classifier.
    Returns (labels, confidences) aligned with `boxes`.
    """
    labels, confs = [], []
    with torch.no_grad():
        for start in range(0, len(boxes), batch_size):
            batch_boxes = boxes[start:start + batch_size]
            pils = []
            for (x1, y1, x2, y2) in batch_boxes:
                crop_bgr = bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    crop_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pils.append(Image.fromarray(crop_rgb))
            tensors = torch.stack([_TRANSFORM(p) for p in pils]).to(device)
            logits  = model(tensors)
            probs   = torch.softmax(logits, dim=1)
            preds   = logits.argmax(1)
            for pred, prob in zip(preds.cpu(), probs.cpu()):
                labels.append(INT_TO_LABEL[int(pred)])
                confs.append(float(prob[pred]))
    return labels, confs

def is_oversized(box: Tuple) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) > MAX_CELL_W or (y2 - y1) > MAX_CELL_H

# Stage 1 — cell-level recall/precision (dataset-agnostic)
def evaluate_stage1(
    gt_df:   pd.DataFrame,
    img_dir: Path,
    parasite_classes: set,
    iou_thr: float = 0.50,
) -> Dict:
    """
    Run watershed on each image, match to GT, report cell recovery rate.
    No Stage 2 classification here — pure detection evaluation.

    Beyond the standard IoU@0.5 recall, this function also computes four
    alternative localization metrics that are fairer to watershed-based
    detectors (which produce organic boundaries, not rectangular proposals):

      1. Centroid-in-GT-box recall  — a WS box is a "hit" if its centroid
         falls inside the GT box, regardless of boundary IoU.  This tests
         spatial localization without penalising non-rectangular regions.

      2. Relaxed IoU recalls        — recall at IoU@0.25 and IoU@0.30 in
         addition to the standard IoU@0.50.  Watershed boundaries grow
         from distance-transform seeds and do not align with rectangular
         GT annotations even when the correct cell is found.

      3. Infected-cell sensitivity  — recall computed only over parasitized
         GT boxes (ring, trophozoite, schizont, gametocyte / all MP-IDB
         species).  For clinical malaria diagnosis, missing infected cells
         is the critical failure mode; missing RBCs is not.

      4. Biological localization recall — a WS box is a "hit" if its centroid
         lies within 0.5 * diagonal(GT box) of the GT centroid.  For a
         typical BBBC041 RBC (65x65 px) this allows ~46 px displacement,
         roughly one cell radius — sufficient for a clinician to verify
         the detection location.
    """
    RELAXED_THRESHOLDS = [0.25, 0.30, 0.50]   # IoU thresholds to report
    BIO_FRACTION       = 0.50                  # fraction of GT diagonal for bio-loc

    images = gt_df["img_name"].unique()
    total_gt   = 0
    total_ws   = 0

    # IoU-based counters (one set per threshold)
    tp_at = {t: 0 for t in RELAXED_THRESHOLDS}
    fp_at = {t: 0 for t in RELAXED_THRESHOLDS}

    per_class_tp = {t: defaultdict(int) for t in RELAXED_THRESHOLDS}
    per_class_gt = defaultdict(int)

    # Infected-cell TP at each IoU threshold
    inf_tp_at  = {t: 0 for t in RELAXED_THRESHOLDS}
    total_inf_gt = 0   # total parasitized GT boxes

    # Alternative metric counters
    centroid_tp    = 0   # metric 1: centroid-in-box
    bio_loc_tp     = 0   # metric 4: biological localization (centroid distance)

    for img_name in tqdm(images, desc="  Stage 1 eval"):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        # GT for this image
        img_gt = gt_df[gt_df["img_name"] == img_name]
        gt_boxes  = [(int(r.x_min), int(r.y_min), int(r.x_max), int(r.y_max))
                     for r in img_gt.itertuples()]
        gt_labels = [r.label for r in img_gt.itertuples()]

        # Stage 1
        ws_boxes = [b for b in _run_stage1(bgr) if not is_oversized(b)]

        total_gt += len(gt_boxes)
        total_ws += len(ws_boxes)

        # Count GT per class
        for lbl in gt_labels:
            per_class_gt[lbl] += 1
        total_inf_gt += sum(1 for l in gt_labels if l in parasite_classes)

        # IoU matching at each threshold (separate greedy match per thr) -
        for thr in RELAXED_THRESHOLDS:
            gt_matched = [False] * len(gt_boxes)
            ws_tp_img  = 0
            inf_tp_img = 0

            for wb in ws_boxes:
                best_iou_val, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if gt_matched[j]:
                        continue
                    v = iou(wb, gb)
                    if v > best_iou_val:
                        best_iou_val, best_j = v, j
                if best_iou_val >= thr and best_j >= 0:
                    gt_matched[best_j] = True
                    ws_tp_img += 1
                    per_class_tp[thr][gt_labels[best_j]] += 1
                    if gt_labels[best_j] in parasite_classes:
                        inf_tp_img += 1

            tp_at[thr] += ws_tp_img
            fp_at[thr] += len(ws_boxes) - ws_tp_img
            inf_tp_at[thr] += inf_tp_img

        # Metric 1: centroid-in-GT-box (each GT matched at most once)
        gt_centroid_matched = [False] * len(gt_boxes)
        for wb in ws_boxes:
            for j, gb in enumerate(gt_boxes):
                if gt_centroid_matched[j]:
                    continue
                if centroid_in_box(wb, gb):
                    gt_centroid_matched[j] = True
                    centroid_tp += 1
                    break  # this WS box can match at most one GT

        # Metric 4: biological localization (each GT matched at most once) -
        gt_bio_matched = [False] * len(gt_boxes)
        for wb in ws_boxes:
            for j, gb in enumerate(gt_boxes):
                if gt_bio_matched[j]:
                    continue
                if biological_localization_match(wb, gb, BIO_FRACTION):
                    gt_bio_matched[j] = True
                    bio_loc_tp += 1
                    break

    # Aggregate metrics

    # Standard IoU@0.5 recall / precision / F1
    tp50      = tp_at[0.50]
    fp50      = fp_at[0.50]
    fn50      = total_gt - tp50
    recall50  = tp50 / total_gt    if total_gt > 0 else 0.0
    prec50    = tp50 / total_ws    if total_ws > 0 else 0.0
    f1_50     = (2 * prec50 * recall50 / (prec50 + recall50)
                 if (prec50 + recall50) > 0 else 0.0)

    # Relaxed IoU recalls
    relaxed_recalls = {}
    for thr in RELAXED_THRESHOLDS:
        relaxed_recalls[f"iou{int(thr*100):02d}"] = round(
            tp_at[thr] / total_gt if total_gt > 0 else 0.0, 4
        )

    # Centroid-in-box recall (metric 1)
    centroid_recall = centroid_tp / total_gt if total_gt > 0 else 0.0

    # Biological localization recall (metric 4)
    bio_recall = bio_loc_tp / total_gt if total_gt > 0 else 0.0

    # Infected-cell sensitivity at each threshold
    inf_sensitivity = {}
    for thr in RELAXED_THRESHOLDS:
        inf_sensitivity[f"iou{int(thr*100):02d}"] = round(
            inf_tp_at[thr] / total_inf_gt if total_inf_gt > 0 else 0.0, 4
        )

    # Per-class stats (at standard IoU@0.5)
    per_class = {}
    for lbl in per_class_gt:
        tp_lbl = per_class_tp[0.50].get(lbl, 0)
        per_class[lbl] = {
            "gt_count": per_class_gt[lbl],
            "tp":       tp_lbl,
            "recall":   round(tp_lbl / per_class_gt[lbl], 4),
        }

    return {
        # Standard IoU@0.5 metrics
        "total_gt_cells":         total_gt,
        "total_watershed_boxes":  total_ws,
        "tp": tp50, "fp": fp50, "fn": fn50,
        "recall_at_iou50":    round(recall50, 4),
        "precision_at_iou50": round(prec50,   4),
        "f1_at_iou50":        round(f1_50,    4),
        # Alternative localization metrics
        # Metric 2: relaxed IoU thresholds (includes iou25, iou30, iou50)
        "recall_at_relaxed_iou":  relaxed_recalls,
        # Metric 1: centroid-in-GT-box (boundary-independent spatial localization)
        "centroid_in_box_recall": round(centroid_recall, 4),
        "centroid_in_box_tp":     centroid_tp,
        # Metric 3: infected-cell sensitivity (parasite-only recall)
        "total_infected_gt":      total_inf_gt,
        "infected_cell_sensitivity": inf_sensitivity,
        # Metric 4: biological localization (centroid within 0.5 * GT diagonal)
        "bio_localization_recall":        round(bio_recall, 4),
        "bio_localization_fraction_used": BIO_FRACTION,
        "bio_localization_tp":            bio_loc_tp,
        # Per-class breakdown (at IoU@0.5)
        "per_class":          per_class,
    }

# End-to-end evaluation
def evaluate_e2e(
    gt_df:            pd.DataFrame,
    img_dir:          Path,
    model:            nn.Module,
    device:           torch.device,
    dataset:          str,
    iou_thr:          float = 0.50,
) -> Dict:
    """
    Full pipeline: watershed → EfficientNet-B0 → per-class AP@0.5 + mAP.

    For BBBC041: evaluates all foreground classes.
    For MP-IDB: collapses species to 'parasitized' for binary AP.
    """
    is_mpidb     = (dataset == "mpidb")
    eval_classes = MPIDB_EVAL_CLASSES if is_mpidb else BBBC041_EVAL_CLASSES
    parasite_set = MPIDB_PARASITE_CLASSES if is_mpidb else BBBC041_PARASITE_CLASSES

    all_records  = []           # flat list of per-prediction match records
    gt_counts    = defaultdict(int)   # per-class GT counts

    images = gt_df["img_name"].unique()

    for img_name in tqdm(images, desc="  E2E eval"):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        # GT
        img_gt = gt_df[gt_df["img_name"] == img_name]
        gt_boxes  = [(int(r.x_min), int(r.y_min), int(r.x_max), int(r.y_max))
                     for r in img_gt.itertuples()]
        raw_gt_labels = [r.label for r in img_gt.itertuples()]

        # For MP-IDB: collapse species → 'parasitized'
        if is_mpidb:
            gt_labels = ["parasitized" if l in parasite_set else l
                         for l in raw_gt_labels]
        else:
            gt_labels = raw_gt_labels

        # Count GT
        for lbl in gt_labels:
            gt_counts[lbl] += 1

        # Stage 1
        ws_boxes = [b for b in _run_stage1(bgr) if not is_oversized(b)]
        if not ws_boxes:
            continue

        # Stage 2
        pred_labels, pred_scores = classify_batch(model, bgr, ws_boxes, device)

        # For MP-IDB: remap Stage 2 output to binary (any parasite → 'parasitized')
        if is_mpidb:
            pred_labels = ["parasitized" if l in BBBC041_PARASITE_CLASSES else l
                           for l in pred_labels]

        # Match predictions → GT for this image
        records = match_predictions_to_gt(
            ws_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels,
            iou_thr=iou_thr,
        )
        all_records.extend(records)

    # Per-class AP
    ap_per_class = {}
    for cls in eval_classes:
        ap, prec_curve, rec_curve = compute_ap(
            all_records, cls, gt_counts.get(cls, 0)
        )
        ap_per_class[cls] = {
            "ap":            round(ap, 4),
            "n_gt":          gt_counts.get(cls, 0),
            "precision_curve": prec_curve.tolist(),
            "recall_curve":    rec_curve.tolist(),
        }

    # mAP (over eval classes with at least 1 GT box)
    valid_aps = [v["ap"] for v in ap_per_class.values() if v["n_gt"] > 0]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

    # Binary parasitized AP (applicable to both datasets)
    if not is_mpidb:
        # For BBBC041: remap to binary parasitized in a copy of records
        binary_records = []
        for r in all_records:
            br = dict(r)
            br["pred_label"] = ("parasitized"
                                if r["pred_label"] in BBBC041_PARASITE_CLASSES
                                else r["pred_label"])
            br["gt_label"]   = ("parasitized"
                                if r["gt_label"] in BBBC041_PARASITE_CLASSES
                                else r["gt_label"])
            binary_records.append(br)
        n_parasitized_gt = sum(1 for l in gt_counts
                               if l in BBBC041_PARASITE_CLASSES
                               for _ in range(gt_counts[l]))
        bin_ap, _, _ = compute_ap(binary_records, "parasitized",
                                   sum(gt_counts[l] for l in gt_counts
                                       if l in BBBC041_PARASITE_CLASSES))
    else:
        bin_ap = mAP  # already binary

    return {
        "iou_threshold":     iou_thr,
        "dataset":           dataset,
        "n_images_evaluated": len(images),
        "gt_counts":         dict(gt_counts),
        "ap_per_class":      ap_per_class,
        "mAP_at_iou50":      round(mAP, 4),
        "binary_parasitized_AP": round(bin_ap, 4),
    }

# Plotting
def plot_pr_curves(e2e_results: Dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping PR curve plot.")
        return

    ap_per_class = e2e_results["ap_per_class"]
    n = len(ap_per_class)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(f"Precision-Recall Curves — {e2e_results['dataset'].upper()} "
                 f"(mAP@0.5 = {e2e_results['mAP_at_iou50']:.3f})", fontsize=13)

    for ax, (cls, data) in zip(axes[0], ap_per_class.items()):
        rec  = np.array(data["recall_curve"])
        prec = np.array(data["precision_curve"])
        if rec.size > 0:
            ax.plot(rec, prec, lw=2, color="steelblue")
            ax.fill_between(rec, prec, alpha=0.15, color="steelblue")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
        ax.set_title(f"{cls}\nAP@0.5 = {data['ap']:.3f}  (n_gt={data['n_gt']})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PR curves saved -> {out_path}")

# Main
def main():
    p = argparse.ArgumentParser(
        description="MalariAI v2 — End-to-End Evaluation Framework"
    )
    p.add_argument("--dataset",    choices=["bbbc041", "mpidb"], required=True)
    p.add_argument("--img-dir",    required=True,
                   help="Directory containing test images")
    p.add_argument("--ann-csv",    required=True,
                   help="Annotation CSV (test_annotations.csv or mpidb_annotations.csv)")
    p.add_argument("--checkpoint", default="Phase3-PipelineB/checkpoints/best.pth")
    p.add_argument("--out-dir",    default="results/v2/e2e")
    p.add_argument("--iou-thr",    type=float, default=0.50,
                   help="IoU threshold for GT matching (default 0.50)")
    p.add_argument("--stage1-version", choices=["v1", "v2", "v3"], default="v1",
                   help="Stage 1 implementation: v1=original watershed, "
                        "v2=CLAHE+resolution-aware (default: v1)")
    p.add_argument("--stage1-only", action="store_true",
                   help="Run Stage 1 evaluation only (skip Stage 2 + AP computation)")
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit evaluation to N images (for quick smoke test)")
    args = p.parse_args()

    # Wire Stage 1 version to the module-level dispatcher
    global _stage1_version
    _stage1_version = args.stage1_version
    print(f"Stage 1 : {_stage1_version.upper()}  "
          f"({'CLAHE + peak_local_max + auto-scale' if _stage1_version == 'v2' else 'original Otsu + dist_ratio'})")

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir  = Path(args.img_dir)
    ann_csv  = Path(args.ann_csv)

    print(f"\n{'='*60}")
    print(f"MalariAI v2 — End-to-End Evaluation")
    print(f"Dataset   : {args.dataset.upper()}")
    print(f"Ann CSV   : {ann_csv}")
    print(f"Image dir : {img_dir}")
    print(f"IoU thr   : {args.iou_thr}")
    print(f"{'='*60}\n")

    # Load GT
    gt_df = pd.read_csv(ann_csv)
    print(f"Loaded {len(gt_df):,} GT boxes from {ann_csv.name}")

    # Optionally limit images
    if args.max_images:
        images = gt_df["img_name"].unique()[:args.max_images]
        gt_df  = gt_df[gt_df["img_name"].isin(images)].reset_index(drop=True)
        print(f"  Limited to {args.max_images} images for smoke test.")

    parasite_set = (MPIDB_PARASITE_CLASSES if args.dataset == "mpidb"
                    else BBBC041_PARASITE_CLASSES)

    # Stage 1 evaluation
    print("\n[Stage 1] Cell detection evaluation (watershed only) ...")
    t0 = time.time()
    s1_results = evaluate_stage1(gt_df, img_dir, parasite_set, args.iou_thr)
    s1_time = time.time() - t0

    s1_path = out_dir / "stage1_stats.json"
    with open(s1_path, "w") as f:
        json.dump(s1_results, f, indent=2)

    print(f"\n  Stage 1 results ({s1_time:.1f}s):")
    print(f"    GT cells        : {s1_results['total_gt_cells']:,}")
    print(f"    Watershed boxes : {s1_results['total_watershed_boxes']:,}")
    print(f"    TP / FP / FN    : {s1_results['tp']} / {s1_results['fp']} / {s1_results['fn']}")

    print(f"\n  -- Standard IoU metrics --")
    print(f"    Recall @IoU0.5  : {s1_results['recall_at_iou50']:.4f}")
    print(f"    Precision@IoU0.5: {s1_results['precision_at_iou50']:.4f}")
    print(f"    F1 @IoU0.5      : {s1_results['f1_at_iou50']:.4f}")

    print(f"\n  -- Alternative localization metrics --")
    ri = s1_results["recall_at_relaxed_iou"]
    print(f"    Recall @IoU0.25 : {ri.get('iou25', 0):.4f}   (relaxed boundary criterion)")
    print(f"    Recall @IoU0.30 : {ri.get('iou30', 0):.4f}   (relaxed boundary criterion)")
    print(f"    Recall @IoU0.50 : {ri.get('iou50', 0):.4f}   (standard criterion)")
    print(f"    Centroid-in-box : {s1_results['centroid_in_box_recall']:.4f}   "
          f"(boundary-free spatial localization, n={s1_results['centroid_in_box_tp']})")
    print(f"    Bio-loc recall  : {s1_results['bio_localization_recall']:.4f}   "
          f"(centroid within {s1_results['bio_localization_fraction_used']} x GT diagonal, "
          f"n={s1_results['bio_localization_tp']})")

    print(f"\n  -- Infected-cell sensitivity (parasitized GT only, n={s1_results['total_infected_gt']}) --")
    inf = s1_results["infected_cell_sensitivity"]
    print(f"    Sensitivity @IoU0.25: {inf.get('iou25', 0):.4f}")
    print(f"    Sensitivity @IoU0.30: {inf.get('iou30', 0):.4f}")
    print(f"    Sensitivity @IoU0.50: {inf.get('iou50', 0):.4f}")

    print(f"\n  Per-class cell recovery (@IoU0.5):")
    for cls, info in sorted(s1_results["per_class"].items(),
                             key=lambda x: -x[1]["gt_count"]):
        print(f"    {cls:<22}: {info['tp']:4d}/{info['gt_count']:4d}  "
              f"({info['recall']:.3f})")

    if args.stage1_only:
        print(f"\nStage-1-only mode — skipping E2E. Results saved to {out_dir}")
        return

    # End-to-end evaluation
    if not _TORCH_AVAILABLE:
        print("\nERROR: PyTorch is not installed. "
              "Use --stage1-only, or install torch + torchvision.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Stage 2] Loading model (device={device}) ...")
    model = load_model(args.checkpoint, device)

    print(f"\n[E2E] Running full pipeline evaluation ...")
    t1 = time.time()
    e2e_results = evaluate_e2e(
        gt_df, img_dir, model, device,
        dataset=args.dataset, iou_thr=args.iou_thr,
    )
    e2e_time = time.time() - t1

    # Attach Stage 1 results
    e2e_results["stage1"] = s1_results
    e2e_results["timing"] = {
        "stage1_seconds": round(s1_time, 1),
        "e2e_seconds":    round(e2e_time, 1),
    }

    metrics_path = out_dir / "metrics.json"
    # Strip raw curve arrays from saved JSON (curves saved separately in plot)
    metrics_save = dict(e2e_results)
    metrics_save["ap_per_class"] = {
        cls: {"ap": v["ap"], "n_gt": v["n_gt"]}
        for cls, v in e2e_results["ap_per_class"].items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"END-TO-END RESULTS — {args.dataset.upper()} ({e2e_time:.1f}s)")
    print(f"{'='*60}")
    print(f"  mAP@0.5                   : {e2e_results['mAP_at_iou50']:.4f}")
    print(f"  Binary Parasitized AP@0.5 : {e2e_results['binary_parasitized_AP']:.4f}")
    print(f"\n  Per-class AP@0.5:")
    for cls, data in e2e_results["ap_per_class"].items():
        bar = "#" * int(data["ap"] * 30)
        print(f"    {cls:<22}: {data['ap']:.4f}  {bar}  (n_gt={data['n_gt']})")

    # PR curves
    plot_pr_curves(e2e_results, out_dir / "pr_curves.png")

    print(f"\n  Results saved to: {out_dir}")
    print("    metrics.json      -- all numeric results")
    print("    stage1_stats.json -- Stage 1 cell recovery stats")
    print("    pr_curves.png     -- precision-recall curves")

if __name__ == "__main__":
    main()
