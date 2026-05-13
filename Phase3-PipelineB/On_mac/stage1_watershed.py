"""
Phase3-PipelineB/On_mac/stage1_watershed.py
===========================================
Stage 1 of the MalariAI pipeline: Annotation-agnostic cell segmentation.
Uses traditional computer vision (Otsu + Watershed) to isolate ALL cells
in a blood smear image, regardless of whether they were annotated in the GT.

Usage
-----
    python Phase3-PipelineB/On_mac/stage1_watershed.py \
        --json data/malaria/training.json \
        --img-dir data/malaria/images \
        --out-dir data/crops
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path for shared imports
# File is at Phase3-PipelineB/On_mac/stage1_watershed.py
# parent -> On_mac
# parent.parent -> Phase3-PipelineB
# parent.parent.parent -> Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))

from shared.label_map import CLASS_COLOUR_RGB


def segment_cells(image_path: str | Path, debug: bool = False):
    """
    Run the watershed pipeline on a single image.
    Returns:
        crops: list of [64x64] numpy arrays
        boxes: list of [x0, y0, x1, y1] coordinates
        mask:  the watershed label mask
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [], [], None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Otsu thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Noise removal (morphological opening)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 4. Sure foreground area (distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    # 5. Unknown region (between bg and fg)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # Now, mark the unknown region with zero
    markers[unknown == 255] = 0
    
    # 7. Watershed
    markers = cv2.watershed(img, markers)
    
    # 8. Extract crops and boxes
    crops = []
    boxes = []
    
    for label in range(2, markers.max() + 1):
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        if w < 10 or h < 10:
            continue
            
        margin = 5
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
        
        crop = img[y0:y1, x0:x1]
        crop_resized = cv2.resize(crop, (64, 64))
        
        crops.append(crop_resized)
        boxes.append([x0, y0, x1, y1])
        
    return crops, boxes, markers


def main():
    parser = argparse.ArgumentParser(description="MalariAI Stage 1: Watershed Segmentation")
    parser.add_argument("--json", required=True, help="Path to training.json or test.json")
    parser.add_argument("--img-dir", required=True, help="Folder with image files")
    parser.add_argument("--out-dir", default="data/crops", help="Folder to save extracted crops")
    parser.add_argument("--save-viz", action="store_true", help="Save segmentation visualization")
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.json) as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} images from {args.json}...")
    
    all_results = []
    
    for item in tqdm(data):
        img_name = Path(item["image"]["pathname"]).name
        img_path = img_dir / img_name
        
        if not img_path.exists():
            continue
            
        crops, boxes, markers = segment_cells(img_path)
        
        img_stem = img_path.stem
        img_crop_dir = out_dir / img_stem
        img_crop_dir.mkdir(exist_ok=True)
        
        for i, crop in enumerate(crops):
            crop_name = f"{img_stem}_crop_{i:03d}.png"
            cv2.imwrite(str(img_crop_dir / crop_name), crop)
            
        if args.save_viz:
            viz_dir = out_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            img = cv2.imread(str(img_path))
            img[markers == -1] = [0, 255, 0]
            cv2.imwrite(str(viz_dir / f"{img_stem}_viz.png"), img)
            
        all_results.append({
            "img_name": img_name,
            "num_cells": len(crops),
            "boxes": boxes
        })
        
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(all_results, f, indent=2)
            
    print(f"\nDone! Extracted crops saved to {out_dir}")


if __name__ == "__main__":
    main()
