"""
Phase3-PipelineB/stage2_inference.py
====================================
End-to-end inference script for Pipeline B:
1. Stage 1: Watershed segmentation of full smear.
2. Stage 2: EfficientNet-B0 classification of each cell.
3. Explainability: Grad-CAM++ generation.

Usage
-----
    python Phase3-PipelineB/stage2_inference.py \
        --img data/malaria/images/sample.png \
        --ckpt Phase3-PipelineB/checkpoints/best.pth \
        --out-dir outputs/inference
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))
sys.path.insert(0, str(PROJECT_ROOT / "Phase3-PipelineB/On_mac"))

from gradcam import GradCAMPlusPlus, overlay_heatmap
from stage1_watershed import segment_cells
from stage2_train import build_model
from shared.label_map import NUM_CLASSES, INT_TO_LABEL, SHORT_NAME, CLASS_COLOUR_RGB


def run_inference(img_path, model, device):
    # 1. Stage 1: Watershed
    crops, boxes, markers = segment_cells(img_path)
    print(f"Detected {len(crops)} cell segments.")
    
    if not crops:
        return None
        
    # 2. Prepare EfficientNet & Grad-CAM++
    target_layer = model.features[-1]  # Last conv layer of EfficientNet-B0
    gcam = GradCAMPlusPlus(model, target_layer)
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    
    # 3. Classify each crop
    model.eval()
    for i, (crop, box) in enumerate(zip(crops, boxes)):
        input_tensor = transform(crop).unsqueeze(0).to(device)
        
        # Inference + Grad-CAM
        heatmap, class_idx = gcam.generate(input_tensor)
        label = INT_TO_LABEL[class_idx]
        
        results.append({
            "box": box,
            "label": label,
            "confidence": 0.0, # Placeholder for softmax confidence
            "heatmap": heatmap,
            "crop": crop
        })
        
    return results, markers


def main():
    parser = argparse.ArgumentParser(description="End-to-End MalariAI Inference")
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--out-dir", default="outputs/inference", help="Where to save results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = build_model(NUM_CLASSES)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    
    # Run pipeline
    results, markers = run_inference(args.img, model, device)
    
    if results:
        # Load original image for drawing
        img = cv2.imread(args.img)
        canvas = img.copy()
        
        # Draw bounding boxes and labels
        for res in results:
            x0, y0, x1, y1 = res["box"]
            lbl = res["label"]
            short = SHORT_NAME[lbl]
            color = CLASS_COLOUR_RGB[lbl][::-1] # RGB to BGR
            
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(canvas, short, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # Save output
        img_stem = Path(args.img).stem
        cv2.imwrite(str(out_dir / f"{img_stem}_results.png"), canvas)
        
        # Save crops with heatmaps for top cells
        # (Just as a demonstration, we'll save the first 5)
        for i, res in enumerate(results[:5]):
            hmap_overlay = overlay_heatmap(res["crop"], res["heatmap"])
            cv2.imwrite(str(out_dir / f"{img_stem}_cell_{i}_{res['label']}_cam.png"), 
                        cv2.cvtColor(hmap_overlay, cv2.COLOR_RGB2BGR))
            
        print(f"Results saved to {out_dir}")
    else:
        print("No cells detected.")

if __name__ == "__main__":
    main()
