"""
Phase4-WebApp/pipeline.py
==========================
Consolidated MalariAI inference pipeline for web deployment.
Integrates Stage 1 (Watershed), Stage 2 (EfficientNet-B0), and Grad-CAM++.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import io
import base64

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_CLASSES = 7
INT_TO_LABEL = {
    0: "background",
    1: "red blood cell",
    2: "trophozoite",
    3: "ring",
    4: "schizont",
    5: "gametocyte",
    6: "leukocyte",
}
PARASITE_CLASSES = ["trophozoite", "ring", "schizont", "gametocyte"]

CLASS_COLOUR_RGB = {
    "red blood cell": (220, 50,  50),
    "trophozoite":    (50,  180, 50),
    "ring":           (50,  50,  220),
    "schizont":       (200, 130, 0),
    "gametocyte":     (160, 0,   200),
    "leukocyte":      (0,   180, 200),
    "background":     (128, 128, 128),
}

CROP_SIZE = 64
MIN_AREA = 150
MIN_DIST = 30
OPEN_KSIZE = 3
DIST_THRESH = 0.25

MAX_CELL_W = 220
MAX_CELL_H = 220
MAX_ASPECT_RATIO = 2.2

_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TRANSFORM = T.Compose([
    T.Resize((CROP_SIZE, CROP_SIZE)),
    T.ToTensor(),
    _NORMALIZE,
])

# ── Stage 1: Watershed ─────────────────────────────────────────────────────────

def watershed_cells_with_labels(bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Runs watershed and returns both bounding boxes and the labels map."""
    g = bgr[:, :, 1]
    blurred = cv2.GaussianBlur(g, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_filled = np.zeros_like(closed)
    cv2.drawContours(binary_filled, contours, -1, 255, -1)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE, OPEN_KSIZE))
    opened = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    dist = ndi.distance_transform_edt(opened)
    coords = peak_local_max(dist, min_distance=MIN_DIST,
                            threshold_abs=DIST_THRESH * dist.max() if dist.max() > 0 else 1,
                            labels=opened.astype(bool))
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-dist, markers, mask=opened.astype(bool), compactness=0.1)

    boxes = []
    # Re-map labels to be sequential and match the boxes list
    final_labels = np.zeros_like(labels)
    new_id = 1
    for region_id in range(1, labels.max() + 1):
        ys, xs = np.where(labels == region_id)
        if len(ys) < MIN_AREA:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        final_labels[labels == region_id] = new_id
        new_id += 1
    return boxes, final_labels

def extract_masked_crop(bgr: np.ndarray, cell_mask: np.ndarray, box: tuple[int, int, int, int], target_size: int = 64) -> np.ndarray:
    """Extracts a square crop, masking out any pixels not belonging to the specific cell."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = int(max(w, h) * 1.5)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = side // 2

    # Force background to white for the specific cell area
    masked_bgr = bgr.copy()
    masked_bgr[cell_mask == 0] = [255, 255, 255]

    padded = cv2.copyMakeBorder(masked_bgr, half, half, half, half, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    px, py = cx + half, cy + half
    
    crop_bgr = padded[py - half: py + half, px - half: px + half]
    if crop_bgr.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8) + 255
        
    crop_resized = cv2.resize(crop_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    return crop_rgb

def is_oversized(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w > MAX_CELL_W or h > MAX_CELL_H:
        return True
    if max(w, h) / max(min(w, h), 1) > MAX_ASPECT_RATIO:
        return True
    return False

# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = model.features[-1]
        self._activations = None
        self._gradients = None
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        device = next(self.model.parameters()).device
        inp = input_tensor.to(device).requires_grad_(True)
        logits = self.model(inp)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(logits.argmax(1).item())
        confidence = float(probs[0, pred_class].item())
        target_cls = class_idx if class_idx is not None else pred_class

        self.model.zero_grad()
        score = logits[0, target_cls]
        score.backward()

        grads = self._gradients
        acts = self._activations
        grads_sq = grads ** 2
        grads_cub = grads ** 3
        numer = grads_sq
        denom = 2.0 * grads_sq + (acts * grads_cub).sum(dim=(2, 3), keepdim=True) + 1e-9
        alpha = numer / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3))
        cam = (weights[..., None, None] * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        H, W = input_tensor.shape[-2:]
        cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        heatmap = cam_up.squeeze().cpu().numpy()
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap.astype(np.float32), pred_class, confidence

    @staticmethod
    def overlay(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
        import matplotlib.cm as cm
        # Resize heatmap to match image if needed
        H, W = original_rgb.shape[:2]
        if heatmap.shape != (H, W):
            heatmap = cv2.resize(heatmap, (W, H))
        cmap = cm.get_cmap("jet")
        coloured = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        blended = (original_rgb.astype(float) * (1 - alpha) + coloured.astype(float) * alpha).astype(np.uint8)
        return blended

# ── Inference Pipeline ────────────────────────────────────────────────────────

class MalariAI:
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.cam = GradCAMPlusPlus(self.model)

    def analyze(self, image_path: str, conf_threshold: float = 0.40):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        H, W = bgr.shape[:2]

        # Stage 1: Watershed
        all_boxes, labels = watershed_cells_with_labels(bgr)
        
        results = []
        oversized = []
        
        for i, box in enumerate(all_boxes):
            if is_oversized(box):
                oversized.append(box)
                continue
            
            # Extract isolated crop
            cell_id = i + 1
            cell_mask = (labels == cell_id).astype(np.uint8)
            crop_rgb = extract_masked_crop(bgr, cell_mask, box)
            
            pil = Image.fromarray(crop_rgb)
            tens = _TRANSFORM(pil).unsqueeze(0).to(self.device)
            
            # Get Grad-CAM and pred
            hmap, pred_idx, conf = self.cam(tens)
            label = INT_TO_LABEL[pred_idx]
            
            results.append({
                "idx": i,
                "box": box,
                "label": label,
                "confidence": conf,
                "heatmap": hmap,
                "crop_rgb": crop_rgb
            })

        # Calculate metrics
        n_parasites = sum(1 for r in results if r["label"] in PARASITE_CLASSES and r["confidence"] >= conf_threshold)
        n_total = len(results)
        infection_rate = (n_parasites / n_total * 100) if n_total > 0 else 0.0

        # Build Annotated Smear (Card 1)
        vis = bgr.copy()
        for res in results:
            x1, y1, x2, y2 = res["box"]
            lbl, conf = res["label"], res["confidence"]
            rgb = CLASS_COLOUR_RGB.get(lbl, (200, 200, 200))
            bgr_col = (rgb[2], rgb[1], rgb[0])
            thickness = 2 if lbl in PARASITE_CLASSES and conf >= conf_threshold else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_col, thickness)
        
        # Oversized
        for box in oversized:
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (140, 140, 140), 1)

        # Output payload
        # For the web app, we'll return:
        # 1. Total counts
        # 2. Base64 of annotated smear
        # 3. List of cells (brief)
        # 4. A few top parasite crops with Grad-CAM

        # Helper to encode CV2 image to base64
        def to_base64(img_bgr):
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')

        # Sort cells for gallery (parasites first)
        priority = {"ring": 0, "trophozoite": 1, "schizont": 2, "gametocyte": 3, "leukocyte": 4, "red blood cell": 5}
        sorted_results = sorted(results, key=lambda x: (priority.get(x["label"], 9), -x["confidence"]))

        payload = {
            "metrics": {
                "total_cells": n_total,
                "infected_cells": n_parasites,
                "healthy_rbcs": sum(1 for r in results if r["label"] == "red blood cell"),
                "leukocytes": sum(1 for r in results if r["label"] == "leukocyte"),
                "infection_rate": round(infection_rate, 2),
                "oversized": len(oversized),
                "dominant_stage": self._get_dominant_stage(results, conf_threshold)
            },
            "class_counts": {lbl: sum(1 for r in results if r["label"] == lbl) for lbl in INT_TO_LABEL.values()},
            "smear_image": to_base64(vis),
            "cells": []
        }

        for r in sorted_results[:48]: # Top 48 for gallery
            cell_data = {
                "label": r["label"],
                "confidence": round(r["confidence"], 4),
                "crop": to_base64(cv2.cvtColor(r["crop_rgb"], cv2.COLOR_RGB2BGR))
            }
            # Add Grad-CAM for parasites
            if r["label"] in PARASITE_CLASSES and r["confidence"] >= conf_threshold:
                overlay = self.cam.overlay(r["crop_rgb"], r["heatmap"])
                cell_data["gradcam"] = to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            payload["cells"].append(cell_data)

        return payload

    def _get_dominant_stage(self, results, threshold):
        p_counts = {c: 0 for c in PARASITE_CLASSES}
        for r in results:
            if r["label"] in PARASITE_CLASSES and r["confidence"] >= threshold:
                p_counts[r["label"]] += 1
        top = max(p_counts, key=p_counts.get)
        return top if p_counts[top] > 0 else "N/A"

if __name__ == "__main__":
    # Minor smoke test
    print("Pipeline code loaded.")
