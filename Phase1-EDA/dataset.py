from __future__ import annotations

"""
Phase1-EDA/dataset.py
=====================
PyTorch Dataset classes for MalariAI.
This file is shared across Phase 2 (Faster R-CNN) and Phase 3 (EfficientNet).
Copy or symlink it into the later phase folders, or import it directly.

Two classes:
  MalariaDataset      — whole-image detection dataset  (used by Faster R-CNN)
  MalariaCropDataset  — per-cell crop dataset          (used by EfficientNet)

Both parse directly from the raw JSON files.
No intermediate CSV is required — simpler, fewer moving parts.

Usage
-----
    from Phase1_EDA.dataset import MalariaDataset, MalariaCropDataset

    # Detection (Faster R-CNN)
    ds = MalariaDataset("data/malaria/training.json", "data/malaria/images")
    image, target = ds[0]
    # image  : FloatTensor [3, H, W]
    # target : {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N], "image_id": ...}

    # Classification (EfficientNet)
    ds = MalariaCropDataset("data/malaria/training.json", "data/malaria/images",
                             train=True)
    crop, label_idx = ds[0]
    # crop      : FloatTensor [3, 64, 64]
    # label_idx : int
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.label_map import LABEL_TO_INT, INT_TO_LABEL, NUM_CLASSES, PARASITE_CLASSES

SKIP_LABELS = {"difficult"}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_detection_transform():
    """
    For Faster R-CNN: just convert PIL -> Tensor.
    torchvision's Faster R-CNN normalises internally — don't double-normalise.
    """
    return T.ToTensor()


def get_classification_transform(train: bool = True, img_size: int = 64):
    """
    For EfficientNet crop classifier.
    Training: augment to simulate stain variation.
    Eval:     resize + normalise only.

    Why ColorJitter?
    ----------------
    Giemsa/MGG staining intensity varies between labs and even between slide
    batches. A ±30% brightness/contrast jitter forces the model to learn
    morphological features (cell shape, nuclear density) rather than colour
    shortcuts. This is the main domain-shift problem in malaria microscopy.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet — fine for transfer learning
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            # Stain normalisation proxy — jitters H/S/V without destroying shape
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── Detection dataset ─────────────────────────────────────────────────────────

class MalariaDataset(Dataset):
    """
    Whole-image dataset for object detection (Faster R-CNN).

    Parses directly from the raw JSON file — no CSV intermediary.

    Parameters
    ----------
    json_path : str | Path
        Path to training.json or test.json
    image_dir : str | Path
        Folder containing the image files (data/malaria/images/)
    transform : callable, optional
        Applied to the PIL image. Defaults to ToTensor().
    use_difficult : bool
        If False (default), "difficult" annotations are skipped.
    """

    def __init__(self, json_path, image_dir, transform=None, use_difficult=False):
        self.image_dir     = Path(image_dir)
        self.transform     = transform if transform else get_detection_transform()
        self.use_difficult = use_difficult

        # Parse JSON once at init — store as list of per-image dicts
        self._records = self._parse(Path(json_path))

    def _parse(self, json_path: Path) -> list[dict]:
        with open(json_path) as f:
            raw = json.load(f)

        records = []
        for item in raw:
            img_name = Path(item["image"]["pathname"]).name
            boxes, labels = [], []

            for obj in item.get("objects", []):
                label = obj["category"]
                if label in SKIP_LABELS and not self.use_difficult:
                    continue
                if label not in LABEL_TO_INT:
                    continue

                bb    = obj["bounding_box"]
                x_min = float(bb["minimum"]["c"])
                y_min = float(bb["minimum"]["r"])
                x_max = float(bb["maximum"]["c"])
                y_max = float(bb["maximum"]["r"])

                if x_max <= x_min or y_max <= y_min:
                    continue

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(LABEL_TO_INT[label])

            if boxes:   # skip images with zero valid annotations
                records.append({
                    "img_name": img_name,
                    "boxes":    boxes,
                    "labels":   labels,
                })
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        rec      = self._records[idx]
        img_path = self.image_dir / rec["img_name"]

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Check image_dir='{self.image_dir}'"
            )

        image  = Image.open(img_path).convert("RGB")
        image  = self.transform(image)

        boxes  = torch.as_tensor(rec["boxes"],  dtype=torch.float32)
        labels = torch.as_tensor(rec["labels"], dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image, target

    def get_class_counts(self) -> dict[str, int]:
        """Number of annotated boxes per class — useful for loss weighting."""
        counts = defaultdict(int)
        for rec in self._records:
            for lbl in rec["labels"]:
                counts[INT_TO_LABEL[lbl]] += 1
        return dict(counts)

    def get_image_names(self) -> list[str]:
        return [r["img_name"] for r in self._records]


# ── Crop dataset (classification) ────────────────────────────────────────────

class MalariaCropDataset(Dataset):
    """
    Per-cell crop dataset for EfficientNet classification (Pipeline B Stage 2).

    For each annotated bounding box in the JSON, crops the corresponding
    region from the full image, applies a margin, and returns it with
    its integer class label.

    Note: This is trained on ANNOTATED boxes only.
    At inference time, Stage 1 (watershed) supplies the boxes — those
    won't have ground-truth labels, which is the whole point of the pipeline.

    Parameters
    ----------
    json_path  : path to training.json or test.json
    image_dir  : folder with image files
    train      : whether to apply training augmentation
    img_size   : crop will be resized to (img_size, img_size)
    margin     : extra pixels added around each box before cropping
    """

    def __init__(self, json_path, image_dir,
                 train: bool = True, img_size: int = 64, margin: int = 6):
        self.image_dir = Path(image_dir)
        self.transform = get_classification_transform(train=train, img_size=img_size)
        self.margin    = margin
        self._crops    = self._parse(Path(json_path))

    def _parse(self, json_path: Path) -> list[dict]:
        with open(json_path) as f:
            raw = json.load(f)

        crops = []
        for item in raw:
            img_name = Path(item["image"]["pathname"]).name
            img_w    = item["image"]["shape"]["c"]
            img_h    = item["image"]["shape"]["r"]

            for obj in item.get("objects", []):
                label = obj["category"]
                if label in SKIP_LABELS or label not in LABEL_TO_INT:
                    continue

                bb    = obj["bounding_box"]
                x_min = int(bb["minimum"]["c"])
                y_min = int(bb["minimum"]["r"])
                x_max = int(bb["maximum"]["c"])
                y_max = int(bb["maximum"]["r"])

                if x_max <= x_min or y_max <= y_min:
                    continue

                # Apply margin and clip to image bounds
                x0 = max(0, x_min - self.margin)
                y0 = max(0, y_min - self.margin)
                x1 = min(img_w, x_max + self.margin)
                y1 = min(img_h, y_max + self.margin)

                crops.append({
                    "img_name":  img_name,
                    "label_idx": LABEL_TO_INT[label],
                    "label":     label,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                })
        return crops

    def __len__(self) -> int:
        return len(self._crops)

    def __getitem__(self, idx: int):
        c        = self._crops[idx]
        img_path = self.image_dir / c["img_name"]
        image    = Image.open(img_path).convert("RGB")
        crop     = image.crop((c["x0"], c["y0"], c["x1"], c["y1"]))
        crop     = self.transform(crop)
        return crop, c["label_idx"]

    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for Focal Loss / WeightedRandomSampler.
        Returns FloatTensor of shape [NUM_CLASSES].
        Background (index 0) weight is 0 — it never appears in crops.
        """
        counts = defaultdict(int)
        for c in self._crops:
            counts[c["label_idx"]] += 1
        total   = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for idx, cnt in counts.items():
            weights[idx] = total / (len(counts) * cnt)
        return weights

    def class_summary(self) -> None:
        counts = defaultdict(int)
        for c in self._crops:
            counts[c["label"]] += 1
        total = sum(counts.values())
        print(f"\nMalariaCropDataset: {total:,} crops")
        for label in (PARASITE_CLASSES + ["red blood cell", "leukocyte"]):
            cnt = counts.get(label, 0)
            print(f"  {label:<22} {cnt:>7,}  ({100*cnt/max(total,1):5.1f}%)")


# ── Collate function for DataLoader (detection) ────────────────────────────────

def detection_collate(batch):
    """
    torchvision detection models need a list, not a stacked tensor.
    Pass this as collate_fn to DataLoader when using MalariaDataset.
    """
    return tuple(zip(*batch))


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json",     required=True, help="Path to training.json")
    p.add_argument("--img-dir",  required=True, help="Path to images folder")
    p.add_argument("--mode",     default="detection",
                   choices=["detection", "classification"])
    args = p.parse_args()

    if args.mode == "detection":
        ds = MalariaDataset(args.json, args.img_dir)
        print(f"\nMalariaDataset: {len(ds)} images")
        img, tgt = ds[0]
        print(f"  image shape : {tuple(img.shape)}")
        print(f"  num boxes   : {len(tgt['boxes'])}")
        print(f"  labels      : {[INT_TO_LABEL[l.item()] for l in tgt['labels'][:6]]}")
        print(f"\nClass counts:")
        for cls, cnt in ds.get_class_counts().items():
            print(f"  {cls:<22} {cnt:>7,}")

    else:
        ds = MalariaCropDataset(args.json, args.img_dir, train=True)
        ds.class_summary()
        crop, lbl = ds[0]
        print(f"\nSample crop shape : {tuple(crop.shape)}")
        print(f"Sample label      : {INT_TO_LABEL[lbl]} ({lbl})")
        w = ds.get_class_weights()
        print(f"Class weights     : {w.tolist()}")
