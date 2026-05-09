"""
dataset.py — PyTorch Dataset classes for MalariAI

Two classes are provided:

MalariaDataset
    For Pipeline A (Faster R-CNN).
    Each item is (image_tensor, target_dict) where target_dict follows the
    torchvision detection API:
        boxes   : FloatTensor[N, 4]  in [x0, y0, x1, y1] format
        labels  : Int64Tensor[N]     class indices from label_map
        image_id: Int64Tensor[1]     dataset index (required by COCO evaluator)

MalariaCropDataset
    For Pipeline B Stage-2 (EfficientNet classifier).
    Each item is (crop_tensor, label_idx) where the crop is a PIL image
    region corresponding to one ground-truth bounding box.
    Used to train the per-cell classification head on annotated crops.

IMPORTANT: label integers come from src/utils/label_map.LABEL_TO_INT.
Never re-derive them from the CSV — that would break reproducibility.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# ── Shared label map ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.label_map import LABEL_TO_INT, INT_TO_LABEL, NUM_CLASSES  # noqa


# ── Default transforms ────────────────────────────────────────────────────────

def detection_transforms(train: bool = True):
    """
    Minimal transform pipeline for the detection dataset.
    torchvision's Faster R-CNN handles its own internal normalisation,
    so we only need ToTensor() here. Add stain-augmentation in augment.py.
    """
    # For training we keep it simple; augmentation is applied separately.
    return T.ToTensor()


def classification_transforms(train: bool = True, img_size: int = 64):
    """
    Transform pipeline for EfficientNet crop classification.
    Crops are small (~20-60 px), so we resize to img_size first.

    Training: random horizontal/vertical flip + colour jitter to simulate
              stain variation (a known domain-shift problem in malaria slides).
    Eval:     only resize + normalise.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats — fine for transfer learning
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05),  # simulates stain variation
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ── MalariaDataset (detection) ────────────────────────────────────────────────

class MalariaDataset(Dataset):
    """
    Loads whole blood-smear images with all their bounding boxes.
    Used by Faster R-CNN (Pipeline A).

    Parameters
    ----------
    csv_file : str | Path
        One of the CSVs produced by prepare_data.py
        (train_annotations.csv / val_annotations.csv / test_annotations.csv)
    image_dir : str | Path
        Root folder that contains the .png/.jpg image files.
        For this project: data/malaria/images/
    transforms : callable, optional
        Applied to the PIL image before returning. Defaults to ToTensor().
    """

    def __init__(self, csv_file, image_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir   = Path(image_dir)
        self.transforms  = transforms if transforms is not None \
                           else detection_transforms(train=False)

        # Group annotations by image so __getitem__ can collect all boxes at once
        self.image_groups = self.annotations.groupby("img_name")
        self.image_names  = sorted(self.image_groups.groups.keys())

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        group    = self.image_groups.get_group(img_name)

        # ── Load image ────────────────────────────────────────────────────
        img_path = self.image_dir / img_name
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Check that image_dir='{self.image_dir}' is correct."
            )
        image = Image.open(img_path).convert("RGB")

        # ── Collect boxes and labels ──────────────────────────────────────
        boxes  = []
        labels = []
        for _, row in group.iterrows():
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])

            # Skip degenerate boxes (should not exist after prepare_data.py,
            # but guard against corrupted CSVs)
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            # Use label_idx column directly — already encoded by prepare_data.py
            labels.append(int(row["label_idx"]))

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # ── Build target dict (torchvision detection API) ─────────────────
        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        # ── Apply transforms ──────────────────────────────────────────────
        image = self.transforms(image)

        return image, target

    def get_class_counts(self) -> dict:
        """Return {label_name: count} for the entire split. Useful for weighting."""
        return self.annotations["label"].value_counts().to_dict()


# ── MalariaCropDataset (classification) ──────────────────────────────────────

class MalariaCropDataset(Dataset):
    """
    Returns individual cell crops for EfficientNet classification (Pipeline B).

    Each row in the CSV becomes one sample: the image is loaded, cropped to
    [x_min, y_min, x_max, y_max], resized, and returned with its integer label.

    Parameters
    ----------
    csv_file : str | Path
        Annotation CSV from prepare_data.py
    image_dir : str | Path
        Root folder containing the source images
    transforms : callable, optional
        Applied to each PIL crop. Defaults to classification_transforms(train=False).
    margin : int
        Extra pixels added around each box before cropping.
        A small margin (4-8 px) gives the model context around the cell boundary.
    """

    def __init__(self, csv_file, image_dir, transforms=None, margin: int = 4):
        self.annotations = pd.read_csv(csv_file).reset_index(drop=True)
        self.image_dir   = Path(image_dir)
        self.transforms  = transforms if transforms is not None \
                           else classification_transforms(train=False)
        self.margin      = margin

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]

        # ── Load full image ───────────────────────────────────────────────
        img_path = self.image_dir / row["img_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        W, H  = image.size

        # ── Crop with margin ──────────────────────────────────────────────
        x0 = max(0, int(row["x_min"]) - self.margin)
        y0 = max(0, int(row["y_min"]) - self.margin)
        x1 = min(W, int(row["x_max"]) + self.margin)
        y1 = min(H, int(row["y_max"]) + self.margin)
        crop = image.crop((x0, y0, x1, y1))

        label_idx = int(row["label_idx"])

        if self.transforms is not None:
            crop = self.transforms(crop)

        return crop, label_idx

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class inverse-frequency weights for Focal Loss / WeightedSampler.

        Returns a 1-D FloatTensor of length NUM_CLASSES (index 0 = background = 0.0).
        Absent classes get weight 0.0 (they won't appear in crop dataset anyway).
        """
        counts = self.annotations["label_idx"].value_counts().to_dict()
        total  = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for idx, count in counts.items():
            weights[int(idx)] = total / (NUM_CLASSES * count)
        return weights


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True, help="Path to an annotation CSV")
    p.add_argument("--img-dir",  required=True, help="Path to images folder")
    p.add_argument("--mode",     default="detection",
                   choices=["detection", "classification"])
    args = p.parse_args()

    if args.mode == "detection":
        ds = MalariaDataset(args.csv, args.img_dir)
        print(f"MalariaDataset: {len(ds)} images")
        img, tgt = ds[0]
        print(f"  image shape : {img.shape}")
        print(f"  num boxes   : {len(tgt['boxes'])}")
        print(f"  labels      : {[INT_TO_LABEL[l.item()] for l in tgt['labels'][:5]]}")
    else:
        ds = MalariaCropDataset(args.csv, args.img_dir,
                                transforms=classification_transforms(train=True))
        print(f"MalariaCropDataset: {len(ds)} crops")
        crop, label = ds[0]
        print(f"  crop shape  : {crop.shape}")
        print(f"  label       : {INT_TO_LABEL[label]} ({label})")
        print(f"  class weights: {ds.get_class_weights()}")
