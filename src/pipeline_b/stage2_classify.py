"""
stage2_classify.py — Pipeline B Stage 2: EfficientNet-B0 classifier (KB)

What this module does
---------------------
Trains a per-cell image classifier on cropped cell patches. Each crop comes
from either:
  (a) ground-truth bounding boxes (for supervised training), OR
  (b) watershed-detected boxes    (for inference on unseen images).

The crop is resized to 64x64, passed through EfficientNet-B0, and the output
logit vector gives the class probabilities: {RBC, trophozoite, ring, schizont,
gametocyte, leukocyte}.

Why EfficientNet-B0?
--------------------
EfficientNet (Tan & Le 2019) scales width, depth, and resolution jointly using
a compound coefficient phi. EfficientNet-B0 is the smallest variant:
    Width multiplier  (w): 1.0
    Depth multiplier  (d): 1.0
    Resolution        (r): 224 (we use 64 because crops are tiny)
    Params: ~5.3M  |  Top-1 ImageNet: 77.1%

For our use case the advantages are:
  - Fewer parameters than ResNet-50 (23M) — important for Colab T4 GPU memory
    when processing many crops per image in batch.
  - MBConv (mobile inverted bottleneck) blocks are efficient on small inputs.
  - Strong pretrained ImageNet features transfer well to cell texture patterns.

Alternative considered: ResNet-18 (11M params). EfficientNet-B0 outperforms it
on small-image benchmarks with fewer parameters, so B0 is the right choice.

Focal Loss
----------
Standard cross-entropy on imbalanced data is dominated by the majority class.
Focal Loss (Lin et al. 2017) down-weights easy negatives:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the loss contribution from easy examples.
    alpha_t is the inverse-frequency weight for class t.

At gamma=2.0 (the canonical value from the RetinaNet paper), a well-classified
example with p_t=0.9 has its loss contribution reduced by (1-0.9)^2 = 0.01x.
The model's gradient budget concentrates on the hard misclassified examples
(the rare parasitic stages), which is exactly what we need.

Usage (training)
----------------
    python src/pipeline_b/stage2_classify.py \\
        --train-csv data/processed/train_annotations.csv \\
        --val-csv   data/processed/val_annotations.csv \\
        --img-dir   data/malaria/images \\
        --epochs    30 \\
        --batch     64 \\
        --out-dir   checkpoints/pipeline_b

Usage (inference — called by pipeline_b_inference.py)
------------------------------------------------------
    from src.pipeline_b.stage2_classify import EfficientNetClassifier
    clf = EfficientNetClassifier.from_checkpoint("checkpoints/pipeline_b/best.pth")
    probs = clf.predict_crops(crops_list)   # list of PIL Images
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.label_map import NUM_CLASSES, INT_TO_LABEL, FOREGROUND_CLASSES  # noqa
from src.models.dataset import MalariaCropDataset, classification_transforms   # noqa


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 reduces to standard cross-entropy.
        gamma=2.0 is the value used in the RetinaNet paper.
    alpha : Tensor | None
        Per-class weights (inverse frequency). Shape: [num_classes].
        If None, all classes are weighted equally.
    reduction : str
        'mean' | 'sum' | 'none'

    Mathematics
    -----------
    For a prediction vector p (after softmax) and one-hot target y:

        p_t  = p[y]               # probability of the true class
        FL   = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The factor (1 - p_t)^gamma is the modulating term.
    When p_t is large (easy example): (1-p_t) is small -> low contribution.
    When p_t is small (hard example): (1-p_t) ≈ 1    -> full contribution.
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha   # will be moved to device in forward()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : FloatTensor [N, C]  — raw (pre-softmax) model outputs
        targets : LongTensor  [N]     — class indices (0 .. C-1)
        """
        log_softmax = nn.functional.log_softmax(logits, dim=1)
        softmax     = log_softmax.exp()

        # Gather log-probability of the true class for each sample
        log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        pt     = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)       # [N]

        # Modulating factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)  # [N]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt   # [N]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── Model ─────────────────────────────────────────────────────────────────────

def build_efficientnet(num_classes: int = NUM_CLASSES,
                       pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 with a custom classification head.

    We replace the default 1000-class head with a Linear(1280, num_classes).
    1280 is EfficientNet-B0's penultimate feature dimension.

    We freeze all layers except the last MBConv block and the head, since
    cell texture features are low-level enough that middle-layer ImageNet
    features transfer very well.
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
    else:
        model   = models.efficientnet_b0(weights=None)

    # Freeze early layers — keep the last MBConv block (features[7]) trainable
    for i, layer in enumerate(model.features):
        for param in layer.parameters():
            param.requires_grad = (i >= 6)   # train last 2 blocks + head

    # Replace classifier
    in_features = model.classifier[1].in_features   # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


class EfficientNetClassifier:
    """
    High-level wrapper for training, checkpointing, and inference.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model.to(device)
        self.device = device

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path,
                        device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt  = torch.load(ckpt_path, map_location=device)
        model = build_efficientnet(num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(ckpt["model"])
        obj   = cls(model, device)
        print(f"Loaded checkpoint from {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
        return obj

    @torch.no_grad()
    def predict_crops(self, crops, transform=None):
        """
        Classify a list of PIL Images (cell crops from Stage 1).

        Parameters
        ----------
        crops : list of PIL.Image.Image
        transform : callable
            Applied to each crop. Defaults to classification_transforms(train=False).

        Returns
        -------
        list of dicts: [{"class": str, "class_idx": int, "probs": Tensor[C]}, ...]
        """
        if transform is None:
            transform = classification_transforms(train=False)

        self.model.eval()
        results = []
        for crop in crops:
            x      = transform(crop).unsqueeze(0).to(self.device)   # [1, C, H, W]
            logits = self.model(x)                                    # [1, num_classes]
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()   # [num_classes]
            idx    = probs.argmax().item()
            results.append({
                "class":     INT_TO_LABEL[idx],
                "class_idx": idx,
                "probs":     probs,
            })
        return results


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for i, (images, labels) in enumerate(loader):
        images  = images.to(device)
        labels  = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images  = images.to(device)
        labels  = labels.to(device)
        logits  = model(images)
        loss    = criterion(logits, labels)
        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 classifier for Pipeline B Stage 2"
    )
    parser.add_argument("--train-csv",  required=True)
    parser.add_argument("--val-csv",    required=True)
    parser.add_argument("--img-dir",    required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--img-size",   type=int,   default=64)
    parser.add_argument("--gamma",      type=float, default=2.0,
                        help="Focal Loss gamma (0 = standard CE)")
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--out-dir",    type=str,   default="checkpoints/pipeline_b")
    parser.add_argument("--no-pretrain", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMalariAI — Pipeline B: EfficientNet-B0 Classifier")
    print(f"Device      : {device}")
    print(f"Focal gamma : {args.gamma}  (0 = standard cross-entropy)")
    print(f"Classes     : {FOREGROUND_CLASSES}")

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = MalariaCropDataset(
        args.train_csv, args.img_dir,
        transforms=classification_transforms(train=True,  img_size=args.img_size),
    )
    val_ds = MalariaCropDataset(
        args.val_csv, args.img_dir,
        transforms=classification_transforms(train=False, img_size=args.img_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    print(f"\nTrain crops : {len(train_ds):,}")
    print(f"Val   crops : {len(val_ds):,}")
    class_weights = train_ds.get_class_weights()
    print(f"Class weights (inverse frequency):")
    for i, w in enumerate(class_weights):
        if w > 0:
            print(f"  [{i}] {INT_TO_LABEL[i]:<20}  {w:.3f}")

    # ── Model & loss ──────────────────────────────────────────────────────
    model = build_efficientnet(
        num_classes=NUM_CLASSES, pretrained=not args.no_pretrain
    )
    model.to(device)

    criterion = FocalLoss(
        gamma=args.gamma,
        alpha=class_weights,   # inverse-frequency weights as alpha
    )

    # Adam + cosine LR schedule
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc  = 0.0

    print(f"\nStarting training for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)"
        )

        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }
        torch.save(ckpt, out_dir / "latest.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  ** New best val_acc: {best_val_acc:.1f}% **")

    print(f"\nTraining complete. Best val_acc = {best_val_acc:.1f}%")
    print(f"Load for inference:")
    print(f"  clf = EfficientNetClassifier.from_checkpoint('{out_dir}/best.pth')")


if __name__ == "__main__":
    main()
