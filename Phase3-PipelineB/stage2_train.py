"""
Phase3-PipelineB/stage2_train.py
==================================
Stage 2: EfficientNet-B0 + Focal Loss classifier for per-cell infection staging.

Training data
-------------
  GT bounding boxes from training.json → 64×64 crops (via MalariaCropDataset).
  No watershed required at training time — we train on the ground-truth crops
  because they have clean labels.  At inference time, crops come from Stage 1.

Model
-----
  torchvision EfficientNet-B0 (ImageNet pretrained)
  Final classifier replaced: Linear(1280 → NUM_CLASSES)
  NUM_CLASSES = 7 (background + 6 cell types)

Loss
----
  Focal Loss  γ=2.0, per-class α = inverse frequency normalised.
  Alpha weights derived from training-set annotation counts (EDA Phase 1).

Training protocol
-----------------
  Optimiser : AdamW  lr=1e-4  weight_decay=1e-4
  Scheduler : CosineAnnealingLR  T_max=epochs
  Epochs    : 30  (configurable via --epochs)
  Batch     : 64  (configurable)
  Val split : 15% of training images (stratified by label)

Outputs (--out-dir)
-------------------
  best.pth          ← best val-accuracy checkpoint
  last.pth          ← last epoch checkpoint (for resuming)
  metrics.json      ← final stats + per-class accuracy
  loss_curves.png   ← train/val loss + accuracy curves

Usage
-----
  # Local / RunPod
  python Phase3-PipelineB/stage2_train.py \
      --train-json data/malaria/training.json \
      --img-dir    data/malaria/images \
      --epochs     30 \
      --out-dir    Phase3-PipelineB/checkpoints

  # Kaggle — override paths only, rest is identical
  python Phase3-PipelineB/stage2_train.py \
      --train-json /kaggle/input/bbbc041-malaria/malaria/training.json \
      --img-dir    /kaggle/input/bbbc041-malaria/malaria/images \
      --out-dir    /kaggle/working/phase3_checkpoints

  # Resume from a previous checkpoint
  python Phase3-PipelineB/stage2_train.py \
      --train-json data/malaria/training.json \
      --img-dir    data/malaria/images \
      --resume     Phase3-PipelineB/checkpoints/last.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))

from shared.label_map import NUM_CLASSES, INT_TO_LABEL, LABEL_TO_INT, FOREGROUND_NAMES
from dataset import MalariaCropDataset


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

# Training annotation counts from Phase 1 EDA (excluding "difficult")
# Used to compute per-class inverse-frequency alpha weights.
_TRAIN_COUNTS = {
    1: 77420,   # red blood cell
    2: 1473,    # trophozoite
    3: 353,     # ring
    4: 179,     # schizont
    5: 144,     # gametocyte
    6: 103,     # leukocyte
}


def compute_focal_alpha(num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Inverse-frequency alpha for Focal Loss.
    Class 0 (background) gets weight 0 — GT crops never contain pure background.
    """
    total = sum(_TRAIN_COUNTS.values())
    alpha = [0.0]  # index 0 = background
    for i in range(1, num_classes):
        count = _TRAIN_COUNTS.get(i, 1)
        alpha.append(total / (len(_TRAIN_COUNTS) * count))

    # Normalise so weights sum to num_foreground
    s = sum(alpha[1:])
    alpha = [a / s * (num_classes - 1) for a in alpha]
    return torch.tensor(alpha, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha  : per-class weight tensor  (shape [num_classes])
    gamma  : focusing exponent (default 2.0)
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per sample, no reduction
        ce   = F.cross_entropy(logits, targets, reduction="none")   # [B]
        pt   = torch.exp(-ce)                                        # [B]
        at   = self.alpha[targets]                                   # [B]
        loss = at * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with replaced classification head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = efficientnet_b0(weights=weights)
    # The default classifier is: Dropout → Linear(1280, 1000)
    # Replace the Linear layer; keep the Dropout for regularisation.
    in_feat = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Training / validation loops
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for crops, labels in loader:
        crops, labels = crops.to(device), labels.to(device)
        logits = model(crops)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    # Per-class tracking
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total   = {i: 0 for i in range(NUM_CLASSES)}

    for crops, labels in loader:
        crops, labels = crops.to(device), labels.to(device)
        logits = model(crops)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss += loss.item() * len(labels)
        correct    += (preds == labels).sum().item()
        total      += len(labels)

        for lbl, pred in zip(labels.cpu(), preds.cpu()):
            class_total[lbl.item()]   += 1
            class_correct[lbl.item()] += int(pred == lbl)

    per_class_acc = {}
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            per_class_acc[INT_TO_LABEL[i]] = round(
                class_correct[i] / class_total[i] * 100, 2)

    return total_loss / total, correct / total, per_class_acc


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_curves(train_losses, val_losses, train_accs, val_accs, out_path: Path):
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, train_losses, "b-o", ms=3, label="Train loss")
    axes[0].plot(epochs, val_losses,   "r-o", ms=3, label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")
    axes[0].set_title("EfficientNet-B0 — Focal Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in train_accs], "b-o", ms=3, label="Train acc")
    axes[1].plot(epochs, [a * 100 for a in val_accs],   "r-o", ms=3, label="Val acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("EfficientNet-B0 — Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f"Loss curves saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2 — EfficientNet-B0 + Focal Loss training")
    p.add_argument("--train-json", required=True)
    p.add_argument("--img-dir",    required=True)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--val-frac",   type=float, default=0.15,
                   help="Fraction of training data used for validation")
    p.add_argument("--out-dir",    default="Phase3-PipelineB/checkpoints")
    p.add_argument("--resume",     default=None,
                   help="Path to last.pth checkpoint to resume from")
    p.add_argument("--no-pretrain", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    p.add_argument("--workers",    type=int, default=4)
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading crops from {args.train_json} ...")
    full_ds = MalariaCropDataset(
        json_path=args.train_json,
        img_dir=args.img_dir,
        train=True,
    )
    print(f"  Total labelled crops: {len(full_ds)}")

    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train}  Val: {n_val}")

    # Disable augmentation on the val split
    val_ds.dataset = MalariaCropDataset(
        json_path=args.train_json,
        img_dir=args.img_dir,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model + Loss + Optimiser ──────────────────────────────────────────────
    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrain).to(device)

    alpha     = compute_focal_alpha(NUM_CLASSES).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_acc  = 0.0
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"]
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
        train_accs   = ckpt.get("train_accs",   [])
        val_accs     = ckpt.get("val_accs",     [])
        # Fast-forward scheduler to correct state
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Resuming from epoch {start_epoch + 1}")

    # ── Print focal-loss alpha weights ────────────────────────────────────────
    print("\nFocal Loss alpha weights:")
    for i in range(NUM_CLASSES):
        print(f"  [{i}] {INT_TO_LABEL[i]:<20} α={alpha[i].item():.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs - start_epoch} epoch(s) "
          f"(total target: {args.epochs})\n")

    for epoch in range(start_epoch, args.epochs):
        ep_start = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, per_class_acc = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        elapsed = time.time() - ep_start
        print(f"Epoch [{epoch+1:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"train_acc={tr_acc*100:.2f}%  val_acc={vl_acc*100:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.0f}s)")

        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  *** New best val acc: {best_val_acc*100:.2f}% — saved best.pth ***")

        # Always save last checkpoint (for resuming)
        ckpt = {
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "train_accs":   train_accs,
            "val_accs":     val_accs,
            "cfg": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pth")

    # ── Final evaluation on val set ───────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation ...")
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
    _, final_acc, final_per_class = evaluate(model, val_loader, criterion, device)

    print(f"\nFinal Val Accuracy (best ckpt): {final_acc*100:.2f}%")
    print("Per-class accuracy:")
    for cls, acc in final_per_class.items():
        print(f"  {cls:<22}: {acc:.2f}%")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "val_accuracy":       round(final_acc * 100, 2),
        "best_val_acc":       round(best_val_acc * 100, 2),
        "total_epochs":       args.epochs,
        "per_class_accuracy": final_per_class,
        "train_losses":       train_losses,
        "val_losses":         val_losses,
        "train_accs":         [round(a, 4) for a in train_accs],
        "val_accs":           [round(a, 4) for a in val_accs],
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Loss curves ───────────────────────────────────────────────────────────
    save_curves(train_losses, val_losses, train_accs, val_accs,
                out_dir / "loss_curves.png")

    print("\nPhase 3 Stage 2 training complete.")
    print(f"Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
