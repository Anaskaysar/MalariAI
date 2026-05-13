"""
Phase3-PipelineB/stage2_train.py
================================
Training script for the Stage 2 EfficientNet-B0 classifier.
Uses Focal Loss to address extreme class imbalance (97% RBC).

Usage
-----
    python Phase3-PipelineB/stage2_train.py \
        --train-json  data/malaria/training.json \
        --img-dir     data/malaria/images \
        --epochs      30 \
        --batch       32 \
        --out-dir     Phase3-PipelineB/checkpoints
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))

from shared.label_map import NUM_CLASSES, INT_TO_LABEL, PARASITE_CLASSES
from dataset import MalariaCropDataset


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    Encourages model to focus on hard, misclassified examples (parasites).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    
    # Replace the classification head
    # EfficientNet-B0 classifier is model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Stage 2: EfficientNet Training")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss gamma")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out-dir", default="Phase3-PipelineB/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMalariAI — Phase 3: Stage 2 EfficientNet Training")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}")
    print(f"Batch  : {args.batch}")

    # ── Data ──────────────────────────────────────────────────────────────
    full_ds = MalariaCropDataset(args.train_json, args.img_dir, train=True)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # Switch validation set to eval mode (no augmentation)
    val_ds.dataset.train = False 

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    print(f"Train crops : {len(train_ds):,}")
    print(f"Val crops   : {len(val_ds):,}")

    # ── Weights & Loss ───────────────────────────────────────────────────
    # Inverse frequency weights for alpha in Focal Loss
    class_weights = full_ds.get_class_weights().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    print(f"Class weights (alpha): {class_weights.cpu().numpy().round(2)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.1f}%"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Acc {train_acc:.1f}% || "
              f"Val Loss {avg_val_loss:.4f} | Acc {val_acc:.1f}%")
        
        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "cfg": vars(args)
        }
        torch.save(ckpt, out_dir / "latest.pth")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  ✓ Saved best model")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
