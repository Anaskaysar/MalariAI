"""
Phase2-BaselineA/train_frcnn.py
================================
Command-line script version of the Phase 2 notebook.
Use this on Kaggle or a cloud GPU instead of running the notebook.

Usage
-----
    python Phase2-BaselineA/train_frcnn.py \
        --train-json  data/malaria/training.json \
        --img-dir     data/malaria/images \
        --epochs      20 \
        --batch       2 \
        --out-dir     Phase2-BaselineA/checkpoints

Kaggle-specific note
--------------------
On Kaggle, paths look like /kaggle/input/bbbc041/training.json.
Pass those as --train-json and --img-dir accordingly.
"""

from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Phase1-EDA"))

from shared.label_map import NUM_CLASSES, INT_TO_LABEL, PARASITE_CLASSES
from dataset import MalariaDataset, detection_collate


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    breakdown  = defaultdict(float)
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            breakdown[k] += v.item()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] loss={losses.item():.4f}  "
                  f"({time.time()-t0:.0f}s)")

    n = len(loader)
    comp = "  ".join(f"{k.replace('loss_','')}={v/n:.3f}" for k, v in breakdown.items())
    return total_loss / n, comp


@torch.no_grad()
def validate_loss(model, loader, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total / len(loader)


@torch.no_grad()
def compute_map(model, loader, device, score_thresh=0.3):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        print("torchmetrics not installed — skipping mAP. Run: pip install torchmetrics")
        return {}

    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)
        filtered = []
        for p in preds:
            keep = p["scores"] >= score_thresh
            filtered.append({
                "boxes":  p["boxes"][keep].cpu(),
                "scores": p["scores"][keep].cpu(),
                "labels": p["labels"][keep].cpu(),
            })
        cpu_tgts = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                    for t in targets]
        metric.update(filtered, cpu_tgts)
    return metric.compute()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Faster R-CNN baseline training"
    )
    parser.add_argument("--train-json",  required=True)
    parser.add_argument("--img-dir",     required=True)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch",       type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=5e-3)
    parser.add_argument("--val-split",   type=float, default=0.2)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--score-thresh",type=float, default=0.3)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--out-dir",     default="Phase2-BaselineA/checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMalariAI — Phase 2: Faster R-CNN Baseline")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs}")
    print(f"Classes : {NUM_CLASSES}")

    # ── Data ──────────────────────────────────────────────────────────────
    full_ds = MalariaDataset(args.train_json, args.img_dir)
    n_val   = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=detection_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=detection_collate,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train images : {len(train_ds):,}")
    print(f"Val   images : {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrain)
    model.to(device)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {total:,}")

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10} Time")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, comp = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_loss(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {lr_now:<10.2e} {time.time()-t0:.0f}s")
        print(f"       {comp}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "val_loss": val_loss,
            "train_losses": train_losses, "val_losses": val_losses,
        }
        torch.save(ckpt, out_dir / "latest.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / "best.pth")
            print(f"       ✓ Best val_loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best val_loss = {best_val_loss:.4f}")

    # ── Final mAP eval ────────────────────────────────────────────────────
    print("\nRunning mAP evaluation ...")
    ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    result = compute_map(model, val_loader, device, score_thresh=args.score_thresh)

    metrics = {"val_loss": best_val_loss}
    if result:
        map50 = result["map_50"].item()
        metrics["map_50"] = map50
        print(f"\nmAP@0.5 : {map50:.4f}  ({100*map50:.1f}%)")
        if "map_per_class" in result and result["map_per_class"] is not None:
            per = result["map_per_class"].tolist()
            metrics["per_class_ap"] = {}
            for i, ap in enumerate(per):
                cls = INT_TO_LABEL.get(i + 1, f"class_{i+1}")
                metrics["per_class_ap"][cls] = round(ap, 4)
                flag = " ← rare" if cls in PARASITE_CLASSES else ""
                print(f"  {cls:<22} {ap:.4f}{flag}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out_dir}/metrics.json")
    print(f"Saved: {out_dir}/best.pth")


if __name__ == "__main__":
    main()
