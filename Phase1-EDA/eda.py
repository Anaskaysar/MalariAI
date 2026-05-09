"""
Phase1-EDA/eda.py
=================
Malaria Blood Smear — Exploratory Data Analysis

Run this first before writing a single line of model code.
Every architectural decision in Phases 2 and 3 should be grounded
in what you see here.

What this script does
---------------------
1.  Parse training.json and test.json — print annotation counts
2.  Class distribution — exact counts + imbalance ratio
3.  Image size distribution — are all images the same size?
4.  Box size distribution — how big are cells on average?
5.  Annotation density — how many cells per image?
6.  Visualise 8 sample images with ground-truth boxes (saved to outputs/)
7.  Visualise the class imbalance as a bar chart (saved to outputs/)
8.  Print a "design decisions" summary that you can paste into the paper

How to run
----------
From the project root:
    python Phase1-EDA/eda.py

On Google Colab:
    !git clone <your-repo>
    %cd MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images
    !python Phase1-EDA/eda.py --data-root /content/drive/MyDrive/malaria
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Colab and headless servers
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Project root on sys.path so we can import shared/ ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.label_map import (
    LABEL_TO_INT, INT_TO_LABEL, FOREGROUND_NAMES,
    PARASITE_CLASSES, SHORT_NAME, CLASS_COLOUR_RGB
)

SKIP_LABELS = {"difficult"}

# ── Output folder ─────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. JSON PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotations(json_path: Path) -> list[dict]:
    """
    Parse one annotation JSON file into a flat list of box records.

    Each record:
        img_name  : str    bare filename, e.g. "abc.png"
        label     : str    class name
        label_idx : int    integer from LABEL_TO_INT
        x_min     : float  left column
        y_min     : float  top row
        x_max     : float  right column
        y_max     : float  bottom row
        width     : float  image width  (from JSON shape)
        height    : float  image height (from JSON shape)
    """
    with open(json_path) as f:
        records = json.load(f)

    rows = []
    skipped = 0
    for rec in records:
        img_name = Path(rec["image"]["pathname"]).name
        img_w    = rec["image"]["shape"]["c"]
        img_h    = rec["image"]["shape"]["r"]

        for obj in rec.get("objects", []):
            label = obj["category"]
            if label in SKIP_LABELS:
                skipped += 1
                continue
            if label not in LABEL_TO_INT:
                continue

            bb    = obj["bounding_box"]
            x_min = bb["minimum"]["c"]
            y_min = bb["minimum"]["r"]
            x_max = bb["maximum"]["c"]
            y_max = bb["maximum"]["r"]

            if x_max <= x_min or y_max <= y_min:
                continue

            rows.append({
                "img_name":  img_name,
                "label":     label,
                "label_idx": LABEL_TO_INT[label],
                "x_min":     float(x_min),
                "y_min":     float(y_min),
                "x_max":     float(x_max),
                "y_max":     float(y_max),
                "width":     float(img_w),
                "height":    float(img_h),
            })

    if skipped:
        print(f"    Skipped {skipped} 'difficult' annotations.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def class_distribution(rows: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    return dict(counts)


def images_per_class(rows: list[dict]) -> dict[str, int]:
    """How many distinct images contain at least one box of each class."""
    img_sets = defaultdict(set)
    for r in rows:
        img_sets[r["label"]].add(r["img_name"])
    return {k: len(v) for k, v in img_sets.items()}


def boxes_per_image_stats(rows: list[dict]) -> dict:
    img_counts = defaultdict(int)
    for r in rows:
        img_counts[r["img_name"]] += 1
    vals = list(img_counts.values())
    return {
        "n_images": len(vals),
        "min":      min(vals),
        "max":      max(vals),
        "mean":     float(np.mean(vals)),
        "median":   float(np.median(vals)),
        "p95":      float(np.percentile(vals, 95)),
    }


def box_size_stats(rows: list[dict]) -> dict:
    widths  = [r["x_max"] - r["x_min"] for r in rows]
    heights = [r["y_max"] - r["y_min"] for r in rows]
    areas   = [w * h for w, h in zip(widths, heights)]
    return {
        "w_mean":    float(np.mean(widths)),
        "w_median":  float(np.median(widths)),
        "h_mean":    float(np.mean(heights)),
        "h_median":  float(np.median(heights)),
        "area_mean": float(np.mean(areas)),
        "area_p5":   float(np.percentile(areas, 5)),
        "area_p95":  float(np.percentile(areas, 95)),
    }


def image_size_check(rows: list[dict]) -> dict:
    sizes = set((r["width"], r["height"]) for r in rows)
    return {"unique_sizes": len(sizes), "sizes": sorted(sizes)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(counts: dict[str, int], split_name: str):
    """Bar chart of class frequencies with imbalance ratio annotated."""
    labels  = [l for l in FOREGROUND_NAMES if l in counts]
    values  = [counts.get(l, 0) for l in labels]
    colours = [tuple(v/255 for v in CLASS_COLOUR_RGB[l]) for l in labels]
    short   = [SHORT_NAME[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short, values, color=colours, edgecolor="black", linewidth=0.5)

    # Annotate each bar with its count
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    rbc   = counts.get("red blood cell", 1)
    troph = counts.get("trophozoite", 1)
    ratio = rbc / max(troph, 1)
    ax.set_title(
        f"Class Distribution — {split_name}\n"
        f"RBC : Trophozoite ratio = {ratio:.0f} : 1",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of annotations")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = OUT_DIR / f"class_distribution_{split_name.lower()}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sample_images(rows: list[dict], image_dir: Path, n: int = 8):
    """
    Show n sample images with bounding boxes drawn on them.
    Selects images that contain at least one parasitic cell so you
    can actually see the annotation quality.
    """
    # Prefer images with parasites
    parasite_imgs = list({r["img_name"] for r in rows if r["label"] in PARASITE_CLASSES})
    rng = np.random.default_rng(seed=0)
    if len(parasite_imgs) >= n:
        chosen = list(rng.choice(parasite_imgs, size=n, replace=False))
    else:
        all_imgs = list({r["img_name"] for r in rows})
        chosen   = list(rng.choice(all_imgs, size=min(n, len(all_imgs)), replace=False))

    # Group annotations by image
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["img_name"]].append(r)

    ncols = 4
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for ax_i, img_name in enumerate(chosen):
        ax = axes[ax_i]
        img_path = image_dir / img_name
        if not img_path.exists():
            ax.axis("off")
            ax.set_title("missing", fontsize=8)
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        ax.imshow(img)

        for r in by_image[img_name]:
            colour = tuple(v/255 for v in CLASS_COLOUR_RGB.get(r["label"], (128, 128, 128)))
            rect   = mpatches.Rectangle(
                (r["x_min"], r["y_min"]),
                r["x_max"] - r["x_min"],
                r["y_max"] - r["y_min"],
                linewidth=1.2, edgecolor=colour, facecolor="none"
            )
            ax.add_patch(rect)

        n_cells   = len(by_image[img_name])
        n_parasit = sum(1 for r in by_image[img_name] if r["label"] in PARASITE_CLASSES)
        ax.set_title(f"{img_name[:12]}…\n{n_cells} cells, {n_parasit} parasites",
                     fontsize=8)
        ax.axis("off")

    for ax_i in range(len(chosen), len(axes)):
        axes[ax_i].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=tuple(v/255 for v in CLASS_COLOUR_RGB[l]),
                       label=f"{SHORT_NAME[l]} — {l}")
        for l in FOREGROUND_NAMES
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sample Blood Smear Images with Ground-Truth Boxes",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT_DIR / "sample_images.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_box_size_histogram(rows: list[dict]):
    """Distribution of cell bounding box areas, split by class."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: overall area distribution (all boxes)
    areas = [(r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"]) for r in rows]
    axes[0].hist(areas, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Bounding Box Area Distribution (all classes)")
    axes[0].set_xlabel("Area (pixels²)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.median(areas), color="red", linestyle="--",
                    label=f"Median = {int(np.median(areas))} px²")
    axes[0].legend()

    # Right: per-class median area (excluding RBC to see parasites clearly)
    parasite_rows  = [r for r in rows if r["label"] in PARASITE_CLASSES]
    by_class_areas = defaultdict(list)
    for r in parasite_rows:
        area = (r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"])
        by_class_areas[r["label"]].append(area)

    cls_names  = [l for l in PARASITE_CLASSES if l in by_class_areas]
    medians    = [float(np.median(by_class_areas[l])) for l in cls_names]
    colours    = [tuple(v/255 for v in CLASS_COLOUR_RGB[l]) for l in cls_names]
    axes[1].barh([SHORT_NAME[l] for l in cls_names], medians,
                 color=colours, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Median Box Area — Parasitic Stages")
    axes[1].set_xlabel("Median Area (pixels²)")

    fig.tight_layout()
    out = OUT_DIR / "box_size_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(data_root: Path, image_dir: Path):
    print("\n" + "=" * 60)
    print("  MalariAI — Phase 1: Exploratory Data Analysis")
    print("=" * 60)

    train_json = data_root / "training.json"
    test_json  = data_root / "test.json"

    # ── 1. Parse ──────────────────────────────────────────────────────────
    print("\n[1] Parsing annotations …")
    train_rows = parse_annotations(train_json)
    test_rows  = parse_annotations(test_json)
    print(f"    Train: {len(train_rows):,} boxes across "
          f"{len({r['img_name'] for r in train_rows})} images")
    print(f"    Test : {len(test_rows):,} boxes across "
          f"{len({r['img_name'] for r in test_rows})} images")

    # ── 2. Class distribution ─────────────────────────────────────────────
    print("\n[2] Class distribution (Training) …")
    train_counts = class_distribution(train_rows)
    rbc          = train_counts.get("red blood cell", 0)
    print(f"\n  {'CLASS':<22} {'COUNT':>7}  {'%':>6}  {'IMBALANCE RATIO':>18}")
    print("  " + "-" * 60)
    total_boxes = sum(train_counts.values())
    for label in FOREGROUND_NAMES:
        cnt   = train_counts.get(label, 0)
        pct   = 100 * cnt / max(total_boxes, 1)
        ratio = f"1 : {rbc // max(cnt, 1)}" if label != "red blood cell" else "—"
        print(f"  {label:<22} {cnt:>7,}  {pct:>5.1f}%  {ratio:>18}")
    print(f"\n  Total annotated boxes: {total_boxes:,}")
    print(f"  Dominant class (RBC): {rbc:,}  ({100*rbc/total_boxes:.1f}% of all boxes)")
    print(f"  Hardest imbalance: RBC vs Gametocyte = {rbc // max(train_counts.get('gametocyte',1), 1)} : 1")

    # ── 3. Image size check ───────────────────────────────────────────────
    print("\n[3] Image size check …")
    size_info = image_size_check(train_rows)
    print(f"    Unique (W, H) combinations: {size_info['unique_sizes']}")
    for s in size_info["sizes"][:5]:
        print(f"      {int(s[0])} x {int(s[1])}")
    if size_info["unique_sizes"] > 5:
        print(f"      … and {size_info['unique_sizes'] - 5} more")

    # ── 4. Boxes per image ────────────────────────────────────────────────
    print("\n[4] Boxes per image …")
    bpi = boxes_per_image_stats(train_rows)
    print(f"    Images:  {bpi['n_images']}")
    print(f"    Min:     {bpi['min']}  boxes/image")
    print(f"    Max:     {bpi['max']}  boxes/image")
    print(f"    Mean:    {bpi['mean']:.1f} boxes/image")
    print(f"    Median:  {bpi['median']:.1f} boxes/image")
    print(f"    95th %:  {bpi['p95']:.1f} boxes/image")
    print(f"    >> This tells us how dense each smear is.")
    print(f"    >> Dense images (high median) are where NMS will struggle most.")

    # ── 5. Box size stats ─────────────────────────────────────────────────
    print("\n[5] Bounding box size stats (all training boxes) …")
    bss = box_size_stats(train_rows)
    print(f"    Width  — mean: {bss['w_mean']:.1f} px  |  median: {bss['w_median']:.1f} px")
    print(f"    Height — mean: {bss['h_mean']:.1f} px  |  median: {bss['h_median']:.1f} px")
    print(f"    Area   — mean: {bss['area_mean']:.0f} px²  |  p5: {bss['area_p5']:.0f}  |  p95: {bss['area_p95']:.0f}")
    crop_size = int(bss['w_median'] * 1.5)
    print(f"    >> Crop size for Stage 2: ~{crop_size}px  (1.5 × median width)")

    # ── 6. Visualisations ─────────────────────────────────────────────────
    print("\n[6] Generating visualisations …")
    plot_class_distribution(train_counts, split_name="Training")
    plot_class_distribution(class_distribution(test_rows), split_name="Test")
    plot_box_size_histogram(train_rows)

    if image_dir.exists():
        plot_sample_images(train_rows, image_dir, n=8)
    else:
        print(f"  [skip] image_dir not found: {image_dir}")
        print(f"         Pass --image-dir to generate sample images.")

    # ── 7. Design decisions summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DESIGN DECISIONS (based on EDA)")
    print("=" * 60)
    print(f"""
  1. CLASS IMBALANCE
     RBC dominates at {100*rbc/total_boxes:.0f}% of all boxes.
     Gametocyte is rarest (~{100*train_counts.get('gametocyte',0)/total_boxes:.1f}%).
     → Use Focal Loss (gamma=2.0) in both pipelines.
     → For Faster R-CNN: use class-balanced sampling or class weights in head.
     → Do NOT merge classes into binary (infected/healthy) — we need per-stage AP.

  2. CELL SIZE
     Median cell area ≈ {int(bss['area_mean'])} px² on {int(size_info['sizes'][0][0])}×{int(size_info['sizes'][0][1])} images.
     → EfficientNet crop size: 64×64 is safe (covers the {bss['w_median']:.0f}px median width with padding).
     → Watershed AREA_MIN = 150 px²  (5% of smallest cells) — tune on val images.

  3. ANNOTATION COMPLETENESS
     BBBC041 dataset is known to have incomplete annotations.
     Unannotated RBCs in dense regions are treated as background in Faster R-CNN.
     → This is Novelty Claim N1: watershed detects them all, FRCNN misses them.

  4. IMAGE SIZE
     Images are {size_info['unique_sizes']} unique size(s): {size_info['sizes'][:2]}
     → Faster R-CNN's FPN handles variable sizes internally — no resize needed.
     → For watershed: process at native resolution, then extract crops.

  5. DENSE REGIONS
     Some images have {bpi['max']} boxes — severe overlap likely.
     95th percentile: {bpi['p95']:.0f} boxes/image.
     → NMS threshold tuning is critical for Baseline A.
     → Watershed + distance transform handles this natively (Novelty N2).
""")

    print(f"All outputs saved to: {OUT_DIR}/")
    print("Next step: Phase 2 — Baseline Faster R-CNN\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — MalariAI EDA")
    parser.add_argument(
        "--data-root", type=str,
        default=str(ROOT / "data" / "malaria"),
        help="Folder containing training.json and test.json"
    )
    parser.add_argument(
        "--image-dir", type=str,
        default=str(ROOT / "data" / "malaria" / "images"),
        help="Folder containing the .png image files"
    )
    args = parser.parse_args()

    run_eda(Path(args.data_root), Path(args.image_dir))
