"""
prepare_data.py — MalariAI annotation pipeline (co-author: data foundations)

What this script does
---------------------
1. Reads training.json and test.json from data/malaria/
2. Parses every bounding-box annotation, keeping all 6 foreground classes
   (drops "difficult" only — nothing else is collapsed or merged)
3. Splits training.json 80/20 at the IMAGE level (stratified-ish, seeded)
   → train_annotations.csv   (≈ 80 % of images)
   → val_annotations.csv     (≈ 20 % of images)
4. Converts test.json whole
   → test_annotations.csv

CSV schema (same for all three files)
--------------------------------------
img_name   – bare filename, e.g. "abc123.png"   (no directory prefix)
label      – class name string, e.g. "trophozoite"
label_idx  – integer from label_map.LABEL_TO_INT
x_min      – left   column  (pixel)
y_min      – top    row     (pixel)
x_max      – right  column  (pixel)
y_max      – bottom row     (pixel)

Why img_name is bare
---------------------
The MalariaDataset class knows the image root directory at construction time,
so concatenating a prefix inside the CSV would create brittle absolute paths.
Keep the CSV pure; let the Dataset handle path assembly.

Usage
-----
    python data/prepare_data.py                             # uses defaults
    python data/prepare_data.py --data-root /path/to/data  # custom root
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Import shared label map ───────────────────────────────────────────────────
# Add project root to sys.path so this script is runnable from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.label_map import LABEL_TO_INT  # noqa: E402

SKIP_LABELS = {"difficult"}   # annotations we intentionally discard


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_json(json_path: Path) -> pd.DataFrame:
    """
    Parse a MalariAI annotation JSON file into a flat DataFrame.

    JSON structure (per element):
        {
          "image": {
              "pathname": "/images/<uuid>.png",
              "shape": {"r": H, "c": W, "channels": 3}
          },
          "objects": [
              {
                "bounding_box": {
                    "minimum": {"r": y_min, "c": x_min},
                    "maximum": {"r": y_max, "c": x_max}
                },
                "category": "red blood cell"
              },
              ...
          ]
        }

    Returns
    -------
    pd.DataFrame with columns:
        img_name, label, label_idx, x_min, y_min, x_max, y_max
    """
    print(f"  Parsing {json_path} ...")
    with open(json_path, "r") as f:
        records = json.load(f)

    rows = []
    skipped_labels = set()
    skipped_invalid_box = 0

    for record in tqdm(records, desc=f"    {json_path.name}", leave=False):
        # ── Image name ────────────────────────────────────────────────────
        pathname = record["image"]["pathname"]          # e.g. "/images/abc.png"
        img_name = Path(pathname).name                  # -> "abc.png"

        # ── Annotations ───────────────────────────────────────────────────
        for obj in record.get("objects", []):
            label = obj["category"]

            # Drop "difficult" and any future unknown labels gracefully
            if label in SKIP_LABELS:
                continue
            if label not in LABEL_TO_INT:
                skipped_labels.add(label)
                continue

            bb = obj["bounding_box"]
            x_min = bb["minimum"]["c"]
            y_min = bb["minimum"]["r"]
            x_max = bb["maximum"]["c"]
            y_max = bb["maximum"]["r"]

            # Sanity check — degenerate boxes cause NaN losses during training
            if x_max <= x_min or y_max <= y_min:
                skipped_invalid_box += 1
                continue

            rows.append({
                "img_name":  img_name,
                "label":     label,
                "label_idx": LABEL_TO_INT[label],
                "x_min":     float(x_min),
                "y_min":     float(y_min),
                "x_max":     float(x_max),
                "y_max":     float(y_max),
            })

    if skipped_labels:
        print(f"  WARNING: Unknown labels skipped (not in label_map): {skipped_labels}")
    if skipped_invalid_box:
        print(f"  WARNING: {skipped_invalid_box} degenerate bounding boxes dropped.")

    df = pd.DataFrame(rows)
    return df


def train_val_split(df: pd.DataFrame, val_fraction: float = 0.20,
                    seed: int = 42):
    """
    Split at the IMAGE level so no image appears in both train and val.

    We shuffle the unique image list with a fixed seed, then take the last
    val_fraction as validation. This is deterministic and reproducible.
    """
    all_images = df["img_name"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_fraction))
    val_images = set(all_images[-n_val:])
    train_images = set(all_images[:-n_val])

    train_df = df[df["img_name"].isin(train_images)].reset_index(drop=True)
    val_df   = df[df["img_name"].isin(val_images)].reset_index(drop=True)

    return train_df, val_df


def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts().sort_index()


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  OK  Saved {len(df):,} rows -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MalariAI JSON annotations to train/val/test CSVs."
    )
    parser.add_argument(
        "--data-root", type=str,
        default=str(PROJECT_ROOT / "data" / "malaria"),
        help="Directory containing training.json, test.json, and images/"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.20,
        help="Fraction of training images held out for validation (default 0.20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val split"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)

    train_json = data_root / "training.json"
    test_json  = data_root / "test.json"

    for p in [train_json, test_json]:
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Check --data-root.")

    # ── Training -> train / val split ─────────────────────────────────────
    print("\n[1/2] Parsing training.json ...")
    train_full = parse_json(train_json)
    train_df, val_df = train_val_split(
        train_full, val_fraction=args.val_fraction, seed=args.seed
    )

    # ── Test ─────────────────────────────────────────────────────────────
    print("\n[2/2] Parsing test.json ...")
    test_df = parse_json(test_json)

    # ── Save ─────────────────────────────────────────────────────────────
    print("\nSaving CSVs ...")
    save(train_df, out_dir / "train_annotations.csv")
    save(val_df,   out_dir / "val_annotations.csv")
    save(test_df,  out_dir / "test_annotations.csv")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("DATASET SUMMARY")
    print("=" * 56)

    split_info = [
        ("Train", train_df),
        ("Val",   val_df),
        ("Test",  test_df),
    ]
    for split_name, df in split_info:
        n_images = df["img_name"].nunique()
        n_boxes  = len(df)
        print(f"\n{split_name}  ({n_images} images, {n_boxes:,} boxes)")
        dist = class_distribution(df)
        for label, count in dist.items():
            pct = 100 * count / n_boxes
            bar = "#" * int(pct / 2)
            idx = LABEL_TO_INT[label]
            print(f"  [{idx}] {label:<20} {count:6,}  ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 56)
    print("Done. Import path reminder:")
    print("  from src.utils.label_map import LABEL_TO_INT, INT_TO_LABEL")
    print("  from src.models.dataset import MalariaDataset")
    print()


if __name__ == "__main__":
    main()
