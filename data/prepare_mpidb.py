"""
prepare_mpidb.py — MP-IDB annotation converter for MalariAI v2

What this script does
---------------------
1. Reads MP-IDB Supervisely JSON annotations from data/MP-IDB/ann/
2. Decodes base64+zlib bitmap masks → tight bounding boxes (origin + mask dims)
3. Outputs a flat CSV with the same schema used by BBBC041 prepare_data.py,
   plus two extra columns for cross-dataset evaluation:
     - species      : the original label (falciparum / vivax / malariae / ovale)
     - parasitized  : always 1 (all annotated objects are infected cells)

CSV schema
----------
img_name    – bare filename, e.g. "1305121398-0001-R_S.jpg"
label       – species string (falciparum | vivax | malariae | ovale)
label_idx   – integer index (1=falciparum, 2=vivax, 3=malariae, 4=ovale)
x_min, y_min, x_max, y_max  – tight bounding box (pixels, from mask + origin)
parasitized – always 1; used for binary cross-dataset evaluation

Cross-dataset evaluation strategy
-----------------------------------
MP-IDB uses SPECIES labels (falciparum / vivax / malariae / ovale).
BBBC041 uses LIFECYCLE STAGE labels (ring / trophozoite / schizont / gametocyte).
These taxonomies do not align at fine grain, so we evaluate BINARY detection:
  - Any annotated object in MP-IDB = parasitized cell (positive)
  - The pipeline's job: find ALL infected cells regardless of species/stage
This lets us report "cell detection recall" on an unseen staining/institution.

Usage
-----
    python data/prepare_mpidb.py
    python data/prepare_mpidb.py --data-root data/MP-IDB --out-dir data/processed
"""

import argparse
import base64
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Label map (MP-IDB species, separate from BBBC041 stage labels) ────────────
MPIDB_LABEL_TO_INT: dict[str, int] = {
    "falciparum": 1,
    "vivax":      2,
    "malariae":   3,
    "ovale":      4,
}


def decode_bitmap(data_b64: str, origin: list) -> tuple[int, int, int, int]:
    """
    Decode a Supervisely bitmap annotation to a tight bounding box.

    Supervisely bitmap encoding:
        base64(zlib(PNG_data))
    The PNG is a binary mask (palette mode 'P', values 0 or 1).
    The bounding box is simply origin + mask dimensions — no pixel iteration
    needed because the mask is already cropped tight by the labeller.

    Parameters
    ----------
    data_b64 : str
        Base64-encoded, zlib-compressed PNG mask.
    origin : list
        [x, y] pixel offset of the mask's top-left corner in the full image.

    Returns
    -------
    (x_min, y_min, x_max, y_max) : int tuple
    """
    raw = base64.b64decode(data_b64)
    png_bytes = zlib.decompress(raw)
    mask = Image.open(io.BytesIO(png_bytes))
    mw, mh = mask.size          # PIL: (width, height)
    ox, oy = int(origin[0]), int(origin[1])
    return ox, oy, ox + mw, oy + mh


def parse_mpidb(ann_dir: Path) -> pd.DataFrame:
    """
    Parse all MP-IDB annotation JSON files into a flat DataFrame.

    JSON structure (Supervisely format):
        {
          "size": {"height": 1944, "width": 2592},
          "tags": [{"name": "ring stage"}, ...],
          "objects": [
            {
              "classTitle": "falciparum",
              "geometryType": "bitmap",
              "bitmap": {
                "origin": [x, y],
                "data": "<base64+zlib+PNG>"
              }
            }, ...
          ]
        }
    """
    files = sorted(ann_dir.glob("*.json"))
    print(f"  Found {len(files)} annotation files.")

    rows = []
    skipped_labels = set()
    skipped_invalid_box = 0
    decode_errors = 0

    for fp in tqdm(files, desc="  Parsing MP-IDB annotations"):
        img_name = fp.stem          # "1305121398-0001-R_S.jpg" (stem removes .json)

        with open(fp) as f:
            ann = json.load(f)

        for obj in ann.get("objects", []):
            label = obj.get("classTitle", "")

            if label not in MPIDB_LABEL_TO_INT:
                skipped_labels.add(label)
                continue

            bm = obj.get("bitmap", {})
            try:
                x_min, y_min, x_max, y_max = decode_bitmap(
                    bm["data"], bm["origin"]
                )
            except Exception as e:
                decode_errors += 1
                continue

            if x_max <= x_min or y_max <= y_min:
                skipped_invalid_box += 1
                continue

            rows.append({
                "img_name":    img_name,
                "label":       label,
                "label_idx":   MPIDB_LABEL_TO_INT[label],
                "x_min":       float(x_min),
                "y_min":       float(y_min),
                "x_max":       float(x_max),
                "y_max":       float(y_max),
                "parasitized": 1,
            })

    if skipped_labels:
        print(f"  WARNING: Unknown labels skipped: {skipped_labels}")
    if skipped_invalid_box:
        print(f"  WARNING: {skipped_invalid_box} degenerate boxes dropped.")
    if decode_errors:
        print(f"  WARNING: {decode_errors} bitmap decode errors.")

    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts().sort_index()


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  OK  Saved {len(df):,} rows -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert MP-IDB Supervisely bitmap annotations to CSV."
    )
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--data-root", type=str,
        default=str(PROJECT_ROOT / "data" / "MP-IDB"),
        help="Directory containing MP-IDB img/ and ann/ subdirectories"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for CSV files"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_dir   = data_root / "ann"
    out_dir   = Path(args.out_dir)

    if not ann_dir.exists():
        sys.exit(f"ERROR: {ann_dir} not found. Check --data-root.")

    print(f"\n[1/1] Parsing MP-IDB annotations from {ann_dir} ...")
    df = parse_mpidb(ann_dir)

    out_path = out_dir / "mpidb_annotations.csv"
    print("\nSaving CSV ...")
    save(df, out_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("MP-IDB ANNOTATION SUMMARY")
    print("=" * 56)
    n_images = df["img_name"].nunique()
    n_boxes  = len(df)
    print(f"\nTotal: {n_images} images, {n_boxes:,} annotated cells")

    dist = class_distribution(df)
    for label, count in dist.items():
        pct = 100 * count / n_boxes
        bar = "#" * int(pct / 2)
        idx = MPIDB_LABEL_TO_INT[label]
        print(f"  [{idx}] {label:<12} {count:5,}  ({pct:5.1f}%)  {bar}")

    # Box size stats
    df["box_w"] = df["x_max"] - df["x_min"]
    df["box_h"] = df["y_max"] - df["y_min"]
    df["box_area"] = df["box_w"] * df["box_h"]
    print(f"\nBounding box area (px²):")
    print(f"  min={df['box_area'].min():.0f}  "
          f"max={df['box_area'].max():.0f}  "
          f"mean={df['box_area'].mean():.0f}  "
          f"median={df['box_area'].median():.0f}")

    print("\n" + "=" * 56)
    print("Done. Next step:")
    print("  python src/pipeline_b_v2/e2e_eval.py --dataset mpidb")
    print()


if __name__ == "__main__":
    main()
