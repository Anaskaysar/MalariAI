# Option B — Stage 1 v3 Ablation: Gaussian Blur + Morphological Close

## What This Is

This folder contains everything needed to extend the CMIG paper with a Stage 1 v3
ablation row, should the co-authors decide to include it. Option A (current paper
submission) uses v2 parameters throughout. Option B would report v3 as the deployed
system and add an ablation discussion explaining the tradeoff.

---

## What Changed in v3 (vs v2)

Three modifications to `stage1_v2.py` → `stage1_v3.py`:

1. **Gaussian blur (5×5) before Otsu thresholding**
   Smooths staining intensity variation inside large infected cells (schizonts,
   trophozoites) before the binary mask is computed, reducing the chance that one
   large cell gets split into multiple binary regions.

2. **Morphological CLOSE after OPEN**
   Seals interior holes in large parasitised cells so they produce a single
   watershed seed rather than several. Sequence: OPEN (removes speckle) → CLOSE
   (fills holes).

3. **`REF_MIN_DIST` raised 12 → 16 px**
   Watershed seeds must be further apart, reducing over-segmentation of
   closely-packed cells.

4. **`REF_AREA_MAX` raised 8,000 → 14,000 px²**
   Captures large schizonts that were previously clipped by the old area ceiling.

---

## Evaluation Results (BBBC041 test set, 120 images)

Run on 2026-05-19 using:
```
python src/pipeline_b_v2/e2e_eval.py --dataset bbbc041 --stage1-only --stage1-version v3 --out-dir results/v3/e2e_bbbc041_v3
```

### Comparison Table

| Metric                          | v2 (paper) | v3         | Δ         |
|---------------------------------|------------|------------|-----------|
| GT cells                        | 5,917      | 5,917      | —         |
| Watershed boxes detected        | —          | 9,531      | —         |
| TP / FP / FN (@IoU0.5)          | 3,957/— /—  | 4,177/5,354/1,740 | —  |
| **Recall @IoU0.5**              | 66.88%     | **70.59%** | +3.71pp ✓ |
| Precision @IoU0.5               | —          | 43.83%     | (many FP) |
| F1 @IoU0.5                      | —          | 54.08%     | —         |
| **Recall @IoU0.25 (M2)**        | 78.08%     | **86.07%** | +7.99pp ✓ |
| Recall @IoU0.30                 | 76.80%     | **82.80%** | +6.00pp ✓ |
| **Centroid-in-box (M1)**        | 79.04%     | **92.73%** | +13.69pp ✓✓ |
| **Bio-loc recall (M4)**         | 79.50%     | **92.97%** | +13.47pp ✓✓ |
| **Infected sensitivity @0.25 (M3)** | **82.84%** | 78.55% | −4.29pp ✗ |
| Infected sensitivity @0.30      | 80.20%     | 70.63%     | −9.57pp ✗✗ |
| Infected sensitivity @0.50      | 66.34%     | 55.12%     | −11.22pp ✗✗ |

### Per-class Recovery @IoU0.5 (v3)

| Class         | Detected | GT  | Rate  |
|---------------|----------|-----|-------|
| Red blood cell| 4,010    |5,614| 71.4% |
| Ring          | 121      | 169 | 71.6% |
| Trophozoite   | 43       | 111 | 38.7% |
| Gametocyte    | 2        | 12  | 16.7% |
| Schizont      | 1        | 11  |  9.1% |

---

## Why Infected Sensitivity Dropped

The morphological CLOSE operation merges large infected cells with neighbouring
RBCs into a single blob. The resulting bounding box is much larger than the tight
GT box around the infected cell alone, so IoU falls below threshold and the
infected GT cell counts as a false negative. Schizont and gametocyte recovery
collapsed to near zero as a result.

Visually on individual demo images v3 looks better (fewer missed cells overall),
but the BBBC041 test-set average penalises the FP cost and the infected-cell IoU drop.

---

## How to Activate Option B in the Paper

### Step 1 — The code is already ready
- `stage1_v3.py` is in this folder and in `src/pipeline_b_v2/`
- `e2e_eval.py` already supports `--stage1-version v3`
- Eval results are in `eval_results_v3/`

### Step 2 — Update the HF Space pipeline.py
In `MalariAI/pipeline.py`, change three constants and the preprocessing block
(see diff below).

```python
# Change these constants:
_REF_MIN_DIST   = 16    # was 12
_REF_AREA_MAX   = 14000 # was 8000

# Change preprocessing (inside watershed_cells_with_labels):
gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)                          # ADD
_, binary = cv2.threshold(blurred, 0, 255,                            # blurred not gray
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened  = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=2)  # ADD
cleaned = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel, iterations=2)  # was binary→OPEN only
```

### Step 3 — Add ablation row to Table 4 (Stage 1 ablation) in the paper

Add a v3 row and a paragraph in Section 4.2 (Stage 1 Localisation: Alternative
Metrics) explaining the tradeoff.

Suggested paragraph:

> We additionally evaluated a v3 configuration that applies Gaussian pre-smoothing
> and morphological closing before the distance transform, and relaxes the minimum
> seed distance from 12 to 16 px. Centroid-in-box recall improves substantially
> (+13.7 pp, reaching 92.7%) and IoU@0.5 recall rises to 70.6%. However,
> infected-cell sensitivity (M3) falls from 82.8% to 78.6% at IoU@0.25 and more
> sharply at stricter thresholds, because the closing operation merges large
> parasitised cells with neighbouring erythrocytes, inflating bounding boxes and
> reducing per-cell IoU. This result illustrates a fundamental tension in
> annotation-agnostic segmentation: morphological operations that improve general
> cell recall can inadvertently reduce sensitivity to the clinically critical
> infected-cell subpopulation. We therefore retain v2 parameters for all reported
> metrics and use v3 in the public demo only.

---

## Current Status (Option A — active)

- Paper reports v2 metrics throughout
- HF Space (`MalariAI/pipeline.py`) uses v2 parameters
- `stage1_v3.py` exists in `src/pipeline_b_v2/` but is not the default
- `e2e_eval.py` supports `--stage1-version v3` for reproducibility

Everything needed to switch to Option B is in this folder.
