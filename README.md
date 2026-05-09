# MalariAI — A Label-Resilient Decoupled Framework for Universal Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears

> **Research Paper** · Kaysarul Anas Apurba (Laurentian University, Canada) · Co-Author (Melbourne Institute of Technology, Australia)

---

## Abstract

Automated malaria diagnosis from peripheral blood smear microscopy remains a critical open problem in global health AI. Three compounding failure modes persist across the literature:

1. **Incomplete Annotation** — End-to-end detectors (Faster R-CNN, YOLO) treat unannotated cells as background, suppressing true positives in sparsely labelled datasets.
2. **Dense Overlap / NMS Failure** — Non-Maximum Suppression discards valid cell detections in high-density smear regions where red blood cells routinely overlap.
3. **Black-Box Output** — Existing pipelines produce opaque class labels without spatial evidence, limiting clinical adoption.

**MalariAI** is a two-stage decoupled framework that addresses all three simultaneously:

- **Stage 1** — An *annotation-agnostic* distance-transform guided watershed algorithm that isolates every cell in the smear regardless of ground-truth completeness.
- **Stage 2** — An **EfficientNet-B0** classifier trained with **Focal Loss** for multi-class infection stage identification (ring, trophozoite, schizont, gametocyte), with **Grad-CAM++** generating per-cell spatial attention heatmaps.

We benchmark against a **Faster R-CNN** baseline on the publicly available **NIH BBBC041** dataset (1,208 training images, 80,113 annotated instances across 6 cell categories).

---

## Why This Approach?

| Problem | Old Approach (Mask R-CNN, 2021) | MalariAI (2025) |
|---|---|---|
| Missing annotations | Treats unannotated cells as background ❌ | Watershed finds *all* cells — label-agnostic ✅ |
| Dense overlapping cells | NMS deletes genuine overlapping detections ❌ | Distance-transform splits touching cells ✅ |
| Clinical explainability | Black-box prediction ❌ | Grad-CAM++ heatmap per cell ✅ |
| Tech stack | TF1 Matterport Mask R-CNN (deprecated) ❌ | Modern PyTorch + torchvision ✅ |
| Multi-class imbalance | Ignored (537:1 RBC to gametocyte ratio) ❌ | Focal Loss + weighted sampling ✅ |

---

## Dataset

**NIH BBBC041** — Giemsa-stained *P. falciparum* thin blood smears  
Source: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC041)

| Split | Images | Boxes |
|---|---|---|
| Training | 1,208 | 79,672 |
| Test | 80 | 5,917 |

**Class distribution (training):**

| Class | Count | % | vs RBC |
|---|---|---|---|
| Red Blood Cell | 77,420 | 97.2% | — |
| Trophozoite | 1,473 | 1.8% | 1:52 |
| Ring | 353 | 0.4% | 1:219 |
| Schizont | 179 | 0.2% | 1:432 |
| Gametocyte | 144 | 0.2% | 1:537 |
| Leukocyte | 103 | 0.1% | 1:751 |

> Data is not included in the repo. Place it under `data/malaria/` — see below.

### Dataset Visualisations

| Class Distribution | Boxes per Image |
|---|---|
| ![Class Distribution](notebooks/outputs/class_distribution.png) | ![Boxes per Image](notebooks/outputs/boxes_per_image.png) |

| Box Size Distributions | Parasite Crops |
|---|---|
| ![Box Sizes](notebooks/outputs/box_sizes.png) | ![Parasite Crops](notebooks/outputs/parasite_crops.png) |

> 📄 See the full EDA analysis: [`notebooks/EDA_Overview.md`](notebooks/EDA_Overview.md)

---

## Project Structure

```text
MalariAI/
├── README.md
├── requirements.txt
├── problem_statement.md
│
├── data/
│   ├── .gitignore          # Keeps data/ tracked but ignores large files
│   └── malaria/            # ← Place dataset here (not committed)
│       ├── training.json
│       ├── test.json
│       └── images/
│
├── shared/
│   └── label_map.py        # Single source of truth: LABEL_TO_INT, NUM_CLASSES, etc.
│
├── Phase1-EDA/             # Exploratory Data Analysis
│   ├── dataset.py          # MalariaDataset + MalariaCropDataset (shared across phases)
│   ├── eda.py              # Standalone EDA script
│   └── outputs/            # EDA plots (class dist, box sizes, sample images)
│
├── notebooks/
│   ├── Phase1_EDA.ipynb    # Interactive EDA notebook
│   └── outputs/            # Notebook plot outputs
│
├── src/
│   ├── pipeline_a/
│   │   └── baseline_frcnn.py     # Baseline A: Faster R-CNN (ResNet-50 FPN)
│   ├── pipeline_b/
│   │   ├── stage1_watershed.py   # Stage 1: Distance-transform watershed segmentation
│   │   └── stage2_classify.py    # Stage 2: EfficientNet-B0 + Focal Loss + Grad-CAM++
│   ├── utils/
│   │   └── label_map.py
│   ├── train.py
│   ├── inference.py
│   └── segment_all_cells.py      # Run Stage 1 over a full image dataset
│
├── paper/
│   ├── MalariAI_Paper_Draft.tex  # LaTeX paper draft
│   ├── MalariAI_Paper_Draft.pdf
│   ├── malaria_architecture_figure_*.png
│   ├── coauthor_reading_guide.md
│   ├── strategi.md               # Architecture rationale
│   └── research_strategy*.md
│
└── archive/                # Legacy Mask R-CNN / TF1 scripts (deprecated)
```

---

## Architecture

### Pipeline B — MalariAI (Proposed)

```
Full Blood Smear Image (1600×1200)
          │
          ▼
┌─────────────────────────────┐
│  STAGE 1: WATERSHED         │
│  • Otsu thresholding        │
│  • Distance transform       │
│  • Marker-based watershed   │
│  → Crop every cell (N crops)│
│    (annotation-agnostic)    │
└─────────────┬───────────────┘
              │  N individual cell crops
              ▼
┌─────────────────────────────┐
│  STAGE 2: EFFICIENTNET-B0   │
│  • Focal Loss (γ=2, α=inv)  │
│  • 7-class head             │
│  • Grad-CAM++ heatmaps      │
│  → Class + attention map    │
└─────────────────────────────┘
```

### Pipeline A — Faster R-CNN Baseline

```
Full Blood Smear Image
          │
          ▼
ResNet-50-FPN Backbone → RPN → RoI Align → Classification Head
          │
          ▼
Bounding boxes + class labels (NMS applied)
```

---

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/Anaskaysar/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images.git
cd MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images

python -m venv malariaenv
# Windows:
malariaenv\Scripts\activate
# macOS/Linux:
source malariaenv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download from [BBBC041](https://bbbc.broadinstitute.org/BBBC041) and place:

```
data/malaria/training.json
data/malaria/test.json
data/malaria/images/   ← all .png files go here
```

### 3. Run EDA

```bash
# Notebook (recommended)
jupyter notebook notebooks/Phase1_EDA.ipynb

# Or standalone script
python Phase1-EDA/eda.py
```

### 4. Train — Pipeline A (Faster R-CNN baseline)

```bash
python src/train.py --pipeline a
```

### 5. Train — Pipeline B (MalariAI two-stage)

```bash
# Stage 1: segment all cells and save crops
python src/segment_all_cells.py \
    --json data/malaria/training.json \
    --img-dir data/malaria/images \
    --out-dir data/crops

# Stage 2: train the EfficientNet-B0 classifier
python src/pipeline_b/stage2_classify.py \
    --crops-dir data/crops \
    --epochs 30
```

### 6. Inference with Grad-CAM++

```bash
python src/inference.py \
    --image data/malaria/images/SOME_IMAGE.png \
    --pipeline b \
    --model models/stage2_classifier.pth \
    --gradcam
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Watershed over Mask R-CNN for Stage 1** | Zero labelled data required; no NMS; splits touching cells by topology |
| **EfficientNet-B0 for Stage 2** | Strong accuracy/speed trade-off; transfers well from ImageNet |
| **Focal Loss** | Addresses 537:1 class imbalance (RBC vs gametocyte) without oversampling |
| **Grad-CAM++** | More accurate multi-activation localisation than standard Grad-CAM; clinically interpretable |
| **Decoupled stages** | Segmentation is label-agnostic; classification sees pre-isolated crops → avoids NMS artefacts |

---

## Research Contributions

**C1 — Label-Resilient Segmentation.** An annotation-agnostic Stage 1 that detects every cell in a blood smear without relying on ground-truth bounding boxes, directly addressing the incomplete-annotation failure mode in the NIH BBBC041 benchmark.

**C2 — Density-Invariant Overlap Handling.** Distance-transform guided watershed separates touching and overlapping cells at the instance level, recovering detections that NMS-based detectors systematically suppress in dense smear regions.

**C3 — Integrated End-to-End Explainability.** Grad-CAM++ spatial attention heatmaps are generated within the full detection-to-classification pipeline, providing per-cell visual evidence of parasite location — a capability absent from all prior whole-slide malaria detection systems.

---

## Current Status

| Phase | Status |
|---|---|
| Phase 1 — EDA & Dataset Analysis | ✅ Complete |
| Phase 2 — Faster R-CNN Baseline (Pipeline A) | 🔧 In progress |
| Phase 3 — Watershed + EfficientNet (Pipeline B) | 🔧 In progress |
| Phase 4 — Evaluation, Ablation, Grad-CAM++ | ⏳ Pending |
| Paper Write-up | 📝 Draft in progress (`paper/MalariAI_Paper_Draft.tex`) |

---

## Citation

```bibtex
@article{apurba2025malariai,
  title   = {MalariAI: A Label-Resilient Decoupled Framework for Universal Cell
             Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears},
  author  = {Apurba, Kaysarul Anas and {Co-Author}},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## Acknowledgements

The authors thank their former supervisor from North South University, Dhaka, Bangladesh (currently Assistant Professor at KFUPM), for guidance during the Junior Design 299 course (2021) where the initial concept was developed.

---

## License

This repository is for academic research purposes. The NIH BBBC041 dataset is subject to its own licence — see [BBBC041](https://bbbc.broadinstitute.org/BBBC041).