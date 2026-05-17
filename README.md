# MalariAI — No Cell Left Behind

### A Label-Resilient Decoupled Framework for Universal Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears

> **Research Paper — Targeting CMIG (Computerized Medical Imaging and Graphics)**
> Kaysarul Anas Apurba · Md Hasibul Hasan (Laurentian University, Canada) · Mohammed Ali (Melbourne Institute of Technology, Australia)
> Paper: `paper_writing/cmig_submission/MalariAI_CMIG.tex`

---

## Abstract

Automated malaria diagnosis from peripheral blood smear microscopy remains a critical open problem in global health AI. Three compounding failure modes persist across the literature:

1. **Incomplete Annotation (P1)** — End-to-end detectors (Faster R-CNN, YOLO) treat unannotated cells as background, suppressing true positives in sparsely labelled datasets.
2. **Dense Overlap / NMS Failure (P2)** — Non-Maximum Suppression discards valid cell detections in high-density smear regions where red blood cells routinely overlap.
3. **Black-Box Output (P3)** — Existing pipelines produce opaque class labels without spatial evidence, limiting clinical adoption.

**MalariAI** is a two-stage decoupled framework that addresses all three simultaneously:

- **Stage 1** — An *annotation-agnostic* distance-transform guided watershed algorithm that isolates every cell in the smear regardless of ground-truth completeness. Stage 1 v2 adds CLAHE contrast normalisation and resolution-aware peak detection for cross-dataset robustness.
- **Stage 2** — An **EfficientNet-B0** classifier trained with **Focal Loss** for multi-class infection stage identification (ring, trophozoite, schizont, gametocyte), with **Grad-CAM++** generating per-cell spatial attention heatmaps.

We evaluate on **NIH BBBC041** (1,208 training images, 79,672 annotated instances) and validate cross-dataset on **MP-IDB** (209 images, 1,407 infected cells from 4 *Plasmodium* species) without any retraining.

---

## Why This Approach?

| Problem | Prior Work | MalariAI |
|---|---|---|
| Missing annotations | Treats unannotated cells as background ❌ | Watershed finds *all* cells — label-agnostic ✅ |
| Dense overlapping cells | NMS deletes genuine overlapping detections ❌ | Distance-transform splits touching cells ✅ |
| Clinical explainability | Black-box prediction ❌ | Grad-CAM++ heatmap per cell ✅ |
| Multi-class imbalance | Ignored (537:1 RBC:gametocyte ratio) ❌ | Focal Loss + per-class inverse-frequency weights ✅ |
| Single-dataset evaluation | Trained and tested on same dataset ❌ | Cross-dataset validation on MP-IDB (unseen stain/lab) ✅ |

---

## Key Results

### BBBC041 — Source Domain (120 test images, 5,917 GT boxes)

| Method | Stage 1 Recall@IoU0.5 | Binary Parasitized AP@0.5 | mAP@0.5 |
|---|---|---|---|
| Baseline A (Faster R-CNN) | N/A | — | 58.99% |
| Pipeline B — Stage 1 v1 | 66.88% | 29.10% | 8.67% |
| Pipeline B — Stage 1 v2 | 41.61% | 7.40% | 0.78% |

> Stage 2 (EfficientNet-B0) crop classification accuracy: **98.36%** overall; schizont **87.5%**, gametocyte **75.0%** — vs Faster R-CNN AP of 24.6% and 26.0% for the same rare classes.

### MP-IDB — Cross-Dataset Zero-Shot (209 images, 1,407 infected cells)

| Method | Stage 1 Recall@IoU0.5 | Binary Parasitized AP@0.5 |
|---|---|---|
| Pipeline B — Stage 1 v1 | 1.28% | 1.82% |
| Pipeline B — Stage 1 v2 (CLAHE) | **20.68%** | **9.09%** |

Stage 1 v2 delivers a **16× recall improvement** on the unseen dataset. Per-species recall (v2): *P. malariae* 86.0%, *P. ovale* 60.6%, *P. vivax* 32.8%, *P. falciparum* 16.8%.

---

## Datasets

### NIH BBBC041 (Primary)
Giemsa-stained *P. falciparum* thin blood smears, 1600×1200 px.
Source: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC041)

| Split | Images | Valid Boxes |
|---|---|---|
| Training | 1,208 | 79,672 |
| Test | 120 | 5,917 |

Class distribution (training): Red Blood Cell 97.2%, Trophozoite 1.8%, Ring 0.4%, Schizont 0.2%, Gametocyte 0.2%, Leukocyte 0.1%.

### MP-IDB (Cross-Dataset Validation)
Giemsa-stained thin blood smears, 2592×1944 px. 4 *Plasmodium* species annotated with Supervisely bitmap masks.
Source: [MP-IDB dataset](https://github.com/lstorchi/pica_rbc)

| Species | Images | Infected Cells |
|---|---|---|
| *P. falciparum* | — | 1,267 (90.1%) |
| *P. vivax* | — | 64 (4.5%) |
| *P. ovale* | — | 33 (2.3%) |
| *P. malariae* | — | 43 (3.1%) |
| **Total** | **209** | **1,407** |

---

## Project Structure

```text
MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── malaria/                        ← BBBC041 images + JSON annotations
│   │   ├── training.json
│   │   ├── test.json
│   │   └── images/
│   ├── MP-IDB/                         ← MP-IDB images + Supervisely annotations
│   │   ├── img/
│   │   └── ann/
│   ├── processed/
│   │   ├── train_annotations.csv       ← BBBC041 train (prepare_data.py)
│   │   ├── test_annotations.csv        ← BBBC041 test  (prepare_data.py)
│   │   └── mpidb_annotations.csv       ← MP-IDB        (prepare_mpidb.py)
│   ├── prepare_data.py                 ← BBBC041 JSON → CSV converter
│   └── prepare_mpidb.py                ← MP-IDB Supervisely bitmap → CSV converter
│
├── shared/
│   └── label_map.py                    ← Class indices, colours, names
│
├── notebooks/
│   ├── Phase1_EDA.ipynb                ← ✅ BBBC041 exploratory data analysis
│   └── mpidb_eda.ipynb                 ← ✅ MP-IDB exploratory data analysis
│
├── Phase1-EDA/                         ← ✅ COMPLETE
├── Phase2-BaselineA/                   ← ✅ COMPLETE — Faster R-CNN baseline
│   ├── train_frcnn.py
│   ├── evaluate.py
│   └── checkpoints/best.pth
│
├── Phase3-PipelineB/                   ← ✅ COMPLETE — MalariAI pipeline
│   ├── stage1_watershed.py             ← Otsu → distance transform → watershed (v1)
│   ├── stage2_train.py                 ← EfficientNet-B0 + Focal Loss training
│   ├── stage2_inference.py             ← Crop classification + Grad-CAM++
│   ├── gradcam.py
│   └── checkpoints/best.pth           ← Trained Stage 2 checkpoint
│
├── src/
│   ├── pipeline_b/
│   │   ├── stage1_watershed.py         ← Stage 1 v1 (original)
│   │   └── stage2_classify.py
│   └── pipeline_b_v2/
│       ├── stage1_v2.py                ← ✅ Stage 1 v2: CLAHE + resolution-aware peak detection
│       └── e2e_eval.py                 ← ✅ End-to-end evaluation framework (IoU matching + AP)
│
├── results/
│   └── v2/
│       ├── e2e_bbbc041/                ← Stage 1 v1 results on BBBC041
│       │   ├── metrics.json
│       │   ├── stage1_stats.json
│       │   └── pr_curves.png
│       ├── e2e_bbbc041_v2/             ← Stage 1 v2 results on BBBC041
│       ├── e2e_mpidb/                  ← Stage 1 v1 results on MP-IDB
│       └── e2e_mpidb_v2/              ← Stage 1 v2 results on MP-IDB
│
├── Phase4-WebApp/                      ← ✅ COMPLETE — Flask UI
│   ├── app.py
│   ├── pipeline.py
│   └── README_deploy.md
│
└── paper_writing/
    ├── cmig_submission/
    │   ├── MalariAI_CMIG.tex           ← ✅ Full paper (1,827 lines, CMIG format)
    │   ├── references.bib
    │   ├── system_diagram_v1.png       ← ⚠ Being replaced (low resolution)
    │   └── fig_*.png                   ← All paper figures
    └── research_strategy_*.md
```

---

## Architecture

### Stage 1 — Annotation-Agnostic Cell Segmentation

```
Input image (any resolution)
        │
        ▼ [v2 only]
CLAHE contrast normalisation (L channel, LAB space)
        │
        ▼
Grayscale + Otsu thresholding  →  binary mask (cell=1, bg=0)
        │
        ▼
Morphological opening  →  noise removal
        │
        ▼
Distance transform  →  "height map" with peaks at cell centres
        │
        ▼ [v1: dist_norm >= 0.35]  [v2: peak_local_max, min_dist auto-scaled]
Seed generation (one seed per cell)
        │
        ▼
Watershed from seeds  →  individual cell regions
        │
        ▼
N bounding boxes (no GT labels required)
```

### Stage 2 — EfficientNet-B0 Crop Classifier

```
N cell crops (64×64 px each)
        │
        ▼
EfficientNet-B0 (ImageNet pretrained, 5.3M params)
        │
        ▼
Focal Loss head (γ=2.0, per-class α)
        │
        ├── Class label (RBC / Ring / Trophozoite / Schizont / Gametocyte / Leukocyte)
        ├── Confidence score
        └── Grad-CAM++ heatmap (spatial evidence of prediction)
```

---

## Reproducing Results

### 1. Setup

```bash
git clone https://github.com/Anaskaysar/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images.git
cd MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images
python -m venv malariaenv
malariaenv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install scipy scikit-image       # required for Stage 1 v2
```

### 2. Prepare annotation CSVs

```bash
python data/prepare_data.py          # BBBC041 → data/processed/test_annotations.csv
python data/prepare_mpidb.py         # MP-IDB  → data/processed/mpidb_annotations.csv
```

### 3. Run end-to-end evaluation — Stage 1 v1 (original)

```bash
# BBBC041
python src/pipeline_b_v2/e2e_eval.py \
    --dataset bbbc041 \
    --img-dir data/malaria/images \
    --ann-csv data/processed/test_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_bbbc041

# MP-IDB (cross-dataset zero-shot)
python src/pipeline_b_v2/e2e_eval.py \
    --dataset mpidb \
    --img-dir data/MP-IDB/img \
    --ann-csv data/processed/mpidb_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_mpidb
```

### 4. Run end-to-end evaluation — Stage 1 v2 (CLAHE + resolution-aware)

```bash
# BBBC041
python src/pipeline_b_v2/e2e_eval.py \
    --dataset bbbc041 --stage1-version v2 \
    --img-dir data/malaria/images \
    --ann-csv data/processed/test_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_bbbc041_v2

# MP-IDB
python src/pipeline_b_v2/e2e_eval.py \
    --dataset mpidb --stage1-version v2 \
    --img-dir data/MP-IDB/img \
    --ann-csv data/processed/mpidb_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_mpidb_v2
```

### 5. Run the web app

```bash
cd Phase4-WebApp
python app.py
# Open http://localhost:5000
```

---

## Research Contributions

**C1 — Label-Resilient Segmentation.** Annotation-agnostic Stage 1 detects every cell without ground-truth bounding boxes, addressing the incomplete-annotation failure mode (P1). Cell recovery rate: 75.95% (centroid-in-box), 66.88% (IoU≥0.5) on BBBC041.

**C2 — Density-Invariant Overlap Handling.** Distance-transform guided watershed separates touching cells at the instance level, recovering detections that NMS-based detectors suppress in dense smear regions (P2). 58% of BBBC041 training images contain overlapping cell pairs (IoU > 0.3).

**C3 — Integrated End-to-End Explainability.** Grad-CAM++ spatial heatmaps generated within the full detection-to-classification pipeline, with dual views (crop detail + full-image overlay) — absent from all prior whole-slide malaria detection systems (P3).

**C4 — Cross-Dataset Generalisation Study.** First evaluation of a malaria detection pipeline on MP-IDB without retraining. Stage 1 v2 (CLAHE) achieves 20.68% recall on unseen staining/institution, a 16× improvement over the unadapted baseline. Per-species analysis reveals morphology-driven recall patterns consistent with the parasitology literature.

---

## Paper Status

| Section | Status |
|---|---|
| Abstract | ✅ Complete |
| §1 Introduction | ✅ Complete |
| §2 Related Work (5 subsections) | ✅ Complete |
| §3 Methodology (Dataset, Stage 1, Stage 2, Baseline A, Metrics, MP-IDB) | ✅ Complete |
| §4 Experiments (Stage 1 eval, Stage 2 eval, E2E BBBC041, Cross-dataset, Ablations, Summary) | ✅ Complete |
| §5 Discussion + Limitations | ✅ Complete |
| §6 Conclusion | ✅ Complete |
| Bibliography | ✅ Complete |
| Appendix (iteration history) | ✅ Complete |
| System diagram | ⚠ Being replaced (high-res version in progress) |

Paper: `paper_writing/cmig_submission/MalariAI_CMIG.tex` (1,827 lines, Elsevier CMIG format)

---

## Citation

```bibtex
@article{apurba2025malariai,
  title   = {{MalariAI}: A Label-Resilient Decoupled Framework for Universal Cell
             Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears},
  author  = {Apurba, Kaysarul Anas and Hasan, Md Hasibul and Ali, Mohammed},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2025}
}
```

---

## Acknowledgements

The authors thank their former supervisor from North South University, Bangladesh, for guidance during the initial concept development (Junior Design 299, 2021).

The authors gratefully acknowledge Prof. Amr Abdel-Dayem (Laurentian University, Canada) for guidance during the Image Processing and Computer Vision course within the M.Sc. programme in Computational Sciences (Fall 2023).

---

## License

This repository is for academic research purposes. The NIH BBBC041 dataset is subject to its own licence — see [BBBC041](https://bbbc.broadinstitute.org/BBBC041). The MP-IDB dataset is subject to its own licence — see [MP-IDB](https://github.com/lstorchi/pica_rbc).
