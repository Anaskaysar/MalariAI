# MalariAI — No Cell Left Behind

### A Label-Resilient Decoupled Framework for Universal Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears

> **Research Paper — Targeting CMIG (Computerized Medical Imaging and Graphics)**  
> Kaysarul Anas Apurba · Md Hasibul Hasan (Laurentian University, Canada) · Mohammed Ali (Melbourne Institute of Technology, Australia)

**Live Demo:** [huggingface.co/spaces/Kaysarulanas/MalariAI](https://huggingface.co/spaces/Kaysarulanas/MalariAI)  
**Code:** [github.com/Anaskaysar/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images](https://github.com/Anaskaysar/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images)

---

## Abstract

Automated malaria diagnosis from peripheral blood smear microscopy remains a critical open problem in global health AI. Three compounding failure modes persist across the literature:

1. **Incomplete Annotation (P1)** — End-to-end detectors (Faster R-CNN, YOLO) treat unannotated cells as background, suppressing true positives in sparsely labelled datasets.
2. **Dense Overlap / NMS Failure (P2)** — Non-Maximum Suppression discards valid cell detections in high-density smear regions where red blood cells routinely overlap.
3. **Black-Box Output (P3)** — Existing pipelines produce opaque class labels without spatial evidence, limiting clinical adoption.

**MalariAI** is a two-stage decoupled framework that addresses all three simultaneously:

- **Stage 1** — An *annotation-agnostic* distance-transform guided watershed algorithm that isolates every cell regardless of ground-truth completeness. Stage 1 v2 adds CLAHE contrast normalisation and resolution-aware peak detection for cross-dataset robustness.
- **Stage 2** — An **EfficientNet-B0** classifier trained with **Focal Loss** for multi-class infection stage identification (ring, trophozoite, schizont, gametocyte), with **Grad-CAM++** generating per-cell spatial attention heatmaps.

Evaluated on **NIH BBBC041** (1,208 training images, 79,672 annotated instances) and validated cross-dataset on **MP-IDB** (209 images, 1,407 infected cells from 4 *Plasmodium* species) without any retraining.

---

## Why This Approach?

| Problem | Prior Work | MalariAI |
|---|---|---|
| Missing annotations | Treats unannotated cells as background | Watershed finds *all* cells — label-agnostic |
| Dense overlapping cells | NMS deletes genuine overlapping detections | Distance-transform splits touching cells |
| Clinical explainability | Black-box prediction | Grad-CAM++ heatmap per cell |
| Multi-class imbalance | Ignored (537:1 RBC:gametocyte ratio) | Focal Loss + per-class inverse-frequency weights |
| Single-dataset evaluation | Trained and tested on same dataset | Cross-dataset validation on MP-IDB (unseen stain/lab) |

---

## Key Results

### BBBC041 — Source Domain (120 test images, 5,917 GT boxes)

| Method | Stage 1 Recall@IoU0.5 | Stage 1 Centroid-in-Box | Binary Parasitized AP@0.5 | mAP@0.5 |
|---|---|---|---|---|
| Baseline A (Faster R-CNN) | N/A | — | — | 58.99% |
| Pipeline B — Stage 1 v1 | 66.88% | 79.04% | 29.10% | 8.67% |
| Pipeline B — Stage 1 v2 | 41.61% | — | 7.40% | 0.78% |

> Stage 2 (EfficientNet-B0) crop classification accuracy: **98.36%** overall; schizont **87.5%**, gametocyte **75.0%**.

### Alternative Localisation Metrics — Stage 1 v1 on BBBC041

| Metric | Value | Description |
|---|---|---|
| Centroid-in-box (M1) | **79.04%** | GT centroid falls inside a WS box — boundary-free |
| Recall @IoU0.25 (M2) | **78.08%** | Relaxed boundary criterion |
| Infected sensitivity @IoU0.25 (M3) | **82.84%** | Parasitised GT cells only (n=303) |
| Bio-localisation recall (M4) | **79.50%** | WS centroid within 0.5x GT diagonal |

### MP-IDB — Cross-Dataset Zero-Shot (209 images, 1,407 infected cells)

| Method | Stage 1 Recall@IoU0.5 | Binary Parasitized AP@0.5 |
|---|---|---|
| Pipeline B — Stage 1 v1 | 1.28% | 1.82% |
| Pipeline B — Stage 1 v2 (CLAHE) | **20.68%** | **9.09%** |

Stage 1 v2 delivers a **16x recall improvement** on the unseen dataset.

---

## Live Demo

The HuggingFace Space runs the full two-stage pipeline (Flask + Docker):

**[huggingface.co/spaces/Kaysarulanas/MalariAI](https://huggingface.co/spaces/Kaysarulanas/MalariAI)**

Upload any Giemsa-stained thin blood smear image (PNG/JPG/TIF) and receive:
- Annotated smear with per-cell bounding boxes coloured by predicted class
- Clinical summary: total cells, infected count, infection rate, dominant stage
- Infected cell crop gallery with EfficientNet-B0 classification labels
- Grad-CAM++ heatmaps per infected cell crop
- Full-image Grad-CAM++ overlay showing spatial distribution of infection

---

## Datasets

### NIH BBBC041 (Primary)
Giemsa-stained *P. falciparum* thin blood smears, 1600x1200 px.  
Source: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC041)

| Split | Images | Valid Boxes |
|---|---|---|
| Training | 1,208 | 79,672 |
| Test | 120 | 5,917 |

### MP-IDB (Cross-Dataset Validation)
Giemsa-stained thin blood smears, 2592x1944 px. 4 *Plasmodium* species.  
Source: [MP-IDB dataset](https://github.com/lstorchi/pica_rbc)

| Species | Infected Cells |
|---|---|
| *P. falciparum* | 1,267 (90.1%) |
| *P. vivax* | 64 (4.5%) |
| *P. ovale* | 33 (2.3%) |
| *P. malariae* | 43 (3.1%) |
| **Total** | **1,407** |

---

## Project Structure

```
MalariAI/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── malaria/                        <- BBBC041 images + JSON annotations
│   ├── MP-IDB/ann/                     <- MP-IDB Supervisely annotations
│   ├── processed/                      <- Derived annotation CSVs
│   ├── prepare_data.py                 <- BBBC041 JSON -> CSV converter
│   └── prepare_mpidb.py                <- MP-IDB bitmap mask -> CSV converter
│
├── shared/label_map.py                 <- Class indices, colours, names
│
├── notebooks/
│   ├── Phase1_EDA.ipynb                <- BBBC041 exploratory data analysis
│   └── mpidb_eda.ipynb                 <- MP-IDB exploratory data analysis
│
├── Phase2-BaselineA/                   <- Faster R-CNN baseline
├── Phase3-PipelineB/                   <- MalariAI two-stage pipeline
│   ├── stage1_watershed.py             <- Stage 1 v1
│   ├── stage2_train.py                 <- EfficientNet-B0 + Focal Loss
│   ├── stage2_inference.py             <- Crop classification + Grad-CAM++
│   └── gradcam.py
│
├── src/
│   └── pipeline_b_v2/
│       ├── stage1_v2.py                <- Stage 1 v2: CLAHE + resolution-aware
│       ├── stage1_v3.py                <- Stage 1 v3: experimental ablation
│       └── e2e_eval.py                 <- End-to-end evaluation framework
│
├── results/v2/                         <- Evaluation outputs (JSON + figures)
│
├── huggingface_space/                  <- HuggingFace Docker Space (Flask)
│   ├── Dockerfile
│   ├── app.py
│   ├── pipeline.py
│   ├── templates/index.html
│   └── static/
│
└── paper_writing/
    ├── cmig_submission/
    │   ├── MalariAI_CMIG.tex           <- Full paper (Elsevier CMIG format)
    │   └── references.bib
    └── option_b_v3_ablation/           <- v3 ablation held in reserve
```

---

## Architecture

### Stage 1 — Annotation-Agnostic Cell Segmentation

```
Input image (any resolution)
        |
        v [v2 only]
CLAHE contrast normalisation (L channel, LAB colour space)
        |
        v
Grayscale + Otsu thresholding  ->  binary mask (cell=1, bg=0)
        |
        v
Morphological opening  ->  noise removal
        |
        v
Distance transform  ->  height map with peaks at cell centres
        |
        v  [v1: dist_norm >= 0.35]  [v2: peak_local_max, auto-scaled min_dist]
Seed generation (one seed per cell)
        |
        v
Watershed from seeds  ->  individual cell regions
        |
        v
N bounding boxes  (no GT labels required)
```

### Stage 2 — EfficientNet-B0 Crop Classifier

```
N cell crops (64x64 px each)
        |
        v
EfficientNet-B0 (ImageNet pretrained, 5.3M params)
        |
        v
Focal Loss head (gamma=2.0, per-class alpha)
        |
        +-- Class label (RBC / Ring / Trophozoite / Schizont / Gametocyte / Leukocyte)
        +-- Confidence score
        +-- Grad-CAM++ heatmap
```

---

## Reproducing Results

### Setup

```bash
git clone https://github.com/Anaskaysar/MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images.git
cd MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images
python -m venv malariaenv
malariaenv\Scripts\activate
pip install -r requirements.txt
pip install scipy scikit-image
```

### Prepare annotation CSVs

```bash
python data/prepare_data.py
python data/prepare_mpidb.py
```

### Run evaluation

```bash
# Stage 1 v1 — BBBC041
python src/pipeline_b_v2/e2e_eval.py \
    --dataset bbbc041 \
    --img-dir data/malaria/images \
    --ann-csv data/processed/test_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_bbbc041

# Stage 1 v2 — BBBC041
python src/pipeline_b_v2/e2e_eval.py \
    --dataset bbbc041 --stage1-version v2 \
    --img-dir data/malaria/images \
    --ann-csv data/processed/test_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_bbbc041_v2

# Stage 1 v2 — MP-IDB (cross-dataset zero-shot)
python src/pipeline_b_v2/e2e_eval.py \
    --dataset mpidb --stage1-version v2 \
    --img-dir data/MP-IDB/img \
    --ann-csv data/processed/mpidb_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_mpidb_v2
```

### Run the web app locally

```bash
cd Phase4-WebApp
python app.py
# Open http://localhost:5000
```

---

## Research Contributions

**C1 — Label-Resilient Segmentation.** Annotation-agnostic Stage 1 detects every cell without ground-truth bounding boxes. Centroid-in-box recall: 79.04% on BBBC041.

**C2 — Density-Invariant Overlap Handling.** Distance-transform watershed separates touching cells at instance level, recovering detections NMS-based detectors suppress in dense regions.

**C3 — Integrated Explainability.** Grad-CAM++ spatial heatmaps within the full pipeline, with dual views (crop detail + full-image overlay).

**C4 — Cross-Dataset Generalisation.** First evaluation of a malaria pipeline on MP-IDB without retraining. Stage 1 v2 achieves 20.68% recall on unseen staining — a 16x improvement over the unadapted baseline.

---

## Acknowledgements

The authors thank their former supervisor from North South University, Bangladesh, for guidance during the initial concept development.

The authors gratefully acknowledge Prof. Amr Abdel-Dayem (Laurentian University, Canada) for guidance during the Image Processing and Computer Vision course within the M.Sc. programme in Computational Sciences (Fall 2023).

---

## License

This repository is for academic research purposes. The NIH BBBC041 dataset is subject to its own licence — see [BBBC041](https://bbbc.broadinstitute.org/BBBC041). The MP-IDB dataset is subject to its own licence — see [MP-IDB](https://github.com/lstorchi/pica_rbc).
