# MalariAI

### A Label-Resilient Decoupled Framework for Annotation-Agnostic Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears

> **Status:** Under review at *Array* (Elsevier) — manuscript ARRAY-D-26-04638. Preprint: [arXiv:2607.00385](https://arxiv.org/abs/2607.00385).
> Kaysarul Anas Apurba¹ · Md Hasibul Hasan² · Mohammed Ali¹ · Tanzilur Rahman¹
> ¹ Dept. of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh
> ² Dept. of Computer Science and Engineering, IUBAT, Dhaka, Bangladesh

**Live Demo:** [huggingface.co/spaces/Kaysarulanas/MalariAI](https://huggingface.co/spaces/Kaysarulanas/MalariAI)
**Code:** [github.com/Anaskaysar/MalariAI](https://github.com/Anaskaysar/MalariAI)
**Preprint:** [arxiv.org/abs/2607.00385](https://arxiv.org/abs/2607.00385)

---

## Abstract

Automated malaria diagnosis from blood smear microscopy is a critical global health AI challenge; expert scarcity remains the primary diagnostic bottleneck. Existing deep learning systems face three compounding failures: end-to-end detectors treat unannotated cells as background, skewing recall by annotation completeness rather than true cell recovery; Non-Maximum Suppression suppresses valid detections in dense smears; and pipelines lack per-cell spatial evidence for clinical audit. We present MalariAI, a two-stage decoupled framework addressing all three. Stage 1 applies an annotation-agnostic watershed algorithm to isolate every cell in a full 1600×1200 image, recovering 75.95% of ground-truth cells without any ground-truth input. End-to-end, the pipeline reaches a binary parasitized AP@0.5 of 29.10% — the clinically relevant metric for flagging any infected cell — while the stricter multi-class mAP@0.5 of 8.67% mainly reflects watershed's organic region boundaries being penalized against axis-aligned ground-truth boxes, not a localisation failure. Stage 2 fine-tunes EfficientNet-B0 with Focal Loss on ground-truth crops, achieving 98.36% classification accuracy — an oracle upper bound once a cell is correctly localised — with 87.5% and 75.0% accuracy on the rare schizont and gametocyte stages, versus 38.45% and 57.27% AP for a modern YOLOv8s detector evaluated end-to-end on the same classes. Grad-CAM++ heatmaps generated per detected cell provide instance-level spatial evidence for clinical audit; a quantitative energy-in-box analysis confirms this activation is concentrated on the annotated cell body significantly above a geometric chance baseline (+0.0485, paired p = 1.4×10⁻³³), letting microscopists verify predictions at the individual parasite level without sacrificing classification performance.

---

## Why This Approach?

| Problem | Prior Work | MalariAI |
|---|---|---|
| Missing annotations | Treats unannotated cells as background | Watershed finds *all* cells — label-agnostic |
| Dense overlapping cells | NMS deletes genuine overlapping detections | Distance-transform splits touching cells |
| Clinical explainability | Black-box prediction, or qualitative heatmaps only | Grad-CAM++ heatmap per cell, quantitatively validated above chance (p = 1.4×10⁻³³) |
| Multi-class imbalance | Ignored (537:1 RBC:gametocyte ratio) | Focal Loss + per-class inverse-frequency weights |
| Single-dataset evaluation | Trained and tested on same dataset | Cross-dataset stress test on MP-IDB (unseen stain/lab) |
| Learned vs. classical segmentation | Assumed learned segmentation is strictly better | Directly benchmarked: a U-Net baseline under-segments dense clusters unless paired with the same watershed-style post-processing used in Stage 1 |

---

## Key Results

### BBBC041 — Source Domain (120 test images, 5,917 GT boxes)

| Method | Stage 1 Recall@IoU0.5 | Cell Recovery (centroid) | Binary Parasitized AP@0.5 | mAP@0.5 |
|---|---|---|---|---|
| Baseline A (Faster R-CNN) | N/A | — | — | 58.99% |
| Baseline B (YOLOv8s) | N/A | — | — | 71.24% |
| MalariAI Stage 1 (watershed) | 66.88% | **75.95%** | 29.10% | 8.67% |

> Stage 2 (EfficientNet-B0) crop classification accuracy: **98.36%** on a single split, **97.33% ± 0.49%** under 5-fold stratified cross-validation. Rare-stage accuracy: schizont **87.5%**, gametocyte **75.0%** — versus Baseline A's 24.57%/25.95% and Baseline B's 38.45%/57.27% AP on the same two classes end-to-end.

### Grad-CAM++ Quantitative Localization Validation

Energy-in-box ratio at the selected target layer: **0.8831** vs. a geometric chance baseline of **0.8346** (+0.0485). Paired t-test p = 1.40×10⁻³³, Wilcoxon p = 1.39×10⁻²⁸ (n = 326 correctly-classified crops); every parasite class individually significant (p < 0.003).

### Learned Segmentation Baseline — U-Net vs. Watershed (same 120-image test set)

| Method | Recall | Precision | F1 |
|---|---|---|---|
| Stage 1 watershed (this paper) | 66.88% | 51.36% | 0.581 |
| U-Net, naive threshold | 25.28% | 79.41% | 0.384 |
| U-Net + watershed-style post-processing | 62.89% | 83.28% | **0.717** |

U-Net alone under-segments dense clusters; watershed-style distance-transform splitting on its own output closes most of the recall gap to Stage 1 while substantially improving precision and F1 — motivating the paper's choice of watershed over a purely learned segmentation stage.

### MP-IDB — Cross-Dataset Domain-Shift Stress Test (209 images, 1,407 infected cells, zero-shot)

| Method | Stage 1 Recall@IoU0.5 | Binary Parasitized AP@0.5 |
|---|---|---|
| Stage 1 v1 | 1.28% | 1.82% |
| Stage 1 v2 (CLAHE contrast normalisation) | **20.68%** | **9.09%** |

Stage 1 v2 delivers a 16x recall / 5x AP improvement on the unseen dataset. This experiment is a domain-shift stress test — not a claim of universal deployment readiness — since MP-IDB's cells are ~11x smaller in relative frame than BBBC041's.

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

### MP-IDB (Cross-Dataset Stress Test)
Giemsa-stained thin blood smears, 2592x1944 px. 4 *Plasmodium* species.
Source: [MP-IDB dataset](http://www.neuroimaging.it/malaria_parasite_image_database) (Loddo et al., 2019)

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
MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── malaria/                        <- BBBC041 images + JSON annotations (images gitignored)
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
├── Phase2-BaselineA/                   <- Faster R-CNN baseline (Baseline A)
├── Phase3-PipelineB/                   <- MalariAI two-stage pipeline
│   ├── stage1_watershed.py             <- Stage 1 v1
│   ├── stage2_train.py                 <- EfficientNet-B0 + Focal Loss
│   ├── stage2_inference.py             <- Crop classification + Grad-CAM++
│   └── gradcam.py
│
├── Phase5-YOLO-Baseline/               <- YOLOv8s baseline (Baseline B), trained on the
│                                           identical image split as Baseline A
│
├── Phase6-Revision-Experiments/        <- Journal-revision experiments: 5-fold Stage 2
│                                           cross-validation, Grad-CAM++ layer ablation +
│                                           quantitative localization metric, U-Net
│                                           segmentation baseline vs. watershed
│
├── src/
│   └── pipeline_b_v2/
│       ├── stage1_v2.py                <- Stage 1 v2: CLAHE + resolution-aware
│       ├── stage1_v3.py                <- Stage 1 v3: experimental ablation
│       └── e2e_eval.py                 <- End-to-end evaluation framework
│
├── results/v2/                         <- Evaluation outputs (JSON + figures)
│
├── Phase4-WebApp/                      <- Local Flask dev copy of the web app
│
└── MalariAI/                           <- Deployed HuggingFace Space (separate git repo,
                                            not tracked in this repo's history — see
                                            huggingface.co/spaces/Kaysarulanas/MalariAI)
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

A learned alternative (U-Net) was benchmarked against this stage during journal revision (see `Phase6-Revision-Experiments/`); watershed remains the better choice unless paired with the same distance-transform splitting as post-processing.

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
        +-- Grad-CAM++ heatmap (quantitatively validated above chance, p = 1.4x10^-33)
```

---

## Reproducing Results

### Setup

```bash
git clone https://github.com/Anaskaysar/MalariAI.git
cd MalariAI-Automated-Malaria-Cell-Segmentation-from-Blood-Smear-Images
python -m venv malariaenv
malariaenv\Scripts\activate
pip install -r requirements.txt
```

### Prepare annotation CSVs

```bash
python data/prepare_data.py
python data/prepare_mpidb.py
```

### Run evaluation (Stage 1 + Stage 2 pipeline)

```bash
# Stage 1 v1 — BBBC041
python src/pipeline_b_v2/e2e_eval.py \
    --dataset bbbc041 \
    --img-dir data/malaria/images \
    --ann-csv data/processed/test_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_bbbc041

# Stage 1 v2 — MP-IDB (cross-dataset stress test)
python src/pipeline_b_v2/e2e_eval.py \
    --dataset mpidb --stage1-version v2 \
    --img-dir data/MP-IDB/img \
    --ann-csv data/processed/mpidb_annotations.csv \
    --checkpoint Phase3-PipelineB/checkpoints/best.pth \
    --out-dir results/v2/e2e_mpidb_v2
```

### Baseline B (YOLOv8s) and revision experiments

```bash
# YOLOv8s baseline — see Phase5-YOLO-Baseline/ for training/eval notebooks

# 5-fold CV, Grad-CAM++ quantitative metric, U-Net baseline
# — see Phase6-Revision-Experiments/MalariAI_Revision_Experiments_Kaggle.ipynb
#   and Phase6-Revision-Experiments/UNet_Sanity_Check.ipynb
```

### Run the web app locally

```bash
cd Phase4-WebApp
python app.py
# Open http://localhost:5000
```

---

## Research Contributions

**C1 — Label-Resilient Segmentation.** Annotation-agnostic Stage 1 recovers 75.95% of ground-truth cells on BBBC041 without any ground-truth bounding boxes.

**C2 — Density-Invariant Overlap Handling.** Distance-transform watershed separates touching cells at instance level, recovering detections NMS-based detectors suppress in dense regions.

**C3 — Quantitatively Validated Explainability.** Grad-CAM++ spatial heatmaps per detected cell, with a quantitative energy-in-box metric confirming activation is concentrated on the annotated cell body significantly above chance (p = 1.4×10⁻³³) — not just a qualitative visualization.

**C4 — Cross-Dataset Domain-Shift Stress Test.** Evaluation of the full pipeline on MP-IDB without retraining, explicitly framed as a stress test rather than a universality claim; Stage 1 v2 achieves a 16x recall improvement over the unadapted baseline on unseen staining.

**C5 — Rigor via Direct Baseline Comparison.** A modern single-stage detector (YOLOv8s) and a learned segmentation model (U-Net) are both benchmarked directly against MalariAI's design choices, rather than assumed inferior.

---

## Related Work Positioning

MalariAI is complementary to Angkoso et al. (2026), *"Research on different automatic segmentation methods for color cascading framework in detecting malaria infection,"* Array 29:100658 — a benchmark of five classical (non-deep-learning) segmentation methods for pixel-wise parasite classification on a private dataset. Their own stated future work identifies U-Net integration and stage classification as open problems; MalariAI addresses both directly, alongside detection, explainability, and cross-dataset generalization.

---

## Citing This Work

A preprint is available while this manuscript is under review at *Array*:

```
Kaysarul Anas Apurba, Md Hasibul Hasan, Mohammed Ali, Tanzilur Rahman.
MalariAI: A Label-Resilient Decoupled Framework for Annotation-Agnostic Cell
Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears.
arXiv:2607.00385, 2026.
```

---

## Acknowledgements

The authors gratefully acknowledge Prof. Amr Abdel-Dayem (Laurentian University, Canada) for guidance during the Image Processing and Computer Vision course within the M.Sc. programme in Computational Sciences (Fall 2023).

---

## License

This repository is for academic research purposes. The NIH BBBC041 dataset is subject to its own licence — see [BBBC041](https://bbbc.broadinstitute.org/BBBC041). The MP-IDB dataset is subject to its own licence — see [MP-IDB](http://www.neuroimaging.it/malaria_parasite_image_database).
