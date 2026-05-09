# MalariAI — A Label-Resilient Decoupled Framework for Universal Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears

> **Research Paper + Portfolio Demo**  
> Kaysarul Anas Apurba (Laurentian University, Canada) · Mohammed Ali (Melbourne Institute of Technology, Australia)  
> Target venue: arXiv / MICCAI · Live demo: Hugging Face Spaces

---

## Abstract

Automated malaria diagnosis from peripheral blood smear microscopy remains a critical open problem in global health AI. Three compounding failure modes persist across the literature:

1. **Incomplete Annotation (P1)** — End-to-end detectors (Faster R-CNN, YOLO) treat unannotated cells as background, suppressing true positives in sparsely labelled datasets.
2. **Dense Overlap / NMS Failure (P2)** — Non-Maximum Suppression discards valid cell detections in high-density smear regions where red blood cells routinely overlap.
3. **Black-Box Output (P3)** — Existing pipelines produce opaque class labels without spatial evidence, limiting clinical adoption.

**MalariAI** is a two-stage decoupled framework that addresses all three simultaneously:

- **Stage 1** — An *annotation-agnostic* distance-transform guided watershed algorithm that isolates every cell in the smear regardless of ground-truth completeness.
- **Stage 2** — An **EfficientNet-B0** classifier trained with **Focal Loss** for multi-class infection stage identification (ring, trophozoite, schizont, gametocyte), with **Grad-CAM++** generating per-cell spatial attention heatmaps.

We benchmark against a **Faster R-CNN** baseline on the publicly available **NIH BBBC041** dataset (1,208 training images, 79,672 valid annotated instances across 6 cell categories).

---

## Why This Approach?

| Problem | Prior Work | MalariAI |
|---|---|---|
| Missing annotations | Treats unannotated cells as background ❌ | Watershed finds *all* cells — label-agnostic ✅ |
| Dense overlapping cells | NMS deletes genuine overlapping detections ❌ | Distance-transform splits touching cells ✅ |
| Clinical explainability | Black-box prediction ❌ | Grad-CAM++ heatmap per cell ✅ |
| Multi-class imbalance | Ignored (537:1 RBC:gametocyte ratio) ❌ | Focal Loss + per-class inverse-frequency weights ✅ |

---

## Dataset

**NIH BBBC041** — Giemsa-stained *P. falciparum* thin blood smears, 1600×1200 px  
Source: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC041)

| Split | Images | Valid Boxes |
|---|---|---|
| Training | 1,208 | 79,672 |
| Test | 120 | 5,917 |

**Class distribution (training, excluding "difficult"):**

| Class | Count | % |
|---|---|---|
| Red Blood Cell | 77,420 | 97.2% |
| Trophozoite | 1,473 | 1.8% |
| Ring | 353 | 0.4% |
| Schizont | 179 | 0.2% |
| Gametocyte | 144 | 0.2% |
| Leukocyte | 103 | 0.1% |

**EDA key findings:**
- 58% of training images contain at least one overlapping cell pair (IoU > 0.3) — empirical motivation for P2
- Median 59 boxes/image, max 223 — "dense smear" defined as > 100 boxes (11% of images)
- Trophozoite median area 17,272 px² vs. RBC 11,544 px² — intraerythrocytic swelling confirmed

> 📊 Full EDA: [`notebooks/Phase1_EDA.ipynb`](notebooks/Phase1_EDA.ipynb)

---

## Project Structure

```text
MalariAI/
├── README.md                        ← Master plan (this file)
├── requirements.txt                 ← Training dependencies
│
├── data/
│   ├── .gitignore
│   └── malaria/                     ← Place dataset here (not committed)
│       ├── training.json
│       ├── test.json
│       └── images/
│
├── shared/
│   └── label_map.py                 ← Single source of truth: class indices, colours, names
│
├── Phase1-EDA/                      ← ✅ COMPLETE
│   ├── dataset.py                   ← MalariaDataset + MalariaCropDataset (shared across phases)
│   ├── eda.py                       ← Standalone EDA script
│   └── outputs/                     ← EDA plots
│
├── notebooks/
│   ├── Phase1_EDA.ipynb             ← ✅ Interactive EDA notebook (primary interface)
│   └── outputs/                     ← class_distribution.png, parasite_crops.png, etc.
│
├── Phase2-BaselineA/                ← 🔧 IN PROGRESS
│   ├── train_frcnn.py               ← Faster R-CNN ResNet-50 FPN training script
│   ├── evaluate.py                  ← mAP@0.5, mAP@[0.5:0.95], per-class AP
│   └── checkpoints/                 ← Saved model weights (not committed)
│
├── Phase3-PipelineB/                ← ⏳ PENDING
│   ├── stage1_watershed.py          ← Otsu → morphological opening → distance transform → watershed
│   ├── stage2_train.py              ← EfficientNet-B0 + Focal Loss training
│   ├── stage2_inference.py          ← Crop classification + Grad-CAM++ generation
│   ├── gradcam.py                   ← Grad-CAM++ implementation
│   └── checkpoints/                 ← Saved model weights (not committed)
│
├── Phase4-WebApp/                   ← ⏳ PENDING — Flask UI + HuggingFace deployment
│   ├── app.py                       ← Flask server (upload → pipeline → results)
│   ├── pipeline.py                  ← Loads Stage1 + Stage2 models, runs full inference
│   ├── templates/
│   │   └── index.html               ← 3-card UI (smear view / cell gallery / report + Grad-CAM++)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   ├── requirements.txt             ← Lightweight: flask, torch (CPU), torchvision, opencv, pillow
│   └── README_deploy.md             ← HuggingFace Spaces deployment steps
│
├── paper_writing/
│   ├── draft/
│   │   ├── MalariAI_Paper_Draft.tex ← LaTeX paper draft (main file)
│   │   ├── fig_class_distribution.png
│   │   ├── fig_parasite_crops.png
│   │   └── fig_density.png
│   ├── previous_study/              ← Reference PDFs (Papers 1–4)
│   └── coauthor_reading_guide.md    ← Mohammed Ali's extraction tables + timeline
│
└── checkpoints/                     ← Top-level symlinks to best models for web app
    ├── stage2_best.pth              ← → Phase3-PipelineB/checkpoints/best.pth
    └── README.md
```

---

## Architecture

### Pipeline B — MalariAI (Proposed)

```
Full Blood Smear Image (1600×1200 px, Giemsa-stained)
          │
          ▼
┌─────────────────────────────────────────┐
│  STAGE 1: ANNOTATION-AGNOSTIC WATERSHED │
│  • Otsu thresholding (global binarise)  │
│  • Morphological opening (noise remove) │
│  • Distance transform (topology map)    │
│  • Marker-based watershed (separation)  │
│  → N cell crops (no GT labels required) │
└─────────────────┬───────────────────────┘
                  │  N individual 64×64 crops
                  ▼
┌─────────────────────────────────────────┐
│  STAGE 2: EFFICIENTNET-B0 CLASSIFIER    │
│  • Focal Loss γ=2.0, α=inv-frequency    │
│  • 7-class head (incl. background)      │
│  • Grad-CAM++ heatmap per cell          │
│  → Class label + confidence + heatmap  │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  PHASE 4: FLASK WEB APP (HF Spaces)     │
│  Card 1 — Smear + watershed outlines    │
│  Card 2 — Cell crop gallery (filtered)  │
│  Card 3 — Report + dual Grad-CAM++ view │
│            (crop detail / full image)   │
└─────────────────────────────────────────┘
```

### Pipeline A — Faster R-CNN Baseline

```
Full Blood Smear Image
          │
          ▼
ResNet-50-FPN Backbone → RPN → RoI Align → Classification Head → NMS
          │
          ▼
Bounding boxes + class labels
```

---

## Phase Roadmap

| Phase | Name | Owner | Status | Key Outputs |
|---|---|---|---|---|
| **Phase 1** | EDA & Dataset Analysis | KB | ✅ Complete | `Phase1_EDA.ipynb`, dataset.py, EDA plots |
| **Phase 2** | Baseline A — Faster R-CNN | KB | 🔧 Next | `train_frcnn.py`, mAP numbers, checkpoint |
| **Phase 3** | Pipeline B — Watershed + EfficientNet | KB | ⏳ Pending | Stage1+2 code, Grad-CAM++ output, checkpoint |
| **Phase 4** | Flask UI + HuggingFace Deployment | KB | ⏳ Pending | Live demo at `hf.co/spaces/…`, PDF report export |
| **Phase 5** | Evaluation, Ablation & Paper | KB + Mohammed Ali | ⏳ Pending | mAP tables, ablation study, final `.tex` → arXiv |

### Phase 4 Detail — Flask Web App

The web app runs **Pipeline B only** (watershed → EfficientNet-B0 → Grad-CAM++) — no GPU required. EfficientNet-B0 (5.3M params) runs in ~2s on CPU, making it viable on Hugging Face Spaces free tier.

**What the demo does:**
1. User uploads a blood smear PNG/JPG
2. Stage 1 (watershed) isolates every cell — outlines overlaid on smear image (Card 1)
3. Every crop shown in a filterable gallery with class label + confidence (Card 2)
4. Summary report: infection rate, dominant stage, per-class counts (Card 3)
5. Grad-CAM++ shown in two modes: 64×64 crop detail view + full-image activation overlay

**HuggingFace Spaces deployment:**
- Platform: [Hugging Face Spaces](https://huggingface.co/spaces) — free, no GPU needed
- SDK: Flask (via Docker space type)
- Model checkpoint uploaded to HF model hub or stored in repo (≤25MB for EfficientNet-B0)
- URL will be `huggingface.co/spaces/[username]/MalariAI`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Watershed over Mask R-CNN for Stage 1** | Zero labelled data required; no NMS; splits touching cells by topology |
| **EfficientNet-B0 for Stage 2** | 5.3M params — runs on CPU; outperforms ResNet-18 on small images; strong ImageNet transfer |
| **Focal Loss (γ=2.0)** | Addresses 537:1 class imbalance without oversampling; per-class α from EDA counts |
| **Grad-CAM++** | More accurate multi-activation localisation than standard Grad-CAM; clinically interpretable |
| **Decoupled stages** | Stage 1 is label-agnostic; Stage 2 sees pre-isolated crops → avoids NMS artefacts entirely |
| **Flask + HuggingFace** | CPU-viable demo, free hosting, strong portfolio signal, sharable URL |
| **Phase-based folder structure** | Clean separation of EDA / training / inference / app — each phase is self-contained |

---

## Research Contributions

**C1 — Label-Resilient Segmentation.** An annotation-agnostic Stage 1 that detects every cell in a blood smear without relying on ground-truth bounding boxes, directly addressing the incomplete-annotation failure mode (P1) in NIH BBBC041.

**C2 — Density-Invariant Overlap Handling.** Distance-transform guided watershed separates touching and overlapping cells at the instance level, recovering detections that NMS-based detectors systematically suppress in dense smear regions (P2). Empirically: 58% of training images contain overlapping cell pairs (IoU > 0.3).

**C3 — Integrated End-to-End Explainability.** Grad-CAM++ spatial attention heatmaps generated within the full detection-to-classification pipeline, with dual views (crop detail + full-image overlay) — a capability absent from all prior whole-slide malaria detection systems (P3).

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

### 3. Run Phase 1 EDA

```bash
jupyter notebook notebooks/Phase1_EDA.ipynb
```

### 4. Train — Phase 2 (Faster R-CNN baseline)

```bash
python Phase2-BaselineA/train_frcnn.py \
    --json data/malaria/training.json \
    --img-dir data/malaria/images \
    --epochs 20 \
    --out-dir Phase2-BaselineA/checkpoints
```

### 5. Train — Phase 3 (Pipeline B)

```bash
# Stage 1: run watershed over all training images, save crops
python Phase3-PipelineB/stage1_watershed.py \
    --json data/malaria/training.json \
    --img-dir data/malaria/images \
    --out-dir data/crops

# Stage 2: train EfficientNet-B0 on saved crops
python Phase3-PipelineB/stage2_train.py \
    --train-json data/malaria/training.json \
    --img-dir data/malaria/images \
    --epochs 30 \
    --out-dir Phase3-PipelineB/checkpoints
```

### 6. Run the web app locally (Phase 4)

```bash
cd Phase4-WebApp
pip install -r requirements.txt
python app.py
# → Open http://localhost:5000
```

---

## Paper Status

| Section | Status | Owner |
|---|---|---|
| Abstract | ✅ Written | KB |
| §1 Introduction | 🔲 Placeholder | Mohammed Ali (after exps) |
| §2 Related Work | ✅ Written (5 subsections) | KB + Mohammed Ali |
| §3.1 Dataset & Setup | ✅ Written | KB |
| §3.2 Stage 1 Watershed | 🔲 Placeholder | KB (after Phase 3) |
| §3.3 Stage 2 EfficientNet | 🔲 Placeholder | KB (after Phase 3) |
| §3.4 Grad-CAM++ | 🔲 Placeholder | KB (after Phase 3) |
| §3.5 Baseline A | 🔲 Placeholder | KB (after Phase 2) |
| §3.6 Evaluation Metrics | 🔲 Placeholder | KB |
| §4 Experiments | 🔲 Placeholder | KB (after training) |
| §5 Results & Discussion | 🔲 Placeholder | KB + Mohammed Ali |
| §6 Conclusion | 🔲 Placeholder | Mohammed Ali |
| Bibliography | ⚠ Partial (some placeholders) | Mohammed Ali |

Paper draft: [`paper_writing/draft/MalariAI_Paper_Draft.tex`](paper_writing/draft/MalariAI_Paper_Draft.tex)  
Co-author guide: [`paper_writing/coauthor_reading_guide.md`](paper_writing/coauthor_reading_guide.md)

---

## Citation

```bibtex
@article{apurba2025malariai,
  title   = {MalariAI: A Label-Resilient Decoupled Framework for Universal Cell
             Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears},
  author  = {Apurba, Kaysarul Anas and Ali, Mohammed},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## Acknowledgements

The authors thank their former supervisor from North South University, Dhaka, Bangladesh (currently Assistant Professor at KFUPM), for guidance during the Junior Design 299 course (2021) where the initial concept was developed.

The authors also gratefully acknowledge Prof. Amr Abdel-Dayem (Laurentian University, Canada) for guidance during the Image Processing and Computer Vision course within the M.Sc. programme in Computational Sciences, where this project was significantly extended in Fall 2023.

---

## License

This repository is for academic research purposes. The NIH BBBC041 dataset is subject to its own licence — see [BBBC041](https://bbbc.broadinstitute.org/BBBC041).
