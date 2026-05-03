# MalariAI : Automated Malaria Cell Segmentation

## Overview
This repository contains a streamlined Python pipeline for detecting and classifying Malaria cells in blood smear images using PyTorch and Faster R-CNN. 

Initially built as a messy Jupyter Notebook pipeline utilizing deprecated Mask R-CNN frameworks, the project has been fully remodeled into a clean, object-oriented PyTorch codebase that scales easily and relies on standard modern computer vision practices.

## Project Structure
```text
MalariAI/
├── README.md              # Project Tracker and Guide
├── requirements.txt       # Environment dependencies
├── data/
│   └── prepare_data.py    # Standardizes original JSON annotations to CSV
├── src/
│   ├── models/
│   │   ├── dataset.py     # Custom PyTorch Dataset
│   │   └── faster_rcnn.py # torchvision Faster R-CNN instance 
│   ├── train.py           # Training Loop using SGD
│   └── inference.py       # Inference and Visualization module
├── notebooks/             # Archived EDA and experimental notebooks
└── archive/               # Legacy scripts that got refactored
```

## Quick Start
1. **Environment Setup**: 
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation**: Ensure `input/training.json` and `input/test.json` are present, then run:
   ```bash
   python data/prepare_data.py
   ```
3. **Training**:
   ```bash
   python src/train.py
   ```
4. **Inference**:
   ```bash
   python src/inference.py --image ImageMal.jpg --model models/faster_rcnn_malariai.pth
   ```

---

## 📈 Work Tracker
*(Always keep this section updated so future iterations can pick up instantly)*

### Completed Tasks
- [x] Analysed current mixed pipeline (TF, Keras, PyTorch, MRCNN) out of notebooks.
- [x] Restructured directories and moved deprecated/experimental scripts into `archive/` and `notebooks/`.
- [x] Wrote unified `prepare_data.py` to extract correct bounding boxes (min_r, max_c etc.) normalized across files.
- [x] Created scalable PyTorch Dataloader `MalariaDataset` that natively groups bounding boxes by class (`RBC`, `non_RBC`).
- [x] Added automated bounding box visualization and testing script `inference.py`.

### Next Steps / To-Do
- [ ] **Data Validation**: Run data loader over the full dataset and check that image dimensions and bounding box boundaries never throw OOB errors.
- [ ] **Hyperparameter Tuning**: Optimize Learning Rate & Step Schedulers (default currently SGD `lr=0.005`).
- [ ] **Model Evaluation**: Reconstruct map computation across the test holdout in `inference.py`.