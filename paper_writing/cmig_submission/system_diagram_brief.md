# MalariAI System Diagram — Design Brief for Collaborator

**For:** System diagram figure in the CMIG journal paper  
**Replacing:** `system_diagram_v1.png` (current version is too low-resolution for print)  
**Deliver as:** `system_diagram_v2.pdf` (preferred) OR `system_diagram_v2.png` at minimum 300 DPI  
**Target width in paper:** Full text width (~170mm / 6.7 inches in a two-column Elsevier journal)

---

## What the Diagram Must Show

The diagram illustrates the full MalariAI two-stage pipeline — from a raw blood smear image as input, through two processing stages, to a clinical output. There are **5 boxes connected left-to-right with arrows**.

```
[INPUT] → [STAGE 1: Watershed] → [STAGE 2: EfficientNet-B0] → [Grad-CAM++ Explainability] → [OUTPUT]
```

---

## Box-by-Box Content

### Box 1 — INPUT
**Label:** Input  
**Visual:** A representative blood smear microscopy image (circular cells on pale background). You can use the placeholder image from the existing diagram, or any freely available Giemsa-stained blood smear (search "BBBC041 malaria blood smear" on Google Images — these are open-access).  
**Caption below box:** "Giemsa-stained thin blood smear (1600×1200 px)"

---

### Box 2 — STAGE 1: Annotation-Agnostic Cell Segmentation
**Label:** Stage 1: Watershed Segmentation  
**Visual:** The same blood smear image but with coloured outlines drawn around each cell (watershed result). Green outlines for normal cells, red/orange outlines for detected parasite regions. This is exactly what the existing diagram shows — just make it cleaner.  
**Text inside box (bullet points, small font):**
- CLAHE contrast normalisation
- Otsu thresholding
- Distance transform
- Marker-based watershed
- → N cell bounding boxes

**Key claim (below or beside):** "No ground-truth labels required"

---

### Box 3 — STAGE 2: EfficientNet-B0 Classification
**Label:** Stage 2: Feature Extraction & Classification  
**Visual:** A small grid of individual cropped cell images (64×64 px each), each with a coloured label badge. Show 6–8 cells. Labels to show: "RBC" (red), "Ring" (yellow), "Trophozoite" (orange), "Schizont" (purple), "Gametocyte" (green).  
**Text inside box:**
- EfficientNet-B0 (5.3M params)
- Focal Loss (γ=2.0)
- 6-class head
- Per-class confidence score

---

### Box 4 — EXPLAINABILITY
**Label:** Explainability (Grad-CAM++)  
**Visual:** Two heatmap images side by side:
  - Left: A single 64×64 cell crop with a red/yellow Grad-CAM++ heatmap overlay (the model is "looking at" the parasite inside the cell)
  - Right: The same heatmap projected back onto the full blood smear image (showing where parasites are spatially located in the whole slide)  
**Text inside box:**
- Grad-CAM++ spatial heatmap
- Per-cell attention map
- Full-image overlay

---

### Box 5 — OUTPUT
**Label:** Output  
**Visual:** A small mock "clinical report" card with:
  - A pie chart or bar showing: RBC 95%, Ring 2%, Trophozoite 2%, etc.
  - A small table with: Infected cells detected: 12, Infection rate: 4.8%
  - A small legend with colour dots for each class

---

## Connecting Arrows

Between each box, draw a right-pointing arrow. Label the arrows with what is being passed:

| Arrow | Label |
|---|---|
| Box 1 → Box 2 | Full smear image |
| Box 2 → Box 3 | N individual cell crops |
| Box 3 → Box 4 | Class scores + feature maps |
| Box 4 → Box 5 | Labels + heatmaps + counts |

---

## Colour Scheme

Use a clean, publication-appropriate palette. Suggested:

| Element | Colour |
|---|---|
| Box borders / headers | Deep navy blue (`#1B3A6B`) or dark teal (`#1A5276`) |
| Box fill | Very light blue-grey (`#EBF5FB`) or white |
| Arrow colour | Dark grey (`#444444`) |
| RBC label | Coral red (`#E74C3C`) |
| Ring label | Gold (`#F1C40F`) |
| Trophozoite label | Orange (`#E67E22`) |
| Schizont label | Purple (`#8E44AD`) |
| Gametocyte label | Green (`#27AE60`) |
| Heatmap gradient | Blue → Green → Yellow → Red (standard Jet colormap) |

Do **not** use bright neon colours, gradients on box backgrounds, or drop shadows. The journal is printed in black-and-white in some formats — check the figure reads clearly in greyscale too.

---

## Technical Specifications (Critical for Publication)

| Specification | Requirement |
|---|---|
| **Resolution** | Minimum 300 DPI for raster. 600 DPI preferred. |
| **Format** | PDF (vector, no pixelation at any zoom) is strongly preferred. PNG at 300+ DPI is acceptable. Do NOT deliver JPEG (lossy compression creates artefacts). |
| **Width** | Design at 170mm wide (full Elsevier text width). This is ~6.7 inches or 2008px at 300 DPI. |
| **Height** | Aim for ~70–90mm tall (roughly half the width). The diagram must be landscape, not square. |
| **Font** | Use a clean sans-serif font: Arial, Helvetica, or Inter. Font size inside boxes: minimum 8pt at final print size. Title/label font: 10pt bold. |
| **File name** | `system_diagram_v2.pdf` or `system_diagram_v2.png` |

---

## Tools You Can Use

Any of these work well:

- **Figma** (free, browser-based) — easiest for clean vector diagrams, export as PDF
- **Adobe Illustrator** — best quality, export as PDF
- **Inkscape** (free, desktop) — open-source vector, export as PDF or high-res PNG
- **PowerPoint / Keynote** — acceptable if you export as PDF (not screenshot)
- **draw.io / diagrams.net** (free, browser-based) — good for flowcharts, export as PDF

Avoid: Microsoft Paint, screenshots of Jupyter notebooks, low-res Google Slides exports.

---

## What the Current Diagram Gets Right (Keep These)

Looking at `system_diagram_v1.png`:
- The left-to-right flow with 5 boxes is correct
- The use of actual microscopy images as visuals (not abstract shapes) is good
- Showing the watershed-outlined smear in Box 2 is correct
- The heatmap visuals in Box 4 are correct

**The only problem is resolution** — the PNG is too small (272KB) and pixelates when printed at 170mm width. Everything else about the layout is right. Feel free to use the existing diagram as a direct template — just recreate it at publication quality.

---

## Reference Figures in the Paper

The diagram is referred to in the paper as:

> "Figure 1: MalariAI system architecture. The full pipeline processes a raw Giemsa-stained blood smear through two decoupled stages: Stage 1 (annotation-agnostic watershed segmentation) and Stage 2 (EfficientNet-B0 crop classification with Grad-CAM++ explainability)."

The LaTeX include line is:
```latex
\includegraphics[width=\linewidth]{system_diagram_v2}
```

So the file must be named `system_diagram_v2.pdf` (or `.png`) and placed in the same folder as `MalariAI_CMIG.tex`.

---

## Questions?

Contact: Kaysarul Anas Apurba  
The existing low-res version is at: `paper_writing/cmig_submission/system_diagram_v1.png`  
Use it as a layout reference.
