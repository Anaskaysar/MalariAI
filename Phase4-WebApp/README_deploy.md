# Phase 4 — Flask Web App — HuggingFace Spaces Deployment

## What this app does

Upload a blood smear image → get back:
- Full smear with watershed-detected cell outlines (Card 1)
- Scrollable cell crop gallery, colour-coded by predicted class (Card 2)
- Clinical report: infection rate, per-class counts, Grad-CAM++ in crop view + full-image view (Card 3)

Pipeline B only (no Faster R-CNN at inference time). EfficientNet-B0 runs in ~2s on CPU.

---

## Local development

```bash
cd Phase4-WebApp
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## HuggingFace Spaces deployment

### One-time setup

1. Create a HuggingFace account at https://huggingface.co
2. Go to https://huggingface.co/new-space
3. Space name: `MalariAI`
4. SDK: **Docker** (gives us full Flask control)
5. Hardware: **CPU basic** (free tier — EfficientNet-B0 runs fine)

### What to push

The Space repo needs:
```
app.py
pipeline.py
templates/index.html
static/css/style.css
static/js/app.js
requirements.txt
Dockerfile
checkpoints/stage2_best.pth     ← upload after Phase 3 training
```

### Dockerfile (to be created in Phase 4)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

Note: HuggingFace Spaces expects port 7860.

### Model checkpoint

After Phase 3 training completes, upload `stage2_best.pth` either:
- Directly in the Space repo (if < 25 MB — EfficientNet-B0 is ~20 MB ✅)
- Or to HF Model Hub: `huggingface.co/[username]/malariai-efficientnet-b0`
  and load it with `hf_hub_download()` at startup

---

## requirements.txt (Phase 4 — lightweight, CPU only)

```
flask>=3.0
torch>=2.1 --index-url https://download.pytorch.org/whl/cpu
torchvision>=0.16 --index-url https://download.pytorch.org/whl/cpu
opencv-python-headless>=4.8
Pillow>=10.0
numpy>=1.24
scipy>=1.11
```

---

## Status

⏳ To be built after Phase 3 (Pipeline B) training is complete and checkpoints are saved.
