"""
Phase3-PipelineB/gradcam.py
============================
Grad-CAM++ implementation for EfficientNet-B0.

Grad-CAM++ improves on standard Grad-CAM by using second-order gradients to
weight each spatial activation more accurately when multiple objects of the
same class contribute to a single prediction. This is important for small
64×64 cell crops where the target structure occupies most of the receptive
field and single-activation bias (the weakness of Grad-CAM) is common.

Reference
---------
  Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual
  Explanations for Deep Convolutional Networks", WACV 2018.

Hook target
-----------
  EfficientNet-B0:  model.features[-1]  (MBConv block 8, output 1280 channels)
  Specifically: the last Conv2d inside model.features[-1][0]
  Feature map shape at this layer: [B, 1280, 2, 2] for 64×64 input.

Usage (standalone)
------------------
  from Phase3-PipelineB.gradcam import GradCAMPlusPlus
  from torchvision.models import efficientnet_b0

  model = efficientnet_b0(weights=None)
  model.load_state_dict(torch.load("Phase3-PipelineB/checkpoints/best.pth"))
  model.eval()

  cam = GradCAMPlusPlus(model)

  # Single crop inference
  import torch
  from PIL import Image
  import torchvision.transforms as T

  transform = T.Compose([
      T.Resize((64, 64)),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  crop  = Image.open("cell.png").convert("RGB")
  inp   = transform(crop).unsqueeze(0)           # [1, 3, 64, 64]

  heatmap, pred_class, confidence = cam(inp)     # heatmap: H×W float32 in [0,1]
  overlay = cam.overlay(crop, heatmap)           # PIL Image with heatmap overlay
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
#  Grad-CAM++
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """
    Grad-CAM++ wrapper for EfficientNet-B0.

    Parameters
    ----------
    model    : EfficientNet-B0 with a loaded checkpoint (in eval mode).
    target_layer : nn.Module to hook.
                   Defaults to model.features[-1] (last MBConv block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()

        # Default target: last feature block of EfficientNet-B0
        self.target_layer = target_layer if target_layer is not None \
                            else model.features[-1]

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ──────────────────────────────────────────────────────────────────

    def _save_activation(self, module, inp, output):
        """Forward hook: cache feature maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: cache gradients of the target layer output."""
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Call this when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core computation ───────────────────────────────────────────────────────

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Compute the Grad-CAM++ heatmap for one crop.

        Parameters
        ----------
        input_tensor : FloatTensor of shape [1, 3, H, W] (normalised).
        class_idx    : Target class. If None, uses the predicted class.

        Returns
        -------
        heatmap    : np.ndarray  shape [H, W], float32, values in [0, 1].
        pred_class : int   — argmax class index.
        confidence : float — softmax probability of the predicted class.
        """
        device = next(self.model.parameters()).device
        inp    = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        logits = self.model(inp)                              # [1, C]
        probs  = torch.softmax(logits, dim=1)

        pred_class = int(logits.argmax(1).item())
        confidence = float(probs[0, pred_class].item())

        target_cls = class_idx if class_idx is not None else pred_class

        # Backward pass — gradients w.r.t. target class score
        self.model.zero_grad()
        score = logits[0, target_cls]
        score.backward()

        # ── Grad-CAM++ formula ──────────────────────────────────────────────
        # grads  : [1, C_feat, h, w]
        # acts   : [1, C_feat, h, w]
        grads = self._gradients                     # [1, K, h, w]
        acts  = self._activations                   # [1, K, h, w]

        grads_sq  = grads ** 2                      # second order
        grads_cub = grads ** 3                      # third  order

        # Alpha_ck  (Chattopadhay et al., Eq. 19)
        numer = grads_sq
        denom = 2.0 * grads_sq + \
                (acts * grads_cub).sum(dim=(2, 3), keepdim=True) + 1e-9
        alpha = numer / denom                       # [1, K, h, w]

        # Weights: sum of (alpha * relu(grad)) over spatial dims
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3))  # [1, K]

        # Linear combination of activations
        cam = (weights[..., None, None] * acts).sum(dim=1, keepdim=True)  # [1,1,h,w]
        cam = F.relu(cam)                           # keep only positive influence

        # Upsample to input spatial size
        H, W   = input_tensor.shape[-2:]
        cam_up = F.interpolate(cam, size=(H, W),
                               mode="bilinear", align_corners=False)  # [1,1,H,W]

        # Normalise to [0, 1]
        heatmap = cam_up.squeeze().cpu().numpy()    # [H, W]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32), pred_class, confidence

    # ── Overlay helper ─────────────────────────────────────────────────────────

    @staticmethod
    def overlay(
        original: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Overlay a normalised heatmap [H, W] onto a PIL Image.

        Parameters
        ----------
        original  : PIL Image (RGB), any size.
        heatmap   : float32 array in [0, 1], same H×W as original OR smaller.
        alpha     : heatmap opacity (0=invisible, 1=fully opaque).
        colormap  : matplotlib colormap name ('jet' is the standard for CAM).

        Returns
        -------
        PIL Image (RGB) with heatmap blended on top.
        """
        import matplotlib.cm as cm

        W, H = original.size

        # Resize heatmap to match image if needed
        if heatmap.shape != (H, W):
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((W, H), Image.BILINEAR)
            heatmap = np.array(heatmap_img) / 255.0

        # Apply colourmap
        cmap   = cm.get_cmap(colormap)
        coloured = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]
        coloured_img = Image.fromarray(coloured)

        # Blend
        blended = Image.blend(original.convert("RGB"), coloured_img, alpha=alpha)
        return blended

    # ── Batch helper ───────────────────────────────────────────────────────────

    def batch_heatmaps(
        self,
        crops: torch.Tensor,           # [N, 3, H, W]
        class_indices: list | None = None,
    ) -> list[tuple[np.ndarray, int, float]]:
        """
        Compute Grad-CAM++ heatmaps for a batch of crops.
        Returns a list of (heatmap, pred_class, confidence) tuples.
        """
        results = []
        for i in range(crops.shape[0]):
            inp   = crops[i:i+1]
            cidx  = class_indices[i] if class_indices else None
            results.append(self(inp, class_idx=cidx))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from shared.label_map import NUM_CLASSES, INT_TO_LABEL

    print("Grad-CAM++ smoke test")
    print("Loading EfficientNet-B0 (ImageNet pretrained — no task checkpoint)...")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace head to match MalariAI class count
    model.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)
    model.eval()

    cam = GradCAMPlusPlus(model)

    # Random 64×64 input
    inp = torch.randn(1, 3, 64, 64)
    heatmap, pred_cls, conf = cam(inp)

    print(f"  Input shape  : {list(inp.shape)}")
    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Pred class   : {pred_cls} ({INT_TO_LABEL[pred_cls]})")
    print(f"  Confidence   : {conf:.4f}")
    print(f"  Heatmap min/max: {heatmap.min():.4f} / {heatmap.max():.4f}")

    cam.remove_hooks()
    print("OK — Grad-CAM++ working correctly.")
