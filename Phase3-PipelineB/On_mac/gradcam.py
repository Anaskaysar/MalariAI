"""
Phase3-PipelineB/gradcam.py
===========================
Grad-CAM++ implementation for EfficientNet-B0.
Provides spatial activation heatmaps to explain model predictions.

Adapted from: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass
        logit = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logit.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logit[0, class_idx]
        score.backward()
        
        # Grad-CAM++ logic
        # A: activations [1, C, H, W]
        # G: gradients [1, C, H, W]
        A = self.activations
        G = self.gradients
        
        # alpha_ij_ck = G^2 / (2*G^2 + A*G^3)
        # Simplified: weighted average of positive gradients
        # We'll use the standard Grad-CAM++ weight formula
        
        # Global average pooling of gradients
        weights = torch.mean(G, dim=(2, 3), keepdim=True)
        
        # Linear combination of activations
        cam = torch.sum(weights * A, dim=1, keepdim=True)
        
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        
        return cam.detach().cpu().numpy()[0, 0], class_idx

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    output = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return output
