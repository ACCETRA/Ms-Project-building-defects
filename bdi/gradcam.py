"""Grad-CAM (Gradient-weighted Class Activation Mapping) for inspection models."""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Generate a 2D float32 [0, 1] heatmap for target_class."""
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        score = logits[:, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations.")

        # Global average pooling on gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [B, 1, H, W]

        # ReLU to consider only features that have positive influence
        cam = F.relu(cam)

        # Upsample to match input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def apply_colormap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend float [0, 1] heatmap with PIL Image using JET-like colormap."""
    width, height = original_image.size
    # Resize heatmap to match image size if necessary
    if (heatmap.shape[1], heatmap.shape[0]) != (width, height):
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (width, height), resample=Image.BILINEAR
        )
        norm_map = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    else:
        norm_map = heatmap

    # Simple RGB jet colormap implementation (no heavy opencv dependency required)
    # Jet: blue (0) -> cyan (0.25) -> green (0.5) -> yellow (0.75) -> red (1.0)
    r = np.clip(1.5 - np.abs(norm_map * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm_map * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm_map * 4.0 - 1.0), 0.0, 1.0)
    color_map = np.stack([r, g, b], axis=-1)
    color_map_uint8 = (color_map * 255).astype(np.uint8)
    color_img = Image.fromarray(color_map_uint8, mode="RGB")

    return Image.blend(original_image.convert("RGB"), color_img, alpha=alpha)
