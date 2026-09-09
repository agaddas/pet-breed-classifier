"""Grad-CAM: which pixels drove the prediction.

Implemented from the paper (Selvaraju et al., 2017) rather than pulled from a
library, because the mechanism is the point: hook the last convolutional
block, average the gradients of the target logit over each channel to get
per-channel importance weights, take the weighted sum of the activation maps,
keep the positive part.

A classifier that reaches high accuracy by reading the background rather than
the animal is a classifier that will fail in production. Grad-CAM is how you
find that out before someone else does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms, denormalize
from .evaluate import load_checkpoint
from .utils import get_device, pretty_class_name


class GradCAM:
    """Grad-CAM for a CNN, hooked on a chosen layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()

    def __call__(self, image: torch.Tensor, class_idx: int | None = None):
        """Return (heatmap HxW in [0,1], predicted index, probabilities)."""
        image = image.unsqueeze(0) if image.dim() == 3 else image
        image = image.requires_grad_(True)

        logits = self.model(image)
        probs = F.softmax(logits, dim=1)[0].detach().cpu()
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        # One weight per channel: the mean gradient over the spatial map.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.cpu().numpy(), class_idx, probs


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The last convolutional stage — the deepest layer that still has geometry."""
    if hasattr(model, "layer4"):      # torchvision ResNet
        return model.layer4[-1]
    if hasattr(model, "features"):    # our SimpleCNN
        return model.features[-1]
    raise ValueError("Could not locate a convolutional layer to hook.")


def overlay(image: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap over the de-normalised image."""
    rgb = denormalize(image.detach().cpu()).permute(1, 2, 0).numpy()
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def explain_image(image_path: str, checkpoint: str, out_path: str, topk: int = 3) -> None:
    device = get_device()
    model, classes, ckpt = load_checkpoint(checkpoint, device)
    _, eval_tf = build_transforms(ckpt.get("image_size", 224))

    image = eval_tf(Image.open(image_path).convert("RGB")).to(device)

    with GradCAM(model, find_target_layer(model)) as cam_fn:
        cam, class_idx, probs = cam_fn(image)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(denormalize(image.detach().cpu()).permute(1, 2, 0).numpy())
    axes[0].set_title("input")
    axes[1].imshow(overlay(image, cam))
    axes[1].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")

    top = probs.topk(topk)
    caption = "   ".join(
        f"{pretty_class_name(classes[i])} {p:.0%}"
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    )
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Predicted: {pretty_class_name(classes[class_idx])} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM explanation for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="reports/gradcam.png")
    args = parser.parse_args()
    explain_image(args.image, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
