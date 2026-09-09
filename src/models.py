"""Model definitions: a small CNN trained from scratch, and a fine-tuned ResNet.

The point of keeping both in one file is the comparison. The scratch CNN is
the honest baseline that answers "how far does a reasonable architecture get
with 5,900 training images?" — and the answer is: not far. The pretrained
ResNet answers "how much of the gap is closed by features learned elsewhere?"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A conventional VGG-style CNN: four conv blocks, then a classifier head.

    Batch norm after each convolution and dropout before the final layer are
    what keep it trainable on a dataset this small. Even so it overfits — see
    the training curves in reports/.
    """

    def __init__(self, num_classes: int = 37, dropout: float = 0.4):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 224 -> 112
            block(32, 64),   # 112 -> 56
            block(64, 128),  # 56  -> 28
            block(128, 256), # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_resnet(
    num_classes: int = 37,
    arch: str = "resnet34",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Load a torchvision ResNet and swap its 1000-way head for ours.

    freeze_backbone=True turns the network into a fixed feature extractor:
    only the new final layer learns. That is the fast first stage — it lets
    the randomly initialised head settle before large gradients are allowed
    to reach the pretrained weights.
    """
    factories = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    }
    if arch not in factories:
        raise ValueError(f"Unknown architecture {arch!r}; expected one of {list(factories)}")

    factory, weights_enum = factories[arch]
    model = factory(weights=weights_enum if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Re-enable gradients everywhere, for the second fine-tuning stage."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def build_model(name: str, num_classes: int = 37, **kwargs) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    return build_resnet(num_classes=num_classes, arch=name, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
