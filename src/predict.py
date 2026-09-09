"""Predict the breed of one image from the command line.

    python -m src.predict --image path/to/photo.jpg --checkpoint runs/resnet34/best.pt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .evaluate import load_checkpoint
from .utils import get_device, pretty_class_name


@torch.no_grad()
def predict(image_path: str, checkpoint: str, topk: int = 5):
    device = get_device()
    model, classes, ckpt = load_checkpoint(checkpoint, device)
    _, eval_tf = build_transforms(ckpt.get("image_size", 224))

    tensor = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(tensor), dim=1)[0].cpu()

    top = probs.topk(min(topk, len(classes)))
    return [
        (pretty_class_name(classes[i]), float(p))
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a pet photo.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="runs/resnet34/best.pt")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    for rank, (name, prob) in enumerate(predict(args.image, args.checkpoint, args.topk), 1):
        print(f"{rank}. {name:32s} {prob:6.2%}")


if __name__ == "__main__":
    main()
