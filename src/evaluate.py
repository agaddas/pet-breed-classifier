"""Evaluate a checkpoint on the held-out test split.

Produces the numbers and figures quoted in the README:
  - top-1 and top-5 accuracy, macro F1
  - a 37x37 confusion matrix
  - the per-class report and the classes the model actually struggles with
  - a grid of the most confident mistakes

Usage:
    python -m src.evaluate --checkpoint runs/resnet34/best.pt --out reports/resnet34
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import DataConfig, build_dataloaders, denormalize
from .models import build_model
from .utils import get_device, pretty_class_name


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, ckpt


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader, returning probabilities and labels."""
    all_probs, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images), dim=1).cpu()
        all_probs.append(probs)
        all_targets.append(targets)
    return torch.cat(all_probs), torch.cat(all_targets)


def topk_accuracy(probs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    topk = probs.topk(k, dim=1).indices
    hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], path: Path) -> None:
    """Row-normalised confusion matrix. A clean diagonal is the goal."""
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([pretty_class_name(c) for c in classes], rotation=90, fontsize=7)
    ax.set_yticklabels([pretty_class_name(c) for c in classes], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confident_mistakes(loader, probs, targets, classes, path: Path, n: int = 12) -> None:
    """The errors the model was most sure about — the interesting failures."""
    preds = probs.argmax(1)
    conf = probs.max(1).values
    wrong = (preds != targets).nonzero(as_tuple=True)[0]
    if wrong.numel() == 0:
        return
    order = wrong[conf[wrong].argsort(descending=True)][:n]

    # The test loader is unshuffled, so dataset index == position in `probs`.
    dataset = loader.dataset
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    for ax, idx in zip(np.array(axes).ravel(), order.tolist()):
        image, _ = dataset[idx]
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        ax.set_title(
            f"true: {pretty_class_name(classes[targets[idx]])}\n"
            f"pred: {pretty_class_name(classes[preds[idx]])} ({conf[idx]:.0%})",
            fontsize=8,
        )
        ax.axis("off")
    for ax in np.array(axes).ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("Most confident mistakes")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="reports/eval")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    model, classes, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {ckpt['arch']} (val accuracy at save time: {ckpt['val_acc']:.4f})")

    cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=ckpt.get("image_size", 224),
    )
    _, _, test_loader, _ = build_dataloaders(cfg)

    probs, targets = collect_predictions(model, test_loader, device)
    preds = probs.argmax(1)

    top1 = (preds == targets).float().mean().item()
    top5 = topk_accuracy(probs, targets, k=5)
    macro_f1 = f1_score(targets.numpy(), preds.numpy(), average="macro")

    print(f"\nTest top-1 accuracy : {top1:.4f}")
    print(f"Test top-5 accuracy : {top5:.4f}")
    print(f"Test macro F1       : {macro_f1:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(targets.numpy(), preds.numpy(), labels=range(len(classes)))
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    plot_confident_mistakes(test_loader, probs, targets, classes,
                            out_dir / "confident_mistakes.png")

    report = classification_report(
        targets.numpy(), preds.numpy(),
        labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )
    per_class = sorted(
        ((c, report[c]["f1-score"], int(report[c]["support"])) for c in classes),
        key=lambda t: t[1],
    )
    print("\nFive hardest classes (by F1):")
    for name, f1, support in per_class[:5]:
        print(f"  {pretty_class_name(name):32s} F1 {f1:.3f}  (n={support})")

    (out_dir / "metrics.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "arch": ckpt["arch"],
        "test_top1": top1,
        "test_top5": top5,
        "test_macro_f1": macro_f1,
        "per_class_f1": {c: report[c]["f1-score"] for c in classes},
    }, indent=2))
    print(f"\nFigures and metrics written to {out_dir}/")


if __name__ == "__main__":
    main()
