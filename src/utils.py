"""Small helpers shared by training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # figures are written to disk, never displayed
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch so a rerun lands in the same place.

    Note: cuDNN kernel selection stays nondeterministic by default, so GPU
    runs reproduce to within a few tenths of a point, not bit-for-bit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_history_plot(history: dict, path: str | Path, title: str = "") -> None:
    """Loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pretty_class_name(raw: str) -> str:
    """'american_pit_bull_terrier' -> 'American Pit Bull Terrier'."""
    return raw.replace("_", " ").title()
