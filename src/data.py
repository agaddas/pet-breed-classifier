"""Dataset and dataloader construction for the Oxford-IIIT Pet dataset.

The dataset contains 7,349 images of 37 cat and dog breeds. It is a
*fine-grained* classification task: the visual differences between an
American Bulldog and a Boxer are far smaller than between a cat and a
plane, which is exactly what makes it a fair test of transfer learning.

torchvision downloads and caches the archive automatically the first time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# ImageNet statistics: the pretrained backbones were trained with these,
# so the inputs we feed them at fine-tuning time must match.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 37


@dataclass
class DataConfig:
    data_dir: str = "data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    val_split: float = 0.2
    seed: int = 42


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return the (train, eval) transform pipelines.

    Augmentation is deliberately mild. Pets are photographed upright, so a
    vertical flip would teach the model something that never occurs at test
    time. Random resized crops and colour jitter cover the realistic
    variation: framing, distance, lighting.
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_datasets(cfg: DataConfig):
    """Split the official trainval set into train/val, keep test untouched.

    The official test split is only ever used for the final numbers reported
    in the README. Every hyperparameter decision is made on the validation
    split, so the test score stays an honest estimate.
    """
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = build_transforms(cfg.image_size)

    trainval_train = OxfordIIITPet(
        root=cfg.data_dir, split="trainval", download=True, transform=train_tf
    )
    trainval_eval = OxfordIIITPet(
        root=cfg.data_dir, split="trainval", download=True, transform=eval_tf
    )
    test_set = OxfordIIITPet(
        root=cfg.data_dir, split="test", download=True, transform=eval_tf
    )

    n_total = len(trainval_train)
    n_val = int(n_total * cfg.val_split)
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Same underlying images, different transform pipelines: the validation
    # subset must not be augmented.
    train_set = Subset(trainval_train, train_idx)
    val_set = Subset(trainval_eval, val_idx)

    return train_set, val_set, test_set, trainval_train.classes


def build_dataloaders(cfg: DataConfig):
    train_set, val_set, test_set, classes = build_datasets(cfg)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, classes


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalization, for displaying an image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)
