"""Training entry point.

Examples
--------
Baseline, trained from scratch:
    python -m src.train --model simple_cnn --epochs 30 --lr 1e-3 --run-name baseline

Transfer learning, two stages (head first, then the whole network):
    python -m src.train --model resnet34 --epochs 12 --freeze-epochs 3 --run-name resnet34
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import DataConfig, build_dataloaders
from .models import build_model, count_parameters, unfreeze_all
from .utils import set_seed, get_device, save_history_plot


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch = targets.size(0)
            total_loss += loss.item() * batch
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += batch

    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, val_loader, _, classes = build_dataloaders(cfg)
    print(f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images, "
          f"{len(classes)} classes")

    staged = args.freeze_epochs > 0 and args.model != "simple_cnn"
    model = build_model(
        args.model,
        num_classes=len(classes),
        **({"freeze_backbone": True} if staged else {}),
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Model {args.model}: {total:,} parameters ({trainable:,} trainable)")

    # Label smoothing costs nothing and consistently buys a little accuracy
    # on fine-grained tasks, where several classes are genuinely confusable.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    # Starts below zero so the first epoch always writes a checkpoint, even
    # if its accuracy is 0.0 — otherwise a failed run leaves nothing on disk.
    best_val_acc, best_epoch = -1.0, -1
    start = time.time()

    for epoch in range(args.epochs):
        # End of stage 1: release the backbone and drop the learning rate,
        # otherwise the first unfrozen step wrecks the pretrained features.
        if staged and epoch == args.freeze_epochs:
            print(f"--- epoch {epoch + 1}: unfreezing the backbone, lr -> {args.lr * 0.1:g}")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - epoch, 1)
            )

        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.model,
                    "classes": classes,
                    "image_size": args.image_size,
                    "val_acc": va_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            marker = "  <- best, saved"

        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train {tr_loss:.3f}/{tr_acc:.3f}  "
            f"val {va_loss:.3f}/{va_acc:.3f}  "
            f"lr {lr_now:.2e}{marker}"
        )

    elapsed = time.time() - start
    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "minutes": round(elapsed / 60, 2),
        "params_total": total,
        "args": vars(args),
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    save_history_plot(history, out_dir / "curves.png", title=args.run_name)

    print(
        f"\nDone in {elapsed / 60:.1f} min. "
        f"Best val accuracy {best_val_acc:.4f} at epoch {best_epoch + 1}. "
        f"Checkpoint: {ckpt_path}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a pet breed classifier.")
    p.add_argument("--model", default="resnet34",
                   choices=["simple_cnn", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--run-name", default=None, help="Subfolder name under --out-dir")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="Epochs with a frozen backbone before full fine-tuning (0 = off)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.run_name is None:
        args.run_name = args.model
    train(args)


if __name__ == "__main__":
    main()
