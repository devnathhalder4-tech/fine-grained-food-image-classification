from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.food101_subset import Food101Subset, read_classes
from src.models import build_model
from src.transforms import eval_transforms, train_transforms
from src.utils import device_from_arg, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Food-101 1B Asian staple classifiers.")
    parser.add_argument("--data-root", default="data", help="Food-101 root directory.")
    parser.add_argument("--class-file", default="configs/classes_1b_asian.txt")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2", "shallow_cnn"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true", help="Download Food-101 if needed.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out-dir", default="outputs")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    classes = read_classes(args.class_file)

    train_set = Food101Subset(
        args.data_root,
        split="train",
        classes=classes,
        transform=train_transforms(args.image_size),
        download=args.download,
    )
    val_set = Food101Subset(
        args.data_root,
        split="test",
        classes=classes,
        transform=eval_transforms(args.image_size),
        download=args.download,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.out_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(row)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "classes": classes,
                    "image_size": args.image_size,
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                run_dir / "best.pt",
            )

    save_json({"best_acc": best_acc, "seconds": time.time() - started, "history": history}, run_dir / "history.json")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
