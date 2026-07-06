from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.food101_subset import Food101Subset
from src.models import build_model
from src.transforms import eval_transforms
from src.utils import device_from_arg, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Food-101 1B model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model_name = ckpt["model_name"]
    image_size = ckpt.get("image_size", 224)

    dataset = Food101Subset(
        args.data_root,
        split="test",
        classes=classes,
        transform=eval_transforms(image_size),
        download=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    out_dir = Path(args.out_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    save_json(report, out_dir / "classification_report.json")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", square=True, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Food-101 1B Confusion Matrix - {model_name}")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=220)
    print(f"accuracy={report['accuracy']:.4f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
