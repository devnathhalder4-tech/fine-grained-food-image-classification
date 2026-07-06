from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.models import build_model
from src.transforms import eval_transforms
from src.utils import device_from_arg


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def predict_image(model, classes: list[str], image: Image.Image, image_size: int, device: torch.device, topk: int = 5):
    transform = eval_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0]
    values, indices = prob.topk(topk)
    return [(classes[i], float(v)) for v, i in zip(values.cpu(), indices.cpu())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a Food-101 1B image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    image = Image.open(args.image)
    results = predict_image(model, ckpt["classes"], image, ckpt.get("image_size", 224), device, args.topk)
    for label, score in results:
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
