from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from predict import load_checkpoint
from src.models import gradcam_target_layer
from src.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transforms
from src.utils import device_from_arg


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(self, tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * self.activations).sum(dim=1).relu()
        cam_map = torch.nn.functional.interpolate(
            cam_map.unsqueeze(1),
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        return cam_map.cpu().numpy(), class_idx


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    image = (tensor[0] * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for Food-101 1B models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="outputs/grad_cam.png")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    image_size = ckpt.get("image_size", 224)
    image = Image.open(args.image).convert("RGB")
    tensor = eval_transforms(image_size)(image).unsqueeze(0).to(device)
    cam_runner = GradCAM(model, gradcam_target_layer(model, ckpt["model_name"]))
    heatmap, class_idx = cam_runner(tensor)

    base = denormalize(tensor)
    color = cm.get_cmap("jet")(heatmap)[..., :3]
    overlay = (0.55 * base + 0.45 * color).clip(0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out)
    print(f"Predicted class: {ckpt['classes'][class_idx]}")
    print(f"Saved Grad-CAM to {out}")


if __name__ == "__main__":
    main()
