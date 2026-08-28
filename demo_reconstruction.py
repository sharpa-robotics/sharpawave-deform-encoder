#!/usr/bin/env python3
"""Run the reconstruction demonstration on deformation images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sharpawave_deform_encoder import DeformAutoencoder, load_autoencoder


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sharpawave_deform_autoencoder.safetensors"
DEFAULT_INPUT = ROOT / "deform_imgs"
DEFAULT_OUTPUT = ROOT / "outputs"


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(item for item in path.iterdir() if item.suffix.lower() in extensions)
    if not images:
        raise ValueError(f"No supported images found in: {path}")
    return images


def read_image(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    if image.shape != (240, 240):
        raise ValueError(f"Expected a 240x240 image, received {image.shape}: {path}")
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    return image, tensor


def save_comparison(source: np.ndarray, prediction: torch.Tensor, path: Path) -> np.ndarray:
    reconstructed = prediction.squeeze().detach().cpu().numpy()
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    comparison = np.concatenate((source, reconstructed), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), comparison):
        raise OSError(f"Unable to write output image: {path}")
    return comparison


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def run(
    model: DeformAutoencoder,
    image_paths: list[Path],
    output_dir: Path,
    device: torch.device,
    show: bool = False,
) -> int:
    count = 0
    with torch.inference_mode():
        for image_path in image_paths:
            image, tensor = read_image(image_path)
            tensor = tensor.to(device)
            feature = model.encode(tensor)
            reconstruction = model.decoder(feature, tensor.abs() > 1e-6)
            output_path = output_dir / f"{image_path.stem}_comparison.png"
            comparison = save_comparison(image, reconstruction, output_path)
            print(
                f"{image_path.name}: feature={list(feature.shape)}, "
                f"reconstruction={list(reconstruction.shape)}, saved={output_path}"
            )
            if show:
                cv2.imshow("Input | reconstruction", comparison)
                cv2.waitKey(0)
            count += 1
    if show:
        cv2.destroyAllWindows()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = load_autoencoder(args.checkpoint, device)
    count = run(model, collect_images(args.input), args.output_dir, device, args.show)
    print(f"Processed {count} image(s) on {device}.")


if __name__ == "__main__":
    main()
