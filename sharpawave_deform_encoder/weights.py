"""Strict selective loading from the unified SafeTensors checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .autoencoder import DeformAutoencoder
from .encoder import DeformEncoder


ENCODER_PREFIX = "deform_encoder."
ENCODER_TENSOR_COUNT = 139
AUTOENCODER_TENSOR_COUNT = 158


def _validate_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError("Checkpoint must use the .safetensors format")
    return path


def _load_strict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], name: str) -> None:
    expected = model.state_dict()
    if state_dict.keys() != expected.keys():
        missing = sorted(expected.keys() - state_dict.keys())
        unexpected = sorted(state_dict.keys() - expected.keys())
        raise RuntimeError(
            f"{name} checkpoint keys do not match: missing={missing}, unexpected={unexpected}"
        )
    mismatched = sorted(
        key for key, value in state_dict.items() if value.shape != expected[key].shape
    )
    if mismatched:
        raise RuntimeError(f"{name} checkpoint tensor shapes do not match: {mismatched}")
    model.load_state_dict(state_dict, strict=True)


def load_encoder(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> DeformEncoder:
    """Load only encoder tensors and return the customer-facing feature model."""
    path = _validate_path(checkpoint_path)
    state_dict: dict[str, torch.Tensor] = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = [key for key in checkpoint.keys() if key.startswith(ENCODER_PREFIX)]
            if len(keys) != ENCODER_TENSOR_COUNT:
                raise RuntimeError(
                    f"Expected {ENCODER_TENSOR_COUNT} encoder tensors, found {len(keys)}"
                )
            state_dict = {
                key.removeprefix(ENCODER_PREFIX): checkpoint.get_tensor(key) for key in keys
            }
    except RuntimeError:
        raise
    except Exception as error:
        raise ValueError(f"Unable to read checkpoint: {path}") from error

    model = DeformEncoder()
    _load_strict(model, state_dict, "Encoder")
    return model.to(device).eval()


def load_autoencoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> DeformAutoencoder:
    """Load all encoder and decoder tensors into the demonstration model."""
    path = _validate_path(checkpoint_path)
    try:
        source = load_file(path, device="cpu")
    except Exception as error:
        raise ValueError(f"Unable to read checkpoint: {path}") from error
    if len(source) != AUTOENCODER_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {AUTOENCODER_TENSOR_COUNT} checkpoint tensors, found {len(source)}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key.startswith(ENCODER_PREFIX):
            mapped_key = "encoder." + key.removeprefix(ENCODER_PREFIX)
        elif key.startswith("upsample_layer."):
            mapped_key = "decoder." + key
        elif key.startswith("deformation_head."):
            mapped_key = "decoder." + key
        elif key.startswith("kmap_conv."):
            mapped_key = "decoder." + key
        else:
            raise RuntimeError(f"Unexpected checkpoint tensor: {key}")
        state_dict[mapped_key] = value

    model = DeformAutoencoder()
    _load_strict(model, state_dict, "Autoencoder")
    return model.to(device).eval()
