"""Compact tactile feature encoder."""

from __future__ import annotations

import torch
from torch import nn
import timm


INPUT_HEIGHT = 240
INPUT_WIDTH = 240
FEATURE_CHANNELS = 512


def validate_deformation_input(deform: torch.Tensor) -> None:
    """Validate the public encoder input contract."""
    if not isinstance(deform, torch.Tensor):
        raise TypeError("deform must be a torch.Tensor")
    if not deform.is_floating_point():
        raise TypeError("deform must use a floating-point dtype")
    expected = (1, INPUT_HEIGHT, INPUT_WIDTH)
    if deform.ndim != 4 or tuple(deform.shape[1:]) != expected:
        raise ValueError(
            "deform must have shape [B, 1, 240, 240]; "
            f"received {list(deform.shape)}"
        )
    if not torch.isfinite(deform).all():
        raise ValueError("deform contains NaN or infinite values")


class DeformEncoder(nn.Module):
    """Encode a scalar deformation image as a compact tactile feature."""

    def __init__(self) -> None:
        super().__init__()
        self.downsample_layer = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        backbone = timm.create_model("convnextv2_pico", pretrained=False)
        stem = backbone.stem
        old_conv = stem[0]
        stem[0] = nn.Conv2d(
            64,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.convnext_encoder = nn.Sequential(stem, *backbone.stages)

    def forward(self, deform: torch.Tensor) -> torch.Tensor:
        """Return a feature tensor with shape ``[B, 512, 1, 1]``."""
        validate_deformation_input(deform)
        return self.convnext_encoder(self.downsample_layer(deform))
