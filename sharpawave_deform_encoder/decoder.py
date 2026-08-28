"""Reconstruction decoder used by the autoencoder demonstration."""

from __future__ import annotations

import torch
from torch import nn


DEFORMATION_SCALE = 2.2e-4


class DeformDecoder(nn.Module):
    """Decode a compact tactile feature into one reconstructed channel."""

    def __init__(self) -> None:
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(512, 256 * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(256, 128 * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(128, 128 * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(128, 128 * 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((15, 15)),
        )
        self.deformation_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
        )
        self.kmap_conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)

    @staticmethod
    def _validate(feature: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if not isinstance(feature, torch.Tensor) or not feature.is_floating_point():
            raise TypeError("feature must be a floating-point torch.Tensor")
        if feature.ndim != 4 or tuple(feature.shape[1:]) != (512, 1, 1):
            raise ValueError(
                "feature must have shape [B, 512, 1, 1]; "
                f"received {list(feature.shape)}"
            )
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean torch.Tensor")
        expected_mask = (feature.shape[0], 1, 240, 240)
        if tuple(valid_mask.shape) != expected_mask:
            raise ValueError(
                f"valid_mask must have shape {list(expected_mask)}; "
                f"received {list(valid_mask.shape)}"
            )
        if not torch.isfinite(feature).all():
            raise ValueError("feature contains NaN or infinite values")

    @staticmethod
    def _dequantize(deform: torch.Tensor) -> torch.Tensor:
        signed_square = torch.where(deform > 0, deform.square(), -deform.square())
        return signed_square * DEFORMATION_SCALE

    def forward(self, feature: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Return one reconstructed channel with shape ``[B, 1, 240, 240]``."""
        self._validate(feature, valid_mask)
        decoded = self.deformation_head(self.upsample_layer(feature))
        decoded = decoded * valid_mask
        decoded = self.kmap_conv(self._dequantize(decoded))
        return decoded[:, :1]
