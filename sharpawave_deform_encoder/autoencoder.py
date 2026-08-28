"""Autoencoder composition used for reconstruction demonstrations."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import DeformDecoder
from .encoder import DeformEncoder


class DeformAutoencoder(nn.Module):
    """Combine the feature encoder and reconstruction-only decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DeformEncoder()
        self.decoder = DeformDecoder()

    def encode(self, deform: torch.Tensor) -> torch.Tensor:
        """Return the compact tactile feature produced by the encoder."""
        return self.encoder(deform)

    def forward(self, deform: torch.Tensor) -> torch.Tensor:
        """Reconstruct one scalar deformation channel."""
        feature = self.encode(deform)
        valid_mask = deform.abs() > 1e-6
        return self.decoder(feature, valid_mask)
