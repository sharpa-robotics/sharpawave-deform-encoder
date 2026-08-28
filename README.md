# SharpaWave Deform Encoder

## Overview

`DeformEncoder` is the primary model in this repository. It converts a
preprocessed single-channel scalar deformation image into a compact learned
tactile feature with shape `[B, 512, 1, 1]`.

`DeformDecoder` and `DeformAutoencoder` are included only to demonstrate
deformation reconstruction from this feature. Downstream applications normally
need only `DeformEncoder`.

## Components

| Component | Purpose |
| --- | --- |
| `DeformEncoder` | Customer-facing tactile feature extractor |
| `DeformDecoder` | Reconstruction-only decoder |
| `DeformAutoencoder` | Encoder and decoder demonstration model |

## Tensor Shapes

| Operation | Input | Output |
| --- | --- | --- |
| Encoder | `[B, 1, 240, 240]` | `[B, 512, 1, 1]` |
| Flatten feature | `[B, 512, 1, 1]` | `[B, 512]` |
| Autoencoder | `[B, 1, 240, 240]` | `[B, 1, 240, 240]` |

The input is a preprocessed scalar deformation image, not a raw RGB camera
image.

## Requirements

Python 3.10+, PyTorch, timm, SafeTensors, Hugging Face Hub, NumPy, and OpenCV.

```bash
python -m pip install -r requirements.txt
```

## Weights

One unified SafeTensors checkpoint contains both encoder and decoder weights.
`load_encoder()` reads only the encoder tensors, while `load_autoencoder()`
loads the complete reconstruction model.

The published checkpoint was trained from random initialization without
upstream pretrained weights.

```bash
python scripts/download_weights.py
```

Model files: <https://huggingface.co/Sharpa-Robotics/sharpawave-deform-encoder>

## Encoder Usage

```python
import torch

from sharpawave_deform_encoder import load_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = load_encoder(
    "checkpoints/sharpawave_deform_autoencoder.safetensors",
    device,
)

deform = torch.zeros(1, 1, 240, 240, device=device)
with torch.inference_mode():
    feature = encoder(deform)          # [B, 512, 1, 1]
    feature_vector = feature.flatten(1)  # [B, 512]
```

## Reconstruction Demo

The bundled decoder demonstrates reconstruction from the compact feature:

```bash
python demo_reconstruction.py
```

Results are saved in `outputs/`. Use `--show` to display them interactively.

## Limitations

The encoder expects the documented 240-by-240 scalar input representation.

## License

Copyright 2026 Sharpa Group. Licensed under Apache License 2.0. See `LICENSE`
and `THIRD_PARTY_NOTICES.md`.
