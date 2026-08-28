#!/usr/bin/env python3
"""Download and verify the unified model checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY_ID = "Sharpa-Robotics/sharpawave-deform-encoder"
FILENAME = "sharpawave_deform_autoencoder.safetensors"
REVISION = "main"
EXPECTED_SHA256 = "031ba006e2c6ab4446fedd8561cc478c8757ea18556054826ee1ad847ddad841"
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = ROOT / "checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    destination = CHECKPOINT_DIRECTORY / FILENAME
    if destination.is_file() and sha256(destination) == EXPECTED_SHA256:
        print(f"Checkpoint is already available and verified: {destination}")
        return
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename=FILENAME,
            revision=REVISION,
            local_dir=CHECKPOINT_DIRECTORY,
        )
    )
    actual = sha256(downloaded)
    if actual != EXPECTED_SHA256:
        downloaded.unlink(missing_ok=True)
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual}; expected {EXPECTED_SHA256}")
    print(f"Downloaded and verified checkpoint: {downloaded}")


if __name__ == "__main__":
    main()
