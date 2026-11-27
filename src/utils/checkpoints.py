"""Record completion and timing on the best checkpoint after training."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def complete_training_checkpoint(
    path: Path, *, epochs: int, optimizer_steps: int, seconds: float, seed: int
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["training"] = {
        "completed_epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "seconds": seconds,
        "seed": seed,
    }
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint
