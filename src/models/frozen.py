"""
Freeze / reload utilities for the reconstruction (history imputation) models.

A *frozen* checkpoint is a self-describing, integrity-checked artifact:
  - only plain Python types + tensors (loadable with torch.load(weights_only=True));
  - architecture hyper-parameters, training metadata, validation metrics and the data
    provenance (scaler hash, train/val tile names) travel with the weights;
  - a SHA-256 over the weights detects any silent modification;
  - on load, the model is put in eval() mode and every parameter gets requires_grad=False.

The payload also exposes `model_state_dict` and `config.{imputer_model|mae_model}` so the
legacy loaders (`load_imputer_checkpoint`, `load_mae_imputer_checkpoint`,
`ModelInterpolator`, `MAEInterpolator`) can read the same file unchanged.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer

ARCH_FIELDS = {
    "bigru": ("input_dim", "climate_dim", "hidden_dim", "num_layers", "output_dim", "dropout"),
    "mae": (
        "input_dim", "climate_dim", "d_model", "nhead", "num_encoder_layers",
        "num_decoder_layers", "dim_feedforward", "output_dim", "dropout",
    ),
}
CONFIG_KEY = {"bigru": "imputer_model", "mae": "mae_model"}


def extract_arch(model_type: str, section: Dict[str, Any]) -> Dict[str, Any]:
    """Selects the constructor arguments of `model_type` from a config section."""
    if model_type not in ARCH_FIELDS:
        raise ValueError(f"Unknown model_type '{model_type}', expected one of {list(ARCH_FIELDS)}")
    return {k: section[k] for k in ARCH_FIELDS[model_type]}


def build_reconstruction_model(model_type: str, arch: Dict[str, Any]) -> nn.Module:
    if model_type == "bigru":
        return BiGRUHistoryImputer(**arch)
    if model_type == "mae":
        return MAEHistoryImputer(**arch)
    raise ValueError(f"Unknown model_type '{model_type}'")


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Deterministic hash of names, dtypes, shapes and raw bytes of all tensors."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _to_plain(obj: Any) -> Any:
    """Converts Paths / numpy scalars / str subclasses / tuples into JSON-like plain types."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, str) and type(obj) is not str:
        return str(obj)
    return obj


def freeze_model(model: nn.Module) -> nn.Module:
    """eval() mode + requires_grad=False on every parameter."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_frozen_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    model_type: str,
    arch: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    full_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves weights + metadata without mutating the live model's requires_grad state.
    Returns the SHA-256 of the weights.
    """
    if model_type not in CONFIG_KEY:
        raise ValueError(f"Unknown model_type '{model_type}', expected one of {list(CONFIG_KEY)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    digest = state_dict_sha256(state)

    cfg_payload: Dict[str, Any] = _to_plain(full_config) if full_config is not None else {}
    cfg_payload[CONFIG_KEY[model_type]] = _to_plain(arch)

    plain_meta = _to_plain(metadata or {})
    payload: Dict[str, Any] = {
        "frozen": True,
        "model_type": model_type,
        "model_state_dict": state,
        "state_dict_sha256": digest,
        "config": cfg_payload,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": str(platform.python_version()),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
        "metadata": plain_meta,
    }

    for legacy_key in ("epoch", "val_loss", "val_mae_orig", "val_mae_per_index"):
        if legacy_key in plain_meta:
            payload[legacy_key] = plain_meta[legacy_key]

    torch.save(payload, path)
    return digest


def load_frozen_model(
    path: Union[str, Path],
    device: Optional[str] = None,
    verify: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Reloads a frozen model: integrity-checked, eval() mode, requires_grad=False.
    Uses weights_only=True, so no arbitrary pickle code can run.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not payload.get("frozen", False):
        raise ValueError(f"{path} is not a frozen reconstruction checkpoint")

    if verify:
        digest = state_dict_sha256(payload["model_state_dict"])
        if digest != payload["state_dict_sha256"]:
            raise ValueError(
                f"Integrity check failed for {path}: weights hash {digest} != {payload['state_dict_sha256']}"
            )

    model_type = payload["model_type"]
    arch = payload["config"][CONFIG_KEY[model_type]]
    model = build_reconstruction_model(model_type, arch)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    freeze_model(model)
    return model, payload


def dump_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_plain(obj), f, indent=2, ensure_ascii=False)
