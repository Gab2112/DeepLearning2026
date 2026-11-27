"""
Transformations and scalers for vegetation forecasting.
Implements the official standardization_arcsinh transform using scaler.json.
"""

import json
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch


class ArcsinhScaler:
    """
    Standardization followed by arcsinh transformation:
        z = (x - mean) / sqrt(var + eps)
        y = arcsinh(z)

    Inverse:
        x = sinh(y) * sqrt(var + eps) + mean
    """

    def __init__(self, scaler_path: str | Path | Mapping):
        if isinstance(scaler_path, Mapping):
            data = scaler_path
        else:
            with open(scaler_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.eps = float(data.get("eps", 1e-5))
        self.provenance = dict(data.get("provenance", {}))
        self.mean = np.array(data["mean"], dtype=np.float32)
        self.var = np.array(data["var"], dtype=np.float32)
        self.std = np.sqrt(self.var + self.eps).astype(np.float32)

        self.target_mean = np.array(data["target_mean"], dtype=np.float32)
        self.target_var = np.array(data["target_var"], dtype=np.float32)
        self.target_std = np.sqrt(self.target_var + self.eps).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mode": "standardization_arcsinh",
            "eps": self.eps,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_var": self.target_var.tolist(),
            "provenance": self.provenance,
        }

    def transform_features(
        self, x: np.ndarray, *, feature_slice: slice = slice(None)
    ) -> np.ndarray:
        """
        Transform all features, or the columns selected by feature_slice.
        Preserves NaNs.
        """
        z = (x - self.mean[feature_slice]) / self.std[feature_slice]
        return np.arcsinh(z)

    def inverse_transform_features(self, y: np.ndarray) -> np.ndarray:
        return np.sinh(y) * self.std + self.mean

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        targets shape: (..., 3)
        """
        z = (targets - self.target_mean) / self.target_std
        return np.arcsinh(z)

    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        return np.sinh(y) * self.target_std + self.target_mean

    # PyTorch Tensor variants
    def torch_transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, device=targets.device, dtype=targets.dtype)
        std = torch.as_tensor(self.target_std, device=targets.device, dtype=targets.dtype)
        z = (targets - mean) / std
        return torch.asinh(z)

    def torch_inverse_transform_targets(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.target_mean, device=y.device, dtype=y.dtype)
        std = torch.as_tensor(self.target_std, device=y.device, dtype=y.dtype)
        return torch.sinh(y) * std + mean


def scaler_for_imputer(model, scaler: ArcsinhScaler | None = None) -> ArcsinhScaler:
    """Recover the training scaler, with a reference-scaler path for legacy weights."""
    trained_scaler = getattr(model, "arcsinh_scaler", None)
    if trained_scaler is not None:
        if scaler is not None and scaler.to_dict() != trained_scaler.to_dict():
            raise ValueError("Supplied scaler differs from the scaler embedded in the checkpoint")
        return trained_scaler
    if scaler is not None:
        return scaler
    root = Path(__file__).resolve().parents[2]
    return ArcsinhScaler(root / "DATA/train_multi_output_clear_sky/scaler.json")
