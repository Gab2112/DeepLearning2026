"""
Transformations and scalers for vegetation forecasting.
Implements the official standardization_arcsinh transform using scaler.json.
"""

import json
from pathlib import Path
import numpy as np
import torch


def standardization_arcsinh(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Normalize values using precomputed feature or target statistics."""
    return np.arcsinh((x - mean) / std)


class ArcsinhScaler:
    """
    Standardization followed by arcsinh transformation:
        z = (x - mean) / sqrt(var + eps)
        y = arcsinh(z)

    Inverse:
        x = sinh(y) * sqrt(var + eps) + mean
    """

    def __init__(self, scaler_path: str or Path):
        scaler_path = Path(scaler_path)
        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.eps = float(data.get("eps", 1e-5))
        self.mean = np.array(data["mean"], dtype=np.float32)
        self.var = np.array(data["var"], dtype=np.float32)
        self.std = np.sqrt(self.var + self.eps).astype(np.float32)

        self.target_mean = np.array(data["target_mean"], dtype=np.float32)
        self.target_var = np.array(data["target_var"], dtype=np.float32)
        self.target_std = np.sqrt(self.target_var + self.eps).astype(np.float32)

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (..., 21) or (..., F)
        Preserves NaNs.
        """
        return standardization_arcsinh(x, self.mean, self.std)

    def inverse_transform_features(self, y: np.ndarray) -> np.ndarray:
        return np.sinh(y) * self.std + self.mean

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        targets shape: (..., 3)
        """
        return standardization_arcsinh(targets, self.target_mean, self.target_std)

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
