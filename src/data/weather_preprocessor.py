"""Shared weather covariate preprocessing for history and future sequences."""

import numpy as np

from src.data.transforms import ArcsinhScaler, standardization_arcsinh

WEATHER_FEATURE_COUNT = 8
COVARIATE_COUNT = 18


def preprocess_weather_covariates(
    raw: np.ndarray, missing_mask: np.ndarray, scaler: ArcsinhScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Return 18 normalized covariates and eight weather flags as float32 arrays.

    Accept matching (..., 18) or (..., 21) sequences in the dataset feature order.
    Only the first 18 columns are used. Source masks use True for missing values.
    Each weather flag is 1 for an unmasked finite value and 0 otherwise.
    Missing covariates use zero in normalized space.
    """
    raw = np.asarray(raw, dtype=np.float32)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    if raw.ndim < 2 or raw.shape[-1] not in (18, 21) or raw.shape != missing_mask.shape:
        raise ValueError("Expected matching (..., 18) or (..., 21) values and masks")

    covariate_slice = slice(COVARIATE_COUNT)
    covariates = raw[..., covariate_slice]
    mean = np.asarray(scaler.mean[covariate_slice], dtype=np.float32)
    std = np.asarray(scaler.std[covariate_slice], dtype=np.float32)
    if (
        mean.shape != (COVARIATE_COUNT,)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Invalid train scaler statistics for covariates")

    observed = ~missing_mask[..., covariate_slice] & np.isfinite(covariates)
    # Substituting the train mean in this calculation gives arcsinh(0).
    safe = np.where(observed, covariates, mean)
    scaled = standardization_arcsinh(safe, mean, std)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite normalized covariates")

    weather_mask = observed[..., :WEATHER_FEATURE_COUNT].astype(np.float32)
    return scaled, weather_mask
