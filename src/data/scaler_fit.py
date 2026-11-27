"""Fit and identify the scaler shared by fitting and internal validation."""

from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.splits import get_experiment_split
from src.data.transforms import ArcsinhScaler
from src.data.weather_preprocessor import COVARIATE_COUNT, WeatherPreprocessor


class _ColumnMoments:
    """Merge per-tile population moments in float64."""

    def __init__(self, columns: int):
        self.count = np.zeros(columns, dtype=np.int64)
        self.mean = np.zeros(columns, dtype=np.float64)
        self.m2 = np.zeros(columns, dtype=np.float64)

    def update(self, values: np.ndarray, available: np.ndarray):
        count = available.sum(axis=0)
        values = np.where(available, values, 0).astype(np.float64)
        mean = values.sum(axis=0) / np.maximum(count, 1)
        m2 = np.square(np.where(available, values - mean, 0)).sum(axis=0)
        total_count = self.count + count
        delta = mean - self.mean
        weight = count / np.maximum(total_count, 1)
        self.m2 += m2 + delta ** 2 * self.count * weight
        self.mean += delta * weight
        self.count = total_count

    def variance(self) -> np.ndarray:
        if np.any(self.count == 0):
            raise ValueError(f"Fitting split has no usable values for columns {np.flatnonzero(self.count == 0)}")
        return self.m2 / self.count


def fit_arcsinh_scaler(
    fit_files: Sequence[Path], validation_files: Sequence[Path], weather: WeatherPreprocessor, seed: int
) -> ArcsinhScaler:
    """Fit the 21 historical features and three target labels on fitting tiles only."""
    provenance = _fit_provenance(fit_files, validation_files, weather, seed)
    features = _ColumnMoments(21)
    targets = _ColumnMoments(3)

    for path in sorted(fit_files):
        with np.load(path, allow_pickle=True) as tile:
            history = np.concatenate(tile["history"]).astype(np.float32)
            missing = np.concatenate(tile["history_mask"]).astype(bool)
            _, _, usable_covariates = weather.availability(history, missing)
            available = np.isfinite(history) & ~missing
            available[:, :COVARIATE_COUNT] = usable_covariates
            features.update(history, available)

            labels = np.asarray(tile["target"], dtype=np.float32).reshape(-1, 3)
            usable_labels = np.isfinite(labels)
            if "target_mask" in tile:
                usable_labels &= ~np.asarray(tile["target_mask"], dtype=bool).reshape(-1, 3)
            targets.update(labels, usable_labels)

    return ArcsinhScaler({
        "mean": features.mean,
        "var": features.variance(),
        "target_mean": targets.mean,
        "target_var": targets.variance(),
        "provenance": {
            **provenance,
            "feature_counts": features.count.tolist(),
            "target_counts": targets.count.tolist(),
        },
    })


def _fit_provenance(
    fit_files: Sequence[Path], validation_files: Sequence[Path], weather: WeatherPreprocessor, seed: int
) -> dict:
    fit_names = sorted(path.name for path in fit_files)
    validation_names = sorted(path.name for path in validation_files)
    if not fit_names or len(fit_names) != len(set(fit_names)) or set(fit_names) & set(validation_names):
        raise ValueError("Fitting tiles must be nonempty, unique and disjoint from validation")
    if len(validation_names) != len(set(validation_names)):
        raise ValueError("Validation tiles must be unique")
    return {
        "fit_tiles": fit_names,
        "validation_tiles": validation_names,
        "split_seed": seed,
        "historical_qc": weather.historical_qc,
    }


def resolve_experiment_scaler(
    cfg: dict,
    weather: WeatherPreprocessor,
    fit_files: Sequence[Path] | None = None,
    validation_files: Sequence[Path] | None = None,
) -> ArcsinhScaler:
    paths = cfg["paths"]
    mode = cfg.get("scaling", {}).get("mode", "distributed")
    if mode == "distributed":
        return ArcsinhScaler(paths["scaler_path"])
    if mode != "fitting_tiles":
        raise ValueError(f"Unknown scaling mode: {mode!r}")
    if fit_files is None or validation_files is None:
        fit_files, validation_files, _ = get_experiment_split(cfg)
    seed = cfg["data"].get("seed", 42)
    fitted_path = paths.get("fitted_scaler_path")
    if fitted_path is not None:
        scaler = ArcsinhScaler(fitted_path)
        expected = _fit_provenance(fit_files, validation_files, weather, seed)
        if any(scaler.provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Scaler at {fitted_path} belongs to a different split or cleaning policy")
        return scaler
    return fit_arcsinh_scaler(fit_files, validation_files, weather, seed)


def experiment_checkpoint_path(cfg: dict, weather: WeatherPreprocessor, path: str | Path) -> Path:
    """Keep fitted-scaler weights separate from distributed-scaler checkpoints."""
    path = weather.imputer_checkpoint_path(path)
    if cfg.get("scaling", {}).get("mode") == "fitting_tiles":
        path = path.with_stem(f"{path.stem}_fit")
    return path
