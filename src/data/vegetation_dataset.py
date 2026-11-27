"""
Complete Dataset and DataLoader for the Vegetation Forecasting task.
Integrates InterpolationFactory to remove missing values from the historical time series.
Produces aligned historical features, future weather covariates, and future multi-step targets.
"""

from pathlib import Path
from typing import List, Optional, Dict, Union, Sequence
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.transforms import ArcsinhScaler
from src.data.climate import encode_climate
from src.data.interpolation import InterpolationFactory, BaseInterpolator, build_interpolator_from_config
from src.data.scaler_fit import resolve_experiment_scaler
from src.data.weather_preprocessor import WeatherPreprocessor

WEATHER_MASK_KEYS = (
    "history_weather_original_missing_mask",
    "history_weather_invalid_mask",
    "history_weather_interpolated_mask",
    "history_weather_mask",
    "future_weather_original_missing_mask",
    "future_weather_mask",
)


class VegetationDataset(Dataset):
    """
    Primary Dataset for Vegetation Forecasting (3 past clear-sky steps -> 3 future clear-sky steps).
    Uses InterpolationFactory to impute / interpolate missing values in the history sequence.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        scaler: ArcsinhScaler,
        file_pattern: str = "*.npz",
        max_files: Optional[int] = None,
        file_list: Optional[List[Path]] = None,
        interpolation_strategy: Union[str, BaseInterpolator] = "linear",
        interpolation_columns: Sequence[int] = (18, 19, 20),
        interpolate_weather: bool = True,
        interpolation_kwargs: Optional[Dict] = None,
        precompute_model_imputation: bool = True,
        imputation_batch_size: int = 128,
        climate_dropout_p: float = 0.0,
        seed: int = 42,
        weather_preprocessor: Optional[WeatherPreprocessor] = None,
    ):
        super().__init__()
        self.scaler = scaler
        self.interpolation_columns = list(interpolation_columns)
        if not set(self.interpolation_columns).issubset((18, 19, 20)):
            raise ValueError("interpolation_columns must select historical vegetation indices")
        self.interpolate_weather = interpolate_weather
        self.weather_preprocessor = weather_preprocessor or WeatherPreprocessor()
        self.precompute_model_imputation = precompute_model_imputation
        self.imputation_batch_size = imputation_batch_size
        self.climate_dropout_p = climate_dropout_p
        self.rng = np.random.RandomState(seed)

        # Build interpolator from factory
        if interpolation_strategy is not None and str(interpolation_strategy).lower() != "none":
            self.interpolator = InterpolationFactory.create(
                interpolation_strategy, **(interpolation_kwargs or {})
            )
        else:
            self.interpolator = InterpolationFactory.create("none")

        data_dir = Path(data_dir)
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted(list(data_dir.glob(file_pattern)))
            if max_files:
                self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        # Index samples across files
        self.samples = []
        self.file_cache: Dict[Path, Dict[str, np.ndarray]] = {}

        use_batch_precompute = (
            self.precompute_model_imputation
            and getattr(self.interpolator, "supports_batch_history_imputation", False)
        )

        for fpath in self.files:
            with np.load(fpath, allow_pickle=True) as d:
                histories = d["history"]
                hist_masks = d["history_mask"]
                futures = d["future"]
                fut_masks = d["future_mask"]
                targets = d["target"]
                target_masks = d["target_mask"] if "target_mask" in d else None
                target_pos = d["future_target_positions"]
                climates = d["climate"] if "climate" in d else np.array(["unknown"] * len(histories))

            num_samples = len(histories)
            cache_entry = {
                "history": histories,
                "history_mask": hist_masks,
                "future": futures,
                "future_mask": fut_masks,
                "target": targets,
                "target_mask": target_masks,
                "target_pos": target_pos,
                "climate": climates,
            }

            if use_batch_precompute and num_samples > 0:
                imputed_list = self.interpolator.interpolate_history_batch(
                    list(histories),
                    list(hist_masks),
                    [str(c) for c in climates],
                    batch_size=self.imputation_batch_size,
                )
                cache_entry["history_imputed"] = imputed_list

            self.file_cache[fpath] = cache_entry

            for s_idx in range(num_samples):
                self.samples.append((fpath, s_idx))

    @classmethod
    def from_config(
        cls,
        cfg: Dict,
        *,
        data_dir: Optional[Union[str, Path]] = None,
        file_list: Optional[List[Path]] = None,
        max_files: Optional[int] = None,
        strict_checkpoint: bool = False,
    ) -> "VegetationDataset":
        """Build the loader from the project config, defaulting to the official train."""
        paths = cfg["paths"]
        interpolation = cfg.get("interpolation", {})
        weather = WeatherPreprocessor(**cfg.get("weather_preprocessing", {}))
        return cls(
            data_dir if data_dir is not None else paths["train_data_dir"],
            resolve_experiment_scaler(cfg, weather),
            file_list=file_list,
            max_files=max_files,
            interpolation_strategy=build_interpolator_from_config(cfg, strict_checkpoint=strict_checkpoint),
            interpolation_columns=interpolation.get("target_columns", (18, 19, 20)),
            interpolate_weather=interpolation.get("interpolate_weather", True),
            seed=cfg.get("data", {}).get("seed", 42),
            weather_preprocessor=weather,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, s_idx = self.samples[idx]
        cached = self.file_cache[fpath]

        h_raw = cached["history"][s_idx].astype(np.float32)       # (L_h, 21)
        m_raw = cached["history_mask"][s_idx].astype(bool)        # (L_h, 21)
        f_raw = cached["future"][s_idx].astype(np.float32)        # (L_f, 21)
        f_mask = cached["future_mask"][s_idx]
        tgt_raw = cached["target"][s_idx].astype(np.float32)      # (3, 3)
        target_mask = ~np.isfinite(tgt_raw)
        if cached["target_mask"] is not None:
            target_mask |= cached["target_mask"][s_idx].astype(bool)
        tgt_pos = cached["target_pos"][s_idx].astype(np.int64)    # (3,)
        raw_climate = str(cached["climate"][s_idx])

        L_h = h_raw.shape[0]
        L_f = f_raw.shape[0]

        # 1. Apply Interpolation Strategy to remove missing values from history targets
        if "history_imputed" in cached:
            h_clean = cached["history_imputed"][s_idx].astype(np.float32).copy()
        elif hasattr(self.interpolator, "interpolate_history_sample"):
            h_clean = self.interpolator.interpolate_history_sample(h_raw, m_raw, raw_climate)
        else:
            h_clean = self.interpolator.interpolate(
                h_raw, mask=m_raw, columns=self.interpolation_columns
            )

        # 2. Weather preprocessing is independent of the vegetation interpolator.
        h_weather = self.weather_preprocessor.transform(
            h_raw, m_raw, self.scaler, interpolate_history=self.interpolate_weather
        )
        f_weather = self.weather_preprocessor.transform(f_raw, f_mask, self.scaler, future=True)
        norm_h_targets = self.scaler.transform_targets(h_clean[:, 18:21])
        norm_h_targets = np.nan_to_num(norm_h_targets, nan=0.0)
        history_parts = [h_weather.values, norm_h_targets]
        future_weather = f_weather.values
        if self.weather_preprocessor.uses_masks:
            history_parts.append(h_weather.weather_mask)
            future_weather = np.concatenate([future_weather, f_weather.weather_mask], axis=-1)
        norm_history = np.concatenate(history_parts, axis=-1)

        # Future Target labels (3 steps, 3 indices: NDVI, GNDVI, EVI)
        norm_target = self.scaler.transform_targets(np.where(target_mask, np.nan, tgt_raw))

        # 3. Climate tokenization
        climate_id = encode_climate(raw_climate)
        if self.climate_dropout_p > 0.0 and self.rng.rand() < self.climate_dropout_p:
            climate_id = 0  # Force to <UNKNOWN>

        # Observation mask for historical clear-sky dates
        obs_mask = ~m_raw[:, 18]

        return {
            "history": torch.from_numpy(norm_history).float(),
            "future_weather": torch.from_numpy(future_weather).float(),
            "history_weather_original_missing_mask": torch.from_numpy(h_weather.original_missing),
            "history_weather_invalid_mask": torch.from_numpy(h_weather.invalid),
            "history_weather_interpolated_mask": torch.from_numpy(h_weather.interpolated),
            "history_weather_mask": torch.from_numpy(h_weather.weather_mask),
            "future_weather_original_missing_mask": torch.from_numpy(f_weather.original_missing),
            "future_weather_mask": torch.from_numpy(f_weather.weather_mask),
            "target": torch.from_numpy(norm_target).float(),
            "target_mask": torch.from_numpy(target_mask),
            "target_positions": torch.from_numpy(tgt_pos).long(),
            "climate_id": torch.tensor(climate_id, dtype=torch.long),
            "history_obs_mask": torch.from_numpy(obs_mask).bool(),
            "history_length": L_h,
            "future_length": L_f,
            "raw_target": torch.from_numpy(tgt_raw).float(),
        }


def collate_vegetation_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate and pad variable-length history and future weather sequences.
    """
    batch_size = len(batch)
    hist_lens = [b["history_length"] for b in batch]
    fut_lens = [b["future_length"] for b in batch]

    max_h_len = max(hist_lens)
    max_f_len = max(fut_lens)
    h_dim = batch[0]["history"].shape[-1]
    f_dim = batch[0]["future_weather"].shape[-1]

    padded_hist = torch.zeros(batch_size, max_h_len, h_dim, dtype=torch.float32)
    hist_padding_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool)
    padded_hist_obs_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool)
    padded_fut = torch.zeros(batch_size, max_f_len, f_dim, dtype=torch.float32)
    fut_padding_mask = torch.zeros(batch_size, max_f_len, dtype=torch.bool)

    targets = torch.stack([b["target"] for b in batch], dim=0)
    target_masks = torch.stack([b.get("target_mask", ~torch.isfinite(b["target"])) for b in batch])
    raw_targets = torch.stack([b["raw_target"] for b in batch], dim=0)
    target_pos = torch.stack([b["target_positions"] for b in batch], dim=0)
    climate_ids = torch.tensor([b["climate_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        lh = b["history_length"]
        lf = b["future_length"]
        padded_hist[i, :lh] = b["history"]
        hist_padding_mask[i, :lh] = True
        padded_hist_obs_mask[i, :lh] = b["history_obs_mask"]
        padded_fut[i, :lf] = b["future_weather"]
        fut_padding_mask[i, :lf] = True

    return {
        "history": padded_hist,
        "history_padding_mask": hist_padding_mask,
        "history_obs_mask": padded_hist_obs_mask,
        "future_weather": padded_fut,
        "future_padding_mask": fut_padding_mask,
        "target": targets,
        "target_mask": target_masks,
        "raw_target": raw_targets,
        "target_positions": target_pos,
        "climate_id": climate_ids,
        "history_lengths": torch.tensor(hist_lens, dtype=torch.long),
        "future_lengths": torch.tensor(fut_lens, dtype=torch.long),
        **{
            key: pad_sequence([b[key] for b in batch], batch_first=True)
            for key in WEATHER_MASK_KEYS
            if key in batch[0]
        },
    }
