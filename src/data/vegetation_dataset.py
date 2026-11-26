"""
Complete Dataset and DataLoader for the Vegetation Forecasting task.
Integrates InterpolationFactory to remove missing values from the historical time series.
Produces aligned historical features, future weather covariates, and future multi-step targets.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import ArcsinhScaler
from src.data.climate import encode_climate, NUM_CLIMATES
from src.data.interpolation import InterpolationFactory, BaseInterpolator, ModelInterpolator


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
        climate_dropout_p: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.scaler = scaler
        self.interpolation_columns = interpolation_columns
        self.interpolate_weather = interpolate_weather
        self.climate_dropout_p = climate_dropout_p
        self.rng = np.random.RandomState(seed)

        # Build interpolator from factory
        if interpolation_strategy is not None and str(interpolation_strategy).lower() != "none":
            self.interpolator = InterpolationFactory.create(interpolation_strategy)
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

        for fpath in self.files:
            d = np.load(fpath, allow_pickle=True)
            histories = d["history"]
            hist_masks = d["history_mask"]
            futures = d["future"]
            fut_masks = d["future_mask"]
            targets = d["target"]
            target_pos = d["future_target_positions"]
            climates = d["climate"] if "climate" in d else np.array(["unknown"] * len(histories))
            fut_noise = d["future_noise"] if "future_noise" in d else None

            num_samples = len(histories)
            self.file_cache[fpath] = {
                "history": histories,
                "history_mask": hist_masks,
                "future": futures,
                "future_mask": fut_masks,
                "target": targets,
                "target_pos": target_pos,
                "climate": climates,
                "future_noise": fut_noise,
            }

            for s_idx in range(num_samples):
                self.samples.append((fpath, s_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, s_idx = self.samples[idx]
        cached = self.file_cache[fpath]

        h_raw = cached["history"][s_idx].astype(np.float32)       # (L_h, 21)
        m_raw = cached["history_mask"][s_idx].astype(bool)        # (L_h, 21)
        f_raw = cached["future"][s_idx].astype(np.float32)        # (L_f, 21)
        tgt_raw = cached["target"][s_idx].astype(np.float32)      # (3, 3)
        tgt_pos = cached["target_pos"][s_idx].astype(np.int64)    # (3,)
        raw_climate = str(cached["climate"][s_idx])

        L_h = h_raw.shape[0]
        L_f = f_raw.shape[0]

        # 1. Apply Interpolation Strategy to remove missing values from history
        if isinstance(self.interpolator, ModelInterpolator):
            h_clean = self.interpolator.interpolate_history_sample(h_raw, m_raw, raw_climate)
        else:
            cols = list(self.interpolation_columns)
            if self.interpolate_weather:
                # Also interpolate any occasional missing weather values in 0..17
                weather_missing_cols = [c for c in range(18) if np.any(m_raw[:, c])]
                cols = sorted(list(set(cols + weather_missing_cols)))

            h_clean = self.interpolator.interpolate(h_raw, mask=m_raw, columns=cols)

        # Fallback safeguard against any remaining NaNs
        h_clean = np.nan_to_num(h_clean, nan=0.0)

        # 2. Arcsinh Normalization
        # History weather + targets
        norm_h_features = self.scaler.transform_features(h_clean)[:, :18]
        norm_h_targets = self.scaler.transform_targets(h_clean[:, 18:21])
        norm_history = np.concatenate([norm_h_features, norm_h_targets], axis=-1)  # (L_h, 21)
        norm_history = np.nan_to_num(norm_history, nan=0.0)

        # Future weather (columns 0..17)
        norm_future_weather = self.scaler.transform_features(f_raw)[:, :18]
        norm_future_weather = np.nan_to_num(norm_future_weather, nan=0.0)

        # Future Target labels (3 steps, 3 indices: NDVI, GNDVI, EVI)
        norm_target = self.scaler.transform_targets(tgt_raw)
        norm_target = np.nan_to_num(norm_target, nan=0.0)

        # 3. Climate tokenization
        climate_id = encode_climate(raw_climate)
        if self.climate_dropout_p > 0.0 and self.rng.rand() < self.climate_dropout_p:
            climate_id = 0  # Force to <UNKNOWN>

        # Observation mask for historical clear-sky dates
        obs_mask = ~m_raw[:, 18]

        return {
            "history": torch.from_numpy(norm_history).float(),
            "future_weather": torch.from_numpy(norm_future_weather).float(),
            "target": torch.from_numpy(norm_target).float(),
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
    padded_fut = torch.zeros(batch_size, max_f_len, f_dim, dtype=torch.float32)
    fut_padding_mask = torch.zeros(batch_size, max_f_len, dtype=torch.bool)

    targets = torch.stack([b["target"] for b in batch], dim=0)
    raw_targets = torch.stack([b["raw_target"] for b in batch], dim=0)
    target_pos = torch.stack([b["target_positions"] for b in batch], dim=0)
    climate_ids = torch.tensor([b["climate_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        lh = b["history_length"]
        lf = b["future_length"]
        padded_hist[i, :lh] = b["history"]
        hist_padding_mask[i, :lh] = True
        padded_fut[i, :lf] = b["future_weather"]
        fut_padding_mask[i, :lf] = True

    return {
        "history": padded_hist,
        "history_padding_mask": hist_padding_mask,
        "future_weather": padded_fut,
        "future_padding_mask": fut_padding_mask,
        "target": targets,
        "raw_target": raw_targets,
        "target_positions": target_pos,
        "climate_id": climate_ids,
        "history_lengths": torch.tensor(hist_lens, dtype=torch.long),
        "future_lengths": torch.tensor(fut_lens, dtype=torch.long),
    }
