"""
Dataset and DataLoader utilities for the History (Past) Sequence.
Implements:
1. Tile-based loading (zero spatial data leakage).
2. ArcsinhScaler normalization.
3. Sentinel-2 5-day orbital revisit cycle dynamics (orbital phase, missed revisits, delta_t).
4. Safe Climate encoding with Climate Dropout for 'unknown' handling.
5. Self-Supervised Masking for reconstruction training.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import ArcsinhScaler
from src.data.climate import encode_climate, NUM_CLIMATES
from src.data.interpolation import InterpolationFactory, BaseInterpolator, ModelInterpolator


class HistoryImputationDataset(Dataset):
    """
    Dataset for learning to impute and reconstruct the missing target days in the past (history).
    Incorporates Sentinel-2 orbital physics (5-day cadence) and climate tokenization.
    Optionally supports plug-and-play interpolation strategies via InterpolationFactory.
    """

    def __init__(
        self,
        data_dir: str or Path,
        scaler: ArcsinhScaler,
        file_pattern: str = "*.npz",
        max_files: Optional[int] = None,
        file_list: Optional[List[Path]] = None,
        mask_dropout: bool = True,
        climate_dropout_p: float = 0.35,
        interpolation_strategy: Optional[Union[str, BaseInterpolator]] = None,
        interpolation_columns: Optional[Sequence[int]] = (18, 19, 20),
        interpolation_kwargs: Optional[Dict] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.scaler = scaler
        self.mask_dropout = mask_dropout
        self.climate_dropout_p = climate_dropout_p
        self.interpolation_columns = interpolation_columns
        self.rng = np.random.RandomState(seed)

        if interpolation_strategy is not None and str(interpolation_strategy).lower() != "none":
            self.interpolator = InterpolationFactory.create(
                interpolation_strategy, **(interpolation_kwargs or {})
            )
        else:
            self.interpolator = None

        data_dir = Path(data_dir)
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted(list(data_dir.glob(file_pattern)))
            if max_files:
                self.files = self.files[:max_files]

        # Preload / index samples across files
        self.samples = []
        self.file_cache: Dict[Path, Dict[str, np.ndarray]] = {}

        for fpath in self.files:
            d = np.load(fpath, allow_pickle=True)
            histories = d["history"]
            masks = d["history_mask"]
            climates = d["climate"] if "climate" in d else np.array(["unknown"] * len(histories))
            num_samples = len(histories)

            self.file_cache[fpath] = {
                "history": histories,
                "history_mask": masks,
                "climate": climates,
            }

            for s_idx in range(num_samples):
                self.samples.append((fpath, s_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_orbital_dynamics(self, obs_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes temporal dynamics aligned with the 5-day Sentinel-2 orbital revisit cycle:
        1. delta_t: elapsed days since last observation
        2. orbital_phase: position within the 5-day orbital grid (t % 5) / 5.0
        3. missed_revisits: number of 5-day revisit opportunities missed due to clouds
        """
        L = len(obs_mask)
        delta_t = np.zeros(L, dtype=np.float32)
        orbital_phase = np.zeros(L, dtype=np.float32)
        missed_revisits = np.zeros(L, dtype=np.float32)

        last_obs = 0.0
        has_seen_first = False

        for t in range(L):
            orbital_phase[t] = (t % 5) / 5.0

            if obs_mask[t]:
                delta_t[t] = 0.0
                missed_revisits[t] = 0.0
                last_obs = float(t)
                has_seen_first = True
            else:
                if has_seen_first:
                    elapsed = float(t) - last_obs
                    delta_t[t] = elapsed
                    # Each 5 days represents one missed satellite pass
                    missed_revisits[t] = float(np.floor(elapsed / 5.0))
                else:
                    delta_t[t] = 0.0
                    missed_revisits[t] = 0.0

        return delta_t, orbital_phase, missed_revisits

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, s_idx = self.samples[idx]
        cached = self.file_cache[fpath]
        h_raw = cached["history"][s_idx].astype(np.float32)  # shape (L, 21)
        m_raw = cached["history_mask"][s_idx]               # shape (L, 21), True where missing
        raw_climate = str(cached["climate"][s_idx])

        L = h_raw.shape[0]

        # 1. Feature normalization with ArcsinhScaler
        norm_weather_calendar = self.scaler.transform_features(h_raw)[:, :18]
        norm_weather_calendar = np.nan_to_num(norm_weather_calendar, nan=0.0)

        # Targets 18..20 (NDVI, GNDVI, EVI)
        targets_raw = h_raw[:, 18:21]
        norm_targets = self.scaler.transform_targets(targets_raw)

        # Boolean mask of valid observations
        obs_mask = ~m_raw[:, 18]  # shape (L,), True where observed (the 3 clear-sky dates)
        valid_indices = np.where(obs_mask)[0]

        # 2. Self-Supervised Mask Dropout (for training reconstruction)
        input_obs_mask = obs_mask.copy()
        eval_supervision_mask = np.zeros(L, dtype=bool)

        if self.mask_dropout and len(valid_indices) >= 2:
            drop_idx = self.rng.choice(valid_indices)
            input_obs_mask[drop_idx] = False
            eval_supervision_mask[drop_idx] = True
        else:
            eval_supervision_mask = obs_mask.copy()

        # 3. Climate tokenization & Climate Dropout (computed before optional interpolator)
        climate_id = encode_climate(raw_climate)
        effective_climate_str = raw_climate
        if self.climate_dropout_p > 0.0 and self.rng.rand() < self.climate_dropout_p:
            climate_id = 0  # Force to <UNKNOWN> to train robustness for OOD-S
            effective_climate_str = "<UNKNOWN>"

        # 4. Handle NaNs in input target / Apply Interpolation Strategy (never leak masked point)
        if self.interpolator is not None:
            effective_mask = m_raw.copy()
            effective_mask[:, 18:21] = ~input_obs_mask[:, None]
            h_for_interp = h_raw.copy()
            h_for_interp[~input_obs_mask, 18:21] = np.nan

            if isinstance(self.interpolator, ModelInterpolator):
                h_interp = self.interpolator.interpolate_history_sample(
                    h_for_interp, effective_mask, effective_climate_str
                )
            else:
                h_interp = self.interpolator.interpolate(
                    h_for_interp, mask=effective_mask, columns=self.interpolation_columns
                )
            input_targets = self.scaler.transform_targets(h_interp[:, 18:21])
            input_targets = np.nan_to_num(input_targets, nan=0.0)
        else:
            input_targets = norm_targets.copy()
            input_targets[~input_obs_mask] = 0.0
            input_targets = np.nan_to_num(input_targets, nan=0.0)

        # Mask feature M_t (1.0 if observed in input, 0.0 otherwise)
        mask_feature = np.zeros((L, 3), dtype=np.float32)
        mask_feature[input_obs_mask] = 1.0

        # 5. Sentinel-2 5-day Orbital dynamics features
        delta_t, orbital_phase, missed_revisits = self._compute_orbital_dynamics(input_obs_mask)
        orbital_feats = np.stack([delta_t, orbital_phase, missed_revisits], axis=-1)  # (L, 3)

        # Concatenate full input vector:
        # [weather_calendar (18), input_targets (3), mask_feature (3), orbital_feats (3)] -> total 27 dims
        model_input = np.concatenate([norm_weather_calendar, input_targets, mask_feature, orbital_feats], axis=-1)
        model_input = np.nan_to_num(model_input, nan=0.0)

        clean_norm_targets = np.nan_to_num(norm_targets, nan=0.0)

        return {
            "input": torch.from_numpy(model_input).float(),
            "climate_id": torch.tensor(climate_id, dtype=torch.long),
            "target": torch.from_numpy(clean_norm_targets).float(),
            "loss_mask": torch.from_numpy(eval_supervision_mask).bool(),
            "all_obs_mask": torch.from_numpy(obs_mask).bool(),
            "length": L,
        }


def collate_history_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad variable-length history sequences in a batch.
    """
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)
    feat_dim = batch[0]["input"].shape[-1]
    target_dim = batch[0]["target"].shape[-1]

    padded_input = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    padded_target = torch.zeros(batch_size, max_len, target_dim, dtype=torch.float32)
    padded_loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_obs_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    climate_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        padded_input[i, :L] = item["input"]
        padded_target[i, :L] = item["target"]
        padded_loss_mask[i, :L] = item["loss_mask"]
        padded_obs_mask[i, :L] = item["all_obs_mask"]
        padding_mask[i, :L] = True
        climate_ids[i] = item["climate_id"]

    return {
        "input": padded_input,
        "climate_id": climate_ids,
        "target": padded_target,
        "loss_mask": padded_loss_mask,
        "obs_mask": padded_obs_mask,
        "padding_mask": padding_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
