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
from src.data.interpolation import InterpolationFactory, BaseInterpolator


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
        loo_all_points: bool = False,
    ):
        """
        loo_all_points=True builds a deterministic leave-one-out evaluation set: every window
        is repeated once per clear-sky observation, and in repetition r the r-th observation
        is hidden. Requires mask_dropout=True. Windows with < 2 observations are skipped.
        """
        super().__init__()
        if loo_all_points and not mask_dropout:
            raise ValueError("loo_all_points=True requires mask_dropout=True")
        self.scaler = scaler
        self.loo_all_points = loo_all_points
        self.drop_ranks: Optional[List[int]] = [] if loo_all_points else None
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

            norm_weather_list = []
            norm_targets_list = []
            obs_mask_list = []
            valid_indices_list = []
            climate_ids_arr = np.empty(num_samples, dtype=np.int64)

            for s_idx in range(num_samples):
                h_s = histories[s_idx].astype(np.float32)
                m_s = masks[s_idx]
                nw = self.scaler.transform_features(h_s)[:, :18]
                nw = np.nan_to_num(nw, nan=0.0).astype(np.float32, copy=False)
                nt = self.scaler.transform_targets(h_s[:, 18:21]).astype(np.float32, copy=False)
                om = (~m_s[:, 18]).astype(bool, copy=False)
                vi = np.where(om)[0].astype(np.int64, copy=False)

                norm_weather_list.append(nw)
                norm_targets_list.append(nt)
                obs_mask_list.append(om)
                valid_indices_list.append(vi)
                climate_ids_arr[s_idx] = encode_climate(str(climates[s_idx]))

            self.file_cache[fpath] = {
                "history": histories,
                "history_mask": masks,
                "climate": climates,
                "norm_weather": norm_weather_list,
                "norm_targets": norm_targets_list,
                "obs_mask": obs_mask_list,
                "valid_indices": valid_indices_list,
                "climate_ids": climate_ids_arr,
            }

            for s_idx in range(num_samples):
                if loo_all_points:
                    n_valid = len(valid_indices_list[s_idx])
                    if n_valid >= 2:
                        for rank in range(n_valid):
                            self.samples.append((fpath, s_idx))
                            self.drop_ranks.append(rank)
                else:
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
        idx = np.arange(L, dtype=np.float32)
        orbital_phase = (idx % 5.0) / 5.0
        obs_pos = np.where(obs_mask, idx, -1.0)
        last_obs = np.maximum.accumulate(obs_pos)
        has_seen = last_obs >= 0.0
        delta_t = np.where(has_seen & (~obs_mask), idx - last_obs, 0.0).astype(np.float32)
        missed_revisits = np.floor(delta_t / 5.0).astype(np.float32)
        return delta_t, orbital_phase, missed_revisits

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, s_idx = self.samples[idx]
        cached = self.file_cache[fpath]

        norm_weather_calendar = cached["norm_weather"][s_idx]
        norm_targets = cached["norm_targets"][s_idx]
        obs_mask = cached["obs_mask"][s_idx]
        valid_indices = cached["valid_indices"][s_idx]
        climate_id = int(cached["climate_ids"][s_idx])
        raw_climate = str(cached["climate"][s_idx])

        L = norm_weather_calendar.shape[0]

        # 2. Self-Supervised Mask Dropout (for training reconstruction)
        input_obs_mask = obs_mask.copy()
        eval_supervision_mask = np.zeros(L, dtype=bool)
        drop_rank = -1

        if self.mask_dropout and len(valid_indices) >= 2:
            if self.drop_ranks is not None:
                drop_rank = int(self.drop_ranks[idx])
                drop_idx = valid_indices[drop_rank]
            else:
                drop_rank = int(self.rng.randint(len(valid_indices)))
                drop_idx = valid_indices[drop_rank]
            input_obs_mask[drop_idx] = False
            eval_supervision_mask[drop_idx] = True
        else:
            eval_supervision_mask = obs_mask.copy()

        # 3. Climate tokenization & Climate Dropout (computed before optional interpolator)
        effective_climate_str = raw_climate
        if self.climate_dropout_p > 0.0 and self.rng.rand() < self.climate_dropout_p:
            climate_id = 0  # Force to <UNKNOWN> to train robustness for OOD-S
            effective_climate_str = "<UNKNOWN>"

        # 4. Handle NaNs in input target / Apply Interpolation Strategy (never leak masked point)
        if self.interpolator is not None:
            h_raw = cached["history"][s_idx].astype(np.float32)
            m_raw = cached["history_mask"][s_idx]
            effective_mask = m_raw.copy()
            effective_mask[:, 18:21] = ~input_obs_mask[:, None]
            h_for_interp = h_raw.copy()
            h_for_interp[~input_obs_mask, 18:21] = np.nan

            if hasattr(self.interpolator, "interpolate_history_sample"):
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
            "drop_rank": torch.tensor(drop_rank, dtype=torch.long),
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
    drop_ranks = torch.full((batch_size,), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        padded_input[i, :L] = item["input"]
        padded_target[i, :L] = item["target"]
        padded_loss_mask[i, :L] = item["loss_mask"]
        padded_obs_mask[i, :L] = item["all_obs_mask"]
        padding_mask[i, :L] = True
        climate_ids[i] = item["climate_id"]
        if "drop_rank" in item:
            drop_ranks[i] = item["drop_rank"]

    return {
        "input": padded_input,
        "climate_id": climate_ids,
        "target": padded_target,
        "loss_mask": padded_loss_mask,
        "obs_mask": padded_obs_mask,
        "padding_mask": padding_mask,
        "drop_ranks": drop_ranks,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
