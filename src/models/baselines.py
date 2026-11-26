"""
Baseline forecasting models for Vegetation Health Forecasting:
1. PersistenceBaseline: deterministic last-observation carried forward across all steps & quantiles.
2. ProbabilisticPersistenceBaseline: persistence median + empirical residual quantiles fitted strictly on train.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from src.metrics import DEFAULT_QUANTILES


def extract_clear_sky_history(
    histories: Sequence[np.ndarray],
    history_masks: Sequence[np.ndarray],
    target_cols: Sequence[int] = (18, 19, 20),
    num_obs: int = 3,
) -> np.ndarray:
    """
    Extracts the clear-sky observed vegetation indices from variable-length history arrays.
    Returns shape (N, num_obs, len(target_cols)) on original scale.
    If a window has fewer than `num_obs` valid observations, left-pads with NaNs.
    If a window has more than `num_obs`, takes the most recent `num_obs` observations.
    """
    n_samples = len(histories)
    n_targets = len(target_cols)
    out = np.full((n_samples, num_obs, n_targets), np.nan, dtype=np.float32)

    for i in range(n_samples):
        h = np.asarray(histories[i], dtype=np.float32)
        m = np.asarray(history_masks[i], dtype=bool)
        obs_rows = (~m[:, target_cols[0]]) & np.all(np.isfinite(h[:, target_cols]), axis=-1)
        obs_vals = h[obs_rows][:, target_cols]

        if len(obs_vals) == 0:
            continue
        if len(obs_vals) >= num_obs:
            out[i] = obs_vals[-num_obs:]
        else:
            out[i, -len(obs_vals):] = obs_vals

    return out


def extract_last_clear_sky(
    histories: Sequence[np.ndarray],
    history_masks: Sequence[np.ndarray],
    target_cols: Sequence[int] = (18, 19, 20),
) -> np.ndarray:
    """
    Extracts the most recent clear-sky observed target values (N, C) per window.
    Falls back column-by-column to the latest finite target value if joint mask is empty.
    """
    n_samples = len(histories)
    n_targets = len(target_cols)
    out = np.zeros((n_samples, n_targets), dtype=np.float32)

    for i in range(n_samples):
        h = np.asarray(histories[i], dtype=np.float32)
        m = np.asarray(history_masks[i], dtype=bool)
        obs_rows = (~m[:, target_cols[0]]) & np.all(np.isfinite(h[:, target_cols]), axis=-1)
        if np.any(obs_rows):
            out[i] = h[obs_rows][-1, target_cols]
        else:
            for c_idx, col in enumerate(target_cols):
                valid_col = (~m[:, col]) & np.isfinite(h[:, col])
                if np.any(valid_col):
                    out[i, c_idx] = h[valid_col, col][-1]
                else:
                    out[i, c_idx] = 0.0

    return out


class PersistenceBaseline:
    """
    Naive persistence baseline: repeats the last clear-sky observation
    across all future steps and all quantiles (0.1, 0.5, 0.9).
    """

    def __init__(
        self,
        future_steps: int = 3,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ):
        self.future_steps = future_steps
        self.quantiles = tuple(quantiles)

    def fit(self, *args, **kwargs) -> "PersistenceBaseline":
        return self

    def predict(
        self,
        last_obs: np.ndarray,
        clip_min: Optional[float] = -1.0,
        clip_max: Optional[float] = 1.0,
    ) -> np.ndarray:
        """
        Args:
            last_obs: (N, C) last observed clear-sky target values on original scale.
        Returns:
            predictions of shape (N, future_steps, C, len(quantiles)).
        """
        obs = np.asarray(last_obs, dtype=np.float32)
        n, c = obs.shape
        preds = np.broadcast_to(
            obs[:, None, :, None],
            (n, self.future_steps, c, len(self.quantiles)),
        ).copy()
        if clip_min is not None or clip_max is not None:
            preds = np.clip(preds, a_min=clip_min, a_max=clip_max)
        return preds


class ProbabilisticPersistenceBaseline:
    """
    Residual-quantile calibrated persistence baseline:
        y_hat(s, c, q) = last_obs(c) + Quantile_q(y_train(s, c) - last_obs_train(c))
    Must be fitted strictly on the training split.
    """

    def __init__(
        self,
        future_steps: int = 3,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ):
        self.future_steps = future_steps
        self.quantiles = tuple(quantiles)
        self.residual_quantiles_: Optional[np.ndarray] = None

    def fit(
        self,
        last_obs_train: np.ndarray,
        targets_train: np.ndarray,
        valid_mask_train: Optional[np.ndarray] = None,
        quantiles: Optional[Sequence[float]] = None,
    ) -> "ProbabilisticPersistenceBaseline":
        if quantiles is not None:
            self.quantiles = tuple(quantiles)

        last_obs = np.asarray(last_obs_train, dtype=np.float64)
        targets = np.asarray(targets_train, dtype=np.float64)

        if targets.ndim != 3:
            raise ValueError(f"Expected targets_train shape (N, S, C), got {targets.shape}")
        if last_obs.shape != (targets.shape[0], targets.shape[2]):
            raise ValueError(
                f"Expected last_obs_train shape {(targets.shape[0], targets.shape[2])}, got {last_obs.shape}"
            )

        self.future_steps = targets.shape[1]
        residuals = targets - last_obs[:, None, :]  # (N, S, C)

        if valid_mask_train is not None:
            vm = np.asarray(valid_mask_train, dtype=bool)
            residuals = np.where(vm, residuals, np.nan)

        # np.nanquantile returns shape (Q, S, C) -> transpose to (S, C, Q)
        q_res = np.nanquantile(residuals, self.quantiles, axis=0)
        self.residual_quantiles_ = np.transpose(q_res, (1, 2, 0)).astype(np.float32)
        return self

    def predict(
        self,
        last_obs: np.ndarray,
        clip_min: Optional[float] = -1.0,
        clip_max: Optional[float] = 1.0,
    ) -> np.ndarray:
        """
        Args:
            last_obs: (N, C) last observed clear-sky target values on original scale.
        Returns:
            predictions of shape (N, future_steps, C, len(quantiles)), guaranteed non-crossing.
        """
        if self.residual_quantiles_ is None:
            raise RuntimeError("ProbabilisticPersistenceBaseline must be fitted before predict().")

        obs = np.asarray(last_obs, dtype=np.float32)
        preds = obs[:, None, :, None] + self.residual_quantiles_[None, :, :, :]

        if clip_min is not None or clip_max is not None:
            preds = np.clip(preds, a_min=clip_min, a_max=clip_max)

        # Enforce monotonicity q_0.1 <= q_0.5 <= q_0.9
        preds = np.sort(preds, axis=-1)
        return preds
