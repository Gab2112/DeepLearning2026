"""
Interpolation strategies and factory for handling missing values in time-series data.
Supports modular strategies: Linear, Forward-Fill, Backward-Fill, Nearest,
Spline/PCHIP, Mean, Median, Constant/Zero, and Model-based (Bi-GRU).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
from pathlib import Path

import numpy as np
import torch


class BaseInterpolator(ABC):
    """
    Abstract Base Class for time-series interpolation strategies.
    Operates on 1D series (L,) or 2D sequences (L, D).
    """

    def __init__(
        self,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        fill_value: float = 0.0,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.fill_value = fill_value

    @abstractmethod
    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Subclasses implement this method for a 1D sequence of length L.
        Args:
            series: 1D array of shape (L,)
            mask: boolean 1D array of shape (L,), True where value is missing/NaN
        Returns:
            Filled 1D array of shape (L,) with no missing values.
        """
        pass

    def interpolate_series(self, series: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolates a single 1D time-series.
        """
        series = np.array(series, dtype=np.float32, copy=True)
        if mask is None:
            mask = np.isnan(series)
        else:
            mask = np.array(mask, dtype=bool) | np.isnan(series)

        # If no values are missing, return as is
        if not np.any(mask):
            return self._apply_clip(series)

        # If all values are missing, fallback to fill_value
        if np.all(mask):
            series[:] = self.fill_value
            return self._apply_clip(series)

        # If only 1 valid observation, constant fill
        valid_indices = np.where(~mask)[0]
        if len(valid_indices) == 1:
            series[:] = series[valid_indices[0]]
            return self._apply_clip(series)

        filled = self._interpolate_1d(series, mask)
        return self._apply_clip(filled)

    def interpolate(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        columns: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Interpolates a 1D or 2D time series sequence.
        Args:
            data: array of shape (L,) or (L, D)
            mask: boolean array of same shape as data, True where missing.
                  If None, computed as np.isnan(data).
            columns: if specified and data is 2D, only these column indices are interpolated.
                     If None, all columns containing missing values are interpolated.
        Returns:
            A new numpy array with missing values filled.
        """
        out = np.array(data, dtype=np.float32, copy=True)

        if out.ndim == 1:
            return self.interpolate_series(out, mask=mask)

        if out.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {out.shape}")

        L, D = out.shape
        if mask is None:
            mask = np.isnan(out)
        else:
            mask = np.array(mask, dtype=bool) | np.isnan(out)

        cols_to_process = list(columns) if columns is not None else list(range(D))

        for col in cols_to_process:
            if col < 0 or col >= D:
                continue
            col_series = out[:, col]
            col_mask = mask[:, col]
            if np.any(col_mask):
                out[:, col] = self.interpolate_series(col_series, col_mask)

        return out

    def _apply_clip(self, arr: np.ndarray) -> np.ndarray:
        if self.clip_min is not None or self.clip_max is not None:
            return np.clip(arr, a_min=self.clip_min, a_max=self.clip_max)
        return arr


class InterpolationFactory:
    """
    Factory registry for creating time-series interpolation strategies.
    Allows easy registration and instantiation of different algorithms.
    """

    _registry: Dict[str, Type[BaseInterpolator]] = {}

    @classmethod
    def register(cls, name: str, aliases: Optional[List[str]] = None):
        """
        Decorator to register a new interpolation strategy.
        """
        def decorator(subclass: Type[BaseInterpolator]):
            key = name.strip().lower().replace("-", "_")
            cls._registry[key] = subclass
            if aliases:
                for alias in aliases:
                    cls._registry[alias.strip().lower().replace("-", "_")] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, strategy: Union[str, BaseInterpolator], **kwargs) -> BaseInterpolator:
        """
        Instantiates an interpolator by name or returns the instance directly.
        Args:
            strategy: name of the strategy (e.g. 'linear', 'spline', 'forward_fill') or an instance
            **kwargs: parameters passed to the constructor (e.g. clip_min, clip_max, etc.)
        """
        if isinstance(strategy, BaseInterpolator):
            return strategy

        key = str(strategy).strip().lower().replace("-", "_")
        if key not in cls._registry:
            available = sorted(list(cls._registry.keys()))
            raise ValueError(
                f"Unknown interpolation strategy: '{strategy}'. "
                f"Available strategies: {available}"
            )

        return cls._registry[key](**kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        """Returns sorted list of unique registered strategy names."""
        return sorted(list(set(cls._registry.keys())))


# =====================================================================
# CONCRETE STRATEGY IMPLEMENTATIONS
# =====================================================================

@InterpolationFactory.register("linear", aliases=["interp1d", "piecewise_linear"])
class LinearInterpolator(BaseInterpolator):
    """
    Standard linear interpolation between known observations.
    Uses flat (constant) boundary extrapolation before first and after last observation.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_idx = np.where(~mask)[0]
        missing_idx = np.where(mask)[0]

        valid_vals = series[valid_idx]
        # np.interp uses constant values at boundaries (series[valid_idx[0]] on left, series[valid_idx[-1]] on right)
        series[missing_idx] = np.interp(missing_idx, valid_idx, valid_vals)
        return series


@InterpolationFactory.register("forward_fill", aliases=["ffill", "locf", "last_observation_carried_forward"])
class ForwardFillInterpolator(BaseInterpolator):
    """
    Forward-fill (LOCF - Last Observation Carried Forward).
    Initial missing points before the first observation are backward-filled.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        L = len(series)
        last_val = None

        # Find first valid value to handle initial missing points
        valid_idx = np.where(~mask)[0]
        first_val = series[valid_idx[0]]

        for t in range(L):
            if not mask[t]:
                last_val = series[t]
            else:
                series[t] = last_val if last_val is not None else first_val

        return series


@InterpolationFactory.register("backward_fill", aliases=["bfill", "nocb", "next_observation_carried_backward"])
class BackwardFillInterpolator(BaseInterpolator):
    """
    Backward-fill (NOCB - Next Observation Carried Backward).
    Trailing missing points after the last observation are forward-filled.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        L = len(series)
        next_val = None

        # Find last valid value to handle trailing missing points
        valid_idx = np.where(~mask)[0]
        last_val = series[valid_idx[-1]]

        for t in range(L - 1, -1, -1):
            if not mask[t]:
                next_val = series[t]
            else:
                series[t] = next_val if next_val is not None else last_val

        return series


@InterpolationFactory.register("nearest", aliases=["nearest_neighbor"])
class NearestInterpolator(BaseInterpolator):
    """
    Nearest neighbor interpolation: each missing point takes the value
    of the temporally closest valid observation.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_idx = np.where(~mask)[0]
        missing_idx = np.where(mask)[0]

        for m in missing_idx:
            nearest_idx = valid_idx[np.argmin(np.abs(valid_idx - m))]
            series[m] = series[nearest_idx]

        return series


@InterpolationFactory.register("spline", aliases=["pchip", "cubic", "akima"])
class SplineInterpolator(BaseInterpolator):
    """
    Smooth spline interpolation.
    Defaults to 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial),
    which preserves shape and monotonicity without wild overshoots,
    ideal for bounded physical vegetation indices (NDVI/GNDVI/EVI).
    Falls back gracefully to linear if fewer than 3 points exist.
    """

    def __init__(
        self,
        kind: str = "pchip",
        clip_min: Optional[float] = -1.0,
        clip_max: Optional[float] = 1.0,
        fill_value: float = 0.0,
    ):
        super().__init__(clip_min=clip_min, clip_max=clip_max, fill_value=fill_value)
        self.kind = kind.lower()

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_idx = np.where(~mask)[0]
        missing_idx = np.where(mask)[0]

        # Need at least 2 points for linear, 3 points for cubic/pchip
        if len(valid_idx) < 3:
            series[missing_idx] = np.interp(missing_idx, valid_idx, series[valid_idx])
            return series

        try:
            from scipy.interpolate import PchipInterpolator, CubicSpline

            if self.kind == "pchip":
                # PCHIP preserves monotonicity, highly recommended for satellite index time series
                pchip = PchipInterpolator(valid_idx, series[valid_idx], extrapolate=False)
                interpolated = pchip(missing_idx)
                # Fill out-of-bounds with nearest boundary values
                left_bounds = missing_idx < valid_idx[0]
                right_bounds = missing_idx > valid_idx[-1]
                interpolated[left_bounds] = series[valid_idx[0]]
                interpolated[right_bounds] = series[valid_idx[-1]]
                series[missing_idx] = interpolated
            elif self.kind == "cubic":
                cs = CubicSpline(valid_idx, series[valid_idx], extrapolate=False)
                interpolated = cs(missing_idx)
                left_bounds = missing_idx < valid_idx[0]
                right_bounds = missing_idx > valid_idx[-1]
                interpolated[left_bounds] = series[valid_idx[0]]
                interpolated[right_bounds] = series[valid_idx[-1]]
                series[missing_idx] = interpolated
            else:
                # Fallback to linear
                series[missing_idx] = np.interp(missing_idx, valid_idx, series[valid_idx])
        except Exception:
            series[missing_idx] = np.interp(missing_idx, valid_idx, series[valid_idx])

        return series


@InterpolationFactory.register("mean", aliases=["average"])
class MeanInterpolator(BaseInterpolator):
    """
    Fills missing values with the mean of observed values in that sequence.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_vals = series[~mask]
        mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else self.fill_value
        series[mask] = mean_val
        return series


@InterpolationFactory.register("median")
class MedianInterpolator(BaseInterpolator):
    """
    Fills missing values with the median of observed values in that sequence.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid_vals = series[~mask]
        med_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else self.fill_value
        series[mask] = med_val
        return series


@InterpolationFactory.register("constant", aliases=["zero", "fixed_fill"])
class ConstantInterpolator(BaseInterpolator):
    """
    Fills missing values with a fixed constant (default: 0.0).
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        series[mask] = self.fill_value
        return series


@InterpolationFactory.register("none", aliases=["identity", "pass_through"])
class NoneInterpolator(BaseInterpolator):
    """
    Leaves the sequence unchanged (keeps NaNs or existing values as-is).
    Useful as an ablation / baseline.
    """

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return series


@InterpolationFactory.register("model", aliases=["bigru", "learned_imputer", "history_imputer"])
class ModelInterpolator(BaseInterpolator):
    """
    Learned neural network imputer using the pre-trained BiGRUHistoryImputer.
    Requires checkpoint path and scaler to perform end-to-end continuous imputation.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/history_imputer_best.pt",
        scaler_path: Union[str, Path] = "DATA/train_multi_output_clear_sky/scaler.json",
        device: Optional[str] = None,
        clip_min: Optional[float] = -1.0,
        clip_max: Optional[float] = 1.0,
        fill_value: float = 0.0,
    ):
        super().__init__(clip_min=clip_min, clip_max=clip_max, fill_value=fill_value)
        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = Path(scaler_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = None
        self._load_resources()

    def _load_resources(self):
        import sys
        import pathlib
        if "pathlib._local" not in sys.modules:
            sys.modules["pathlib._local"] = pathlib

        from src.data.transforms import ArcsinhScaler
        from src.models.history_imputer import BiGRUHistoryImputer

        if self.scaler_path.exists():
            self.scaler = ArcsinhScaler(self.scaler_path)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            m_cfg = ckpt.get("config", {}).get("imputer_model", {
                "input_dim": 27,
                "climate_dim": 8,
                "hidden_dim": 64,
                "num_layers": 2,
                "output_dim": 3,
                "dropout": 0.1,
            })
            self.model = BiGRUHistoryImputer(
                input_dim=m_cfg.get("input_dim", 27),
                climate_dim=m_cfg.get("climate_dim", 8),
                hidden_dim=m_cfg.get("hidden_dim", 64),
                num_layers=m_cfg.get("num_layers", 2),
                output_dim=m_cfg.get("output_dim", 3),
                dropout=m_cfg.get("dropout", 0.1),
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

    def _interpolate_1d(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Fallback to linear if 1D called without full features context
        valid_idx = np.where(~mask)[0]
        missing_idx = np.where(mask)[0]
        series[missing_idx] = np.interp(missing_idx, valid_idx, series[valid_idx])
        return series

    def interpolate_history_sample(
        self,
        h_raw: np.ndarray,
        m_raw: np.ndarray,
        climate_str: Optional[str] = None,
    ) -> np.ndarray:
        """
        Imputes missing past target values in a full history sample (L, 21) using the trained Bi-GRU.
        Replaces targets in columns 18..20 where missing.
        """
        if self.model is None or self.scaler is None:
            # Fallback to linear if model or scaler checkpoint is unavailable
            linear = LinearInterpolator(clip_min=self.clip_min, clip_max=self.clip_max)
            return linear.interpolate(h_raw, mask=m_raw, columns=[18, 19, 20])

        L = h_raw.shape[0]
        norm_weather = self.scaler.transform_features(h_raw)[:, :18]
        norm_weather = np.nan_to_num(norm_weather, nan=0.0)

        targets_raw = h_raw[:, 18:21]
        norm_targets = self.scaler.transform_targets(targets_raw)

        obs_mask = ~m_raw[:, 18]
        input_targets = norm_targets.copy()
        input_targets[~obs_mask] = 0.0
        input_targets = np.nan_to_num(input_targets, nan=0.0)

        mask_feature = np.zeros((L, 3), dtype=np.float32)
        mask_feature[obs_mask] = 1.0

        # Orbital dynamics
        delta_t = np.zeros(L, dtype=np.float32)
        orbital_phase = np.zeros(L, dtype=np.float32)
        missed_revisits = np.zeros(L, dtype=np.float32)
        last_obs = 0.0
        has_seen = False
        for t in range(L):
            orbital_phase[t] = (t % 5) / 5.0
            if obs_mask[t]:
                last_obs = float(t)
                has_seen = True
            else:
                if has_seen:
                    elapsed = float(t) - last_obs
                    delta_t[t] = elapsed
                    missed_revisits[t] = float(np.floor(elapsed / 5.0))

        orbital_feats = np.stack([delta_t, orbital_phase, missed_revisits], axis=-1)
        model_input = np.concatenate([norm_weather, input_targets, mask_feature, orbital_feats], axis=-1)
        model_input = np.nan_to_num(model_input, nan=0.0)

        from src.data.climate import encode_climate
        climate_id = encode_climate(climate_str)

        x_t = torch.from_numpy(model_input).float().unsqueeze(0).to(self.device)
        c_t = torch.tensor([climate_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds_norm = self.model(x_t, climate_id=c_t).squeeze(0).cpu().numpy()

        preds_orig = self.scaler.inverse_transform_targets(preds_norm)

        out = h_raw.copy()
        for idx, col in enumerate([18, 19, 20]):
            col_missing = m_raw[:, col]
            out[col_missing, col] = preds_orig[col_missing, idx]

        return self._apply_clip(out)
