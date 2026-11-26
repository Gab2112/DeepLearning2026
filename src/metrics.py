"""
Official evaluation metrics and probabilistic quantile losses for Vegetation Forecasting.
All evaluation metrics operate strictly on the original physical index scale
(NDVI, GNDVI, EVI) over shape conventions:
    - y_true: (N, S, C)
    - y_pred_quantiles: (N, S, C, Q) ordered as quantiles=(0.1, 0.5, 0.9)
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:  # avoid importing src.data at runtime (prevents circular imports)
    from src.data.transforms import ArcsinhScaler

DEFAULT_TARGET_NAMES: Tuple[str, ...] = ("NDVI", "GNDVI", "EVI")
DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)

METRIC_COLUMNS: Tuple[str, ...] = (
    "RMSE",
    "MAE",
    "WMAPE",
    "MASE",
    "Pinball",
    "Pinball_q10",
    "Pinball_q50",
    "Pinball_q90",
    "CRPS",
    "Coverage",
    "IntervalWidth",
    "CrossingRate",
)


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts a NumPy array or PyTorch tensor to a float64 NumPy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def _resolve_valid_mask(
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]],
    *arrays: np.ndarray,
) -> np.ndarray:
    """Builds a finite boolean mask matching the broadcast shape of inputs."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    if valid_mask is not None:
        vm = _to_numpy(valid_mask).astype(bool)
        mask &= vm
    return mask


def inverse_transform_quantiles(
    y_pred_norm: Union[np.ndarray, torch.Tensor],
    scaler: "ArcsinhScaler",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Inverts arcsinh standardization on multi-quantile predictions of shape (..., C, Q)
    back to the original physical scale. Supports both NumPy arrays and PyTorch tensors.
    """
    if isinstance(y_pred_norm, torch.Tensor):
        moved = torch.movedim(y_pred_norm, -1, -2)  # (..., Q, C)
        inv = scaler.torch_inverse_transform_targets(moved)
        return torch.movedim(inv, -2, -1)  # (..., C, Q)

    arr = np.asarray(y_pred_norm, dtype=np.float32)
    moved = np.moveaxis(arr, -1, -2)  # (..., Q, C)
    inv = scaler.inverse_transform_targets(moved)
    return np.moveaxis(inv, -2, -1)  # (..., C, Q)


def compute_naive_history_scale(
    history_obs: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes per-index Hyndman-Koehler naive 1-step scale from historical
    clear-sky observations of shape (N, H_obs, C):
        scale_c = mean(|obs[:, 1:, c] - obs[:, :-1, c]|)
    Returns shape (C,) strictly >= eps.
    """
    obs = _to_numpy(history_obs)
    if obs.ndim != 3 or obs.shape[1] < 2:
        raise ValueError(f"Expected history_obs shape (N, >=2, C), got {obs.shape}")
    diffs = np.abs(obs[:, 1:, :] - obs[:, :-1, :])
    scale = np.nanmean(diffs, axis=(0, 1))
    scale = np.where(np.isfinite(scale), np.maximum(scale, eps), eps)
    return scale


def rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_median: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred_median)
    mask = _resolve_valid_mask(valid_mask, yt, yp)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def mae(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_median: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred_median)
    mask = _resolve_valid_mask(valid_mask, yt, yp)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def wmape(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_median: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred_median)
    mask = _resolve_valid_mask(valid_mask, yt, yp)
    if not np.any(mask):
        return float("nan")
    num = np.sum(np.abs(yt[mask] - yp[mask]))
    den = np.sum(np.abs(yt[mask])) + eps
    return float(num / den)


def mase(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_median: Union[np.ndarray, torch.Tensor],
    scale: float,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    err_mae = mae(y_true, y_pred_median, valid_mask=valid_mask)
    if not np.isfinite(err_mae) or not np.isfinite(scale):
        return float("nan")
    return float(err_mae / (float(scale) + eps))


def pinball_loss(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_q: Union[np.ndarray, torch.Tensor],
    quantile: float,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred_q)
    mask = _resolve_valid_mask(valid_mask, yt, yp)
    if not np.any(mask):
        return float("nan")
    diff = yt[mask] - yp[mask]
    q = float(quantile)
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def crps_from_quantiles(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_quantiles: Union[np.ndarray, torch.Tensor],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Approximates Continuous Ranked Probability Score (CRPS) from quantile predictions:
        CRPS ~= 2 * mean_{q in quantiles}(PinballLoss_q(y, y_hat_q))
    Note: equals MAE in the deterministic limit where all quantiles are identical
    and quantiles are symmetric around 0.5 (e.g. (0.1, 0.5, 0.9)).
    """
    yt = _to_numpy(y_true)
    ypq = _to_numpy(y_pred_quantiles)
    if ypq.shape[-1] != len(quantiles):
        raise ValueError(
            f"Last dim of y_pred_quantiles ({ypq.shape[-1]}) != len(quantiles) ({len(quantiles)})."
        )
    losses = [
        pinball_loss(yt, ypq[..., idx], q, valid_mask=valid_mask)
        for idx, q in enumerate(quantiles)
    ]
    if any(not np.isfinite(val) for val in losses):
        return float("nan")
    return float(2.0 * np.mean(losses))


def coverage(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_low: Union[np.ndarray, torch.Tensor],
    y_pred_high: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    yt = _to_numpy(y_true)
    ylow = _to_numpy(y_pred_low)
    yhigh = _to_numpy(y_pred_high)
    mask = _resolve_valid_mask(valid_mask, yt, ylow, yhigh)
    if not np.any(mask):
        return float("nan")
    inside = (yt[mask] >= ylow[mask]) & (yt[mask] <= yhigh[mask])
    return float(np.mean(inside))


def interval_width(
    y_pred_low: Union[np.ndarray, torch.Tensor],
    y_pred_high: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    ylow = _to_numpy(y_pred_low)
    yhigh = _to_numpy(y_pred_high)
    mask = _resolve_valid_mask(valid_mask, ylow, yhigh)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(yhigh[mask] - ylow[mask]))


def quantile_crossing_rate(
    y_pred_quantiles: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Returns fraction of predictions violating monotonicity (q_i > q_{i+1}).
    """
    ypq = _to_numpy(y_pred_quantiles)
    finite = np.all(np.isfinite(ypq), axis=-1)
    if valid_mask is not None:
        finite &= _to_numpy(valid_mask).astype(bool)
    if not np.any(finite):
        return float("nan")
    crossing = np.any(np.diff(ypq, axis=-1) < -1e-7, axis=-1)
    return float(np.mean(crossing[finite]))


def _compute_single_target_metrics(
    yt: np.ndarray,
    ypq: np.ndarray,
    quantiles: Sequence[float],
    mase_scale_val: Optional[float],
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    q_arr = np.asarray(quantiles, dtype=np.float64)
    q10_idx = int(np.argmin(np.abs(q_arr - 0.1)))
    q50_idx = int(np.argmin(np.abs(q_arr - 0.5)))
    q90_idx = int(np.argmin(np.abs(q_arr - 0.9)))

    yp_low = ypq[..., q10_idx]
    yp_med = ypq[..., q50_idx]
    yp_high = ypq[..., q90_idx]

    base_mask = np.isfinite(yt) & np.all(np.isfinite(ypq), axis=-1)
    if valid_mask is not None:
        base_mask &= valid_mask.astype(bool)

    p_q10 = pinball_loss(yt, yp_low, quantiles[q10_idx], valid_mask=base_mask)
    p_q50 = pinball_loss(yt, yp_med, quantiles[q50_idx], valid_mask=base_mask)
    p_q90 = pinball_loss(yt, yp_high, quantiles[q90_idx], valid_mask=base_mask)
    pinball_mean = float(np.mean([p_q10, p_q50, p_q90])) if np.all(np.isfinite([p_q10, p_q50, p_q90])) else float("nan")

    return {
        "RMSE": rmse(yt, yp_med, valid_mask=base_mask),
        "MAE": mae(yt, yp_med, valid_mask=base_mask),
        "WMAPE": wmape(yt, yp_med, valid_mask=base_mask),
        "MASE": mase(yt, yp_med, scale=mase_scale_val, valid_mask=base_mask) if mase_scale_val is not None else float("nan"),
        "Pinball": pinball_mean,
        "Pinball_q10": p_q10,
        "Pinball_q50": p_q50,
        "Pinball_q90": p_q90,
        "CRPS": crps_from_quantiles(yt, ypq, quantiles=quantiles, valid_mask=base_mask),
        "Coverage": coverage(yt, yp_low, yp_high, valid_mask=base_mask),
        "IntervalWidth": interval_width(yp_low, yp_high, valid_mask=base_mask),
        "CrossingRate": quantile_crossing_rate(ypq, valid_mask=base_mask),
    }


def _macro_average(metrics_by_index: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in METRIC_COLUMNS:
        vals = [m[col] for m in metrics_by_index.values() if np.isfinite(m[col])]
        out[col] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_forecast(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_quantiles: Union[np.ndarray, torch.Tensor],
    mase_scale: Optional[Union[np.ndarray, Sequence[float]]] = None,
    target_names: Sequence[str] = DEFAULT_TARGET_NAMES,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    include_per_step: bool = True,
) -> Dict:
    """
    Computes all official point and probabilistic metrics on original scale.
    Args:
        y_true: (N, S, C)
        y_pred_quantiles: (N, S, C, Q)
        mase_scale: optional (C,) naive scale per index
    Returns:
        Dictionary with keys: 'per_index', 'overall', and optionally 'per_step'.
    """
    yt = _to_numpy(y_true)
    ypq = _to_numpy(y_pred_quantiles)
    vm = _to_numpy(valid_mask).astype(bool) if valid_mask is not None else None

    if yt.ndim != 3:
        raise ValueError(f"Expected y_true shape (N, S, C), got {yt.shape}")
    if ypq.shape[:3] != yt.shape or ypq.shape[3] != len(quantiles):
        raise ValueError(
            f"Expected y_pred_quantiles shape {yt.shape + (len(quantiles),)}, got {ypq.shape}"
        )

    _, S, C = yt.shape
    if len(target_names) != C:
        raise ValueError(f"len(target_names)={len(target_names)} != C={C}")

    scale_arr = _to_numpy(mase_scale) if mase_scale is not None else None
    if scale_arr is not None and scale_arr.shape != (C,):
        raise ValueError(f"Expected mase_scale shape ({C},), got {scale_arr.shape}")

    per_index: Dict[str, Dict[str, float]] = {}
    for c_idx, name in enumerate(target_names):
        s_val = float(scale_arr[c_idx]) if scale_arr is not None else None
        vm_c = vm[:, :, c_idx] if vm is not None else None
        per_index[name] = _compute_single_target_metrics(
            yt[:, :, c_idx],
            ypq[:, :, c_idx, :],
            quantiles=quantiles,
            mase_scale_val=s_val,
            valid_mask=vm_c,
        )

    result: Dict = {
        "per_index": per_index,
        "overall": _macro_average(per_index),
    }

    if include_per_step:
        per_step: Dict[str, Dict[str, Dict[str, float]]] = {}
        for s_idx in range(S):
            step_dict: Dict[str, Dict[str, float]] = {}
            for c_idx, name in enumerate(target_names):
                s_val = float(scale_arr[c_idx]) if scale_arr is not None else None
                vm_sc = vm[:, s_idx, c_idx] if vm is not None else None
                step_dict[name] = _compute_single_target_metrics(
                    yt[:, s_idx, c_idx],
                    ypq[:, s_idx, c_idx, :],
                    quantiles=quantiles,
                    mase_scale_val=s_val,
                    valid_mask=vm_sc,
                )
            step_dict["overall"] = _macro_average(step_dict)
            per_step[f"step_{s_idx + 1}"] = step_dict
        result["per_step"] = per_step

    return result


def metrics_to_dataframe(
    results_by_split: Dict[str, Dict],
    model_name: str = "",
) -> pd.DataFrame:
    """
    Converts nested evaluation results `{split_name: evaluate_forecast_dict}`
    into a tidy pandas DataFrame.
    """
    rows: List[Dict] = []
    for split_name, res in results_by_split.items():
        for idx_name, m_dict in res["per_index"].items():
            row = {"model": model_name, "split": split_name, "index": idx_name}
            row.update(m_dict)
            rows.append(row)

        overall_row = {"model": model_name, "split": split_name, "index": "OVERALL"}
        overall_row.update(res["overall"])
        rows.append(overall_row)

    cols = ["model", "split", "index"] + list(METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=cols)


def format_metrics_table(metrics_dict: Dict, split_name: str = "") -> str:
    """Formats per-index and overall metrics into a readable console table."""
    header = f"=== {split_name} ===" if split_name else "=== Evaluation Metrics ==="
    lines = [
        header,
        f"{'Index':<8} | {'RMSE':>7} | {'MAE':>7} | {'WMAPE':>7} | {'MASE':>7} | {'Pinball':>7} | {'CRPS':>7} | {'Cov80%':>7} | {'Width':>7} | {'Cross':>6}",
        "-" * 96,
    ]

    entries = list(metrics_dict["per_index"].items()) + [("OVERALL", metrics_dict["overall"])]
    for name, m in entries:
        if name == "OVERALL":
            lines.append("-" * 96)
        cov_pct = m["Coverage"] * 100.0 if np.isfinite(m["Coverage"]) else float("nan")
        cross_pct = m["CrossingRate"] * 100.0 if np.isfinite(m["CrossingRate"]) else float("nan")
        lines.append(
            f"{name:<8} | "
            f"{m['RMSE']:7.4f} | "
            f"{m['MAE']:7.4f} | "
            f"{m['WMAPE']:7.4f} | "
            f"{m['MASE']:7.4f} | "
            f"{m['Pinball']:7.4f} | "
            f"{m['CRPS']:7.4f} | "
            f"{cov_pct:6.2f}% | "
            f"{m['IntervalWidth']:7.4f} | "
            f"{cross_pct:5.2f}%"
        )
    return "\n".join(lines)


class PinballLoss(nn.Module):
    """
    Multi-quantile Pinball (tilted L1) loss for probabilistic PyTorch forecasters.
    Expects:
        preds: (B, ..., Q)
        targets: (B, ...) matching preds.shape[:-1]
    """

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if preds.shape[:-1] != targets.shape:
            raise ValueError(
                f"Expected preds.shape[:-1] == targets.shape, got {preds.shape} vs {targets.shape}"
            )
        if preds.shape[-1] != len(self.quantiles):
            raise ValueError(
                f"Last dim of preds ({preds.shape[-1]}) != num quantiles ({len(self.quantiles)})"
            )

        q = self.quantiles.to(device=preds.device, dtype=preds.dtype)
        q = q.view(*([1] * targets.ndim), -1)

        safe_targets = torch.nan_to_num(targets, nan=0.0)
        errors = safe_targets.unsqueeze(-1) - preds
        losses = torch.maximum(q * errors, (q - 1.0) * errors)

        if self.reduction == "none":
            return losses

        mask = torch.isfinite(targets) & torch.all(torch.isfinite(preds), dim=-1)
        if valid_mask is not None:
            mask = mask & valid_mask.bool()

        mask_exp = mask.unsqueeze(-1).expand_as(losses)
        if not torch.any(mask_exp):
            return preds.sum() * 0.0

        valid_losses = losses[mask_exp]
        if self.reduction == "sum":
            return valid_losses.sum()
        return valid_losses.mean()
