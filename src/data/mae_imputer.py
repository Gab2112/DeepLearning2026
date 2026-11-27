"""
Inference, checkpoint loading, and evaluation utilities for the MAE History Imputer.
Reuses feature construction and loss utilities from src.models.bigru_imputer.
"""

import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Cross-Python compatibility shim for pathlib unpickling (Python 3.13 / 3.14)
if "pathlib._local" not in sys.modules:
    sys.modules["pathlib._local"] = pathlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


from src.models.bigru_imputer import (
    build_imputer_features,
    compute_imputation_metrics,
    compute_masked_loss,
)
from src.data.climate import encode_climate
from src.data.transforms import ArcsinhScaler, scaler_for_imputer
from src.data.weather_preprocessor import (
    WeatherPreprocessor,
    checkpoint_weather_preprocessor,
    model_weather_preprocessor,
)
from src.models.mae_imputer import MAEHistoryImputer


def load_mae_imputer_checkpoint(
    checkpoint_path: Union[str, Path] = "checkpoints/mae_imputer_best.pt",
    device: Optional[str] = None,
    *,
    expected_weather: Optional[WeatherPreprocessor] = None,
) -> Tuple[MAEHistoryImputer, Dict]:
    """
    Loads a trained MAEHistoryImputer model and metadata from checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_project_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found at: {ckpt_path.resolve()}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    m_cfg = config.get("mae_model", {})
    input_dim = m_cfg.get("input_dim", 27)
    weather = checkpoint_weather_preprocessor(config, input_dim, expected_weather)

    model = MAEHistoryImputer(
        input_dim=input_dim,
        climate_dim=m_cfg.get("climate_dim", 8),
        d_model=m_cfg.get("d_model", 64),
        nhead=m_cfg.get("nhead", 4),
        num_encoder_layers=m_cfg.get("num_encoder_layers", 3),
        num_decoder_layers=m_cfg.get("num_decoder_layers", 1),
        dim_feedforward=m_cfg.get("dim_feedforward", 128),
        output_dim=m_cfg.get("output_dim", 3),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.weather_preprocessor = weather
    if "scaler" in ckpt:
        model.arcsinh_scaler = ArcsinhScaler(ckpt["scaler"])
    model.eval()
    return model, ckpt


def impute_sequence_mae(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    climate_str: Optional[str] = None,
    model: Optional[MAEHistoryImputer] = None,
    scaler: Optional[ArcsinhScaler] = None,
    device: Optional[str] = None,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> np.ndarray:
    """
    Imputes missing target values (columns 18..20) in a history array (L, 21) using MAE.
    Preserves exact ground-truth values on observed clear-sky dates and leaves weather untouched.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, _ = load_mae_imputer_checkpoint(device=device)
    scaler = scaler_for_imputer(model, scaler)

    features = build_imputer_features(
        h_raw, m_raw, scaler, weather_preprocessor=model_weather_preprocessor(model)
    )
    climate_id = encode_climate(climate_str)

    x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    c_tensor = torch.tensor([climate_id], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        preds_norm = model(x_tensor, climate_id=c_tensor).squeeze(0).cpu().numpy()

    preds_orig = scaler.inverse_transform_targets(preds_norm)

    out = h_raw.copy()
    for idx, col in enumerate([18, 19, 20]):
        missing_mask = m_raw[:, col]
        out[missing_mask, col] = preds_orig[missing_mask, idx]

    if clip_min is not None or clip_max is not None:
        out[:, 18:21] = np.clip(out[:, 18:21], a_min=clip_min, a_max=clip_max)
    out[:, 18:21] = np.nan_to_num(out[:, 18:21], nan=0.0)
    return out


def impute_batch_sequences_mae(
    histories: List[np.ndarray],
    masks: List[np.ndarray],
    climates: List[str],
    model: MAEHistoryImputer,
    scaler: Optional[ArcsinhScaler] = None,
    device: Optional[str] = None,
    batch_size: int = 128,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> List[np.ndarray]:
    """
    Batched MAE imputation for a list of history sequences (L_i, 21).
    Modifies only missing target entries (columns 18..20) and leaves weather (0..17) untouched.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = scaler_for_imputer(model, scaler)
    model.eval()
    weather = model_weather_preprocessor(model)

    num_samples = len(histories)
    results: List[np.ndarray] = [None] * num_samples

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_h = histories[start_idx:end_idx]
        batch_m = masks[start_idx:end_idx]
        batch_c = climates[start_idx:end_idx]

        feats = [build_imputer_features(h, m, scaler, weather_preprocessor=weather) for h, m in zip(batch_h, batch_m)]
        lengths = [f.shape[0] for f in feats]
        max_l = max(lengths)
        B = len(feats)

        x_padded = torch.zeros(B, max_l, feats[0].shape[-1], dtype=torch.float32)
        pad_mask = torch.zeros(B, max_l, dtype=torch.bool)
        c_ids = torch.zeros(B, dtype=torch.long)

        for i in range(B):
            L = lengths[i]
            x_padded[i, :L] = torch.from_numpy(feats[i])
            pad_mask[i, :L] = True
            c_ids[i] = encode_climate(batch_c[i])

        x_padded = x_padded.to(device)
        c_ids = c_ids.to(device)
        pad_mask = pad_mask.to(device)

        with torch.no_grad():
            preds = model(x_padded, climate_id=c_ids, padding_mask=pad_mask).cpu().numpy()

        for i in range(B):
            L = lengths[i]
            preds_i = scaler.inverse_transform_targets(preds[i, :L])
            h_out = batch_h[i].copy()
            m_i = batch_m[i]
            for c_idx, col in enumerate([18, 19, 20]):
                missing = m_i[:, col]
                h_out[missing, col] = preds_i[missing, c_idx]
            if clip_min is not None or clip_max is not None:
                h_out[:, 18:21] = np.clip(h_out[:, 18:21], a_min=clip_min, a_max=clip_max)
            h_out[:, 18:21] = np.nan_to_num(h_out[:, 18:21], nan=0.0)
            results[start_idx + i] = h_out

    return results


def evaluate_mae_imputer_on_sample(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    climate_str: str,
    model: MAEHistoryImputer,
    scaler: ArcsinhScaler,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Leave-one-out cross-validation on the clear-sky observations of a single sample.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs_mask = ~m_raw[:, 18]
    valid_indices = np.where(obs_mask)[0]

    if len(valid_indices) < 2:
        return {}

    target_names = ["NDVI", "GNDVI", "EVI"]
    preds_all: Dict[str, List[float]] = {n: [] for n in target_names}
    trues_all: Dict[str, List[float]] = {n: [] for n in target_names}

    scaler = scaler_for_imputer(model, scaler)
    model.eval()
    weather = model_weather_preprocessor(model)
    for drop_idx in valid_indices:
        perturbed_mask = obs_mask.copy()
        perturbed_mask[drop_idx] = False

        feats = build_imputer_features(
            h_raw, m_raw, scaler, obs_mask_override=perturbed_mask, weather_preprocessor=weather
        )
        x_t = torch.from_numpy(feats).float().unsqueeze(0).to(device)
        c_t = torch.tensor([encode_climate(climate_str)], dtype=torch.long, device=device)

        with torch.no_grad():
            preds_norm = model(x_t, climate_id=c_t).squeeze(0).cpu().numpy()

        preds_orig = scaler.inverse_transform_targets(preds_norm[drop_idx : drop_idx + 1])
        true_orig = h_raw[drop_idx : drop_idx + 1, 18:21]

        for i, name in enumerate(target_names):
            preds_all[name].append(float(preds_orig[0, i]))
            trues_all[name].append(float(true_orig[0, i]))

    return {
        name: compute_imputation_metrics(
            np.asarray(trues_all[name], dtype=np.float32),
            np.asarray(preds_all[name], dtype=np.float32),
        )
        for name in target_names
    }


__all__ = [
    "MAEHistoryImputer",
    "compute_masked_loss",
    "compute_imputation_metrics",
    "load_mae_imputer_checkpoint",
    "impute_sequence_mae",
    "impute_batch_sequences_mae",
    "evaluate_mae_imputer_on_sample",
]
