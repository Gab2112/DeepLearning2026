from __future__ import annotations

"""
Unified module for the Bi-GRU History Imputer.
Contains all components, utilities, and functions needed to train, load,
evaluate, and perform inference/imputation on satellite time-series data.

Features:
1. Neural Model Architecture (BiGRUHistoryImputer with Climate Embedding).
2. Orbital Mechanics Calculations (5-day Sentinel-2 cadence, phase, revisit gaps).
3. Feature Engineering & Preprocessing (Arcsinh transform, 27-dim vector).
4. Safe Model Loading & Checkpoint Management.
5. Sequence & Batch Imputation (replaces NaNs with continuous physical predictions).
6. File-level Imputation (processes full .npz datasets).
7. Evaluation Metrics (MAE, RMSE, WMAPE on original scale).
8. Training & Loss Functions (smoothness-regularized masked reconstruction).
9. Command-line interface for standalone execution.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Cross-Python compatibility shim for pathlib unpickling (Python 3.13 / 3.14)
import pathlib
if "pathlib._local" not in sys.modules:
    sys.modules["pathlib._local"] = pathlib

# Project root in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.data.transforms import ArcsinhScaler

from src.models.history_imputer import BiGRUHistoryImputer


# =====================================================================
# 2. ORBITAL DYNAMICS & FEATURE ENGINEERING
# =====================================================================

def compute_orbital_dynamics(obs_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes temporal dynamics aligned with the 5-day Sentinel-2 orbital revisit cycle:
    1. delta_t: elapsed days since last clear-sky observation.
    2. orbital_phase: position within the 5-day orbital grid (t % 5) / 5.0.
    3. missed_revisits: number of 5-day revisit opportunities missed due to clouds.
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


def build_imputer_features(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    scaler: ArcsinhScaler,
    obs_mask_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Builds the standardized 27-dimensional feature matrix for a history sequence:
        [18 weather/calendar, 3 normalized targets, 3 mask flags, 3 orbital dynamics].
    """
    L = h_raw.shape[0]

    # 1. Weather + Calendar DOY features (0..17)
    norm_weather = scaler.transform_features(h_raw)[:, :18]
    norm_weather = np.nan_to_num(norm_weather, nan=0.0)

    # 2. Observed targets (18..20)
    targets_raw = h_raw[:, 18:21]
    norm_targets = scaler.transform_targets(targets_raw)

    if obs_mask_override is not None:
        obs_mask = obs_mask_override
    else:
        obs_mask = ~m_raw[:, 18]

    input_targets = norm_targets.copy()
    input_targets[~obs_mask] = 0.0
    input_targets = np.nan_to_num(input_targets, nan=0.0)

    # 3. Mask indicator (1.0 = observed, 0.0 = missing)
    mask_feat = np.zeros((L, 3), dtype=np.float32)
    mask_feat[obs_mask] = 1.0

    # 4. Orbital dynamics
    delta_t, phase, missed = compute_orbital_dynamics(obs_mask)
    orbital_feats = np.stack([delta_t, phase, missed], axis=-1)

    # Concatenate full vector
    features = np.concatenate([norm_weather, input_targets, mask_feat, orbital_feats], axis=-1)
    return np.nan_to_num(features, nan=0.0)


# =====================================================================
# 3. MODEL LOADING & INFERENCE FUNCTIONS
# =====================================================================

def _resolve_project_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_imputer_checkpoint(
    checkpoint_path: Union[str, Path] = "checkpoints/history_imputer_best.pt",
    device: Optional[str] = None,
) -> Tuple[BiGRUHistoryImputer, Dict]:
    """
    Loads trained BiGRUHistoryImputer model and metadata from checkpoint.
    Freezes parameters (eval mode + requires_grad=False).
    """
    from src.models.frozen import freeze_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_project_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    m_cfg = ckpt.get("config", {}).get("imputer_model", {
        "input_dim": 27,
        "climate_dim": 8,
        "hidden_dim": 64,
        "num_layers": 2,
        "output_dim": 3,
        "dropout": 0.1,
    })

    model = BiGRUHistoryImputer(
        input_dim=m_cfg.get("input_dim", 27),
        climate_dim=m_cfg.get("climate_dim", 8),
        hidden_dim=m_cfg.get("hidden_dim", 64),
        num_layers=m_cfg.get("num_layers", 2),
        output_dim=m_cfg.get("output_dim", 3),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    freeze_model(model)
    return model, ckpt


def impute_sequence(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    climate_str: Optional[str] = None,
    model: Optional[BiGRUHistoryImputer] = None,
    scaler: Optional[ArcsinhScaler] = None,
    device: Optional[str] = None,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> np.ndarray:
    """
    Imputes missing target values (columns 18, 19, 20) in a history array (L, 21).
    Preserves exact ground truth values on observed clear-sky dates and leaves weather columns untouched.
    Returns:
        History array (L, 21) on original physical scale with imputed targets.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, _ = load_imputer_checkpoint(device=device)

    if scaler is None:
        from src.data.transforms import ArcsinhScaler
        scaler = ArcsinhScaler(_resolve_project_path("DATA/train_multi_output_clear_sky/scaler.json"))

    # Build features
    from src.data.climate import encode_climate
    features = build_imputer_features(h_raw, m_raw, scaler)
    climate_id = encode_climate(climate_str)

    x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    c_tensor = torch.tensor([climate_id], dtype=torch.long, device=device)

    with torch.no_grad():
        preds_norm = model(x_tensor, climate_id=c_tensor).squeeze(0).cpu().numpy()

    # Inverse transform to original scale
    preds_orig = scaler.inverse_transform_targets(preds_norm)

    out = h_raw.copy()
    for idx, col in enumerate([18, 19, 20]):
        missing_mask = m_raw[:, col]
        out[missing_mask, col] = preds_orig[missing_mask, idx]

    if clip_min is not None or clip_max is not None:
        out[:, 18:21] = np.clip(out[:, 18:21], a_min=clip_min, a_max=clip_max)
    out[:, 18:21] = np.nan_to_num(out[:, 18:21], nan=0.0)
    return out


def impute_batch_sequences(
    histories: List[np.ndarray],
    masks: List[np.ndarray],
    climates: List[str],
    model: BiGRUHistoryImputer,
    scaler: ArcsinhScaler,
    device: Optional[str] = None,
    batch_size: int = 64,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> List[np.ndarray]:
    """
    Batched imputation for a list of history sequences (L_i, 21).
    Only modifies missing target columns (18..20) and leaves weather columns (0..17) untouched.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_samples = len(histories)
    results: List[np.ndarray] = [None] * num_samples

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_h = histories[start_idx:end_idx]
        batch_m = masks[start_idx:end_idx]
        batch_c = climates[start_idx:end_idx]

        feats = [build_imputer_features(h, m, scaler) for h, m in zip(batch_h, batch_m)]
        lengths = [f.shape[0] for f in feats]
        max_l = max(lengths)
        B = len(feats)

        x_padded = torch.zeros(B, max_l, 27, dtype=torch.float32)
        pad_mask = torch.zeros(B, max_l, dtype=torch.bool)
        c_ids = torch.zeros(B, dtype=torch.long)

        from src.data.climate import encode_climate
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


def impute_npz_file(
    npz_path: Union[str, Path],
    scaler: ArcsinhScaler,
    model: BiGRUHistoryImputer,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads an .npz file, imputes all past history sequences using the Bi-GRU model,
    and returns a dictionary matching the .npz structure with imputed history.
    """
    d = np.load(npz_path, allow_pickle=True)
    histories = list(d["history"])
    masks = list(d["history_mask"])
    climates = list(d["climate"]) if "climate" in d else ["unknown"] * len(histories)

    imputed_histories = impute_batch_sequences(
        histories, masks, climates, model=model, scaler=scaler, device=device
    )

    data_out = dict(d)
    data_out["history"] = np.array(imputed_histories, dtype=object)
    # Target columns in history_mask are now all False (observed/imputed)
    dense_masks = []
    for m in masks:
        m_copy = m.copy()
        m_copy[:, 18:21] = False
        dense_masks.append(m_copy)
    data_out["history_mask"] = np.array(dense_masks, dtype=object)

    return data_out


# =====================================================================
# 4. EVALUATION & METRICS
# =====================================================================

def compute_imputation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and WMAPE on the original physical scale.
    """
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sum_true = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(err)) / max(sum_true, 1e-6))

    return {"MAE": mae, "RMSE": rmse, "WMAPE": wmape}


def evaluate_imputer_on_sample(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    climate_str: str,
    model: BiGRUHistoryImputer,
    scaler: ArcsinhScaler,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Performs leave-one-out cross-validation on the clear-sky observations of a sample.
    Masks out 1 observed point, predicts it with the model, and calculates error against ground truth.
    """
    obs_mask = ~m_raw[:, 18]
    valid_indices = np.where(obs_mask)[0]

    if len(valid_indices) < 2:
        return {}

    errors = {"NDVI": [], "GNDVI": [], "EVI": []}
    target_names = ["NDVI", "GNDVI", "EVI"]

    from src.data.climate import encode_climate
    for drop_idx in valid_indices:
        perturbed_mask = obs_mask.copy()
        perturbed_mask[drop_idx] = False

        feats = build_imputer_features(h_raw, m_raw, scaler, obs_mask_override=perturbed_mask)
        x_t = torch.from_numpy(feats).float().unsqueeze(0).to(device)
        c_t = torch.tensor([encode_climate(climate_str)], dtype=torch.long, device=device)

        with torch.no_grad():
            preds_norm = model(x_t, climate_id=c_t).squeeze(0).cpu().numpy()

        preds_orig = scaler.inverse_transform_targets(preds_norm[drop_idx:drop_idx + 1])
        true_orig = h_raw[drop_idx:drop_idx + 1, 18:21]

        for i, name in enumerate(target_names):
            errors[name].append(abs(preds_orig[0, i] - true_orig[0, i]))

    results = {}
    for name in target_names:
        vals = errors[name]
        results[name] = {"MAE": float(np.mean(vals)) if vals else 0.0}

    return results


# =====================================================================
# 5. TRAINING & LOSS FUNCTIONS
# =====================================================================

def compute_masked_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    smoothness_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes masked Smooth L1 reconstruction loss on supervised points
    plus temporal smoothness regularization penalty.
    """
    if loss_mask.sum() > 0:
        p_masked = preds[loss_mask]
        t_masked = targets[loss_mask]
        recon_loss = nn.functional.smooth_l1_loss(p_masked, t_masked, reduction="mean")
    else:
        recon_loss = torch.tensor(0.0, device=preds.device)

    # Temporal Smoothness Regularization (penalizes sharp unnatural day-to-day jumps)
    diff = preds[:, 1:, :] - preds[:, :-1, :]
    valid_diff_mask = (padding_mask[:, 1:] & padding_mask[:, :-1]).unsqueeze(-1).expand_as(diff)
    if valid_diff_mask.sum() > 0:
        smooth_loss = torch.mean(diff[valid_diff_mask] ** 2)
    else:
        smooth_loss = torch.tensor(0.0, device=preds.device)

    total_loss = recon_loss + smoothness_weight * smooth_loss
    return total_loss, recon_loss


def train_imputer(
    config_path: Union[str, Path] = "configs/config.yaml",
    device: Optional[str] = None,
):
    """
    Delegates to the canonical training pipeline in src.train_bigru_imputer.
    """
    from src.train_bigru_imputer import train_bigru_imputer as _train_bigru_imputer

    return _train_bigru_imputer(config_path=config_path, device=device)


train_bigru_imputer = train_imputer


# =====================================================================
# 6. CLI ENTRYPOINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Bi-GRU History Imputer CLI")
    parser.add_argument("--mode", type=str, choices=["test", "evaluate", "train"], default="test",
                        help="Execution mode: test (impute 1 sample), evaluate (leave-one-out metrics), or train")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/history_imputer_best.pt",
                        help="Path to trained imputer checkpoint")
    parser.add_argument("--scaler", type=str, default="DATA/train_multi_output_clear_sky/scaler.json",
                        help="Path to scaler.json")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "train":
        train_bigru_imputer(args.config, device=device)
        return

    # For test and evaluate, load sample data
    from src.data.transforms import ArcsinhScaler
    scaler = ArcsinhScaler(args.scaler)
    model, _ = load_imputer_checkpoint(args.checkpoint, device=device)

    data_files = sorted(list(Path("DATA/train_multi_output_clear_sky").glob("*.npz")))
    if not data_files:
        print("[!] No .npz files found in DATA/train_multi_output_clear_sky.")
        return

    d = np.load(data_files[0], allow_pickle=True)
    h0 = d["history"][0]
    m0 = d["history_mask"][0]
    climate0 = str(d["climate"][0]) if "climate" in d else "unknown"

    if args.mode == "test":
        print(f"[*] Running test imputation on sample 0 of {data_files[0].name} (Seq len: {h0.shape[0]} days)...")
        imputed = impute_sequence(h0, m0, climate0, model=model, scaler=scaler, device=device)
        obs_idx = np.where(~m0[:, 18])[0]
        print(f"[*] Clear-sky observation days: {obs_idx}")
        print(f"[*] Original shape: {h0.shape} | Imputed shape: {imputed.shape}")
        print(f"[*] NaNs remaining in targets: {np.isnan(imputed[:, 18:21]).sum()}")
        print(f"[*] Sample imputed NDVI (first 5 days): {imputed[:5, 18].round(4)}")
        print("[+] Test imputation completed successfully.")

    elif args.mode == "evaluate":
        print(f"[*] Running leave-one-out evaluation on sample 0...")
        metrics = evaluate_imputer_on_sample(h0, m0, climate0, model=model, scaler=scaler, device=device)
        print("[*] Leave-one-out MAE per index:")
        for k, v in metrics.items():
            print(f"    - {k}: MAE = {v['MAE']:.4f}")


if __name__ == "__main__":
    main()
