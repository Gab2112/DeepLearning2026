"""
Training and Validation Pipeline for the Masked Autoencoder (MAE) History Imputer.
Configured via configs/config.yaml with optional CLI overrides.

Features:
1. Stratified tile-based Train/Val splitting (no spatial leakage).
2. ArcsinhScaler normalization using train scaler.json.
3. Self-supervised target masking (mask_dropout=True on both train & val for honest MAE eval).
4. Auxiliary self-supervised weather masking (weather_mask_ratio) for representation learning.
5. Original-scale overall & per-index (NDVI, GNDVI, EVI) validation MAE logging + checkpointing.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.bigru_imputer import compute_masked_loss
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.splits import get_train_val_tile_split
from src.data.transforms import ArcsinhScaler
from src.models.frozen import extract_arch, file_sha256, load_frozen_model, save_frozen_checkpoint
from src.models.mae_imputer import MAEHistoryImputer
from src.utils.config import load_config


def _sample_weather_mask(
    padding_mask: torch.Tensor,
    ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Samples boolean weather mask over valid (non-padded) timesteps."""
    if ratio <= 0.0:
        return torch.zeros_like(padding_mask, dtype=torch.bool)
    probs = torch.rand(padding_mask.shape, device=padding_mask.device, generator=generator)
    return (probs < ratio) & padding_mask.bool()


def train_mae_imputer(
    config_path: Union[str, Path] = "configs/config.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_train_tiles: Optional[int] = None,
    max_val_tiles: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Path:
    cfg = load_config(config_path)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    m_cfg = cfg["mae_model"]
    t_cfg = cfg["mae_training"]
    interp_cfg = cfg.get("interpolation", {})

    data_dir = paths_cfg["train_data_dir"]
    scaler_path = paths_cfg["scaler_path"]
    target_save_path = (
        Path(save_path)
        if save_path is not None
        else Path(paths_cfg["checkpoints_dir"]) / t_cfg["checkpoint_name"]
    )

    num_train_tiles = data_cfg["num_train_tiles"]
    num_val_tiles = data_cfg["num_val_tiles"]
    climate_dropout_p = data_cfg["climate_dropout_p"]
    seed = int(data_cfg["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    epochs_val = epochs if epochs is not None else int(t_cfg["epochs"])
    batch_size_val = batch_size if batch_size is not None else int(t_cfg["batch_size"])
    lr = float(t_cfg["learning_rate"])
    weight_decay = float(t_cfg["weight_decay"])
    smoothness_weight = float(t_cfg["smoothness_weight"])
    visible_recon_weight = float(t_cfg.get("visible_recon_weight", 0.25))
    weather_mask_ratio = float(t_cfg.get("weather_mask_ratio", 0.15))
    weather_recon_weight = float(t_cfg.get("weather_recon_weight", 0.1))
    clip_min = interp_cfg.get("clip_min", -1.0)
    clip_max = interp_cfg.get("clip_max", 1.0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" TRAINING MAE HISTORY IMPUTER (Transformer Masked Autoencoder)")
    print("=" * 72)
    print(f"[*] Config File      : {config_path}")
    print(f"[*] Device           : {device}")
    print(f"[*] Data Directory   : {data_dir}")
    print(f"[*] Checkpoint Target: {target_save_path}")

    train_files, val_files, split_summary = get_train_val_tile_split(
        data_dir, num_val_tiles=num_val_tiles, seed=seed
    )
    if num_train_tiles < len(train_files):
        train_files = train_files[:num_train_tiles]
    if max_train_tiles is not None:
        train_files = train_files[:max_train_tiles]
    if max_val_tiles is not None:
        val_files = val_files[:max_val_tiles]

    assert set(train_files).isdisjoint(set(val_files)), "Data leakage: train_files and val_files overlap!"

    print(f"[*] Train Tiles: {len(train_files)} | Val Tiles: {len(val_files)}")
    if max_train_tiles is None and max_val_tiles is None:
        print(f"[*] Climi Train: {split_summary['train']}")
        print(f"[*] Climi Val  : {split_summary['val']}")

    scaler = ArcsinhScaler(scaler_path)
    scaler_sha = file_sha256(scaler_path)

    print("[*] Indexing datasets...")
    train_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=train_files,
        mask_dropout=True,
        climate_dropout_p=climate_dropout_p,
        seed=seed,
    )
    # Deterministic Leave-One-Out validation across all 3 clear-sky observations per window.
    val_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=val_files,
        mask_dropout=True,
        climate_dropout_p=0.0,
        seed=seed + 1,
        loo_all_points=True,
    )
    print(f"[*] Train Windows: {len(train_ds):,} | Val LOO Targets: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_val,
        shuffle=True,
        collate_fn=collate_history_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size_val, 256),
        shuffle=False,
        collate_fn=collate_history_batch,
    )

    arch = extract_arch("mae", m_cfg)
    model = MAEHistoryImputer(**arch).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_mae = float("inf")
    target_save_path.parent.mkdir(parents=True, exist_ok=True)
    target_names = ("NDVI", "GNDVI", "EVI")

    print("\n" + "-" * 72)
    print(" START TRAINING")
    print("-" * 72)

    for epoch in range(1, epochs_val + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_target_recon = 0.0
        total_weather_recon = 0.0
        batches = 0

        for batch in train_loader:
            x = batch["input"].to(device)
            climate_id = batch["climate_id"].to(device)
            target = batch["target"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            weather_mask = _sample_weather_mask(padding_mask, weather_mask_ratio)

            optimizer.zero_grad()
            preds, pred_weather = model(
                x,
                climate_id=climate_id,
                padding_mask=padding_mask,
                weather_mask=weather_mask,
                return_weather=True,
            )

            target_loss, target_recon = compute_masked_loss(
                preds,
                target,
                loss_mask,
                padding_mask,
                smoothness_weight=smoothness_weight,
            )

            visible_mask = obs_mask & (~loss_mask)
            if visible_recon_weight > 0.0 and torch.any(visible_mask):
                vis_loss = F.smooth_l1_loss(
                    preds[visible_mask],
                    target[visible_mask],
                    reduction="mean",
                )
            else:
                vis_loss = torch.tensor(0.0, device=device)

            if weather_recon_weight > 0.0 and torch.any(weather_mask):
                w_loss = F.smooth_l1_loss(
                    pred_weather[weather_mask],
                    x[:, :, :11][weather_mask],
                    reduction="mean",
                )
            else:
                w_loss = torch.tensor(0.0, device=device)

            loss = target_loss + visible_recon_weight * vis_loss + weather_recon_weight * w_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_target_recon += target_recon.item()
            total_weather_recon += w_loss.item()
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)
        avg_target_recon = total_target_recon / max(1, batches)
        avg_weather_recon = total_weather_recon / max(1, batches)

        # Honest Validation on held-out tiles with deterministic LOO clear-sky targets
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        abs_err_sum = np.zeros(3, dtype=np.float64)
        count_masked = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                climate_id = batch["climate_id"].to(device)
                target = batch["target"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                padding_mask = batch["padding_mask"].to(device)

                preds = model(x, climate_id=climate_id, padding_mask=padding_mask)
                val_loss, _ = compute_masked_loss(
                    preds,
                    target,
                    loss_mask,
                    padding_mask,
                    smoothness_weight=smoothness_weight,
                )
                total_val_loss += val_loss.item()
                val_batches += 1

                if torch.any(loss_mask):
                    p_orig = scaler.torch_inverse_transform_targets(preds[loss_mask])
                    if clip_min is not None or clip_max is not None:
                        p_orig = torch.clamp(p_orig, min=clip_min, max=clip_max)
                    t_orig = scaler.torch_inverse_transform_targets(target[loss_mask])
                    abs_err = torch.abs(p_orig - t_orig).sum(dim=0).cpu().numpy()
                    abs_err_sum += abs_err
                    count_masked += int(loss_mask.sum().item())

        avg_val_loss = total_val_loss / max(1, val_batches)
        per_idx_mae: Dict[str, float] = {
            name: float(abs_err_sum[i] / max(1, count_masked))
            for i, name in enumerate(target_names)
        }
        avg_val_mae = float(np.mean(list(per_idx_mae.values())))
        elapsed = time.time() - t0

        scheduler.step(avg_val_mae)

        is_best = avg_val_mae < best_val_mae
        if is_best:
            best_val_mae = avg_val_mae
            save_frozen_checkpoint(
                target_save_path,
                model=model,
                model_type="mae",
                arch=arch,
                metadata={
                    "epoch": epoch,
                    "val_loss": float(avg_val_loss),
                    "val_mae_orig": float(avg_val_mae),
                    "val_mae_per_index": per_idx_mae,
                    "seed": seed,
                    "scaler_sha256": scaler_sha,
                    "train_tiles": [p.name for p in train_files],
                    "val_tiles": [p.name for p in val_files],
                },
                full_config=cfg,
            )

        flag = " [BEST SAVED]" if is_best else ""
        idx_str = " ".join(f"{n}={per_idx_mae[n]:.4f}" for n in target_names)
        print(
            f"Epoch {epoch:02d}/{epochs_val:02d} [{elapsed:4.1f}s] | "
            f"Train: {avg_train_loss:.4f} (Tgt={avg_target_recon:.4f}, W={avg_weather_recon:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} [{idx_str}]{flag}"
        )

    print("-" * 72)
    print(f"[*] Training completed! Best frozen checkpoint saved to:\n    {target_save_path}")

    if target_save_path.exists():
        model, _ = load_frozen_model(target_save_path, device=device, verify=True)

    demonstrate_mae_imputation(model, val_ds, scaler, device)
    return target_save_path


def demonstrate_mae_imputation(
    model: MAEHistoryImputer,
    dataset: HistoryImputationDataset,
    scaler: ArcsinhScaler,
    device: str,
):
    """Prints a sample history reconstruction on original physical scale."""
    if len(dataset) == 0:
        return
    model.eval()
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).to(device)
    climate_id = sample["climate_id"].unsqueeze(0).to(device)
    obs = sample["all_obs_mask"].numpy()
    loss_m = sample["loss_mask"].numpy()
    target_clean = sample["target"].unsqueeze(0).to(device)
    L = sample["length"]

    with torch.no_grad():
        imputed_norm = model(x, climate_id=climate_id)[0, :L]
        imputed_orig = scaler.torch_inverse_transform_targets(imputed_norm).cpu().numpy()
        real_orig = scaler.torch_inverse_transform_targets(target_clean[0, :L]).cpu().numpy()

    print("\n--- SAMPLE MAE RECONSTRUCTION (Val Sample 0) ---")
    print(
        f"{'Day':<6} | {'Status':<18} | {'NDVI Real':<10} | "
        f"{'NDVI Imp':<10} | {'GNDVI Imp':<10} | {'EVI Imp':<10}"
    )
    print("-" * 78)
    for t in range(min(14, L)):
        if loss_m[t]:
            status = "Masked (Val Held)"
        elif obs[t]:
            status = "Clear-Sky (Input)"
        else:
            status = "Cloud/Gap (NaN)"
        real_str = f"{real_orig[t, 0]:.4f}" if obs[t] else "   ---    "
        print(
            f"Day {t:02d} | {status:<18} | {real_str:<10} | "
            f"{imputed_orig[t, 0]:10.4f} | {imputed_orig[t, 1]:10.4f} | {imputed_orig[t, 2]:10.4f}"
        )
    print("-" * 78)


def main():
    parser = argparse.ArgumentParser(description="Train MAE History Imputer")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-tiles", type=int, default=None)
    parser.add_argument("--max-val-tiles", type=int, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_mae_imputer(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_tiles=args.max_train_tiles,
        max_val_tiles=args.max_val_tiles,
        save_path=args.save_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
