"""
Training and Validation Pipeline for the Bi-GRU Past Time-Series Imputer.
Configured centrally via configs/config.yaml.
Executes:
1. Spatial Tile-based Train/Val splitting.
2. ArcsinhScaler normalization using existing train scaler.json.
3. Sentinel-2 5-day orbital revisit dynamics + Climate Embedding.
4. Self-supervised training with masked reconstruction loss + smoothness penalty.
5. Checkpointing and evaluation on original scale.
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config
from src.data.transforms import ArcsinhScaler
from src.data.splits import get_train_val_tile_split
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.models.bigru_imputer import compute_masked_loss
from src.models.frozen import extract_arch, file_sha256, load_frozen_model, save_frozen_checkpoint
from src.models.history_imputer import BiGRUHistoryImputer


def train_bigru_imputer(
    config_path: Union[str, Path] = "configs/config.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_train_tiles: Optional[int] = None,
    max_val_tiles: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Path:
    # 1. Caricamento configurazione centralizzata da YAML
    cfg = load_config(config_path)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    m_cfg = cfg["imputer_model"]
    t_cfg = cfg["imputer_training"]
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
    clip_min = interp_cfg.get("clip_min", -1.0)
    clip_max = interp_cfg.get("clip_max", 1.0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(" TRAINING HISTORY IMPUTER (Bi-GRU + 5d Orbit + Climate Embedding)")
    print("=" * 65)
    print(f"[*] Config File      : {config_path}")
    print(f"[*] Project Root     : {cfg['project_root']}")
    print(f"[*] Device           : {device}")
    print(f"[*] Data Directory   : {data_dir}")
    print(f"[*] Checkpoint Target: {target_save_path}")

    # 2. Tile Partitioning (Split geografico disgiunto e stratificato per clima)
    all_files = sorted(list(data_dir.glob("*.npz")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"Nessun file .npz trovato nella cartella:\n  {data_dir.resolve()}")

    required_tiles = num_train_tiles + num_val_tiles
    if len(all_files) < required_tiles:
        raise ValueError(
            f"Trovati {len(all_files)} file .npz in {data_dir}, ma ne sono richiesti almeno {required_tiles} "
            f"(train={num_train_tiles}, val={num_val_tiles})."
        )

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

    print(f"[*] Train Tiles: {len(train_files)} | Val Tiles: {len(val_files)} (su {len(all_files)} totali)")
    if max_train_tiles is None and max_val_tiles is None:
        print(f"[*] Climi Train: {split_summary['train']}")
        print(f"[*] Climi Val  : {split_summary['val']}")

    # 3. Scaler
    scaler = ArcsinhScaler(scaler_path)
    scaler_sha = file_sha256(scaler_path)
    print(f"[*] Scaler caricato da: {scaler_path}")

    # 4. Datasets & Loaders
    print("[*] Indicizzazione dataset in corso...")
    train_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=train_files,
        mask_dropout=True,
        climate_dropout_p=climate_dropout_p,
        seed=seed,
    )
    val_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=val_files,
        mask_dropout=True,
        climate_dropout_p=0.0,
        seed=seed + 1,
        loo_all_points=True,
    )
    print(f"[*] Campioni Train: {len(train_ds):,} | Val LOO Targets: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size_val, shuffle=True, collate_fn=collate_history_batch)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size_val, 256), shuffle=False, collate_fn=collate_history_batch)

    # 5. Modello & Optimizer
    arch = extract_arch("bigru", m_cfg)
    model = BiGRUHistoryImputer(**arch).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_mae = float("inf")
    target_save_path.parent.mkdir(parents=True, exist_ok=True)
    target_names = ("NDVI", "GNDVI", "EVI")

    print("\n" + "-" * 65)
    print(" INIZIO CICLO DI ADDESTRAMENTO")
    print("-" * 65)

    for epoch in range(1, epochs_val + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_recon = 0.0
        batches = 0

        for batch in train_loader:
            x = batch["input"].to(device)
            climate_id = batch["climate_id"].to(device)
            target = batch["target"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            optimizer.zero_grad()
            preds = model(x, climate_id=climate_id, padding_mask=padding_mask)
            loss, recon_loss = compute_masked_loss(
                preds, target, loss_mask, padding_mask, smoothness_weight=smoothness_weight
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)
        avg_train_recon = total_train_recon / max(1, batches)

        # Validazione deterministica LOO su tutti i punti clear-sky
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
                loss, _ = compute_masked_loss(
                    preds, target, loss_mask, padding_mask, smoothness_weight=smoothness_weight
                )
                total_val_loss += loss.item()
                val_batches += 1

                if loss_mask.sum() > 0:
                    p_masked = preds[loss_mask]
                    t_masked = target[loss_mask]
                    p_orig = scaler.torch_inverse_transform_targets(p_masked)
                    if clip_min is not None or clip_max is not None:
                        p_orig = torch.clamp(p_orig, min=clip_min, max=clip_max)
                    t_orig = scaler.torch_inverse_transform_targets(t_masked)
                    abs_err_sum += torch.abs(p_orig - t_orig).sum(dim=0).cpu().numpy()
                    count_masked += int(loss_mask.sum().item())

        avg_val_loss = total_val_loss / max(1, val_batches)
        per_idx_mae = {
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
                model_type="bigru",
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
        print(f"Epoch {epoch:02d}/{epochs_val:02d} [{elapsed:4.1f}s] | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} [{idx_str}]{flag}")

    print("-" * 65)
    print(f"[*] Training completato! Miglior checkpoint frozen salvato in:\n    {target_save_path}")

    if target_save_path.exists():
        model, _ = load_frozen_model(target_save_path, device=device, verify=True)

    demonstrate_imputation(model, val_ds, scaler, device)
    return target_save_path


def demonstrate_imputation(model, dataset, scaler, device):
    """
    Mostra un esempio concreto di sequenza passata prima e dopo l'imputazione.
    """
    model.eval()
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).to(device)
    climate_id = sample["climate_id"].unsqueeze(0).to(device)
    obs = sample["all_obs_mask"].numpy()
    target_clean = sample["target"].unsqueeze(0).to(device)
    L = sample["length"]

    with torch.no_grad():
        imputed_norm = model(x, climate_id=climate_id)[0, :L]
        imputed_orig = scaler.torch_inverse_transform_targets(imputed_norm).cpu().numpy()
        real_orig = scaler.torch_inverse_transform_targets(target_clean[0, :L]).cpu().numpy()

    print("\n--- ESEMPIO DI RICOSTRUZIONE/IMPUTAZIONE DELLA STORIA (Campione Val 0) ---")
    print(f"{'Giorno':<8} | {'Stato Reale':<14} | {'NDVI Reale':<12} | {'NDVI Imputato':<14} | {'GNDVI Imputato':<14} | {'EVI Imputato':<14}")
    print("-" * 84)

    for t in range(min(12, L)):
        is_clear = obs[t]
        status = "Clear-Sky (Obs)" if is_clear else "NaN (Nubi/Gap)"
        real_str = f"{real_orig[t, 0]:.4f}" if is_clear else "   ---   "
        imp_ndvi = f"{imputed_orig[t, 0]:.4f}"
        imp_gndvi = f"{imputed_orig[t, 1]:.4f}"
        imp_evi = f"{imputed_orig[t, 2]:.4f}"
        print(f"Giorno {t:02d} | {status:<14} | {real_str:<12} | {imp_ndvi:<14} | {imp_gndvi:<14} | {imp_evi:<14}")
    print("-" * 84)


train_imputer = train_bigru_imputer


def main():
    parser = argparse.ArgumentParser(description="Train Bi-GRU History Imputer")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-tiles", type=int, default=None)
    parser.add_argument("--max-val-tiles", type=int, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_bigru_imputer(
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
