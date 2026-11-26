"""
Training and Validation Pipeline for the Past Time-Series Imputer.
Configured centrally via configs/config.yaml.
Executes:
1. Spatial Tile-based Train/Val splitting.
2. ArcsinhScaler normalization using existing train scaler.json.
3. Sentinel-2 5-day orbital revisit dynamics + Climate Embedding.
4. Self-supervised training with masked reconstruction loss + smoothness penalty.
5. Checkpointing and evaluation on original scale.
"""

import sys
from pathlib import Path
import time
from typing import Optional, Union, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config
from src.data.transforms import ArcsinhScaler
from src.data.splits import get_train_val_tile_split
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.bigru_imputer import compute_masked_loss
from src.models.history_imputer import BiGRUHistoryImputer


def train_imputer(
    config_path: Union[str, Path] = "configs/config.yaml",
    device: Optional[str] = None,
):
    # 1. Caricamento configurazione centralizzata da YAML
    cfg = load_config(config_path)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    m_cfg = cfg["imputer_model"]
    t_cfg = cfg["imputer_training"]

    data_dir = paths_cfg["train_data_dir"]
    scaler_path = paths_cfg["scaler_path"]
    save_path = paths_cfg["checkpoints_dir"] / t_cfg["checkpoint_name"]

    num_train_tiles = data_cfg["num_train_tiles"]
    num_val_tiles = data_cfg["num_val_tiles"]
    climate_dropout_p = data_cfg["climate_dropout_p"]
    seed = data_cfg["seed"]

    epochs = t_cfg["epochs"]
    batch_size = t_cfg["batch_size"]
    lr = t_cfg["learning_rate"]
    weight_decay = t_cfg["weight_decay"]
    smoothness_weight = t_cfg["smoothness_weight"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(" TRAINING HISTORY IMPUTER (Bi-GRU + 5d Orbit + Climate Embedding)")
    print("=" * 65)
    print(f"[*] Config File      : {config_path}")
    print(f"[*] Project Root     : {cfg['project_root']}")
    print(f"[*] Device           : {device}")
    print(f"[*] Data Directory   : {data_dir}")
    print(f"[*] Checkpoint Target: {save_path}")

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
    print(f"[*] Train Tiles: {len(train_files)} | Val Tiles: {len(val_files)} (su {len(all_files)} totali)")
    print(f"[*] Climi Train: {split_summary['train']}")
    print(f"[*] Climi Val  : {split_summary['val']}")

    # 3. Scaler
    scaler = ArcsinhScaler(scaler_path)
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
        mask_dropout=False,
        climate_dropout_p=0.0,
        seed=seed,
    )
    print(f"[*] Campioni Train: {len(train_ds):,} | Campioni Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_history_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_history_batch)

    # 5. Modello & Optimizer
    model = BiGRUHistoryImputer(
        input_dim=m_cfg["input_dim"],
        climate_dim=m_cfg["climate_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        output_dim=m_cfg["output_dim"],
        dropout=m_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 65)
    print(" INIZIO CICLO DI ADDESTRAMENTO")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
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

        # Validazione
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        val_mae_orig = []

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
                    t_orig = scaler.torch_inverse_transform_targets(t_masked)
                    val_mae_orig.append(torch.mean(torch.abs(p_orig - t_orig)).item())

        avg_val_loss = total_val_loss / max(1, val_batches)
        avg_val_mae = np.mean(val_mae_orig) if val_mae_orig else 0.0
        elapsed = time.time() - t0

        scheduler.step(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "val_loss": best_val_loss,
                "val_mae_orig": avg_val_mae,
            }, save_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:4.1f}s] | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE (Scala Reale): {avg_val_mae:.4f}{flag}")

    print("-" * 65)
    print(f"[*] Training completato! Miglior checkpoint salvato in:\n    {save_path}")

    # Dimostrazione pratica su un campione di validazione
    demonstrate_imputation(model, val_ds, scaler, device)


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


if __name__ == "__main__":
    train_imputer()
