"""
Training and Validation Pipeline for the Past Time-Series Imputer.
Executes:
1. Spatial Tile-based Train/Val splitting.
2. ArcsinhScaler normalization using existing train scaler.json.
3. Self-supervised training with masked reconstruction loss + smoothness penalty.
4. Checkpointing and evaluation on original scale.
"""

from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.data.transforms import ArcsinhScaler
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.models.history_imputer import BiGRUHistoryImputer


def compute_masked_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    smoothness_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes masked L1 reconstruction loss on supervised points plus temporal smoothness.
    """
    # 1. Masked Reconstruction Loss
    # loss_mask is (B, L)
    if loss_mask.sum() > 0:
        p_masked = preds[loss_mask]      # shape (K, 3)
        t_masked = targets[loss_mask]    # shape (K, 3)
        recon_loss = nn.functional.smooth_l1_loss(p_masked, t_masked, reduction="mean")
    else:
        recon_loss = torch.tensor(0.0, device=preds.device)

    # 2. Temporal Smoothness Regularization (vegetation does not jump discontinuously day-to-day)
    # Difference between consecutive timesteps
    diff = preds[:, 1:, :] - preds[:, :-1, :]  # (B, L-1, 3)
    valid_diff_mask = (padding_mask[:, 1:] & padding_mask[:, :-1]).unsqueeze(-1).expand_as(diff)
    if valid_diff_mask.sum() > 0:
        smooth_loss = torch.mean(diff[valid_diff_mask] ** 2)
    else:
        smooth_loss = torch.tensor(0.0, device=preds.device)

    total_loss = recon_loss + smoothness_weight * smooth_loss
    return total_loss, recon_loss


def train_imputer(
    data_dir: str = "DATA/train_multi_output_clear_sky",
    scaler_path: str = "DATA/train_multi_output_clear_sky/scaler.json",
    num_train_tiles: int = 15,
    num_val_tiles: int = 3,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_path: str = "checkpoints/history_imputer_best.pt",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training History Imputer on device: {device}")

    # 1. Tile Partitioning (Punto 1: Split a prova di Data Leakage)
    all_files = sorted(list(Path(data_dir).glob("*.npz")))
    assert len(all_files) >= (num_train_tiles + num_val_tiles), "Not enough tile files!"

    train_files = all_files[:num_train_tiles]
    val_files = all_files[num_train_tiles : num_train_tiles + num_val_tiles]
    print(f"[*] Train Tiles: {len(train_files)} | Val Tiles: {len(val_files)}")

    # 2. Scaler (Punto 2: uso dello scaler esistente)
    scaler = ArcsinhScaler(scaler_path)
    print(f"[*] Scaler loaded successfully from {scaler_path}")

    # 3. Datasets & Loaders (Punto 3: Gestione NaN e mascheramento)
    print("[*] Indexing datasets...")
    train_ds = HistoryImputationDataset(data_dir, scaler=scaler, file_list=train_files, mask_dropout=True)
    val_ds = HistoryImputationDataset(data_dir, scaler=scaler, file_list=val_files, mask_dropout=False)
    print(f"[*] Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_history_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_history_batch)

    # 4. Model & Optimizer
    model = BiGRUHistoryImputer(input_dim=27, climate_dim=8, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" INIZIO TRAINING MODELLO IMPUTAZIONE PASSATO (Bi-GRU + 5d Orbit + Climate)")
    print("=" * 65)

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
            loss, recon_loss = compute_masked_loss(preds, target, loss_mask, padding_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)
        avg_train_recon = total_train_recon / max(1, batches)

        # Validation
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
                loss, _ = compute_masked_loss(preds, target, loss_mask, padding_mask)
                total_val_loss += loss.item()
                val_batches += 1

                # Calcolo MAE su scala originale (denormalizzata)
                if loss_mask.sum() > 0:
                    p_masked = preds[loss_mask]      # shape (K, 3)
                    t_masked = target[loss_mask]     # shape (K, 3)
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
                "val_loss": best_val_loss,
                "val_mae_orig": avg_val_mae,
            }, save_path)

        flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:4.1f}s] | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE (Orig Scale): {avg_val_mae:.4f}{flag}")

    print("=" * 65)
    print(f"[*] Training completato! Miglior checkpoint salvato in: {save_path}")

    # Dimostrazione pratica su un campione
    demonstrate_imputation(model, val_ds, scaler, device)


def demonstrate_imputation(model, dataset, scaler, device):
    """
    Mostra un esempio concreto di sequenza passata prima e dopo l'imputazione.
    """
    model.eval()
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).to(device)
    climate_id = sample["climate_id"].unsqueeze(0).to(device)
    obs_mask = sample["all_obs_mask"].unsqueeze(0).to(device)
    target_clean = sample["target"].unsqueeze(0).to(device)
    L = sample["length"]

    with torch.no_grad():
        imputed_norm = model(x, climate_id=climate_id)[0, :L]
        imputed_orig = scaler.torch_inverse_transform_targets(imputed_norm).cpu().numpy()
        real_orig = scaler.torch_inverse_transform_targets(target_clean[0, :L]).cpu().numpy()
        obs = sample["all_obs_mask"][:L].numpy()

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
