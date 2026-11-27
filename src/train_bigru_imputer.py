"""
Training and Validation Pipeline for the Bi-GRU Past Time-Series Imputer.
Configured centrally via configs/config.yaml.
Executes:
1. Spatial Tile-based Train/Val splitting.
2. ArcsinhScaler normalization fitted on the fitting tiles.
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
from src.data.scaler_fit import experiment_checkpoint_path, resolve_experiment_scaler
from src.data.splits import get_experiment_split
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.weather_preprocessor import configure_imputer_weather
from src.models.bigru_imputer import compute_masked_loss
from src.models.history_imputer import BiGRUHistoryImputer
from src.utils.checkpoints import complete_training_checkpoint


def train_bigru_imputer(
    config_path: Union[str, Path] = "configs/config.yaml",
    device: Optional[str] = None,
):
    # 1. Caricamento configurazione centralizzata da YAML
    cfg = load_config(config_path)
    weather = configure_imputer_weather(cfg, "imputer_model")

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    m_cfg = cfg["imputer_model"]
    t_cfg = cfg["imputer_training"]

    data_dir = paths_cfg["train_data_dir"]
    save_path = experiment_checkpoint_path(cfg, weather, paths_cfg["checkpoints_dir"] / t_cfg["checkpoint_name"])

    climate_dropout_p = data_cfg["climate_dropout_p"]
    seed = data_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    train_files, val_files, split_summary = get_experiment_split(cfg)
    print(f"[*] Train Tiles: {len(train_files)} | Val Tiles: {len(val_files)}")
    print(f"[*] Climi Train: {split_summary['train']}")
    print(f"[*] Climi Val  : {split_summary['val']}")

    # 3. Scaler
    scaler = resolve_experiment_scaler(cfg, weather, train_files, val_files)

    # 4. Datasets & Loaders
    print("[*] Indicizzazione dataset in corso...")
    train_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=train_files,
        mask_dropout=True,
        climate_dropout_p=climate_dropout_p,
        seed=seed,
        weather_preprocessor=weather,
    )
    val_ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=val_files,
        mask_dropout=True,
        climate_dropout_p=0.0,
        seed=seed + 1,
        weather_preprocessor=weather,
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

    training_start = time.perf_counter()
    optimizer_steps = 0
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
            optimizer_steps += 1

            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)
        avg_train_recon = total_train_recon / max(1, batches)

        # Validazione onesta su punti clear-sky mascherati
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        abs_err_sum = np.zeros(3, dtype=np.float64)
        count_masked = 0
        target_names = ("NDVI", "GNDVI", "EVI")

        val_ds.rng = np.random.RandomState(seed + 1)
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
                    abs_err_sum += torch.abs(p_orig - t_orig).sum(dim=0).cpu().numpy()
                    count_masked += int(loss_mask.sum().item())

        avg_val_loss = total_val_loss / max(1, val_batches)
        per_idx_mae = {
            name: float(abs_err_sum[i] / max(1, count_masked))
            for i, name in enumerate(target_names)
        }
        avg_val_mae = float(np.mean(list(per_idx_mae.values())))
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
                "scaler": scaler.to_dict(),
                "val_loss": best_val_loss,
                "val_mae_orig": avg_val_mae,
                "val_mae_per_index": per_idx_mae,
            }, save_path)

        flag = " [BEST SAVED]" if is_best else ""
        idx_str = " ".join(f"{n}={per_idx_mae[n]:.4f}" for n in target_names)
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:4.1f}s] | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} [{idx_str}]{flag}")

    if best_val_loss == float("inf") or optimizer_steps == 0:
        raise RuntimeError("Training produced no usable Bi-GRU checkpoint")
    best_ckpt = complete_training_checkpoint(
        save_path, epochs=epochs, optimizer_steps=optimizer_steps,
        seconds=time.perf_counter() - training_start, seed=seed,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    print("-" * 65)
    print(f"[*] Training completato! Miglior checkpoint salvato in:\n    {save_path}")

    val_ds.rng = np.random.RandomState(seed + 1)
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


train_imputer = train_bigru_imputer


if __name__ == "__main__":
    train_bigru_imputer()
