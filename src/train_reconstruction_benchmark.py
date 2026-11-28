"""
Unified Data-Integrity Audit, Apples-to-Apples Training, Freezing, and Multi-Split
Leave-One-Out Benchmark for History Reconstruction (Imputation) Models:
  - BiGRUHistoryImputer
  - MAEHistoryImputer
  - Classical interpolation baselines (linear, spline/pchip, forward_fill,
    backward_fill, nearest, mean, median)

Protocol guarantees:
  1. Training & model selection use ONLY the 70-tile train / 14-tile val_internal
     partition inside DATA/train_multi_output_clear_sky.
  2. External splits (id, ood_s, ood_t) are strictly read-only evaluation splits.
  3. Only `history` (3 sparse clear-sky past observations + daily past weather) is
     ever read; `future`, `future_noise`, and `target` are never touched.
  4. Winning & per-architecture checkpoints are saved in a SHA-256 verified frozen
     format (weights_only=True compatible, eval mode, requires_grad=False).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bigru_imputer import compute_masked_loss
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.interpolation import InterpolationFactory
from src.data.splits import get_train_val_tile_split
from src.data.transforms import ArcsinhScaler
from src.models.frozen import (
    build_reconstruction_model,
    dump_json,
    extract_arch,
    file_sha256,
    freeze_model,
    load_frozen_model,
    save_frozen_checkpoint,
)
from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer
from src.utils.config import load_config

TARGET_NAMES: Tuple[str, ...] = ("NDVI", "GNDVI", "EVI")
POS_NAMES: Dict[int, str] = {0: "first", 1: "middle", 2: "last"}


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_weather_mask(padding_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0.0:
        return torch.zeros_like(padding_mask, dtype=torch.bool)
    probs = torch.rand(padding_mask.shape, device=padding_mask.device)
    return (probs < ratio) & padding_mask.bool()


def _collect_csv_sources(file_list: Sequence[Path]) -> set:
    csvs = set()
    for fpath in file_list:
        jpath = fpath.with_suffix(".json")
        if jpath.exists():
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for rec in meta.get("metadata_records", []):
                if "csv" in rec:
                    csvs.add(rec["csv"])
    return csvs


def run_data_integrity_audit(
    cfg: Dict[str, Any],
    train_files: List[Path],
    val_files: List[Path],
    scaler: ArcsinhScaler,
    max_samples_check: int = 32,
) -> Dict[str, Any]:
    """
    Verifies zero data leakage, strict scaler provenance, mask-dropout integrity,
    and batch-padding invariance before any training starts.
    """
    paths_cfg = cfg["paths"]
    train_dir = Path(paths_cfg["train_data_dir"]).resolve()
    scaler_path = Path(paths_cfg["scaler_path"]).resolve()

    # 1. Scaler must come strictly from train_data_dir
    assert scaler_path.exists(), f"Scaler not found: {scaler_path}"
    assert scaler_path.parent == train_dir, (
        f"Scaler provenance violation: {scaler_path} is not inside {train_dir}"
    )
    scaler_sha = file_sha256(scaler_path)

    # 2. Disjoint train vs val_internal by path, tile stem, and source minicube CSV
    train_resolved = {p.resolve() for p in train_files}
    val_resolved = {p.resolve() for p in val_files}
    assert train_resolved.isdisjoint(val_resolved), "Train and val resolved paths overlap!"

    train_stems = {p.stem for p in train_files}
    val_stems = {p.stem for p in val_files}
    assert train_stems.isdisjoint(val_stems), "Train and val tile stems overlap!"

    train_csvs = _collect_csv_sources(train_files)
    val_csvs = _collect_csv_sources(val_files)
    assert train_csvs.isdisjoint(val_csvs), "Train and val source CSVs overlap!"

    # 3. Disjoint resolved paths against external evaluation splits
    ext_dirs = {
        "id": Path(paths_cfg["val_id_data_dir"]),
        "ood_s": Path(paths_cfg["ood_s_data_dir"]),
        "ood_t": Path(paths_cfg["ood_t_data_dir"]),
    }
    ext_counts: Dict[str, int] = {}
    for split_name, split_dir in ext_dirs.items():
        ext_files = sorted(split_dir.glob("*.npz"))
        ext_resolved = {p.resolve() for p in ext_files}
        assert train_resolved.isdisjoint(ext_resolved), f"Train overlaps {split_name} files!"
        assert val_resolved.isdisjoint(ext_resolved), f"Val overlaps {split_name} files!"
        if split_name == "ood_s":
            ext_stems = {p.stem for p in ext_files}
            assert train_stems.isdisjoint(ext_stems), "Train and ood_s share spatial tiles!"
            assert val_stems.isdisjoint(ext_stems), "Val and ood_s share spatial tiles!"
        ext_counts[split_name] = len(ext_files)

    # 4. Mask-dropout & no-future-leakage check on HistoryImputationDataset
    audit_ds = HistoryImputationDataset(
        train_dir,
        scaler=scaler,
        file_list=val_files[:1],
        mask_dropout=True,
        climate_dropout_p=0.0,
        seed=123,
        loo_all_points=True,
    )
    for fpath, cache_entry in audit_ds.file_cache.items():
        assert "future" not in cache_entry and "target" not in cache_entry, (
            f"HistoryImputationDataset loaded future/target fields from {fpath}!"
        )

    checked = 0
    for i in range(min(max_samples_check, len(audit_ds))):
        item = audit_ds[i]
        x = item["input"].numpy()
        loss_m = item["loss_mask"].numpy()
        obs_m = item["all_obs_mask"].numpy()
        assert int(loss_m.sum()) == 1, f"Expected 1 masked point, got {loss_m.sum()}"
        drop_t = int(np.where(loss_m)[0][0])
        assert np.allclose(x[drop_t, 18:21], 0.0, atol=1e-7), "Target leaked into input at dropped timestep!"
        assert np.allclose(x[drop_t, 21:24], 0.0, atol=1e-7), "Mask flag leaked at dropped timestep!"
        earlier_obs = np.where(obs_m[:drop_t])[0]
        if len(earlier_obs) > 0:
            expected_dt = float(drop_t - int(earlier_obs[-1]))
            assert np.isclose(x[drop_t, 24], expected_dt, atol=1e-5), (
                f"Orbital delta_t leaked dropped observation: {x[drop_t, 24]} != {expected_dt}"
            )
        checked += 1

    # 5. Batch-padding invariance check for both BiGRU and MAE
    padding_diffs: Dict[str, float] = {}
    lengths = [audit_ds[i]["length"] for i in range(min(len(audit_ds), 120))]
    short_idx = int(np.argmin(lengths))
    long_idx = int(np.argmax(lengths))
    short_item = audit_ds[short_idx]
    long_item = audit_ds[long_idx]
    solo = collate_history_batch([short_item])
    mixed = collate_history_batch([short_item, long_item])

    for mtype, cfg_key in (("bigru", "imputer_model"), ("mae", "mae_model")):
        arch = extract_arch(mtype, cfg[cfg_key])
        m = build_reconstruction_model(mtype, arch).eval()
        with torch.no_grad():
            p_solo = m(solo["input"], climate_id=solo["climate_id"], padding_mask=solo["padding_mask"])[
                0, : short_item["length"]
            ]
            p_mix = m(mixed["input"], climate_id=mixed["climate_id"], padding_mask=mixed["padding_mask"])[
                0, : short_item["length"]
            ]
        diff = float((p_solo - p_mix).abs().max().item())
        assert diff < 1e-5, f"{mtype} failed batch-padding invariance: max diff={diff:.3e}"
        padding_diffs[mtype] = diff

    return {
        "scaler_path": str(scaler_path),
        "scaler_sha256": scaler_sha,
        "train_tiles": len(train_files),
        "val_tiles": len(val_files),
        "train_csvs": len(train_csvs),
        "val_csvs": len(val_csvs),
        "tile_overlap": 0,
        "csv_overlap": 0,
        "external_split_tile_counts": ext_counts,
        "mask_dropout_samples_verified": checked,
        "padding_invariance_max_diff": padding_diffs,
    }


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drop_ranks: np.ndarray,
    split_name: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    Computes MAE, RMSE, WMAPE on original scale:
      - scope in [ALL, NDVI, GNDVI, EVI]
      - position in [ALL, first, middle, last]
    """
    rows: List[Dict[str, Any]] = []
    pos_filters: List[Tuple[str, np.ndarray]] = [("ALL", np.ones(len(y_true), dtype=bool))]
    for rank, pname in POS_NAMES.items():
        mask = drop_ranks == rank
        if np.any(mask):
            pos_filters.append((pname, mask))

    for pos_name, pmask in pos_filters:
        yt = y_true[pmask].astype(np.float64)
        yp = y_pred[pmask].astype(np.float64)
        n_pts = int(len(yt))
        err = yp - yt

        # Overall across the 3 vegetation indices
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "scope": "ALL",
                "position": pos_name,
                "n_points": n_pts,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "WMAPE": float(np.sum(np.abs(err)) / max(float(np.sum(np.abs(yt))), 1e-8)),
            }
        )
        # Per vegetation index
        for idx, tname in enumerate(TARGET_NAMES):
            e_i = err[:, idx]
            yt_i = yt[:, idx]
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "scope": tname,
                    "position": pos_name,
                    "n_points": n_pts,
                    "MAE": float(np.mean(np.abs(e_i))),
                    "RMSE": float(np.sqrt(np.mean(e_i ** 2))),
                    "WMAPE": float(np.sum(np.abs(e_i)) / max(float(np.sum(np.abs(yt_i))), 1e-8)),
                }
            )
    return rows


def evaluate_neural_loo(
    model: torch.nn.Module,
    val_loader: DataLoader,
    scaler: ArcsinhScaler,
    device: str,
    smoothness_weight: float = 0.05,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs deterministic Leave-One-Out evaluation and returns:
      (avg_val_loss, y_true_orig (N, 3), y_pred_orig (N, 3), drop_ranks (N,))
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    trues_parts: List[np.ndarray] = []
    preds_parts: List[np.ndarray] = []
    ranks_parts: List[np.ndarray] = []

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
            total_loss += float(loss.item())
            batches += 1

            if torch.any(loss_mask):
                p_orig = scaler.torch_inverse_transform_targets(preds[loss_mask])
                if clip_min is not None or clip_max is not None:
                    p_orig = torch.clamp(p_orig, min=clip_min, max=clip_max)
                t_orig = scaler.torch_inverse_transform_targets(target[loss_mask])
                preds_parts.append(p_orig.cpu().numpy())
                trues_parts.append(t_orig.cpu().numpy())
                ranks_parts.append(batch["drop_ranks"].numpy())

    avg_loss = total_loss / max(1, batches)
    y_true = np.concatenate(trues_parts, axis=0)
    y_pred = np.concatenate(preds_parts, axis=0)
    ranks = np.concatenate(ranks_parts, axis=0)
    return avg_loss, y_true, y_pred, ranks


def evaluate_classical_baseline_loo(
    file_list: Sequence[Path],
    strategy: str,
    clip_min: Optional[float] = -1.0,
    clip_max: Optional[float] = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates a classical interpolation strategy in exact Leave-One-Out mode on all
    windows in `file_list`. Uses a vectorized fast path for standard 3-observation
    windows and falls back to InterpolationFactory for any window with != 3 obs.
    Returns (y_true_orig (N, 3), y_pred_orig (N, 3), drop_ranks (N,)).
    """
    strat = strategy.strip().lower().replace("-", "_")
    fallback_interp = InterpolationFactory.create(strat, clip_min=clip_min, clip_max=clip_max)

    trues_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []
    ranks_list: List[np.ndarray] = []

    for fpath in file_list:
        with np.load(fpath, allow_pickle=True) as d:
            histories = d["history"]
            masks = d["history_mask"]

        n_win = len(histories)
        t_pos = np.empty((n_win, 3), dtype=np.float64)
        v_obs = np.empty((n_win, 3, 3), dtype=np.float64)
        fast_idx: List[int] = []
        slow_idx: List[int] = []

        for i in range(n_win):
            obs_i = np.where(~masks[i][:, 18])[0]
            if len(obs_i) == 3:
                t_pos[len(fast_idx)] = obs_i
                v_obs[len(fast_idx)] = histories[i][obs_i, 18:21]
                fast_idx.append(i)
            elif len(obs_i) >= 2:
                slow_idx.append(i)

        nf = len(fast_idx)
        if nf > 0:
            tp = t_pos[:nf]
            vo = v_obs[:nf]
            t0, t1, t2 = tp[:, 0:1], tp[:, 1:2], tp[:, 2:3]
            v0, v1, v2 = vo[:, 0, :], vo[:, 1, :], vo[:, 2, :]

            if strat in ("linear", "interp1d", "piecewise_linear", "spline", "pchip", "cubic", "akima"):
                p0 = v1
                alpha = (t1 - t0) / np.maximum(t2 - t0, 1e-8)
                p1 = v0 + alpha * (v2 - v0)
                p2 = v1
            elif strat in ("forward_fill", "ffill", "locf", "last_observation_carried_forward"):
                p0 = v1
                p1 = v0
                p2 = v1
            elif strat in ("backward_fill", "bfill", "nocb", "next_observation_carried_backward"):
                p0 = v1
                p1 = v2
                p2 = v1
            elif strat in ("nearest", "nearest_neighbor"):
                p0 = v1
                choose_left = (t1 - t0) <= (t2 - t1)
                p1 = np.where(choose_left, v0, v2)
                p2 = v1
            elif strat in ("mean", "average", "median"):
                p0 = 0.5 * (v1 + v2)
                p1 = 0.5 * (v0 + v2)
                p2 = 0.5 * (v0 + v1)
            else:
                p0 = p1 = p2 = None

            if p0 is not None:
                pred_3 = np.stack([p0, p1, p2], axis=1)  # (nf, 3, 3)
                if clip_min is not None or clip_max is not None:
                    pred_3 = np.clip(pred_3, a_min=clip_min, a_max=clip_max)
                preds_list.append(pred_3.reshape(-1, 3).astype(np.float32))
                trues_list.append(vo.reshape(-1, 3).astype(np.float32))
                ranks_list.append(np.tile(np.array([0, 1, 2], dtype=np.int64), nf))
            else:
                slow_idx = fast_idx + slow_idx

        for i in slow_idx:
            h_raw = histories[i].astype(np.float32)
            m_raw = masks[i].astype(bool)
            obs_i = np.where(~m_raw[:, 18])[0]
            for r, drop_t in enumerate(obs_i):
                h_in = h_raw.copy()
                m_in = m_raw.copy()
                h_in[drop_t, 18:21] = np.nan
                m_in[drop_t, 18:21] = True
                filled = fallback_interp.interpolate(h_in, mask=m_in, columns=[18, 19, 20])
                preds_list.append(filled[drop_t : drop_t + 1, 18:21])
                trues_list.append(h_raw[drop_t : drop_t + 1, 18:21])
                ranks_list.append(np.array([r], dtype=np.int64))

    return (
        np.concatenate(trues_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(ranks_list, axis=0),
    )


def train_model_run(
    model_type: str,
    seed: int,
    cfg: Dict[str, Any],
    train_ds: HistoryImputationDataset,
    val_loader: DataLoader,
    scaler: ArcsinhScaler,
    scaler_sha: str,
    train_files: Sequence[Path],
    val_files: Sequence[Path],
    save_path: Path,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    device: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Trains either 'bigru' or 'mae' under unified budget and saves the best epoch as
    a frozen checkpoint at `save_path`.
    """
    _set_seed(seed)
    train_ds.rng = np.random.RandomState(seed)
    dl_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_history_batch,
        generator=dl_gen,
    )

    interp_cfg = cfg.get("interpolation", {})
    clip_min = interp_cfg.get("clip_min", -1.0)
    clip_max = interp_cfg.get("clip_max", 1.0)

    if model_type == "bigru":
        arch = extract_arch("bigru", cfg["imputer_model"])
        t_cfg = cfg["imputer_training"]
        visible_recon_weight = 0.0
        weather_mask_ratio = 0.0
        weather_recon_weight = 0.0
    elif model_type == "mae":
        arch = extract_arch("mae", cfg["mae_model"])
        t_cfg = cfg["mae_training"]
        visible_recon_weight = float(t_cfg.get("visible_recon_weight", 0.25))
        weather_mask_ratio = float(t_cfg.get("weather_mask_ratio", 0.15))
        weather_recon_weight = float(t_cfg.get("weather_recon_weight", 0.1))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    lr = float(t_cfg["learning_rate"])
    weight_decay = float(t_cfg["weight_decay"])
    smoothness_weight = float(t_cfg["smoothness_weight"])

    model = build_reconstruction_model(model_type, arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_mae = float("inf")
    best_summary: Dict[str, Any] = {}
    history_rows: List[Dict[str, Any]] = []
    no_improve = 0

    print("\n" + "=" * 78)
    print(f" TRAINING {model_type.upper()} | seed={seed} | params={n_params:,} | epochs={epochs} | device={device}")
    print("=" * 78)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        total_target_recon = 0.0
        batches = 0

        for batch in train_loader:
            x = batch["input"].to(device)
            climate_id = batch["climate_id"].to(device)
            target = batch["target"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            optimizer.zero_grad()
            if model_type == "mae":
                weather_mask = _sample_weather_mask(padding_mask, weather_mask_ratio)
                preds, pred_weather = model(
                    x,
                    climate_id=climate_id,
                    padding_mask=padding_mask,
                    weather_mask=weather_mask,
                    return_weather=True,
                )
                target_loss, target_recon = compute_masked_loss(
                    preds, target, loss_mask, padding_mask, smoothness_weight=smoothness_weight
                )
                visible_mask = obs_mask & (~loss_mask)
                if visible_recon_weight > 0.0 and torch.any(visible_mask):
                    vis_loss = F.smooth_l1_loss(preds[visible_mask], target[visible_mask], reduction="mean")
                else:
                    vis_loss = torch.tensor(0.0, device=device)

                if weather_recon_weight > 0.0 and torch.any(weather_mask):
                    w_loss = F.smooth_l1_loss(
                        pred_weather[weather_mask], x[:, :, :11][weather_mask], reduction="mean"
                    )
                else:
                    w_loss = torch.tensor(0.0, device=device)

                loss = target_loss + visible_recon_weight * vis_loss + weather_recon_weight * w_loss
            else:
                preds = model(x, climate_id=climate_id, padding_mask=padding_mask)
                loss, target_recon = compute_masked_loss(
                    preds, target, loss_mask, padding_mask, smoothness_weight=smoothness_weight
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += float(loss.item())
            total_target_recon += float(target_recon.item())
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)
        avg_train_recon = total_target_recon / max(1, batches)

        val_loss, yt_val, yp_val, _ = evaluate_neural_loo(
            model=model,
            val_loader=val_loader,
            scaler=scaler,
            device=device,
            smoothness_weight=smoothness_weight,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        err = yp_val.astype(np.float64) - yt_val.astype(np.float64)
        per_idx_mae = {
            name: float(np.mean(np.abs(err[:, i]))) for i, name in enumerate(TARGET_NAMES)
        }
        val_mae_orig = float(np.mean(np.abs(err)))
        val_rmse_orig = float(np.sqrt(np.mean(err ** 2)))
        cur_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - t0

        scheduler.step(val_mae_orig)

        is_best = val_mae_orig < best_val_mae
        if is_best:
            best_val_mae = val_mae_orig
            no_improve = 0
            best_summary = {
                "model_type": model_type,
                "seed": seed,
                "best_epoch": epoch,
                "n_parameters": n_params,
                "val_loss": float(val_loss),
                "val_mae_orig": val_mae_orig,
                "val_rmse_orig": val_rmse_orig,
                "val_mae_per_index": per_idx_mae,
                "checkpoint_path": str(save_path),
            }
            sha = save_frozen_checkpoint(
                save_path,
                model=model,
                model_type=model_type,
                arch=arch,
                metadata={
                    "epoch": epoch,
                    "seed": seed,
                    "val_loss": float(val_loss),
                    "val_mae_orig": val_mae_orig,
                    "val_rmse_orig": val_rmse_orig,
                    "val_mae_per_index": per_idx_mae,
                    "scaler_sha256": scaler_sha,
                    "train_tiles": [p.name for p in train_files],
                    "val_tiles": [p.name for p in val_files],
                },
                full_config=cfg,
            )
            best_summary["state_dict_sha256"] = sha
        else:
            no_improve += 1

        history_rows.append(
            {
                "model": model_type,
                "seed": seed,
                "epoch": epoch,
                "lr": cur_lr,
                "elapsed_s": round(elapsed, 2),
                "train_loss": avg_train_loss,
                "train_target_recon": avg_train_recon,
                "val_loss": float(val_loss),
                "val_mae_orig": val_mae_orig,
                "val_rmse_orig": val_rmse_orig,
                "val_mae_NDVI": per_idx_mae["NDVI"],
                "val_mae_GNDVI": per_idx_mae["GNDVI"],
                "val_mae_EVI": per_idx_mae["EVI"],
                "is_best": is_best,
            }
        )

        flag = " [BEST SAVED]" if is_best else ""
        idx_str = " ".join(f"{n}={per_idx_mae[n]:.5f}" for n in TARGET_NAMES)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:5.1f}s] lr={cur_lr:.1e} | "
            f"Train={avg_train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"ValMAE={val_mae_orig:.5f} [{idx_str}]{flag}"
        )

        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            print(f"[*] Early stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
            break

    return best_summary, history_rows


def _build_markdown_leaderboard(
    metrics_df: pd.DataFrame,
    winner_info: Dict[str, Any],
    audit_report: Dict[str, Any],
    run_summaries: List[Dict[str, Any]],
) -> str:
    lines: List[str] = [
        "# Reconstruction (History Imputation) Benchmark Leaderboard",
        "",
        "## Winner (Selected Strictly on `val_internal`)",
        f"- **Winning Architecture**: `{winner_info['best_model']}` (seed `{winner_info['best_seed']}`)",
        f"- **`val_internal` Overall MAE**: `{winner_info['val_internal_mae']:.6f}`",
        f"- **Frozen Checkpoint**: `{winner_info['best_checkpoint']}`",
        f"- **Weights SHA-256**: `{winner_info['state_dict_sha256']}`",
        "",
        "## Data Integrity Audit",
        f"- Train tiles / minicubes: `{audit_report['train_tiles']}` / `{audit_report['train_csvs']}`",
        f"- Val tiles / minicubes: `{audit_report['val_tiles']}` / `{audit_report['val_csvs']}`",
        f"- Tile overlap: `{audit_report['tile_overlap']}` | CSV overlap: `{audit_report['csv_overlap']}`",
        f"- Scaler SHA-256: `{audit_report['scaler_sha256']}`",
        f"- Padding invariance max diff: `{audit_report['padding_invariance_max_diff']}`",
        "",
        "## Overall LOO Metrics (`scope=ALL`, `position=ALL`, original scale `[-1, 1]`)",
        "",
    ]

    overall = metrics_df[(metrics_df["scope"] == "ALL") & (metrics_df["position"] == "ALL")]
    for split in ("val_internal", "id", "ood_s", "ood_t"):
        sub = overall[overall["split"] == split].sort_values("MAE")
        if sub.empty:
            continue
        lines.append(f"### Split: `{split}` (n={int(sub.iloc[0]['n_points']):,} masked clear-sky points)")
        lines.append("| Rank | Model | MAE | RMSE | WMAPE |")
        lines.append("|---:|---|---:|---:|---:|")
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            lines.append(
                f"| {rank} | `{r['model']}` | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['WMAPE']:.6f} |"
            )
        lines.append("")

    if run_summaries:
        lines.append("## Trained Neural Runs")
        lines.append("| Model | Seed | Best Epoch | Params | Val MAE (orig) | Weights SHA-256 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for s in run_summaries:
            lines.append(
                f"| `{s['model_type']}` | {s['seed']} | {s['best_epoch']} | "
                f"{s['n_parameters']:,} | {s['val_mae_orig']:.6f} | `{s['state_dict_sha256'][:16]}...` |"
            )
        lines.append("")

    return "\n".join(lines)


def run_benchmark(
    config_path: Union[str, Path] = "configs/config.yaml",
    epochs: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    max_train_tiles: Optional[int] = None,
    max_val_tiles: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    frozen_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    interp_cfg = cfg.get("interpolation", {})
    bench_cfg = cfg.get("reconstruction_benchmark", {})

    project_root = Path(cfg["project_root"])
    model_list = [m.lower() for m in (models or bench_cfg.get("models", ["bigru", "mae"]))]
    seed_list = [int(s) for s in (seeds or bench_cfg.get("seeds", [data_cfg["seed"]]))]
    epochs_val = int(epochs if epochs is not None else bench_cfg.get("epochs", 20))
    patience = int(bench_cfg.get("early_stopping_patience", 6))
    eval_bs = int(eval_batch_size if eval_batch_size is not None else bench_cfg.get("eval_batch_size", 512))
    baselines = list(
        bench_cfg.get(
            "baselines",
            ["linear", "spline", "forward_fill", "backward_fill", "nearest", "mean", "median"],
        )
    )

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else project_root / bench_cfg.get("output_dir", "results/reconstruction_benchmark")
    )
    frz_dir = (
        Path(frozen_dir)
        if frozen_dir is not None
        else project_root / bench_cfg.get("frozen_dir", "checkpoints/reconstruction")
    )
    write_canonical = frozen_dir is None and max_train_tiles is None and max_val_tiles is None

    out_dir.mkdir(parents=True, exist_ok=True)
    frz_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_min = interp_cfg.get("clip_min", -1.0)
    clip_max = interp_cfg.get("clip_max", 1.0)

    train_dir = Path(paths_cfg["train_data_dir"])
    scaler_path = Path(paths_cfg["scaler_path"])
    scaler = ArcsinhScaler(scaler_path)
    scaler_sha = file_sha256(scaler_path)

    train_files, val_files, split_summary = get_train_val_tile_split(
        train_dir,
        num_val_tiles=int(data_cfg["num_val_tiles"]),
        seed=int(data_cfg["seed"]),
    )
    if int(data_cfg["num_train_tiles"]) < len(train_files):
        train_files = train_files[: int(data_cfg["num_train_tiles"])]
    if max_train_tiles is not None:
        train_files = train_files[:max_train_tiles]
    if max_val_tiles is not None:
        val_files = val_files[:max_val_tiles]

    print("=" * 78)
    print(" DATA INTEGRITY & ZERO-LEAKAGE AUDIT")
    print("=" * 78)
    audit_report = run_data_integrity_audit(cfg, train_files, val_files, scaler)
    print(f"[+] Train tiles: {audit_report['train_tiles']} ({audit_report['train_csvs']} csvs) | "
          f"Val tiles: {audit_report['val_tiles']} ({audit_report['val_csvs']} csvs)")
    print(f"[+] Tile/CSV/Split overlap: 0 | Mask-dropout verified: {audit_report['mask_dropout_samples_verified']} samples")
    print(f"[+] Padding invariance max diff: {audit_report['padding_invariance_max_diff']}")
    print(f"[+] Scaler SHA-256: {scaler_sha}")

    print("\n[*] Indexing train and deterministic LOO val_internal datasets...")
    train_ds = HistoryImputationDataset(
        train_dir,
        scaler=scaler,
        file_list=train_files,
        mask_dropout=True,
        climate_dropout_p=float(data_cfg["climate_dropout_p"]),
        seed=int(data_cfg["seed"]),
    )
    val_ds = HistoryImputationDataset(
        train_dir,
        scaler=scaler,
        file_list=val_files,
        mask_dropout=True,
        climate_dropout_p=0.0,
        seed=int(data_cfg["seed"]) + 1,
        loo_all_points=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_history_batch
    )
    print(f"[*] Train windows: {len(train_ds):,} | Val LOO masked targets: {len(val_ds):,}")

    # Train all requested (model_type, seed) combinations
    run_summaries: List[Dict[str, Any]] = []
    all_history_rows: List[Dict[str, Any]] = []
    best_by_type: Dict[str, Dict[str, Any]] = {}

    for mtype in model_list:
        default_bs = int(
            bench_cfg.get(
                "batch_size",
                cfg["imputer_training" if mtype == "bigru" else "mae_training"]["batch_size"],
            )
        )
        bs_val = int(batch_size if batch_size is not None else default_bs)
        for seed in seed_list:
            run_ckpt = frz_dir / f"{mtype}_seed{seed}_frozen.pt"
            summary, hist_rows = train_model_run(
                model_type=mtype,
                seed=seed,
                cfg=cfg,
                train_ds=train_ds,
                val_loader=val_loader,
                scaler=scaler,
                scaler_sha=scaler_sha,
                train_files=train_files,
                val_files=val_files,
                save_path=run_ckpt,
                epochs=epochs_val,
                batch_size=bs_val,
                early_stopping_patience=patience,
                device=device,
            )
            run_summaries.append(summary)
            all_history_rows.extend(hist_rows)

            if mtype not in best_by_type or summary["val_mae_orig"] < best_by_type[mtype]["val_mae_orig"]:
                best_by_type[mtype] = summary

    # Copy best per-architecture frozen checkpoints and verify every saved checkpoint
    frozen_model_paths: Dict[str, Path] = {}
    canonical_map = {
        "bigru": Path(paths_cfg["checkpoints_dir"]) / cfg["imputer_training"]["checkpoint_name"],
        "mae": Path(paths_cfg["checkpoints_dir"]) / cfg["mae_training"]["checkpoint_name"],
    }
    for mtype, best_info in best_by_type.items():
        src_ckpt = Path(best_info["checkpoint_path"])
        dst_frz = frz_dir / f"{mtype}_frozen.pt"
        if src_ckpt.resolve() != dst_frz.resolve():
            shutil.copy2(src_ckpt, dst_frz)
        frozen_model_paths[mtype] = dst_frz

        if write_canonical and mtype in canonical_map:
            canonical_map[mtype].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst_frz, canonical_map[mtype])

    # Crown winner strictly on val_internal overall MAE
    winner_type = min(best_by_type, key=lambda k: best_by_type[k]["val_mae_orig"])
    winner_summary = best_by_type[winner_type]
    best_frz_path = frz_dir / "best_imputer_frozen.pt"
    shutil.copy2(frozen_model_paths[winner_type], best_frz_path)

    best_overall_rel = bench_cfg.get("best_overall_checkpoint", "checkpoints/best_reconstruction_imputer.pt")
    best_overall_path = Path(best_overall_rel)
    if not best_overall_path.is_absolute():
        best_overall_path = project_root / best_overall_path
    if write_canonical:
        best_overall_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_frz_path, best_overall_path)

    # Verify reload + frozen status on all saved checkpoints
    loaded_models: Dict[str, torch.nn.Module] = {}
    verify_paths = list(frozen_model_paths.values()) + [best_frz_path]
    if write_canonical:
        verify_paths.extend([canonical_map[m] for m in best_by_type if m in canonical_map])
        verify_paths.append(best_overall_path)

    for vp in verify_paths:
        m_loaded, _ = load_frozen_model(vp, device=device, verify=True)
        assert not m_loaded.training, f"Loaded model {vp} is not in eval mode!"
        assert all(not p.requires_grad for p in m_loaded.parameters()), (
            f"Loaded model {vp} still has requires_grad=True!"
        )
    for mtype, fpath in frozen_model_paths.items():
        loaded_models[mtype], _ = load_frozen_model(fpath, device=device, verify=True)

    # Comprehensive Leave-One-Out Benchmark across splits (history ONLY)
    print("\n" + "=" * 78)
    print(" MULTI-SPLIT LEAVE-ONE-OUT RECONSTRUCTION BENCHMARK (HISTORY ONLY)")
    print("=" * 78)

    eval_splits: Dict[str, Tuple[Path, List[Path]]] = {
        "val_internal": (train_dir, val_files),
    }
    ext_split_dirs = {
        "id": Path(paths_cfg["val_id_data_dir"]),
        "ood_s": Path(paths_cfg["ood_s_data_dir"]),
        "ood_t": Path(paths_cfg["ood_t_data_dir"]),
    }
    for sp_name, sp_dir in ext_split_dirs.items():
        sp_files = sorted(sp_dir.glob("*.npz"))
        if max_val_tiles is not None:
            sp_files = sp_files[:max_val_tiles]
        eval_splits[sp_name] = (sp_dir, sp_files)

    metric_rows: List[Dict[str, Any]] = []
    for sp_name, (sp_dir, sp_files) in eval_splits.items():
        print(f"\n[*] Evaluating split '{sp_name}' ({len(sp_files)} tiles)...")
        sp_ds = (
            val_ds
            if sp_name == "val_internal"
            else HistoryImputationDataset(
                sp_dir,
                scaler=scaler,
                file_list=sp_files,
                mask_dropout=True,
                climate_dropout_p=0.0,
                seed=int(data_cfg["seed"]) + 1,
                loo_all_points=True,
            )
        )
        sp_loader = (
            val_loader
            if sp_name == "val_internal"
            else DataLoader(sp_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_history_batch)
        )

        for mtype, frozen_m in loaded_models.items():
            sw = float(cfg["imputer_training" if mtype == "bigru" else "mae_training"]["smoothness_weight"])
            _, yt, yp, dr = evaluate_neural_loo(
                model=frozen_m,
                val_loader=sp_loader,
                scaler=scaler,
                device=device,
                smoothness_weight=sw,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            metric_rows.extend(_compute_regression_metrics(yt, yp, dr, sp_name, mtype))

        for b_strat in baselines:
            yt_b, yp_b, dr_b = evaluate_classical_baseline_loo(
                sp_files, strategy=b_strat, clip_min=clip_min, clip_max=clip_max
            )
            metric_rows.extend(_compute_regression_metrics(yt_b, yp_b, dr_b, sp_name, b_strat))

    metrics_df = pd.DataFrame(metric_rows)
    history_df = pd.DataFrame(all_history_rows)

    metrics_csv = out_dir / "reconstruction_metrics.csv"
    history_csv = out_dir / "training_history.csv"
    summary_json = out_dir / "reconstruction_summary.json"
    leaderboard_md = out_dir / "leaderboard.md"

    metrics_df.to_csv(metrics_csv, index=False)
    history_df.to_csv(history_csv, index=False)

    winner_info = {
        "best_model": winner_type,
        "best_seed": int(winner_summary["seed"]),
        "best_epoch": int(winner_summary["best_epoch"]),
        "val_internal_mae": float(winner_summary["val_mae_orig"]),
        "val_internal_rmse": float(winner_summary["val_rmse_orig"]),
        "val_internal_mae_per_index": winner_summary["val_mae_per_index"],
        "best_checkpoint": str(best_overall_path if write_canonical else best_frz_path),
        "frozen_dir_checkpoint": str(best_frz_path),
        "state_dict_sha256": winner_summary["state_dict_sha256"],
    }

    summary_payload = {
        "audit": audit_report,
        "split_climate_summary": split_summary,
        "winner": winner_info,
        "best_per_architecture": best_by_type,
        "all_runs": run_summaries,
        "frozen_checkpoints": {k: str(v) for k, v in frozen_model_paths.items()},
        "canonical_checkpoints": {
            **({k: str(canonical_map[k]) for k in best_by_type if write_canonical}),
            **({"best_overall": str(best_overall_path)} if write_canonical else {}),
        },
        "overall_all_positions_metrics": metrics_df[
            (metrics_df["scope"] == "ALL") & (metrics_df["position"] == "ALL")
        ].to_dict(orient="records"),
    }
    dump_json(summary_payload, summary_json)

    md_text = _build_markdown_leaderboard(metrics_df, winner_info, audit_report, run_summaries)
    leaderboard_md.write_text(md_text, encoding="utf-8")

    # Print final summary leaderboard to stdout
    print("\n" + "=" * 78)
    print(f" WINNER (on val_internal): {winner_type.upper()} (seed={winner_info['best_seed']}, "
          f"MAE={winner_info['val_internal_mae']:.6f})")
    print("=" * 78)
    overall = metrics_df[(metrics_df["scope"] == "ALL") & (metrics_df["position"] == "ALL")]
    for sp in ("val_internal", "id", "ood_s", "ood_t"):
        sub = overall[overall["split"] == sp].sort_values("MAE")
        print(f"\n[{sp}]")
        for _, r in sub.iterrows():
            print(f"  {r['model']:<15s} | MAE={r['MAE']:.6f} | RMSE={r['RMSE']:.6f} | WMAPE={r['WMAPE']:.6f}")

    print(f"\n[+] Saved metrics CSV    : {metrics_csv}")
    print(f"[+] Saved training CSV   : {history_csv}")
    print(f"[+] Saved summary JSON   : {summary_json}")
    print(f"[+] Saved leaderboard MD : {leaderboard_md}")
    print(f"[+] Frozen checkpoints   : {frz_dir}")
    if write_canonical:
        print(f"[+] Canonical winner     : {best_overall_path}")

    return summary_payload


def main():
    parser = argparse.ArgumentParser(
        description="Unified audit, training, freezing, and LOO benchmark for history reconstruction models"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-train-tiles", type=int, default=None)
    parser.add_argument("--max-val-tiles", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--frozen-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        epochs=args.epochs,
        seeds=args.seeds,
        models=args.models,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_train_tiles=args.max_train_tiles,
        max_val_tiles=args.max_val_tiles,
        output_dir=args.output_dir,
        frozen_dir=args.frozen_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
