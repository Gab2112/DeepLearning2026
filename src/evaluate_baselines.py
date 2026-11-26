"""
Multi-split evaluation script for persistence baselines:
  - Fits ProbabilisticPersistenceBaseline strictly on the 70-tile training subset.
  - Evaluates PersistenceBaseline and ProbabilisticPersistenceBaseline on:
      1. val_internal (14 stratified tiles held out from train)
      2. id           (val_chopped_multi_output_clear_sky)
      3. ood_s        (ood-s_chopped_multi_output_clear_sky)
      4. ood_t        (ood-t_chopped_multi_output_clear_sky)
  - Saves tidy metrics to results/baselines_metrics.csv and results/baselines_metrics.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import get_train_val_tile_split
from src.metrics import (
    DEFAULT_QUANTILES,
    DEFAULT_TARGET_NAMES,
    compute_naive_history_scale,
    evaluate_forecast,
    format_metrics_table,
    metrics_to_dataframe,
)
from src.models.baselines import (
    PersistenceBaseline,
    ProbabilisticPersistenceBaseline,
    extract_clear_sky_history,
    extract_last_clear_sky,
)
from src.utils.config import load_config


def load_split_arrays(file_list: List[Path]) -> Dict[str, np.ndarray]:
    """
    Loads history clear-sky observations, last observation, targets, valid masks,
    and per-index MASE naive scale from a list of .npz tile files.
    """
    if not file_list:
        raise ValueError("file_list cannot be empty.")

    hist_obs_parts: List[np.ndarray] = []
    last_obs_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []

    for fpath in file_list:
        with np.load(fpath, allow_pickle=True) as d:
            histories = d["history"]
            hist_masks = d["history_mask"]
            targets = d["target"].astype(np.float32)
            t_masks = d["target_mask"].astype(bool) if "target_mask" in d else np.isnan(targets)

            hist_obs_parts.append(extract_clear_sky_history(histories, hist_masks))
            last_obs_parts.append(extract_last_clear_sky(histories, hist_masks))
            target_parts.append(targets)
            mask_parts.append((~t_masks) & np.isfinite(targets))

    history_obs = np.concatenate(hist_obs_parts, axis=0)
    last_obs = np.concatenate(last_obs_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    valid_mask = np.concatenate(mask_parts, axis=0)
    mase_scale = compute_naive_history_scale(history_obs)

    return {
        "history_obs": history_obs,
        "last_obs": last_obs,
        "targets": targets,
        "valid_mask": valid_mask,
        "mase_scale": mase_scale,
    }


def run_baseline_evaluation(
    config_path: Union[str, Path] = "configs/config.yaml",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    f_cfg = cfg["forecasting"]

    target_names = tuple(f_cfg.get("target_names", DEFAULT_TARGET_NAMES))
    quantiles = tuple(f_cfg.get("quantiles", DEFAULT_QUANTILES))
    future_steps = int(f_cfg.get("future_steps", 3))
    interp_cfg = cfg.get("interpolation", {})
    clip_min = interp_cfg.get("clip_min", -1.0)
    clip_max = interp_cfg.get("clip_max", 1.0)

    results_dir = Path(output_dir) if output_dir is not None else Path(paths_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Train/Val tile split (fit ONLY on train_files)
    train_files, val_internal_files, split_summary = get_train_val_tile_split(
        paths_cfg["train_data_dir"],
        num_val_tiles=data_cfg["num_val_tiles"],
        seed=data_cfg["seed"],
    )
    if data_cfg["num_train_tiles"] < len(train_files):
        train_files = train_files[: data_cfg["num_train_tiles"]]

    print("=" * 96)
    print(" BASELINE BENCHMARK: PERSISTENCE & PROBABILISTIC PERSISTENCE")
    print("=" * 96)
    print(f"[*] Fitting exclusively on train split ({len(train_files)} tiles)...")
    print(f"    Train tile climates       : {split_summary['train']}")
    print(f"    Val internal tile climates: {split_summary['val']}")

    train_data = load_split_arrays(train_files)
    print(f"    Train windows loaded      : {len(train_data['targets']):,}")

    # 2. Prepare evaluation splits
    eval_file_map = {
        "val_internal": val_internal_files,
        "id": sorted(Path(paths_cfg["val_id_data_dir"]).glob("*.npz")),
        "ood_s": sorted(Path(paths_cfg["ood_s_data_dir"]).glob("*.npz")),
        "ood_t": sorted(Path(paths_cfg["ood_t_data_dir"]).glob("*.npz")),
    }
    eval_splits = {name: load_split_arrays(files) for name, files in eval_file_map.items()}
    for name, data in eval_splits.items():
        scale_str = ", ".join(f"{n}={v:.4f}" for n, v in zip(target_names, data["mase_scale"]))
        print(f"    Split {name:<12}: {len(data['targets']):>6,} windows | MASE naive scale: [{scale_str}]")

    # 3. Fit baselines strictly on train
    models = {
        "persistence": PersistenceBaseline(future_steps=future_steps, quantiles=quantiles).fit(),
        "probabilistic_persistence": ProbabilisticPersistenceBaseline(
            future_steps=future_steps, quantiles=quantiles
        ).fit(
            last_obs_train=train_data["last_obs"],
            targets_train=train_data["targets"],
            valid_mask_train=train_data["valid_mask"],
            quantiles=quantiles,
        ),
    }

    # 4. Evaluate across all 4 splits
    all_results: Dict[str, Dict] = {}
    dfs: List[pd.DataFrame] = []

    for model_name, model in models.items():
        print("\n" + "#" * 96)
        print(f" MODEL: {model_name}")
        print("#" * 96)
        model_results: Dict[str, Dict] = {}
        for split_name, data in eval_splits.items():
            preds = model.predict(data["last_obs"], clip_min=clip_min, clip_max=clip_max)
            metrics = evaluate_forecast(
                y_true=data["targets"],
                y_pred_quantiles=preds,
                mase_scale=data["mase_scale"],
                target_names=target_names,
                quantiles=quantiles,
                valid_mask=data["valid_mask"],
                include_per_step=True,
            )
            metrics["num_windows"] = int(len(data["targets"]))
            metrics["mase_scale"] = {
                t_name: float(val) for t_name, val in zip(target_names, data["mase_scale"])
            }
            model_results[split_name] = metrics
            print(format_metrics_table(metrics, split_name=f"{model_name} | {split_name}"))
            print()

        all_results[model_name] = model_results
        dfs.append(metrics_to_dataframe(model_results, model_name=model_name))

    # 5. Save results
    full_df = pd.concat(dfs, ignore_index=True)
    csv_path = results_dir / "baselines_metrics.csv"
    json_path = results_dir / "baselines_metrics.json"
    full_df.to_csv(csv_path, index=False)

    payload = {
        "config_path": str(config_path),
        "quantiles": list(quantiles),
        "target_names": list(target_names),
        "train_num_tiles": len(train_files),
        "train_num_windows": int(len(train_data["targets"])),
        "models": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[+] Saved baseline metrics CSV : {csv_path}")
    print(f"[+] Saved baseline metrics JSON: {json_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate persistence baselines across all splits.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional results directory override")
    args = parser.parse_args()
    run_baseline_evaluation(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
