"""
Comprehensive Training and Benchmarking Pipeline for History Imputation Strategies.

Evaluates all imputation strategies on the dataset using a self-supervised
leave-one-out masking protocol on clear-sky observations:
1. Classical / Heuristic Baselines:
   - None (zero-fill / missing)
   - Mean (historical sample mean)
   - Median (historical sample median)
   - Forward Fill (last observation carried forward)
   - Backward Fill (next observation carried backward)
   - Nearest Neighbor
   - Linear Interpolation
   - Spline Interpolation (monotonic PCHIP)
2. Learned Deep Learning Imputers:
   - Bi-GRU History Imputer (bidirectional GRU + orbital dynamics + climate embedding)
   - MAE History Imputer (Transformer Masked Autoencoder + learnable mask tokens)

Can train models from scratch (--train-all, --train-bigru, --train-mae)
and produces a tidy comparative leaderboard in results/imputation_benchmark.csv
and results/imputation_benchmark.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.interpolation import InterpolationFactory, NoneInterpolator
from src.data.scaler_fit import experiment_checkpoint_path, resolve_experiment_scaler
from src.data.splits import get_experiment_split
from src.data.transforms import ArcsinhScaler
from src.data.weather_preprocessor import WeatherPreprocessor
from src.train_bigru_imputer import train_bigru_imputer
from src.train_mae_imputer import train_mae_imputer
from src.utils.config import load_config


# Evaluated strategies and their display labels
DEFAULT_STRATEGIES = [
    ("none", "Zero Fill (None)"),
    ("mean", "Historical Mean"),
    ("median", "Historical Median"),
    ("forward_fill", "Forward Fill (LOCF)"),
    ("backward_fill", "Backward Fill (NOCB)"),
    ("nearest", "Nearest Neighbor"),
    ("linear", "Linear Interpolation"),
    ("spline", "Spline (PCHIP)"),
    ("bigru", "Bi-GRU Imputer (Learned)"),
    ("mae", "MAE Transformer (Learned)"),
]

TARGET_NAMES = ["NDVI", "GNDVI", "EVI"]
TARGET_COLS = [18, 19, 20]


def evaluate_sample_leave_one_out(
    h_raw: np.ndarray,
    m_raw: np.ndarray,
    climate_str: str,
    interpolators: Dict[str, any],
    scaler: ArcsinhScaler,
    rng: Optional[np.random.RandomState] = None,
    mode: str = "all_obs",
    inference_seconds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Evaluates interpolators on a single sequence by masking clear-sky observations.

    Modes:
      - 'all_obs': Leave-one-out for every observed clear-sky point.
      - 'random_drop': Randomly drop one observation (faster for massive datasets).

    Returns:
      Dict mapping strategy_name -> {target_name: (true_values, pred_values)}
    """
    obs_mask = (~m_raw[:, TARGET_COLS] & np.isfinite(h_raw[:, TARGET_COLS])).all(axis=1)
    valid_indices = np.where(obs_mask)[0]

    if len(valid_indices) < 2:
        return {}

    if mode == "random_drop":
        chosen_indices = [int(rng.choice(valid_indices))] if rng is not None else [int(valid_indices[0])]
    else:
        chosen_indices = [int(idx) for idx in valid_indices]

    results: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {
        strat_key: {name: ([], []) for name in TARGET_NAMES}
        for strat_key in interpolators
    }

    for drop_idx in chosen_indices:
        # Create perturbed sequence with drop_idx masked out
        pert_m = m_raw.copy()
        pert_m[drop_idx, TARGET_COLS] = True

        pert_h = h_raw.copy()
        pert_h[drop_idx, TARGET_COLS] = np.nan

        true_vals = h_raw[drop_idx, TARGET_COLS]

        for strat_key, interp in interpolators.items():
            if interp is None:
                continue

            start = time.perf_counter()
            if hasattr(interp, "interpolate_history_sample"):
                imputed = interp.interpolate_history_sample(
                    pert_h.copy(), pert_m.copy(), climate_str
                )
            else:
                imputed = interp.interpolate(
                    pert_h.copy(), mask=pert_m.copy(), columns=TARGET_COLS
                )
            if inference_seconds is not None:
                inference_seconds[strat_key] += time.perf_counter() - start

            pred_vals = imputed[drop_idx, TARGET_COLS]
            if isinstance(interp, NoneInterpolator):
                pred_vals = np.nan_to_num(pred_vals, nan=0.0)
            if not np.isfinite(pred_vals).all():
                raise ValueError(f"Strategy {strat_key!r} produced non-finite predictions")

            for i, name in enumerate(TARGET_NAMES):
                results[strat_key][name][0].append(float(true_vals[i]))
                results[strat_key][name][1].append(float(pred_vals[i]))

    # Convert to numpy arrays
    final_results = {}
    for strat_key, target_dict in results.items():
        final_results[strat_key] = {
            name: (
                np.asarray(target_dict[name][0], dtype=np.float32),
                np.asarray(target_dict[name][1], dtype=np.float32),
            )
            for name in TARGET_NAMES
        }
    return final_results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and WMAPE between true and predicted arrays."""
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "WMAPE": np.nan}

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    denom = np.sum(np.abs(y_true))
    wmape = float(np.sum(abs_errors) / denom) if denom > 1e-8 else np.nan

    return {"MAE": mae, "RMSE": rmse, "WMAPE": wmape}


def evaluate_imputation_strategies(
    file_list: List[Path],
    scaler: ArcsinhScaler,
    strategies: Optional[List[Tuple[str, str]]] = None,
    max_samples_per_tile: Optional[int] = None,
    eval_mode: str = "random_drop",
    device: Optional[str] = None,
    seed: int = 42,
    *,
    interpolator_kwargs: Optional[Dict[str, Dict]] = None,
    required_training_epochs: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Evaluates all specified imputation strategies on a given set of tile files.
    """
    strategies = strategies or DEFAULT_STRATEGIES
    rng = np.random.RandomState(seed)

    # 1. Initialize all interpolator instances via Factory
    print("[*] Initializing interpolation strategies...")
    interpolators: Dict[str, any] = {}
    training_seconds = {}
    for strat_key, strat_label in strategies:
        try:
            extra_kwargs = {}
            learned = strat_key in ("model", "bigru", "learned_imputer", "history_imputer", "mae", "mae_imputer", "masked_autoencoder")
            if learned:
                extra_kwargs["device"] = device
                extra_kwargs["strict_checkpoint"] = True
            extra_kwargs.update((interpolator_kwargs or {}).get(strat_key, {}))

            interp = InterpolationFactory.create(
                strat_key,
                **extra_kwargs,
            )
            if learned and required_training_epochs is not None:
                _check_training(strat_key, interp, scaler, seed, required_training_epochs[strat_key])
            training = getattr(interp, "training_info", None)
            training_seconds[strat_key] = training["seconds"] if training else (None if learned else 0)
            interpolators[strat_key] = interp
            print(f"  [+] Loaded strategy: {strat_label:<28} (key: {strat_key})")
        except Exception as e:
            if required_training_epochs is not None:
                raise RuntimeError(f"Cannot benchmark strategy {strat_key!r}: {e}") from e
            print(f"  [!] Could not load strategy '{strat_key}': {e}. Skipping.")
            interpolators[strat_key] = None

    # Aggregated true and predicted arrays per strategy and index
    aggregated: Dict[str, Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]] = {
        strat_key: {name: ([], []) for name in TARGET_NAMES}
        for strat_key in interpolators
        if interpolators[strat_key] is not None
    }

    # 2. Iterate through files and evaluate
    total_samples = 0
    total_eval_points = 0
    inference_seconds = dict.fromkeys(aggregated, 0.0)
    t0 = time.time()

    print(f"\n[*] Evaluating on {len(file_list)} tile files (mode: {eval_mode})...")
    for f_idx, fpath in enumerate(sorted(file_list)):
        with np.load(fpath, allow_pickle=True) as d:
            histories = d["history"]
            masks = d["history_mask"]
            climates = d["climate"] if "climate" in d else np.array(["unknown"] * len(histories))

            n_samples = len(histories)
            if max_samples_per_tile:
                n_samples = min(n_samples, max_samples_per_tile)

            for i in range(n_samples):
                h = histories[i].astype(np.float32)
                m = masks[i].astype(bool)
                c = str(climates[i])

                sample_res = evaluate_sample_leave_one_out(
                    h_raw=h,
                    m_raw=m,
                    climate_str=c,
                    interpolators=interpolators,
                    scaler=scaler,
                    rng=rng,
                    mode=eval_mode,
                    inference_seconds=inference_seconds,
                )

                if not sample_res:
                    continue

                total_samples += 1

                for strat_key, targets_dict in sample_res.items():
                    if strat_key not in aggregated:
                        continue
                    for name in TARGET_NAMES:
                        t_arr, p_arr = targets_dict[name]
                        if len(t_arr) > 0:
                            aggregated[strat_key][name][0].append(t_arr)
                            aggregated[strat_key][name][1].append(p_arr)

        if (f_idx + 1) % max(1, len(file_list) // 5) == 0 or (f_idx + 1) == len(file_list):
            elapsed = time.time() - t0
            print(f"  - Progress: [{f_idx+1}/{len(file_list)}] tiles processed ({total_samples} samples, {elapsed:.1f}s)")

    # 3. Compute final metrics per strategy
    rows = []
    strat_labels_dict = dict(strategies)

    for strat_key, targets_dict in aggregated.items():
        all_true_parts = []
        all_pred_parts = []
        per_index_metrics = {}

        for name in TARGET_NAMES:
            t_list, p_list = targets_dict[name]
            if len(t_list) > 0:
                t_concat = np.concatenate(t_list)
                p_concat = np.concatenate(p_list)
                all_true_parts.append(t_concat)
                all_pred_parts.append(p_concat)
                per_index_metrics[name] = compute_metrics(t_concat, p_concat)
            else:
                per_index_metrics[name] = {"MAE": np.nan, "RMSE": np.nan, "WMAPE": np.nan}

        if all_true_parts:
            overall_true = np.concatenate(all_true_parts)
            overall_pred = np.concatenate(all_pred_parts)
            overall_m = compute_metrics(overall_true, overall_pred)
            total_eval_points = len(overall_true) // 3
        else:
            overall_m = {"MAE": np.nan, "RMSE": np.nan, "WMAPE": np.nan}

        row = {
            "strategy": strat_key,
            "label": strat_labels_dict.get(strat_key, strat_key),
            "Overall_MAE": overall_m["MAE"],
            "Overall_RMSE": overall_m["RMSE"],
            "Overall_WMAPE": overall_m["WMAPE"],
            "NDVI_MAE": per_index_metrics["NDVI"]["MAE"],
            "NDVI_RMSE": per_index_metrics["NDVI"]["RMSE"],
            "NDVI_WMAPE": per_index_metrics["NDVI"]["WMAPE"],
            "GNDVI_MAE": per_index_metrics["GNDVI"]["MAE"],
            "GNDVI_RMSE": per_index_metrics["GNDVI"]["RMSE"],
            "GNDVI_WMAPE": per_index_metrics["GNDVI"]["WMAPE"],
            "EVI_MAE": per_index_metrics["EVI"]["MAE"],
            "EVI_RMSE": per_index_metrics["EVI"]["RMSE"],
            "EVI_WMAPE": per_index_metrics["EVI"]["WMAPE"],
            "TrainingSeconds": training_seconds[strat_key],
            "InferenceSeconds": inference_seconds[strat_key],
            "MaskedDates": total_eval_points,
        }
        rows.append(row)

    if not rows:
        raise ValueError("No imputation strategies are available for this benchmark")
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Overall_MAE", ascending=True).reset_index(drop=True)
    df["Rank"] = df.index + 1

    return df


def _check_training(strategy, interpolator, scaler: ArcsinhScaler, seed: int, epochs: int) -> None:
    training = interpolator.training_info
    if (
        not training
        or training.get("completed_epochs", 0) != epochs
        or training.get("optimizer_steps", 0) <= 0
        or not np.isfinite(training.get("seconds", 0))
        or training.get("seconds", 0) <= 0
        or training.get("seed") != seed
    ):
        raise ValueError(f"{strategy} checkpoint lacks a completed training run for the configured epochs and seed")
    embedded_scaler = getattr(interpolator.model, "arcsinh_scaler", None)
    if embedded_scaler is None or embedded_scaler.to_dict() != scaler.to_dict():
        raise ValueError(f"{strategy} checkpoint scaler does not match the configured fitting split")


def format_leaderboard(df: pd.DataFrame) -> str:
    """Formats the benchmark dataframe as a neat, aligned ASCII/Markdown leaderboard."""
    lines = []
    lines.append("=" * 105)
    lines.append(f"{'RANK':<5} | {'STRATEGY':<26} | {'OVERALL MAE':<12} | {'RMSE':<10} | {'WMAPE':<10} | {'NDVI MAE':<10} | {'GNDVI MAE':<10} | {'EVI MAE':<10}")
    lines.append("=" * 105)

    for _, row in df.iterrows():
        rank_str = f"#{int(row['Rank'])}"
        best_marker = " [BEST]" if int(row['Rank']) == 1 else ""
        label_str = f"{row['label']}{best_marker}"

        lines.append(
            f"{rank_str:<5} | {label_str:<26} | {row['Overall_MAE']:<12.4f} | {row['Overall_RMSE']:<10.4f} | {row['Overall_WMAPE']:<10.4f} | "
            f"{row['NDVI_MAE']:<10.4f} | {row['GNDVI_MAE']:<10.4f} | {row['EVI_MAE']:<10.4f}"
        )
    lines.append("=" * 105)
    return "\n".join(lines)


def run_benchmark(
    config_path: Union[str, Path] = "configs/config.yaml",
    train_models: bool = False,
    train_bigru_only: bool = False,
    train_mae_only: bool = False,
    eval_only: bool = False,
    max_samples_per_tile: Optional[int] = None,
    eval_mode: str = "random_drop",
    use_val_split_only: bool = True,
    device: Optional[str] = None,
    smoke: bool = False,
) -> pd.DataFrame:
    """
    Main benchmark execution function:
    1. Optionally trains Bi-GRU and MAE models.
    2. Runs leave-one-out cross-validation across all strategies.
    3. Prints and saves results.
    """
    cfg = load_config(config_path)
    weather = WeatherPreprocessor(**cfg.get("weather_preprocessing", {}))
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]

    scaler_path = _resolve_path(paths_cfg["scaler_path"])
    results_dir = _resolve_path(paths_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Project Root: {PROJECT_ROOT}")
    print(f"[*] Execution Device: {device.upper()}")
    print(f"[*] {'SMOKE TEST' if smoke else 'EXPERIMENT'} — weather profile {weather.profile}")
    if not smoke and (max_samples_per_tile is not None or not use_val_split_only):
        raise ValueError("Use --smoke for sample caps or evaluation on the full train")
    if not smoke and cfg.get("scaling", {}).get("mode") != "fitting_tiles":
        raise ValueError("The definitive benchmark requires a scaler fitted on the fitting tiles")
    if (smoke or eval_only) and (train_models or train_bigru_only or train_mae_only):
        raise ValueError("Smoke and evaluation-only runs do not train models")

    # 1. Train only when requested explicitly.
    checkpoint_dir = _resolve_path(paths_cfg["checkpoints_dir"])
    bigru_ckpt = experiment_checkpoint_path(cfg, weather, checkpoint_dir / cfg["imputer_training"]["checkpoint_name"])
    mae_ckpt = experiment_checkpoint_path(cfg, weather, checkpoint_dir / cfg["mae_training"]["checkpoint_name"])

    if train_models or train_bigru_only:
        train_bigru_imputer(config_path=config_path, device=device)
    if train_models or train_mae_only:
        train_mae_imputer(config_path=config_path, device=device)
    if not smoke:
        missing = [str(path) for path in (bigru_ckpt, mae_ckpt) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required trained checkpoints missing: " + ", ".join(missing))

    # 2. Dataset partitioning (stratified by climate, zero spatial leakage)
    train_tiles, val_tiles, _ = get_experiment_split(cfg)

    eval_files = val_tiles if use_val_split_only else (train_tiles + val_tiles)
    split_name = "Internal stratified validation" if use_val_split_only else "Full train"
    print(f"\n[*] Evaluating on: {split_name} ({len(eval_files)} tiles)")

    scaler = resolve_experiment_scaler(cfg, weather, train_tiles, val_tiles)

    # 3. Run Benchmark
    df_results = evaluate_imputation_strategies(
        file_list=eval_files,
        scaler=scaler,
        max_samples_per_tile=max_samples_per_tile,
        eval_mode=eval_mode,
        device=device,
        seed=data_cfg.get("seed", 42),
        interpolator_kwargs={
            strategy: {
                "checkpoint_path": checkpoint_path,
                "scaler_path": scaler_path,
                "expected_weather": weather,
            }
            for strategy, checkpoint_path in (("bigru", bigru_ckpt), ("mae", mae_ckpt))
        },
        required_training_epochs=None if smoke else {
            "bigru": int(cfg["imputer_training"]["epochs"]),
            "mae": int(cfg["mae_training"]["epochs"]),
        },
    )
    df_results["weather_profile"] = weather.profile
    df_results["run_kind"] = "smoke" if smoke else "experiment"
    df_results["seed"] = data_cfg.get("seed", 42)
    df_results["eval_mode"] = eval_mode
    df_results["tiles"] = ",".join(path.stem for path in sorted(eval_files))

    # 4. Display Leaderboard
    table = format_leaderboard(df_results)
    print("\n" + table + "\n")

    print("[*] Il ranking misura la ricostruzione storica. Il contributo al forecasting richiede una valutazione dedicata.")

    # 5. Save artifacts
    suffix = "" if weather.profile == "M0" else f"_{weather.profile.lower()}"
    suffix += "_smoke" if smoke else "_fit"
    csv_path = results_dir / f"imputation_benchmark{suffix}.csv"
    json_path = results_dir / f"imputation_benchmark{suffix}.json"

    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)

    print(f"[+] Risultati salvati in: {csv_path} e {json_path}")
    return df_results


def _resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main():
    parser = argparse.ArgumentParser(description="Benchmark & Train History Imputation Strategies")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--train-all", action="store_true", help="Train both Bi-GRU and MAE before benchmarking")
    parser.add_argument("--train-bigru", action="store_true", help="Train Bi-GRU before benchmarking")
    parser.add_argument("--train-mae", action="store_true", help="Train MAE before benchmarking")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoints without re-training")
    parser.add_argument("--smoke", action="store_true", help="Test pipeline execution and label outputs as smoke results")
    parser.add_argument("--full-dataset", action="store_true", help="Evaluate on all 84 tiles instead of just the 14 val tiles")
    parser.add_argument("--max-samples-per-tile", type=int, default=None, help="Cap samples per tile for rapid benchmarking")
    parser.add_argument("--eval-mode", type=str, choices=["random_drop", "all_obs"], default="random_drop",
                        help="'random_drop' for fast representative sampling, 'all_obs' for exhaustive leave-one-out")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_models = args.train_all

    run_benchmark(
        config_path=args.config,
        train_models=train_models,
        train_bigru_only=args.train_bigru,
        train_mae_only=args.train_mae,
        eval_only=args.eval_only,
        max_samples_per_tile=args.max_samples_per_tile,
        eval_mode=args.eval_mode,
        use_val_split_only=not args.full_dataset,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
