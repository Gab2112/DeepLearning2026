"""Prepare ablation configurations and score future forecast predictions on internal validation."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.interpolation import InterpolationFactory
from src.data.scaler_fit import fit_arcsinh_scaler, resolve_experiment_scaler
from src.data.splits import get_experiment_split
from src.data.weather_preprocessor import WEATHER_PROFILES, WeatherPreprocessor
from src.metrics import DEFAULT_QUANTILES, DEFAULT_TARGET_NAMES, evaluate_forecast, inverse_transform_quantiles
from src.utils.config import load_config


def prepare_experiments(
    cfg: dict, output_dir: Path, *, imputer_profile: str = "M0", candidates: tuple[str, ...] = ()
) -> list[Path]:
    """Save shared fitting statistics plus A, B and user-chosen C configurations."""
    output_dir = output_dir.resolve()
    options = cfg.get("weather_preprocessing", {})
    pairs = []
    for candidate in candidates:
        profile, _, strategy = candidate.partition(":")
        if profile not in WEATHER_PROFILES or strategy not in InterpolationFactory.available_strategies():
            raise ValueError(f"Expected PROFILE:STRATEGY, got {candidate!r}")
        pairs.append((profile, strategy))
    if imputer_profile not in WEATHER_PROFILES:
        raise ValueError(f"Unknown imputer benchmark weather profile: {imputer_profile!r}")

    fit, validation, _ = get_experiment_split(cfg)
    seed = cfg["data"].get("seed", 42)
    _write_json(output_dir / "split.json", {
        "seed": seed,
        "fit_tiles": [path.name for path in fit],
        "validation_tiles": [path.name for path in validation],
    })
    scaler_paths = {}
    for profile, name in (("M0", "scaler_fitting.json"), ("M2", "scaler_fitting_qc.json")):
        weather = WeatherPreprocessor(**{**options, "profile": profile})
        scaler = fit_arcsinh_scaler(fit, validation, weather, seed)
        path = output_dir / name
        _write_json(path, scaler.to_dict())
        scaler_paths[weather.historical_qc is not None] = path

    configurations = []
    conditions = [(f"A_{profile.lower()}_linear", profile, "linear") for profile in WEATHER_PROFILES]
    conditions.append((f"B_{imputer_profile.lower()}_imputers", imputer_profile, "linear"))
    conditions.extend((f"C_{profile.lower()}_{strategy}", profile, strategy) for profile, strategy in pairs)
    for name, profile, strategy in conditions:
        weather = WeatherPreprocessor(**{**options, "profile": profile})
        paths = {
            **cfg["paths"],
            "checkpoints_dir": output_dir / "checkpoints",
            "results_dir": output_dir / "results",
            "fitted_scaler_path": scaler_paths[weather.historical_qc is not None],
        }
        generated = {
            **cfg,
            "paths": paths,
            "scaling": {"mode": "fitting_tiles"},
            "weather_preprocessing": {**options, "profile": profile},
            "interpolation": {
                **cfg.get("interpolation", {}),
                "strategy": strategy,
                "interpolate_weather": True,
                "model_checkpoint": paths["checkpoints_dir"] / cfg["imputer_training"]["checkpoint_name"],
                "mae_checkpoint": paths["checkpoints_dir"] / cfg["mae_training"]["checkpoint_name"],
            },
        }
        generated.pop("project_root", None)
        path = output_dir / f"{name}.yaml"
        _write_json(path, generated)
        configurations.append(path)
    return configurations


def _validation_targets(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets, masks, tile_names, window_indices = [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=True) as tile:
            labels = np.asarray(tile["target"], dtype=np.float32)
            valid = np.isfinite(labels)
            if "target_mask" in tile:
                valid &= ~np.asarray(tile["target_mask"], dtype=bool)
        targets.append(labels)
        masks.append(valid)
        tile_names.extend([path.stem] * len(labels))
        window_indices.extend(range(len(labels)))
    return np.concatenate(targets), np.concatenate(masks), np.asarray(tile_names), np.asarray(window_indices)


def score_internal_forecast(cfg: dict, predictions_path: Path, *, normalized: bool = False) -> dict:
    """Evaluate saved predictions in the fixed validation tile/window order."""
    fit, validation, _ = get_experiment_split(cfg)
    targets, valid, tile_names, window_indices = _validation_targets(validation)
    with np.load(predictions_path, allow_pickle=False) as archive:
        predictions = archive["y_pred_quantiles"]
        if not (
            np.array_equal(archive["tile_names"], tile_names)
            and np.array_equal(archive["window_indices"], window_indices)
        ):
            raise ValueError("Prediction IDs do not match the fixed internal validation windows and order")
    forecast = cfg.get("forecasting", {})
    quantiles = tuple(forecast.get("quantiles", DEFAULT_QUANTILES))
    target_names = tuple(forecast.get("target_names", DEFAULT_TARGET_NAMES))
    if predictions.shape != targets.shape + (len(quantiles),):
        raise ValueError(f"Expected forecast shape {targets.shape + (len(quantiles),)}, got {predictions.shape}")
    weather = WeatherPreprocessor(**cfg.get("weather_preprocessing", {}))
    if normalized:
        scaler = resolve_experiment_scaler(cfg, weather, fit, validation)
        predictions = inverse_transform_quantiles(predictions, scaler)
    if not np.isfinite(predictions[valid]).all():
        raise ValueError("Non-finite forecasts for available validation targets")

    scores = evaluate_forecast(
        targets, predictions, quantiles=quantiles, target_names=target_names, valid_mask=valid
    )
    scores["quantile_calibration"] = {
        name: {
            str(q): float(np.mean(targets[..., j][valid[..., j]] <= predictions[..., j, i][valid[..., j]]))
            if valid[..., j].any() else None
            for i, q in enumerate(quantiles)
        }
        for j, name in enumerate(target_names)
    }
    scores["split"] = "internal_validation"
    scores["seed"] = cfg["data"].get("seed", 42)
    scores["weather_profile"] = weather.profile
    scores["imputer"] = cfg.get("interpolation", {}).get("strategy", "linear")
    scores["windows"] = len(targets)
    return scores


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_json_metrics(payload), indent=2, default=str, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"Output already exists with different contents: {path}")
    with path.open("x", encoding="utf-8") as f:
        f.write(serialized)


def _json_metrics(value):
    if isinstance(value, dict):
        return {key: _json_metrics(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", default="configs/config.yaml")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--imputer-profile", choices=WEATHER_PROFILES, default="M0")
    prepare.add_argument("--candidate", action="append", default=[], metavar="PROFILE:STRATEGY")
    score = commands.add_parser("score-forecast")
    score.add_argument("--config", required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--normalized", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        paths = prepare_experiments(
            cfg, args.output_dir, imputer_profile=args.imputer_profile, candidates=tuple(args.candidate)
        )
        for path in paths:
            print(path)
    else:
        _write_json(args.output, score_internal_forecast(cfg, args.predictions, normalized=args.normalized))
        print(args.output)


if __name__ == "__main__":
    main()
