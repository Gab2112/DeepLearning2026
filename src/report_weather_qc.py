"""Count original meteorological gaps and historical QC invalidations on train."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.weather_preprocessor import WEATHER_FEATURE_COUNT, WEATHER_FEATURES, WeatherPreprocessor
from src.utils.config import load_config


def summarize_weather_qc(train_dir: Path, weather: WeatherPreprocessor) -> list[dict]:
    rows = []
    paths = sorted(train_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No training tiles in {train_dir}")

    for path in paths:
        with np.load(path, allow_pickle=True) as tile:
            histories = tile["history"]
            raw = np.concatenate(histories).astype(np.float32)
            source_mask = np.concatenate(tile["history_mask"]).astype(bool)
        original, invalid, _ = weather.availability(raw, source_mask)
        weather_values = raw[:, :WEATHER_FEATURE_COUNT]
        originally_valid = np.isfinite(weather_values) & ~source_mask[:, :WEATHER_FEATURE_COUNT]
        row = {
            "tile": path.stem,
            "windows": len(histories),
            "historical_steps": len(raw),
            "steps_with_invalidations": int(invalid.any(axis=1).sum()),
            "invalidated_previously_valid": int((invalid & originally_valid).sum()),
            "nan_mask_disagreements": int(np.count_nonzero(
                np.isnan(weather_values) != source_mask[:, :WEATHER_FEATURE_COUNT]
            )),
        }
        row.update({f"{name}_original_missing": int(count) for name, count in zip(WEATHER_FEATURES, original.sum(axis=0))})
        row.update({f"{name}_invalidated": int(count) for name, count in zip(WEATHER_FEATURES, invalid.sum(axis=0))})
        rows.append(row)

    rows.append({
        "tile": "TOTAL",
        **{key: sum(row[key] for row in rows) for key in rows[0] if key != "tile"},
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV report")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must be a CSV path")
    cfg = load_config(args.config)
    weather = WeatherPreprocessor(**{**cfg.get("weather_preprocessing", {}), "profile": "M2"})
    rows = summarize_weather_qc(cfg["paths"]["train_data_dir"], weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = rows[-1]
    print(f"{len(rows) - 1} train tiles, {total['windows']} windows, QC tolerance {weather.temperature_tolerance_c:g} C")
    for name in WEATHER_FEATURES:
        print(f"{name}: {total[f'{name}_original_missing']} unavailable originally, {total[f'{name}_invalidated']} invalidated")
    print(f"{total['invalidated_previously_valid']} invalidations of previously usable values")


if __name__ == "__main__":
    main()
