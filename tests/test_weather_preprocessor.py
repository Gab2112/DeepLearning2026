import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.interpolation import LinearInterpolator, build_interpolator_from_config
from src.data.transforms import ArcsinhScaler
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.data.weather_preprocessor import WeatherPreprocessor, configure_imputer_weather
from src.benchmark_imputers import run_benchmark
from src.models.bigru_imputer import build_imputer_features
from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer


class WeatherPreprocessorTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.scaler_path = self.directory / "scaler.json"
        mean = [2, 70, 100, 1, 1013, 15, 10, 20, 10, 10, 10, *([0] * 10)]
        self.scaler_path.write_text(
            json.dumps(dict(mean=mean, var=[4] * 21, target_mean=[0] * 3, target_var=[1] * 3, eps=0)),
            encoding="utf-8",
        )
        self.scaler = ArcsinhScaler(self.scaler_path)
        self.raw = np.tile(self.scaler.mean, (5, 1))
        self.mask = np.zeros_like(self.raw, dtype=bool)

    def test_missingness_finite_masks_and_observed_zero(self):
        self.raw[:, 4] = np.nan
        self.mask[:, 4] = True
        self.raw[0, 1] = np.nan
        self.raw[2, 0] = np.inf
        self.mask[3, 2] = True
        self.raw[4, 0] += 4
        original = self.raw.copy()
        original_mask = self.mask.copy()

        result = WeatherPreprocessor("M1").transform(self.raw, self.mask, self.scaler)

        np.testing.assert_array_equal(result.values[:, 4], 0)
        np.testing.assert_array_equal(result.weather_mask[:, 4], False)
        self.assertEqual(result.values[1, 1], 0)
        self.assertTrue(result.weather_mask[1, 1])
        for row, col in ((0, 1), (2, 0), (3, 2)):
            self.assertEqual(result.values[row, col], 0)
            self.assertFalse(result.weather_mask[row, col])
        self.assertTrue(result.original_missing[0, 1])
        self.assertFalse(result.original_missing[2, 0])
        self.assertTrue(result.original_missing[3, 2])
        self.assertFalse(result.invalid.any())
        self.assertAlmostEqual(result.values[4, 0], np.arcsinh(2), places=6)
        self.assertTrue(np.isfinite(result.values).all())
        np.testing.assert_array_equal(self.raw, original)
        np.testing.assert_array_equal(self.mask, original_mask)

    def test_historical_qc_flags_physical_bounds_only(self):
        self.raw[0, :5] = [-1, 105, -1, -0.1, 0]
        self.raw[1, :5] = [100, 100, 2000, 500, 350]
        self.raw[2, :5] = [0, 0, 0, 0, 1]
        self.raw[3, 3] = -np.inf
        self.mask[0, 0] = True

        result = WeatherPreprocessor("M2").transform(self.raw, self.mask, self.scaler)
        np.testing.assert_array_equal(result.invalid[0, :5], [0, 1, 1, 1, 1])
        self.assertTrue(result.original_missing[0, 0])
        self.assertFalse(result.invalid[1:3].any())
        self.assertTrue(result.invalid[3, 3])
        np.testing.assert_array_equal(result.values[:, :8][result.invalid], 0)
        np.testing.assert_array_equal(result.weather_mask, ~(result.original_missing | result.invalid))
        self.assertTrue(np.isfinite(result.values).all())

    def test_temperature_tolerance_and_triplet_policy(self):
        # Column order is average, minimum, maximum.
        self.raw[:, 5:8] = [[10, 11, 15], [10, 11.1, 15], [12.5, 13.1, 11.9], [10, 12, np.nan], [20, 10, 18]]
        result = WeatherPreprocessor("M2").transform(self.raw, self.mask, self.scaler)
        np.testing.assert_array_equal(result.invalid[:, 5:8], [
            [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1],
        ])
        self.assertTrue(result.original_missing[3, 7])
        np.testing.assert_array_equal(result.weather_mask[1:, 5:8], False)

        for options in ({"temperature_policy": "ignore"}, {"temperature_tolerance_c": 3.0}):
            with self.subTest(options=options):
                relaxed = WeatherPreprocessor("M2", **options).transform(self.raw, self.mask, self.scaler)
                self.assertFalse(relaxed.invalid.any())

    def test_m0_preserves_linear_history_and_normalized_residual_fill(self):
        self.raw[:3, 3] = [1, np.nan, 5]
        self.raw[:, 4] = np.nan
        self.mask = np.isnan(self.raw)
        baseline = WeatherPreprocessor().transform(self.raw, self.mask, self.scaler, interpolate_history=True)
        legacy = LinearInterpolator().interpolate(self.raw, self.mask, columns=[3, 4])
        expected = np.nan_to_num(self.scaler.transform_features(legacy)[:, :18], nan=0)
        np.testing.assert_array_equal(baseline.values, expected)
        self.assertFalse(baseline.weather_mask[1, 3])
        future = WeatherPreprocessor().transform(self.raw, self.mask, self.scaler, future=True)
        self.assertEqual(future.values[1, 3], 0)

    def test_m4_keeps_interpolation_and_marks_original_availability(self):
        self.raw[:3, 3] = [1, np.nan, 5]
        self.raw[:, 4] = np.nan
        self.mask = np.isnan(self.raw)
        legacy = WeatherPreprocessor("M0").transform(self.raw, self.mask, self.scaler, interpolate_history=True)
        masked = WeatherPreprocessor("M4").transform(self.raw, self.mask, self.scaler)
        np.testing.assert_array_equal(masked.values, legacy.values)
        self.assertFalse(masked.weather_mask[1, 3])
        self.assertTrue(masked.original_missing[1, 3])
        self.assertTrue(masked.interpolated[1, 3])
        self.assertFalse(masked.interpolated[:, 4].any())
        self.assertFalse(masked.invalid.any())

    def test_preserves_future_noise_and_optionally_clips_verified_ranges(self):
        self.raw[0, :11] = [-1, 108, -25, -0.5, 980, 3, 12, 6, -4, 103, 12]
        self.raw[:, 18:] = np.nan
        self.mask = np.isnan(self.raw)
        expected = self.scaler.transform_features(self.raw)[:, :18]
        for profile in ("M0", "M1", "M2", "M4"):
            with self.subTest(profile=profile):
                result = WeatherPreprocessor(profile).transform(self.raw, self.mask, self.scaler, future=True)
                np.testing.assert_array_equal(result.values, expected)
                np.testing.assert_array_equal(result.weather_mask, True)
                self.assertFalse(result.invalid.any())

        clipped = self.raw.copy()
        clipped[0, :11] = [0, 100, 0, 0, 980, 3, 12, 6, -4, 103, 12]
        result = WeatherPreprocessor("M3").transform(self.raw[:, :18], self.mask[:, :18], self.scaler, future=True)
        np.testing.assert_array_equal(result.values, self.scaler.transform_features(clipped)[:, :18])
        np.testing.assert_array_equal(result.weather_mask, True)
        self.assertTrue(np.isnan(self.raw[:, 18:]).all())
        self.assertEqual(self.raw[0, 3], -0.5)

    def _make_tile(self):
        history = self.raw.copy()
        history[:, 18:] = [0.3, 0.4, 0.5]
        history[1, 18:] = np.nan
        history[0, 5:8] = [10, 15, 20]
        history[1, 1] = np.nan
        future = self.raw.copy()
        future[:, 18:] = np.nan
        future[0, 0] = -2

        histories = np.empty(2, dtype=object)
        histories[:] = [history[:3], history]
        futures = np.empty(2, dtype=object)
        futures[:] = [future[:3], future[:4]]
        targets = np.ones((2, 3, 3), dtype=np.float32) * 0.3
        targets[0, 0, 1] = np.nan
        target_mask = np.isnan(targets)
        target_mask[0, 0, 2] = True

        path = self.directory / "tile.npz"
        np.savez(
            path,
            history=histories,
            history_mask=np.fromiter((np.isnan(x) for x in histories), dtype=object, count=2),
            future=futures,
            future_mask=np.fromiter((np.isnan(x) for x in futures), dtype=object, count=2),
            future_noise=np.full((2, 4, 8), 999.0),
            target=targets,
            target_mask=target_mask,
            future_target_positions=np.tile(np.arange(3), (2, 1)),
        )
        return path, histories

    def test_vegetation_profiles_preserve_targets_and_variable_length_masks(self):
        path, histories = self._make_tile()
        before = path.read_bytes()
        cfg = {"paths": {"train_data_dir": self.directory, "scaler_path": self.scaler_path}}
        targets = LinearInterpolator().interpolate(histories[0], columns=[18, 19, 20])[:, 18:]
        for profile in ("M0", "M1", "M2", "M3", "M4"):
            with self.subTest(profile=profile):
                cfg["weather_preprocessing"] = {"profile": profile}
                dataset = VegetationDataset.from_config(cfg, file_list=[path])
                sample = dataset[0]
                batch = collate_vegetation_batch([sample, dataset[1]])
                extra_channels = 0 if profile == "M0" else 8
                self.assertEqual(batch["history"].shape, (2, 5, 21 + extra_channels))
                self.assertEqual(batch["future_weather"].shape, (2, 4, 18 + extra_channels))
                np.testing.assert_array_equal(sample["history"][:, 18:21], self.scaler.transform_targets(targets))
                np.testing.assert_array_equal(batch["history_padding_mask"][0], [1, 1, 1, 0, 0])
                self.assertFalse(batch["history_weather_mask"][0, 3:].any())
                self.assertFalse(batch["history_weather_interpolated_mask"][0, 3:].any())
                self.assertFalse(batch["future_weather_mask"][0, 3:].any())
                self.assertTrue(batch["target_mask"][0, 0, 1:].all())
                self.assertTrue(torch.isnan(batch["target"][0, 0, 1:]).all())
                self.assertTrue(torch.isfinite(batch["raw_target"][0, 0, 2]))
                future_wind = 0 if profile == "M3" else -2
                self.assertAlmostEqual(sample["future_weather"][0, 0].item(), np.arcsinh((future_wind - 2) / 2), places=6)
                if extra_channels:
                    torch.testing.assert_close(batch["history"][:, :, 21:], batch["history_weather_mask"].float())
                    torch.testing.assert_close(batch["future_weather"][:, :, 18:], batch["future_weather_mask"].float())
                self.assertEqual(sample["history_weather_invalid_mask"][0, 5:8].sum().item(), 3 if profile in ("M2", "M3") else 0)
                self.assertTrue(torch.isfinite(batch["history"]).all())
                self.assertTrue(torch.isfinite(batch["future_weather"]).all())
        self.assertEqual(path.read_bytes(), before)

    def test_history_features_match_inference_in_both_widths(self):
        path, histories = self._make_tile()
        for profile in ("M0", "M2"):
            with self.subTest(profile=profile):
                weather = WeatherPreprocessor(profile)
                dataset = HistoryImputationDataset(
                    self.directory, self.scaler, file_list=[path],
                    mask_dropout=False, climate_dropout_p=0.0, weather_preprocessor=weather,
                )
                features = build_imputer_features(
                    histories[0], np.isnan(histories[0]), self.scaler, weather_preprocessor=weather
                )
                np.testing.assert_array_equal(dataset[0]["input"], features)
                batch = collate_history_batch([dataset[0], dataset[1]])
                self.assertEqual(batch["input"].shape, (2, 5, weather.imputer_input_dim))
                self.assertTrue(torch.isfinite(batch["input"]).all())
                self.assertFalse(batch["weather_reconstruction_mask"][0, 3:].any())

    def test_neural_checkpoint_round_trips_and_profiles(self):
        _, histories = self._make_tile()
        masks = [np.isnan(h) for h in histories]
        model_types = [
            ("model", "model_checkpoint", "imputer_model", BiGRUHistoryImputer),
            ("mae", "mae_checkpoint", "mae_model", MAEHistoryImputer),
        ]
        for strategy, path_key, model_key, model_type in model_types:
            for profile in ("M0", "M3"):
                with self.subTest(strategy=strategy, profile=profile):
                    base_checkpoint = self.directory / f"{strategy}.pt"
                    cfg = {
                        "paths": {"scaler_path": self.scaler_path},
                        "interpolation": {"strategy": strategy, path_key: base_checkpoint},
                        "weather_preprocessing": {"profile": profile},
                        model_key: {"input_dim": 27},
                    }
                    checkpoint_cfg = {model_key: cfg[model_key].copy(), "weather_preprocessing": cfg["weather_preprocessing"]}
                    weather = configure_imputer_weather(checkpoint_cfg, model_key)
                    model = model_type(**checkpoint_cfg[model_key])
                    if profile == "M0":
                        checkpoint_cfg.pop("weather_preprocessing")
                    checkpoint = weather.imputer_checkpoint_path(base_checkpoint)
                    torch.save({"config": checkpoint_cfg, "model_state_dict": model.state_dict()}, checkpoint)

                    adapter = build_interpolator_from_config(cfg, strict_checkpoint=True)
                    expected_dim = 27 if profile == "M0" else 35
                    self.assertEqual(adapter.checkpoint_path, checkpoint)
                    self.assertEqual(adapter.model.input_dim, expected_dim)
                    if strategy == "mae":
                        self.assertEqual(adapter.model.weather_proj[0].in_features, 11 + expected_dim - 27)
                    else:
                        self.assertEqual(adapter.model.input_proj[0].in_features, expected_dim + 8)
                    single = adapter.interpolate_history_sample(histories[0], masks[0], "Cfa")
                    batch = adapter.interpolate_history_batch(histories, masks, ["Cfa", "Dfb"])
                    for actual, raw in zip([single, *batch], [histories[0], *histories]):
                        np.testing.assert_array_equal(actual[:, :18], raw[:, :18])
                        self.assertTrue(np.isfinite(actual[:, 18:]).all())
                    if profile != "M0":
                        cfg["weather_preprocessing"]["temperature_tolerance_c"] = 2.0
                        with self.assertRaisesRegex(ValueError, "Checkpoint weather preprocessing"):
                            build_interpolator_from_config(cfg, strict_checkpoint=True)

    def test_benchmark_uses_profile_checkpoints_and_skips_missing_model(self):
        tile, _ = self._make_tile()
        tile.with_name("tile_b.npz").write_bytes(tile.read_bytes())
        config = {
            "paths": {
                "train_data_dir": str(self.directory),
                "scaler_path": str(self.scaler_path),
                "checkpoints_dir": str(self.directory),
                "results_dir": str(self.directory),
            },
            "data": {"num_val_tiles": 1},
            "weather_preprocessing": {"profile": "M3"},
            "imputer_training": {"checkpoint_name": "bigru.pt"},
            "mae_training": {"checkpoint_name": "mae.pt"},
        }
        config_path = self.directory / "benchmark.yaml"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        model_config = {**config, "imputer_model": {}}
        weather = configure_imputer_weather(model_config, "imputer_model")
        model = BiGRUHistoryImputer(**model_config["imputer_model"])
        checkpoint = weather.imputer_checkpoint_path(self.directory / "bigru.pt")
        torch.save({"config": model_config, "model_state_dict": model.state_dict()}, checkpoint)

        with redirect_stdout(StringIO()) as output:
            results = run_benchmark(config_path, eval_only=True, max_samples_per_tile=1, device="cpu", smoke=True)

        self.assertIn("bigru", set(results["strategy"]))
        self.assertNotIn("mae", set(results["strategy"]))
        self.assertIn("MAE checkpoint not found", output.getvalue())
        self.assertEqual(set(results["weather_profile"]), {"M3"})
        self.assertEqual(set(results["run_kind"]), {"smoke"})
        self.assertTrue((self.directory / "imputation_benchmark_m3_smoke.csv").is_file())


if __name__ == "__main__":
    unittest.main()
