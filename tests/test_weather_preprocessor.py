import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.data.bigru_imputer import build_imputer_features, load_imputer_checkpoint
from src.data.history_dataset import HistoryImputationDataset
from src.data.interpolation import ModelInterpolator
from src.data.transforms import ArcsinhScaler
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.data.weather_preprocessor import preprocess_weather_covariates
from src.models.history_imputer import BiGRUHistoryImputer


class WeatherPreprocessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_dir = TemporaryDirectory()
        cls.addClassCleanup(temp_dir.cleanup)
        cls.data_dir = Path(temp_dir.name)
        cls.scaler_path = cls.data_dir / "scaler.json"
        cls.scaler_path.write_text(
            json.dumps({
                "eps": 0.0,
                "mean": list(range(21)),
                "var": [4.0] * 21,
                "target_mean": [0.0] * 3,
                "target_var": [1.0] * 3,
            }),
            encoding="utf-8",
        )
        cls.scaler = ArcsinhScaler(cls.scaler_path)

    def setUp(self):
        self.raw = np.tile(self.scaler.mean, (4, 1))
        self.mask = np.zeros_like(self.raw, dtype=bool)

    def test_observed_mean_is_distinct_from_missing(self):
        self.raw[:, 4] = np.nan
        self.mask[:, 4] = True
        self.raw[1, 1] = np.nan
        self.raw[2, 3] = np.inf
        self.mask[3, 2] = True
        self.raw[0, 0] += 4
        self.raw[0, 3] = -5  # Finite anomalies remain observed.
        original = self.raw.copy()
        original_mask = self.mask.copy()

        values, obs = preprocess_weather_covariates(self.raw, self.mask, self.scaler)
        self.assertEqual(values.shape, (4, 18))
        self.assertEqual(obs.shape, (4, 8))
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:, 4], 0)
        np.testing.assert_array_equal(obs[:, 4], 0)
        self.assertEqual(values[0, 1], 0)
        self.assertEqual(obs[0, 1], 1)
        for row, col in [(1, 1), (2, 3), (3, 2)]:
            self.assertEqual(values[row, col], 0)
            self.assertEqual(obs[row, col], 0)
        self.assertAlmostEqual(float(values[0, 0]), float(np.arcsinh(2)), places=6)
        self.assertAlmostEqual(float(values[0, 3]), float(np.arcsinh(-4)), places=6)
        self.assertEqual(obs[0, 3], 1)
        np.testing.assert_array_equal(self.raw, original)
        np.testing.assert_array_equal(self.mask, original_mask)

    def test_future_vegetation_placeholders_are_ignored(self):
        self.raw[:, 18:] = np.nan
        self.mask[:, 18:] = True
        values, obs = preprocess_weather_covariates(self.raw, self.mask, self.scaler)
        np.testing.assert_array_equal(values, 0)
        np.testing.assert_array_equal(obs, 1)
        v18, m18 = preprocess_weather_covariates(self.raw[:, :18], self.mask[:, :18], self.scaler)
        np.testing.assert_array_equal(v18, values)
        np.testing.assert_array_equal(m18, obs)
        self.assertTrue(np.isnan(self.raw[:, 18:]).all())

    def test_all_weather_missing(self):
        self.raw[:, :8] = np.nan
        values, obs = preprocess_weather_covariates(self.raw, self.mask, self.scaler)
        np.testing.assert_array_equal(values[:, :8], 0)
        np.testing.assert_array_equal(obs, 0)

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            preprocess_weather_covariates(self.raw, self.mask[:, :8], self.scaler)

    def test_history_dataset_matches_inference_features(self):
        self.raw[:, 18:] = 0.4
        self.raw[1, 18:] = np.nan
        self.mask[1, 18:] = True
        self.mask[0, 2] = True
        path = self.data_dir / "history.npz"
        np.savez(path, history=self.raw[None], history_mask=self.mask[None])

        dataset = HistoryImputationDataset(
            self.data_dir, self.scaler, file_list=[path], mask_dropout=False, climate_dropout_p=0.0
        )
        features = dataset[0]["input"]
        self.assertEqual(features.shape, (4, 35))
        self.assertEqual(features[0, 2].item(), 0.0)
        self.assertEqual(features[0, 29].item(), 0.0)
        self.assertEqual(features[1, 29].item(), 1.0)
        np.testing.assert_array_equal(features.numpy(), build_imputer_features(self.raw, self.mask, self.scaler))

        model = BiGRUHistoryImputer()
        self.assertEqual(model.input_proj[0].in_features, 43)
        with torch.inference_mode():
            predictions = model(features.unsqueeze(0))
        self.assertEqual(predictions.shape, (1, 4, 3))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_forecasting_dataset_uses_future_mask_and_perturbed_values(self):
        self.raw[:, 18:] = 0.4
        self.raw[0, 4] = np.nan
        future = self.raw[:3].copy()
        future[:, 4] = self.scaler.mean[4]
        future[:, 18:] = np.nan
        future[0, 0] += 4
        future_mask = np.isnan(future)
        future_mask[1, 0] = True
        path = self.data_dir / "forecasting.npz"
        np.savez(
            path,
            history=self.raw[None],
            history_mask=np.isnan(self.raw)[None],
            future=future[None],
            future_mask=future_mask[None],
            future_noise=np.full((1, 3, 8), 100.0),
            target=np.full((1, 3, 3), 0.4),
            future_target_positions=np.arange(3)[None],
        )
        dataset = VegetationDataset(self.data_dir, self.scaler, file_list=[path])
        sample = dataset[0]
        self.assertEqual(sample["history"][0, 4].item(), 0.0)
        self.assertEqual(sample["history_weather_mask"][0, 4].item(), 0.0)
        self.assertEqual(sample["future_weather"].shape, (3, 18))
        self.assertAlmostEqual(sample["future_weather"][0, 0].item(), float(np.arcsinh(2)), places=6)
        self.assertEqual(sample["future_weather"][1, 0].item(), 0.0)
        np.testing.assert_array_equal(sample["future_weather_mask"][:, 0], [1, 0, 1])

        batch = collate_vegetation_batch([sample])
        torch.testing.assert_close(batch["history_weather_mask"][0], sample["history_weather_mask"])
        torch.testing.assert_close(batch["future_weather_mask"][0], sample["future_weather_mask"])

    def test_legacy_checkpoint_is_rejected(self):
        path = self.data_dir / "legacy.pt"
        torch.save({"model_state_dict": BiGRUHistoryImputer(input_dim=27).state_dict()}, path)

        with self.assertRaisesRegex(ValueError, "35-channel history imputer"):
            load_imputer_checkpoint(path, device="cpu")

    def test_model_interpolator_preserves_weather(self):
        path = self.data_dir / "weather_mask.pt"
        torch.save({"model_state_dict": BiGRUHistoryImputer().state_dict()}, path)
        self.raw[:, 18:] = 0.4
        self.raw[1, 18:] = np.nan
        self.mask[1, 18:] = True
        self.raw[1, 1] = np.nan
        self.raw[0, 0] = 12.0
        interpolator = ModelInterpolator(path, self.scaler_path, device="cpu")

        result = interpolator.interpolate_history_sample(self.raw, self.mask)
        np.testing.assert_array_equal(result[:, :18], self.raw[:, :18])
        self.assertTrue(np.isfinite(result[:, 18:]).all())
        observed = ~self.mask[:, 18:]
        np.testing.assert_array_equal(result[:, 18:][observed], self.raw[:, 18:][observed])


if __name__ == "__main__":
    unittest.main()
