import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.benchmark_imputers import evaluate_sample_leave_one_out, run_benchmark
from src.data.interpolation import build_interpolator_from_config
from src.data.scaler_fit import experiment_checkpoint_path, fit_arcsinh_scaler, resolve_experiment_scaler
from src.data.splits import get_experiment_split
from src.data.transforms import ArcsinhScaler, scaler_for_imputer
from src.data.weather_preprocessor import WeatherPreprocessor
from src.experiment_protocol import prepare_experiments, score_internal_forecast
from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer
from src.utils.config import load_config


class ExperimentProtocolTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data_dir = self.root / "tiles"
        self.data_dir.mkdir()
        self.scaler = ArcsinhScaler(dict(mean=[0] * 21, var=[1] * 21, target_mean=[0] * 3, target_var=[1] * 3))
        reference = self.root / "reference.json"
        reference.write_text(json.dumps(self.scaler.to_dict()), encoding="utf-8")
        self.cfg = {
            "paths": {
                "train_data_dir": self.data_dir,
                "scaler_path": reference,
                "checkpoints_dir": self.root,
                "results_dir": self.root,
            },
            "data": {"num_train_tiles": 1, "num_val_tiles": 1, "seed": 42},
            "weather_preprocessing": {"profile": "M0"},
            "scaling": {"mode": "fitting_tiles"},
            "imputer_training": {"checkpoint_name": "bigru.pt", "epochs": 5},
            "mae_training": {"checkpoint_name": "mae.pt", "epochs": 5},
        }
        self.history = np.tile(np.array(
            [3, 50, 100, 2, 1013, 12, 10, 18, *([0] * 10), 0.2, 0.3, 0.4], dtype=np.float32
        ), (4, 1))
        self.history[:, 0] = [1, 3, 999, 5]
        self.history[:, 5] = [12, 2, 12, 12]
        self.history[3, 2] = np.nan
        self.history_mask = np.isnan(self.history)
        self.history_mask[2, 0] = True
        self.target = np.full((1, 3, 3), 0.6, dtype=np.float32)
        self.target[0, 1, 1] = 999
        self.target_mask = np.zeros_like(self.target, dtype=bool)
        self.target_mask[0, 1, 1] = True
        for name, delta in (("tile_a", 0), ("tile_b", 2)):
            np.savez(
                self.data_dir / f"{name}.npz",
                history=self.history[None] + delta,
                history_mask=self.history_mask[None],
                target=self.target + delta,
                target_mask=self.target_mask,
            )

    def test_fit_reads_only_fitting_tiles_and_excludes_qc_and_masks(self):
        fit = [self.data_dir / "tile_a.npz"]
        nonexistent_validation = [self.data_dir / "validation.npz"]
        scaler = fit_arcsinh_scaler(fit, nonexistent_validation, WeatherPreprocessor("M2"), 42)
        self.assertEqual(scaler.mean[0], 3.0)
        np.testing.assert_allclose(scaler.var[0], 8 / 3)
        self.assertEqual(scaler.mean[5], 12.0)
        self.assertEqual(scaler.provenance["feature_counts"][5:8], [3, 3, 3])
        self.assertEqual(scaler.provenance["feature_counts"][2], 3)
        self.assertEqual(scaler.provenance["target_counts"], [3, 2, 3])
        np.testing.assert_allclose(scaler.target_mean, [0.6, 0.6, 0.6])
        self.assertEqual(scaler.to_dict(), ArcsinhScaler(scaler.to_dict()).to_dict())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_arcsinh_scaler(fit, fit, WeatherPreprocessor("M1"), 42)

    def test_protocol_preparation_and_scoring_use_same_validation_windows(self):
        output = self.root / "study"
        paths = prepare_experiments(self.cfg, output, imputer_profile="M2", candidates=("M4:bigru",))
        self.assertEqual(len(paths), 7)
        self.assertEqual(paths, prepare_experiments(self.cfg, output, imputer_profile="M2", candidates=("M4:bigru",)))
        manifest = json.loads((output / "split.json").read_text(encoding="utf-8"))
        self.assertFalse(set(manifest["fit_tiles"]) & set(manifest["validation_tiles"]))
        m0 = load_config(output / "A_m0_linear.yaml")
        m4 = load_config(output / "A_m4_linear.yaml")
        self.assertEqual(m0["paths"]["fitted_scaler_path"], m4["paths"]["fitted_scaler_path"])
        cfg = load_config(output / "C_m4_bigru.yaml")
        self.assertEqual(cfg["interpolation"]["strategy"], "bigru")
        self.assertEqual(cfg["weather_preprocessing"]["profile"], "M4")
        _, validation, _ = get_experiment_split(cfg)
        self.assertEqual(manifest["validation_tiles"], [p.name for p in validation])
        scaler = resolve_experiment_scaler(cfg, WeatherPreprocessor("M4"))
        self.assertEqual(scaler.provenance["fit_tiles"], manifest["fit_tiles"])
        with self.assertRaisesRegex(ValueError, "different split or cleaning policy"):
            resolve_experiment_scaler(cfg, WeatherPreprocessor("M2"))
        with np.load(validation[0]) as tile:
            labels = tile["target"]
        path = self.root / "predictions.npz"
        np.savez(
            path, y_pred_quantiles=labels[..., None] + [-0.1, 0.0, 0.1],
            tile_names=[validation[0].stem], window_indices=[0],
        )
        scores = score_internal_forecast(cfg, path)
        self.assertAlmostEqual(scores["overall"]["MAE"], 0.0)
        self.assertLess(scores["overall"]["Pinball"], 0.01)
        self.assertEqual(scores["quantile_calibration"]["NDVI"], {"0.1": 0.0, "0.5": 1.0, "0.9": 1.0})
        np.savez(path, y_pred_quantiles=labels[..., None] + [-0.1, 0, 0.1], tile_names=["other"], window_indices=[0])
        with self.assertRaisesRegex(ValueError, "Prediction IDs"):
            score_internal_forecast(cfg, path)

    def test_benchmark_supplies_same_hidden_dates_without_embedded_targets(self):
        class Probe:
            def __init__(self):
                self.hidden = []

            def interpolate(self, history, mask, columns):
                hidden = np.flatnonzero(mask[:, columns].all(axis=1))
                assert len(hidden) == 1 and np.isnan(history[hidden][:, columns]).all()
                self.hidden.append(int(hidden[0]))
                history[np.ix_(hidden, columns)] = 0.5
                return history

        first, second = Probe(), Probe()
        originals = self.history.copy()
        evaluate_sample_leave_one_out(
            self.history, self.history_mask, "Cfa", {"first": first, "second": second}, self.scaler,
        )
        self.assertEqual(first.hidden, [0, 1, 2, 3])
        self.assertEqual(first.hidden, second.hidden)
        np.testing.assert_array_equal(self.history, originals)

    def test_definitive_benchmark_rejects_missing_and_untrained_checkpoints(self):
        cfg_path = self.root / "config.yaml"
        cfg_path.write_text(json.dumps(self.cfg, default=str), encoding="utf-8")
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(FileNotFoundError, "Required trained checkpoints missing"):
                run_benchmark(cfg_path, eval_only=True, device="cpu")
        weather = WeatherPreprocessor("M0")
        scaler = resolve_experiment_scaler(self.cfg, weather)
        for name, model in (("bigru.pt", BiGRUHistoryImputer()), ("mae.pt", MAEHistoryImputer())):
            path = experiment_checkpoint_path(self.cfg, weather, self.root / name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler": scaler.to_dict(),
                "training": {"completed_epochs": 5, "optimizer_steps": 1, "seconds": 1.0, "seed": 42},
            }, path)
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "lacks a completed training run"):
                run_benchmark(cfg_path, eval_only=True, device="cpu")
        self.assertFalse(list(self.root.glob("imputation_benchmark*.csv")))

    def test_wrappers_use_embedded_scaler_even_without_legacy_file(self):
        scaler = resolve_experiment_scaler(self.cfg, WeatherPreprocessor("M0"))
        for strategy, key, model in (
            ("bigru", "model_checkpoint", BiGRUHistoryImputer()),
            ("mae", "mae_checkpoint", MAEHistoryImputer()),
        ):
            base_path = self.root / f"{strategy}.pt"
            path = experiment_checkpoint_path(self.cfg, WeatherPreprocessor("M0"), base_path)
            torch.save({"model_state_dict": model.state_dict(), "scaler": scaler.to_dict()}, path)
            config = {
                **self.cfg,
                "paths": {**self.cfg["paths"], "scaler_path": self.root / "missing.json"},
                "interpolation": {"strategy": strategy, key: base_path},
            }
            wrapper = build_interpolator_from_config(config, strict_checkpoint=True)
            self.assertEqual(wrapper.scaler.to_dict(), scaler.to_dict())
            with self.assertRaisesRegex(ValueError, "differs from the scaler embedded"):
                scaler_for_imputer(wrapper.model, self.scaler)


if __name__ == "__main__":
    unittest.main()
