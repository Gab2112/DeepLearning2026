"""
Unit and integration tests for InterpolationFactory, individual strategies,
and integration with the dataset loader.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.interpolation import (
    InterpolationFactory,
    BaseInterpolator,
    LinearInterpolator,
    ForwardFillInterpolator,
    BackwardFillInterpolator,
    NearestInterpolator,
    SplineInterpolator,
    MeanInterpolator,
    MedianInterpolator,
    ConstantInterpolator,
    NoneInterpolator,
    ModelInterpolator,
)
from src.data.transforms import ArcsinhScaler
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.models.history_imputer import HISTORY_IMPUTER_CHECKPOINT_NAME


def test_factory_registration():
    strategies = InterpolationFactory.available_strategies()
    print(f"[*] Registered strategies: {strategies}")
    expected = [
        "linear", "forward_fill", "backward_fill", "nearest",
        "spline", "mean", "median", "constant", "none", "model"
    ]
    for exp in expected:
        assert exp in strategies, f"Strategy {exp} missing from factory!"

    # Test factory instantiation by name and aliases
    interp_lin = InterpolationFactory.create("linear")
    assert isinstance(interp_lin, LinearInterpolator)

    interp_ffill = InterpolationFactory.create("ffill")
    assert isinstance(interp_ffill, ForwardFillInterpolator)

    interp_pchip = InterpolationFactory.create("pchip")
    assert isinstance(interp_pchip, SplineInterpolator)

    interp_zero = InterpolationFactory.create("zero")
    assert isinstance(interp_zero, ConstantInterpolator)

    print("[+] Factory registration test PASSED.")


def test_1d_strategies():
    # Sequence of length 7 with observations at t=1 (0.2) and t=5 (0.6)
    # [NaN, 0.2, NaN, NaN, NaN, 0.6, NaN]
    raw = np.array([np.nan, 0.2, np.nan, np.nan, np.nan, 0.6, np.nan], dtype=np.float32)
    mask = np.isnan(raw)

    # 1. Linear: flat boundary, slope = (0.6 - 0.2) / 4 = 0.1
    # Expected: [0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6]
    lin = InterpolationFactory.create("linear")
    res_lin = lin.interpolate_series(raw, mask)
    np.testing.assert_allclose(res_lin, [0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6], rtol=1e-5)
    assert not np.any(np.isnan(res_lin))

    # 2. Forward fill: [0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.6]
    ffill = InterpolationFactory.create("forward_fill")
    res_ffill = ffill.interpolate_series(raw, mask)
    np.testing.assert_allclose(res_ffill, [0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.6], rtol=1e-5)
    assert not np.any(np.isnan(res_ffill))

    # 3. Backward fill: [0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.6]
    bfill = InterpolationFactory.create("backward_fill")
    res_bfill = bfill.interpolate_series(raw, mask)
    np.testing.assert_allclose(res_bfill, [0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.6], rtol=1e-5)
    assert not np.any(np.isnan(res_bfill))

    # 4. Nearest:
    # t=0 -> t=1 (0.2)
    # t=2 -> t=1 (0.2)
    # t=3 -> t=1 or 5 (0.2 or 0.6)
    # t=4 -> t=5 (0.6)
    # t=6 -> t=5 (0.6)
    nearest = InterpolationFactory.create("nearest")
    res_nearest = nearest.interpolate_series(raw, mask)
    assert res_nearest[0] == 0.2
    assert res_nearest[2] == 0.2
    assert res_nearest[4] == 0.6
    assert res_nearest[6] == 0.6
    assert not np.any(np.isnan(res_nearest))

    # 5. Spline / Pchip with 3 points
    raw_3pts = np.array([np.nan, 0.2, np.nan, 0.5, np.nan, 0.7, np.nan], dtype=np.float32)
    spline = InterpolationFactory.create("spline", kind="pchip", clip_min=0.0, clip_max=1.0)
    res_spline = spline.interpolate_series(raw_3pts)
    assert not np.any(np.isnan(res_spline))
    assert 0.0 <= np.min(res_spline) and np.max(res_spline) <= 1.0

    # 6. Mean & Median
    mean_int = InterpolationFactory.create("mean")
    res_mean = mean_int.interpolate_series(raw, mask)
    assert res_mean[0] == 0.4
    assert res_mean[2] == 0.4

    # 7. Zero / Constant
    zero_int = InterpolationFactory.create("constant", fill_value=-999.0)
    res_zero = zero_int.interpolate_series(raw, mask)
    assert res_zero[0] == -999.0
    assert res_zero[1] == 0.2

    print("[+] 1D strategies test PASSED.")


def test_2d_and_edge_cases():
    # 2D array of shape (5, 3)
    arr = np.array([
        [np.nan, 10.0, np.nan],
        [1.0,    np.nan, np.nan],
        [2.0,    np.nan, np.nan],
        [np.nan, 20.0,   np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    mask = np.isnan(arr)

    # Test interpolating all columns
    lin = InterpolationFactory.create("linear")
    res = lin.interpolate(arr, mask=mask)
    assert res.shape == (5, 3)
    # Column 2 was completely NaN -> should be filled with fill_value (0.0)
    np.testing.assert_allclose(res[:, 2], 0.0)
    # Column 0: 1.0 at t=1, 2.0 at t=2 -> boundary t=0 is 1.0, boundaries t=3,4 are 2.0
    np.testing.assert_allclose(res[:, 0], [1.0, 1.0, 2.0, 2.0, 2.0])
    # Column 1: 10.0 at t=0, 20.0 at t=3 -> slope = 10/3
    assert not np.any(np.isnan(res))

    # Test column restriction: only interpolate column 0
    res_sub = lin.interpolate(arr, mask=mask, columns=[0])
    assert not np.any(np.isnan(res_sub[:, 0]))
    assert np.isnan(res_sub[1, 1])  # col 1 still has NaN

    print("[+] 2D and edge cases test PASSED.")


def test_dataset_integration():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"

    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping dataset integration test.")
        return

    scaler = ArcsinhScaler(scaler_path)

    for strat in ["linear", "forward_fill", "spline", "mean"]:
        ds = VegetationDataset(
            data_dir=data_dir,
            scaler=scaler,
            max_files=1,
            interpolation_strategy=strat,
            interpolation_columns=[18, 19, 20],
        )

        sample = ds[0]
        hist = sample["history"]
        fut = sample["future_weather"]
        tgt = sample["target"]

        # Check no NaNs exist in any returned tensor
        assert not torch.isnan(hist).any(), f"NaN found in history using strategy {strat}!"
        assert not torch.isnan(fut).any(), f"NaN found in future weather using strategy {strat}!"
        assert not torch.isnan(tgt).any(), f"NaN found in target using strategy {strat}!"
        assert hist.shape[-1] == 21, f"Expected 21 features in history, got {hist.shape[-1]}"
        assert tgt.shape == (3, 3), f"Expected (3, 3) target, got {tgt.shape}"
        assert sample["history_weather_mask"].shape == (hist.shape[0], 8)
        assert sample["future_weather_mask"].shape == (fut.shape[0], 8)

        print(f"[+] VegetationDataset with strategy='{strat}' PASSED (sample seq len: {hist.shape[0]}).")

    # Test batch collate
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_vegetation_batch)
    batch = next(iter(loader))
    assert batch["history"].ndim == 3  # (B, max_L, 21)
    assert not torch.isnan(batch["history"]).any()
    assert batch["target"].shape == (4, 3, 3)
    print("[+] Collate batch test PASSED.")


def test_model_interpolator():
    ckpt_path = PROJECT_ROOT / "checkpoints" / HISTORY_IMPUTER_CHECKPOINT_NAME
    scaler_path = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky" / "scaler.json"

    if not ckpt_path.exists():
        print("[!] Bi-GRU Checkpoint not found, skipping model interpolator test.")
        return

    model_int = InterpolationFactory.create(
        "model",
        checkpoint_path=ckpt_path,
        scaler_path=scaler_path,
        clip_min=-1.0,
        clip_max=1.0,
    )
    assert isinstance(model_int, ModelInterpolator)

    # Test on real npz sample
    sample_npz = sorted(list((PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky").glob("*.npz")))[0]
    d = np.load(sample_npz, allow_pickle=True)
    h0 = d["history"][0]
    m0 = d["history_mask"][0]
    climate0 = str(d["climate"][0]) if "climate" in d else "unknown"

    imputed = model_int.interpolate_history_sample(h0, m0, climate0)
    assert imputed.shape == h0.shape
    assert not np.any(np.isnan(imputed[:, 18:21]))
    np.testing.assert_array_equal(imputed[:, :18], h0[:, :18])
    # Original valid observations must be preserved or within valid bounds
    obs_mask = ~m0[:, 18]
    np.testing.assert_allclose(imputed[obs_mask, 18], h0[obs_mask, 18], rtol=1e-5)
    print(f"[+] ModelInterpolator (Bi-GRU) test PASSED (imputed {h0.shape[0]} days).")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING INTERPOLATION FACTORY UNIT & INTEGRATION TESTS")
    print("=" * 60)
    test_factory_registration()
    test_1d_strategies()
    test_2d_and_edge_cases()
    test_model_interpolator()
    test_dataset_integration()
    print("\n[***] ALL TESTS COMPLETED SUCCESSFULLY! [***]")
