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
from src.data.history_dataset import HistoryImputationDataset
from src.data.splits import get_train_val_tile_split, get_tile_dominant_climate
from src.data.bigru_imputer import BiGRUHistoryImputer as BigruModuleImputer
from src.models.history_imputer import BiGRUHistoryImputer as ModelsModuleImputer


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
            interpolate_weather=True,
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

        print(f"[+] VegetationDataset with strategy='{strat}' PASSED (sample seq len: {hist.shape[0]}).")

    # Test batch collate
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_vegetation_batch)
    batch = next(iter(loader))
    assert batch["history"].ndim == 3  # (B, max_L, 21)
    assert not torch.isnan(batch["history"]).any()
    assert batch["target"].shape == (4, 3, 3)
    print("[+] Collate batch test PASSED.")


def test_model_interpolator():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "history_imputer_best.pt"
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
    assert not np.any(np.isnan(imputed))
    # Original valid observations must be preserved or within valid bounds
    obs_mask = ~m0[:, 18]
    np.testing.assert_allclose(imputed[obs_mask, 18], h0[obs_mask, 18], rtol=1e-5)
    print(f"[+] ModelInterpolator (Bi-GRU) test PASSED (imputed {h0.shape[0]} days).")


def test_collate_includes_history_obs_mask():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"
    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping collate history_obs_mask test.")
        return

    scaler = ArcsinhScaler(scaler_path)
    ds = VegetationDataset(
        data_dir=data_dir,
        scaler=scaler,
        max_files=1,
        interpolation_strategy="linear",
    )
    items = [ds[i] for i in range(4)]
    batch = collate_vegetation_batch(items)
    assert "history_obs_mask" in batch, "history_obs_mask missing from collate_vegetation_batch!"
    obs_mask = batch["history_obs_mask"]
    pad_mask = batch["history_padding_mask"]
    assert obs_mask.shape == pad_mask.shape
    assert obs_mask.dtype == torch.bool
    # No padded position can be marked observed
    assert not torch.any(obs_mask & ~pad_mask)
    for i, item in enumerate(items):
        lh = item["history_length"]
        assert torch.equal(obs_mask[i, :lh], item["history_obs_mask"])
        assert obs_mask[i, :lh].sum().item() == item["history_obs_mask"].sum().item()
    print("[+] Collate history_obs_mask test PASSED.")


def test_none_strategy_normalizes_before_zero_fill():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"
    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping none strategy normalization test.")
        return

    scaler = ArcsinhScaler(scaler_path)
    ds = VegetationDataset(
        data_dir=data_dir,
        scaler=scaler,
        max_files=1,
        interpolation_strategy="none",
    )
    sample = ds[0]
    hist = sample["history"].numpy()
    obs_mask = sample["history_obs_mask"].numpy()

    # Unobserved target timesteps must be exactly 0.0 (not arcsinh((0 - mean)/std))
    unobserved_targets = hist[~obs_mask, 18:21]
    assert unobserved_targets.shape[0] > 0
    np.testing.assert_allclose(unobserved_targets, 0.0, atol=1e-6)

    # Observed target timesteps must match scaler.transform_targets(raw_obs)
    fpath, s_idx = ds.samples[0]
    h_raw = ds.file_cache[fpath]["history"][s_idx].astype(np.float32)
    expected_obs = scaler.transform_targets(h_raw[obs_mask, 18:21])
    np.testing.assert_allclose(hist[obs_mask, 18:21], expected_obs, rtol=1e-5, atol=1e-5)
    print("[+] None strategy normalizes-before-zero-fill test PASSED.")


def test_weather_not_clipped_by_interpolators():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"
    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping weather clipping test.")
        return

    scaler = ArcsinhScaler(scaler_path)
    sample_npz = sorted(list(data_dir.glob("*.npz")))[0]
    d = np.load(sample_npz, allow_pickle=True)
    h0 = d["history"][0].astype(np.float32).copy()
    m0 = d["history_mask"][0].astype(bool).copy()
    climate0 = str(d["climate"][0]) if "climate" in d else "unknown"

    # Inject a synthetic missing weather value (pressure at col 4, ~1013 hPa)
    if h0.shape[0] >= 3:
        m0[1, 4] = True
        h0[1, 4] = np.nan

    # 1. Test ModelInterpolator fallback & in-memory BiGRU model
    model_int = ModelInterpolator(
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "nonexistent_for_test.pt",
        scaler_path=scaler_path,
        device="cpu",
        clip_min=-1.0,
        clip_max=1.0,
    )
    res_fallback = model_int.interpolate_history_sample(h0, m0, climate0)
    assert np.nanmax(np.abs(res_fallback[:, 4])) > 100.0, "Weather pressure was clipped in fallback!"

    # Attach an untrained in-memory BiGRUHistoryImputer to exercise neural path
    dummy_model = ModelsModuleImputer(
        input_dim=27, climate_dim=8, hidden_dim=16, num_layers=1, output_dim=3, dropout=0.0
    ).to("cpu")
    dummy_model.eval()
    model_int.model = dummy_model

    res_sample = model_int.interpolate_history_sample(h0, m0, climate0)
    assert np.nanmax(np.abs(res_sample[:, 4])) > 100.0, "Weather pressure was clipped by ModelInterpolator!"
    assert np.nanmax(np.abs(res_sample[:, 11])) > 1.0, "day_of_year was clipped by ModelInterpolator!"
    assert not np.any(np.isnan(res_sample[:, 18:21]))

    res_batch = model_int.interpolate_history_batch([h0, h0], [m0, m0], [climate0, climate0], batch_size=2)
    assert len(res_batch) == 2
    assert np.nanmax(np.abs(res_batch[0][:, 4])) > 100.0, "Weather pressure was clipped in batch imputation!"

    # 2. Test VegetationDataset with Spline/Linear + interpolate_weather=True and with precomputed ModelInterpolator
    for strat in ["linear", "spline", model_int]:
        ds = VegetationDataset(
            data_dir=data_dir,
            scaler=scaler,
            max_files=1,
            interpolation_strategy=strat,
            interpolation_columns=[18, 19, 20],
            interpolate_weather=True,
            precompute_model_imputation=True,
            imputation_batch_size=32,
        )
        sample = ds[0]
        hist = sample["history"].numpy()
        assert not np.any(np.isnan(hist))
        # Verify weather columns in dataset were NOT clipped before arcsinh scaling
        fpath, s_idx = ds.samples[0]
        raw_h = ds.file_cache[fpath]["history"][s_idx].astype(np.float32)
        expected_weather = np.nan_to_num(scaler.transform_features(raw_h)[:, :18], nan=0.0)
        # At observed weather rows, normalized weather must match expected_weather (only NaNs interpolated)
        raw_w_mask = ds.file_cache[fpath]["history_mask"][s_idx][:, :18]
        valid_w = ~raw_w_mask
        np.testing.assert_allclose(hist[:, :18][valid_w], expected_weather[valid_w], rtol=1e-4, atol=1e-4)

    print("[+] Weather unclipped across interpolators & batched VegetationDataset test PASSED.")


def test_history_dataset_no_mask_dropout_leakage():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"
    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping mask dropout leakage test.")
        return

    scaler = ArcsinhScaler(scaler_path)
    ds = HistoryImputationDataset(
        data_dir=data_dir,
        scaler=scaler,
        max_files=1,
        mask_dropout=True,
        climate_dropout_p=0.0,
        interpolation_strategy="linear",
        seed=42,
    )

    checked = 0
    for idx in range(min(25, len(ds))):
        item = ds[idx]
        loss_mask = item["loss_mask"].numpy()
        obs_mask = item["all_obs_mask"].numpy()
        dropped_indices = np.where(loss_mask)[0]
        if len(dropped_indices) != 1:
            continue
        drop_idx = int(dropped_indices[0])
        remaining_obs = obs_mask.copy()
        remaining_obs[drop_idx] = False
        rem_indices = np.where(remaining_obs)[0]
        if len(rem_indices) < 2:
            continue

        fpath, s_idx = ds.samples[idx]
        h_raw = ds.file_cache[fpath]["history"][s_idx].astype(np.float32)

        # Compute expected linear interpolation using ONLY the remaining observed points
        interp_vals = np.zeros(3, dtype=np.float32)
        for c_i, col in enumerate([18, 19, 20]):
            interp_vals[c_i] = np.interp(drop_idx, rem_indices, h_raw[rem_indices, col])
        expected_norm = scaler.transform_targets(interp_vals.reshape(1, 3))[0]
        actual_norm = item["input"][drop_idx, 18:21].numpy()
        np.testing.assert_allclose(actual_norm, expected_norm, rtol=1e-4, atol=1e-4)
        checked += 1

    assert checked > 0, "Could not find a suitable sample for mask_dropout leakage check."
    print(f"[+] HistoryImputationDataset no-leakage test PASSED (verified {checked} samples).")


def test_bigru_class_unification():
    assert BigruModuleImputer is ModelsModuleImputer, (
        "BiGRUHistoryImputer should be defined once in src.models.history_imputer and imported in src.data.bigru_imputer."
    )
    print("[+] BiGRUHistoryImputer class unification test PASSED.")


def test_stratified_tile_split():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    if not data_dir.exists():
        print("[!] Train dataset path not found, skipping stratified tile split test.")
        return

    train_1, val_1, summary_1 = get_train_val_tile_split(data_dir, num_val_tiles=14, seed=42)
    train_2, val_2, summary_2 = get_train_val_tile_split(data_dir, num_val_tiles=14, seed=42)

    assert len(train_1) == 70
    assert len(val_1) == 14
    assert len(set(train_1).intersection(set(val_1))) == 0
    assert train_1 == train_2 and val_1 == val_2 and summary_1 == summary_2

    for clim in ["Cfb", "Dfb", "Csa", "BSk"]:
        assert clim in summary_1["train"] and summary_1["train"][clim] >= 1
        assert clim in summary_1["val"] and summary_1["val"][clim] >= 1

    # Singleton climate (e.g. Dfa with 1 tile) must stay in train
    all_climates = [get_tile_dominant_climate(p) for p in sorted(data_dir.glob("*.npz"))]
    for clim in set(all_climates):
        if all_climates.count(clim) == 1:
            assert summary_1["train"].get(clim, 0) == 1
            assert summary_1["val"].get(clim, 0) == 0

    print(f"[+] Stratified tile split test PASSED (val climates: {summary_1['val']}).")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING INTERPOLATION FACTORY UNIT & INTEGRATION TESTS")
    print("=" * 60)
    test_factory_registration()
    test_1d_strategies()
    test_2d_and_edge_cases()
    test_model_interpolator()
    test_dataset_integration()
    test_collate_includes_history_obs_mask()
    test_none_strategy_normalizes_before_zero_fill()
    test_weather_not_clipped_by_interpolators()
    test_history_dataset_no_mask_dropout_leakage()
    test_bigru_class_unification()
    test_stratified_tile_split()
    print("\n[***] ALL TESTS COMPLETED SUCCESSFULLY! [***]")
