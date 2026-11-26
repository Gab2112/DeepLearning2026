"""
Unit and integration tests for MAEHistoryImputer, MAEInterpolator,
checkpoint round-trip, factory registration, and dataset integration.
"""

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.interpolation import (
    InterpolationFactory,
    LinearInterpolator,
    MAEInterpolator,
    ModelInterpolator,
    NoneInterpolator,
    SplineInterpolator,
    build_interpolator_from_config,
)
from src.data.mae_imputer import (
    evaluate_mae_imputer_on_sample,
    impute_batch_sequences_mae,
    impute_sequence_mae,
    load_mae_imputer_checkpoint,
)
from src.data.transforms import ArcsinhScaler
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.models.mae_imputer import MAEHistoryImputer


def test_mae_factory_registration():
    strategies = InterpolationFactory.available_strategies()
    for alias in ["mae", "mae_imputer", "masked_autoencoder"]:
        assert alias in strategies, f"Missing '{alias}' in InterpolationFactory!"
        inst = InterpolationFactory.create(
            alias,
            checkpoint_path=PROJECT_ROOT / "checkpoints" / "nonexistent_mae.pt",
            device="cpu",
        )
        assert isinstance(inst, MAEInterpolator)
        assert inst.supports_batch_history_imputation is True
    print("[+] MAE factory registration test PASSED.")


def test_mae_forward_padding_gradients_and_blend():
    torch.manual_seed(42)
    B, L = 3, 15
    x = torch.randn(B, L, 27)
    # Build clean observation flags in 21..24 and padding mask
    obs_mask = torch.zeros(B, L, dtype=torch.bool)
    obs_mask[:, [1, 6, 10]] = True
    x[:, :, 21:24] = obs_mask.unsqueeze(-1).float()
    x[:, :, 18:21][~obs_mask] = 0.0

    padding_mask = torch.ones(B, L, dtype=torch.bool)
    padding_mask[1, 11:] = False
    padding_mask[2, 13:] = False

    climate_id = torch.tensor([0, 3, 7], dtype=torch.long)
    weather_mask = torch.zeros(B, L, dtype=torch.bool)
    weather_mask[:, [2, 5]] = True
    weather_mask &= padding_mask

    model = MAEHistoryImputer(
        input_dim=27,
        climate_dim=8,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=64,
        output_dim=3,
        dropout=0.0,
    )

    # 1. Forward without weather return
    preds = model(x, climate_id=climate_id, padding_mask=padding_mask)
    assert preds.shape == (B, L, 3)
    assert torch.isfinite(preds).all()
    assert torch.all(preds[1, 11:] == 0.0)
    assert torch.all(preds[2, 13:] == 0.0)

    # 2. Forward with weather_mask and return_weather=True + gradient flow
    preds_t, preds_w = model(
        x,
        climate_id=climate_id,
        padding_mask=padding_mask,
        weather_mask=weather_mask,
        return_weather=True,
    )
    assert preds_t.shape == (B, L, 3) and preds_w.shape == (B, L, 11)
    loss = preds_t.pow(2).mean() + preds_w.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_norms) == sum(1 for p in model.parameters() if p.requires_grad)
    assert any(g > 0.0 for g in grad_norms)

    # 3. impute_and_blend preserves exact observed values
    raw_targets = x[:, :, 18:21].clone()
    blended = model.impute_and_blend(
        x, raw_targets, obs_mask, climate_id=climate_id, padding_mask=padding_mask
    )
    torch.testing.assert_close(blended[obs_mask], raw_targets[obs_mask])
    assert torch.all(blended[~padding_mask] == 0.0)
    print("[+] MAE forward, padding, gradients & impute_and_blend test PASSED.")


def test_mae_checkpoint_roundtrip_and_interpolator():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler_path = data_dir / "scaler.json"
    if not data_dir.exists() or not scaler_path.exists():
        print("[!] Dataset path not found, skipping MAE checkpoint & interpolator test.")
        return

    scaler = ArcsinhScaler(scaler_path)
    sample_npz = sorted(list(data_dir.glob("*.npz")))[0]
    d = np.load(sample_npz, allow_pickle=True)
    h0 = d["history"][0].astype(np.float32).copy()
    m0 = d["history_mask"][0].astype(bool).copy()
    h1 = d["history"][1].astype(np.float32).copy()
    m1 = d["history"][1].astype(bool).copy()
    c0 = str(d["climate"][0]) if "climate" in d else "unknown"
    c1 = str(d["climate"][1]) if "climate" in d else "unknown"

    # Fallback mode (checkpoint absent)
    fallback_int = MAEInterpolator(
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "nonexistent_mae.pt",
        scaler_path=scaler_path,
        device="cpu",
    )
    out_fb = fallback_int.interpolate_history_sample(h0, m0, c0)
    assert not np.any(np.isnan(out_fb[:, 18:21]))
    assert np.nanmax(np.abs(out_fb[:, 4])) > 100.0, "Weather pressure clipped in MAE fallback!"

    # Save a dummy MAE checkpoint to temp dir and load via MAEInterpolator + load_mae_imputer_checkpoint
    mae_cfg = {
        "input_dim": 27,
        "climate_dim": 8,
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "output_dim": 3,
        "dropout": 0.0,
    }
    model = MAEHistoryImputer(**mae_cfg)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = Path(tmpdir) / "mae_test.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "config": {"mae_model": mae_cfg},
            },
            ckpt_file,
        )

        loaded_model, _ = load_mae_imputer_checkpoint(ckpt_file, device="cpu")
        assert isinstance(loaded_model, MAEHistoryImputer)

        mae_int = InterpolationFactory.create(
            "mae",
            checkpoint_path=ckpt_file,
            scaler_path=scaler_path,
            device="cpu",
            clip_min=-1.0,
            clip_max=1.0,
        )
        assert mae_int.model is not None

        # Single sample imputation
        imp_single = mae_int.interpolate_history_sample(h0, m0, c0)
        assert imp_single.shape == h0.shape
        assert not np.any(np.isnan(imp_single[:, 18:21]))
        assert np.all(imp_single[:, 18:21] >= -1.0) and np.all(imp_single[:, 18:21] <= 1.0)
        obs0 = ~m0[:, 18]
        np.testing.assert_allclose(imp_single[obs0, 18:21], h0[obs0, 18:21], rtol=1e-5)
        np.testing.assert_allclose(imp_single[:, :18], h0[:, :18], equal_nan=True)

        # Batched imputation (handles variable lengths)
        imp_batch = mae_int.interpolate_history_batch([h0, h1], [m0, m1], [c0, c1], batch_size=2)
        assert len(imp_batch) == 2
        obs1 = ~m1[:, 18]
        np.testing.assert_allclose(imp_batch[1][obs1, 18:21], h1[obs1, 18:21], rtol=1e-5)
        np.testing.assert_allclose(imp_batch[1][:, :18], h1[:, :18], equal_nan=True)

        # Direct helper functions + leave-one-out eval
        imp_seq = impute_sequence_mae(h0, m0, c0, model=loaded_model, scaler=scaler, device="cpu")
        assert not np.any(np.isnan(imp_seq[:, 18:21]))
        imp_b2 = impute_batch_sequences_mae(
            [h0, h1], [m0, m1], [c0, c1], model=loaded_model, scaler=scaler, device="cpu", batch_size=1
        )
        assert len(imp_b2) == 2
        loo = evaluate_mae_imputer_on_sample(h0, m0, c0, model=loaded_model, scaler=scaler, device="cpu")
        assert set(loo.keys()) == {"NDVI", "GNDVI", "EVI"}

        # VegetationDataset integration with batched precomputation
        ds = VegetationDataset(
            data_dir=data_dir,
            scaler=scaler,
            max_files=1,
            interpolation_strategy=mae_int,
            precompute_model_imputation=True,
            imputation_batch_size=32,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_vegetation_batch)
        batch = next(iter(loader))
        assert not torch.isnan(batch["history"]).any()
        assert "history_obs_mask" in batch
        assert batch["history_obs_mask"].sum() > 0

        # Ensure weather columns were not clipped by MAE precomputation
        raw_h = ds.file_cache[ds.files[0]]["history"][0].astype(np.float32)
        expected_w = np.nan_to_num(scaler.transform_features(raw_h)[:, :18], nan=0.0)
        valid_w = ~ds.file_cache[ds.files[0]]["history_mask"][0][:, :18]
        np.testing.assert_allclose(
            ds[0]["history"].numpy()[:, :18][valid_w],
            expected_w[valid_w],
            rtol=1e-4,
            atol=1e-4,
        )

    print("[+] MAE checkpoint round-trip, interpolator & VegetationDataset integration test PASSED.")


def test_mae_no_circular_imports():
    statements = [
        "import src.models.mae_imputer",
        "from src.models import MAEHistoryImputer",
        "from src.data.mae_imputer import load_mae_imputer_checkpoint",
        "import src.data",
    ]
    for stmt in statements:
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Circular import for '{stmt}':\n{proc.stderr}"
    print("[+] MAE no-circular-imports test PASSED.")


def test_build_interpolator_from_config():
    from src.utils.config import load_config

    base_cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")

    for strat, expected_cls in [
        ("linear", LinearInterpolator),
        ("spline", SplineInterpolator),
        ("none", NoneInterpolator),
    ]:
        cfg = {**base_cfg, "interpolation": {**base_cfg["interpolation"], "strategy": strat}}
        inst = build_interpolator_from_config(cfg)
        assert isinstance(inst, expected_cls), f"Expected {expected_cls}, got {type(inst)}"

    # Neural strategies warn cleanly on missing checkpoint when strict_checkpoint=False
    for strat, expected_cls, ckpt_key in [
        ("model", ModelInterpolator, "model_checkpoint"),
        ("mae", MAEInterpolator, "mae_checkpoint"),
    ]:
        cfg = {
            **base_cfg,
            "interpolation": {
                **base_cfg["interpolation"],
                "strategy": strat,
                ckpt_key: "checkpoints/definitely_missing_ckpt.pt",
            },
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inst = build_interpolator_from_config(cfg, strict_checkpoint=False)
            assert isinstance(inst, expected_cls)
            assert any(issubclass(w.category, RuntimeWarning) for w in caught)

        raised = False
        try:
            build_interpolator_from_config(cfg, strict_checkpoint=True)
        except FileNotFoundError:
            raised = True
        assert raised, f"Expected FileNotFoundError with strict_checkpoint=True for {strat}"

    print("[+] build_interpolator_from_config test PASSED.")


if __name__ == "__main__":
    print("=" * 64)
    print(" RUNNING MAE HISTORY IMPUTER UNIT & INTEGRATION TESTS")
    print("=" * 64)
    test_mae_factory_registration()
    test_mae_forward_padding_gradients_and_blend()
    test_mae_checkpoint_roundtrip_and_interpolator()
    test_mae_no_circular_imports()
    test_build_interpolator_from_config()
    print("\n[***] ALL MAE IMPUTER TESTS PASSED! [***]")
