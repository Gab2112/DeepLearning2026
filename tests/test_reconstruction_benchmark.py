"""
Unit and smoke tests for the unified reconstruction benchmark, vectorized orbital
dynamics, padding invariance, deterministic LOO dataset, frozen checkpoint integrity,
and classical baseline parity.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bigru_imputer import compute_orbital_dynamics
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.interpolation import InterpolationFactory, MAEInterpolator, ModelInterpolator
from src.data.transforms import ArcsinhScaler
from src.models.frozen import (
    build_reconstruction_model,
    extract_arch,
    load_frozen_model,
    save_frozen_checkpoint,
)
from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer
from src.train_reconstruction_benchmark import (
    evaluate_classical_baseline_loo,
    run_benchmark,
)
from src.utils.config import load_config


def _reference_orbital_dynamics(obs_mask: np.ndarray):
    L = len(obs_mask)
    delta_t = np.zeros(L, dtype=np.float32)
    orbital_phase = np.zeros(L, dtype=np.float32)
    missed_revisits = np.zeros(L, dtype=np.float32)
    last_obs = 0.0
    has_seen = False
    for t in range(L):
        orbital_phase[t] = (t % 5) / 5.0
        if obs_mask[t]:
            delta_t[t] = 0.0
            missed_revisits[t] = 0.0
            last_obs = float(t)
            has_seen = True
        elif has_seen:
            elapsed = float(t) - last_obs
            delta_t[t] = elapsed
            missed_revisits[t] = float(np.floor(elapsed / 5.0))
    return delta_t, orbital_phase, missed_revisits


def test_vectorized_orbital_dynamics_parity():
    rng = np.random.RandomState(42)
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler = ArcsinhScaler(data_dir / "scaler.json")
    ds = HistoryImputationDataset(data_dir, scaler=scaler, max_files=1, mask_dropout=False)

    for _ in range(50):
        L = int(rng.randint(5, 60))
        mask = rng.rand(L) > 0.8
        ref = _reference_orbital_dynamics(mask)
        vec1 = compute_orbital_dynamics(mask)
        vec2 = ds._compute_orbital_dynamics(mask)
        for r, v1, v2 in zip(ref, vec1, vec2):
            np.testing.assert_allclose(v1, r, atol=1e-6)
            np.testing.assert_allclose(v2, r, atol=1e-6)

    # Also all-False edge case
    empty_mask = np.zeros(12, dtype=bool)
    for r, v1 in zip(_reference_orbital_dynamics(empty_mask), compute_orbital_dynamics(empty_mask)):
        np.testing.assert_allclose(v1, r, atol=1e-6)
    print("[+] Vectorized orbital dynamics parity test PASSED.")


def test_padding_invariance_bigru_and_mae():
    torch.manual_seed(7)
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler = ArcsinhScaler(data_dir / "scaler.json")
    ds = HistoryImputationDataset(
        data_dir, scaler=scaler, max_files=1, mask_dropout=True, climate_dropout_p=0.0, seed=42
    )
    lens = [ds[i]["length"] for i in range(min(len(ds), 150))]
    short_item = ds[int(np.argmin(lens))]
    long_item = ds[int(np.argmax(lens))]
    assert short_item["length"] < long_item["length"]

    solo = collate_history_batch([short_item])
    mixed = collate_history_batch([short_item, long_item])

    for name, model in [
        ("BiGRU", BiGRUHistoryImputer(dropout=0.1).eval()),
        ("MAE", MAEHistoryImputer(dropout=0.1).eval()),
    ]:
        with torch.no_grad():
            p_solo = model(solo["input"], climate_id=solo["climate_id"], padding_mask=solo["padding_mask"])[
                0, : short_item["length"]
            ]
            p_mixed = model(mixed["input"], climate_id=mixed["climate_id"], padding_mask=mixed["padding_mask"])[
                0, : short_item["length"]
            ]
        diff = float((p_solo - p_mixed).abs().max().item())
        assert diff < 1e-5, f"{name} padding invariance failed: max diff={diff:.3e}"
    print("[+] BiGRU & MAE batch-padding invariance test PASSED.")


def test_loo_all_points_deterministic_no_leakage():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    scaler = ArcsinhScaler(data_dir / "scaler.json")
    first_file = sorted(data_dir.glob("*.npz"))[0]
    n_win = len(np.load(first_file, allow_pickle=True)["history"])

    ds = HistoryImputationDataset(
        data_dir,
        scaler=scaler,
        file_list=[first_file],
        mask_dropout=True,
        climate_dropout_p=0.0,
        loo_all_points=True,
    )
    assert len(ds) == 3 * n_win, f"Expected {3 * n_win} LOO points, got {len(ds)}"

    for idx in range(min(15, len(ds))):
        item = ds[idx]
        assert int(item["drop_rank"].item()) == idx % 3
        loss_m = item["loss_mask"].numpy()
        obs_m = item["all_obs_mask"].numpy()
        assert int(loss_m.sum()) == 1
        drop_t = int(np.where(loss_m)[0][0])
        obs_pos = np.where(obs_m)[0]
        assert drop_t == obs_pos[idx % 3]
        x = item["input"].numpy()
        np.testing.assert_allclose(x[drop_t, 18:21], 0.0, atol=1e-7)
        np.testing.assert_allclose(x[drop_t, 21:24], 0.0, atol=1e-7)
    print("[+] Deterministic loo_all_points no-leakage test PASSED.")


def test_frozen_checkpoint_integrity_and_compatibility():
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    scaler_path = cfg["paths"]["scaler_path"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for mtype, cfg_key, interp_name, interp_cls in [
            ("bigru", "imputer_model", "model", ModelInterpolator),
            ("mae", "mae_model", "mae", MAEInterpolator),
        ]:
            arch = extract_arch(mtype, cfg[cfg_key])
            model = build_reconstruction_model(mtype, arch)
            assert all(p.requires_grad for p in model.parameters())

            ckpt_path = tmp / f"{mtype}_frozen.pt"
            digest = save_frozen_checkpoint(
                ckpt_path,
                model=model,
                model_type=mtype,
                arch=arch,
                metadata={"epoch": 3, "val_mae_orig": 0.031},
                full_config=cfg,
            )
            # Saving mid-training must NOT freeze the live model
            assert all(p.requires_grad for p in model.parameters())

            loaded, payload = load_frozen_model(ckpt_path, device="cpu", verify=True)
            assert not loaded.training
            assert all(not p.requires_grad for p in loaded.parameters())
            assert payload["state_dict_sha256"] == digest

            # Compatibility with InterpolationFactory
            interp = InterpolationFactory.create(
                interp_name, checkpoint_path=ckpt_path, scaler_path=scaler_path, device="cpu"
            )
            assert isinstance(interp, interp_cls)
            assert not interp.model.training
            assert all(not p.requires_grad for p in interp.model.parameters())

            # Tamper detection
            tampered = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            first_key = next(iter(tampered["model_state_dict"]))
            tampered["model_state_dict"][first_key] = tampered["model_state_dict"][first_key] + 1.0
            tampered_path = tmp / f"{mtype}_tampered.pt"
            torch.save(tampered, tampered_path)

            raised = False
            try:
                load_frozen_model(tampered_path, device="cpu", verify=True)
            except ValueError:
                raised = True
            assert raised, f"Expected ValueError on tampered {mtype} checkpoint!"
    print("[+] Frozen checkpoint integrity, tamper detection & compatibility test PASSED.")


def test_vectorized_classical_loo_parity():
    data_dir = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky"
    first_file = sorted(data_dir.glob("*.npz"))[0]
    with np.load(first_file, allow_pickle=True) as d:
        histories = d["history"][:10]
        masks = d["history_mask"][:10]

    with tempfile.TemporaryDirectory() as tmpdir:
        sub_npz = Path(tmpdir) / "subset.npz"
        np.savez(
            sub_npz,
            history=np.array(histories, dtype=object),
            history_mask=np.array(masks, dtype=object),
        )
        for strat in ["linear", "spline", "forward_fill", "backward_fill", "nearest", "mean", "median"]:
            yt_v, yp_v, dr_v = evaluate_classical_baseline_loo([sub_npz], strategy=strat, clip_min=-1.0, clip_max=1.0)
            interp = InterpolationFactory.create(strat, clip_min=-1.0, clip_max=1.0)

            ref_true, ref_pred, ref_rank = [], [], []
            for h_raw, m_raw in zip(histories, masks):
                obs_idx = np.where(~m_raw[:, 18])[0]
                for r, drop_t in enumerate(obs_idx):
                    h_in = h_raw.astype(np.float32).copy()
                    m_in = m_raw.astype(bool).copy()
                    h_in[drop_t, 18:21] = np.nan
                    m_in[drop_t, 18:21] = True
                    filled = interp.interpolate(h_in, mask=m_in, columns=[18, 19, 20])
                    ref_pred.append(filled[drop_t, 18:21])
                    ref_true.append(h_raw[drop_t, 18:21])
                    ref_rank.append(r)

            np.testing.assert_allclose(yt_v, np.array(ref_true, dtype=np.float32), atol=1e-6)
            np.testing.assert_allclose(yp_v, np.array(ref_pred, dtype=np.float32), atol=1e-5)
            np.testing.assert_array_equal(dr_v, np.array(ref_rank, dtype=np.int64))
    print("[+] Vectorized classical LOO baseline parity test PASSED.")


def test_benchmark_smoke_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_dir = tmp / "results"
        frz_dir = tmp / "frozen"
        summary = run_benchmark(
            config_path=PROJECT_ROOT / "configs" / "config.yaml",
            epochs=1,
            seeds=[42],
            models=["bigru", "mae"],
            batch_size=128,
            eval_batch_size=256,
            max_train_tiles=2,
            max_val_tiles=1,
            output_dir=out_dir,
            frozen_dir=frz_dir,
            device="cpu",
        )
        assert (out_dir / "reconstruction_metrics.csv").exists()
        assert (out_dir / "training_history.csv").exists()
        assert (out_dir / "reconstruction_summary.json").exists()
        assert (out_dir / "leaderboard.md").exists()
        assert (frz_dir / "bigru_frozen.pt").exists()
        assert (frz_dir / "mae_frozen.pt").exists()
        assert (frz_dir / "best_imputer_frozen.pt").exists()
        assert summary["winner"]["best_model"] in ("bigru", "mae")
        m, _ = load_frozen_model(frz_dir / "best_imputer_frozen.pt", device="cpu", verify=True)
        assert not m.training and all(not p.requires_grad for p in m.parameters())
    print("[+] Benchmark smoke run test PASSED.")


if __name__ == "__main__":
    print("=" * 68)
    print(" RUNNING RECONSTRUCTION BENCHMARK UNIT & SMOKE TESTS")
    print("=" * 68)
    test_vectorized_orbital_dynamics_parity()
    test_padding_invariance_bigru_and_mae()
    test_loo_all_points_deterministic_no_leakage()
    test_frozen_checkpoint_integrity_and_compatibility()
    test_vectorized_classical_loo_parity()
    test_benchmark_smoke_run()
    print("\n[***] ALL RECONSTRUCTION BENCHMARK TESTS PASSED! [***]")
