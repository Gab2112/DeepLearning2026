"""
Unit and integration tests for src/metrics.py and src/models/baselines.py.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import ArcsinhScaler
from src.metrics import (
    PinballLoss,
    compute_naive_history_scale,
    coverage,
    crps_from_quantiles,
    evaluate_forecast,
    interval_width,
    inverse_transform_quantiles,
    mae,
    mase,
    metrics_to_dataframe,
    pinball_loss,
    quantile_crossing_rate,
    rmse,
    wmape,
)
from src.models.baselines import (
    PersistenceBaseline,
    ProbabilisticPersistenceBaseline,
    extract_clear_sky_history,
    extract_last_clear_sky,
)


def test_exact_point_and_probabilistic_metrics():
    # Shape (N=2, S=3, C=3)
    y_true = np.array(
        [
            [[0.4, 0.5, 0.3], [0.5, 0.6, 0.4], [0.6, 0.7, 0.5]],
            [[0.2, 0.3, 0.1], [0.3, 0.4, 0.2], [0.4, 0.5, 0.3]],
        ],
        dtype=np.float64,
    )

    # 1. Perfect prediction -> all errors 0, coverage 1.0, crossing 0.0
    perfect_q = np.broadcast_to(y_true[..., None], y_true.shape + (3,)).copy()
    scale = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    res_perf = evaluate_forecast(y_true, perfect_q, mase_scale=scale)
    ov = res_perf["overall"]
    assert np.isclose(ov["RMSE"], 0.0)
    assert np.isclose(ov["MAE"], 0.0)
    assert np.isclose(ov["WMAPE"], 0.0)
    assert np.isclose(ov["MASE"], 0.0)
    assert np.isclose(ov["Pinball"], 0.0)
    assert np.isclose(ov["CRPS"], 0.0)
    assert np.isclose(ov["Coverage"], 1.0)
    assert np.isclose(ov["IntervalWidth"], 0.0)
    assert np.isclose(ov["CrossingRate"], 0.0)

    # 2. Known deterministic error: y_hat = y_true - 0.1 -> MAE=0.1, RMSE=0.1, MASE=1.0
    shifted_q = perfect_q - 0.1
    assert np.isclose(rmse(y_true, shifted_q[..., 1]), 0.1)
    assert np.isclose(mae(y_true, shifted_q[..., 1]), 0.1)
    expected_wmape = (18 * 0.1) / np.sum(np.abs(y_true))
    assert np.isclose(wmape(y_true, shifted_q[..., 1]), expected_wmape)
    assert np.isclose(mase(y_true, shifted_q[..., 1], scale=0.1), 1.0)

    # Deterministic identity: CRPS == MAE for symmetric quantiles (0.1, 0.5, 0.9)
    crps_det = crps_from_quantiles(y_true, shifted_q, quantiles=(0.1, 0.5, 0.9))
    assert np.isclose(crps_det, 0.1)

    # Asymmetric pinball check: y - y_hat = +0.1 (under-prediction) -> loss = q * 0.1
    assert np.isclose(pinball_loss(y_true, shifted_q[..., 0], 0.1), 0.01)
    assert np.isclose(pinball_loss(y_true, shifted_q[..., 2], 0.9), 0.09)

    # Over-prediction: y_hat = y_true + 0.1 -> y - y_hat = -0.1 -> loss = (1 - q) * 0.1
    over_q = perfect_q + 0.1
    assert np.isclose(pinball_loss(y_true, over_q[..., 0], 0.1), 0.09)
    assert np.isclose(pinball_loss(y_true, over_q[..., 2], 0.9), 0.01)

    # Coverage, IntervalWidth, CrossingRate, and valid_mask with NaNs
    yt_nan = y_true.copy()
    yt_nan[0, 0, 0] = np.nan
    pred_q = perfect_q.copy()
    pred_q[..., 0] -= 0.05
    pred_q[..., 2] += 0.05
    # Make one element violate coverage & crossing
    pred_q[1, 2, 2, 0] = yt_nan[1, 2, 2] + 0.2
    pred_q[1, 2, 2, 2] = yt_nan[1, 2, 2] + 0.1  # q10 > q90

    valid_mask = np.isfinite(yt_nan)
    cov_val = coverage(yt_nan, pred_q[..., 0], pred_q[..., 2], valid_mask=valid_mask)
    assert np.isclose(cov_val, 16.0 / 17.0)
    cross_val = quantile_crossing_rate(pred_q, valid_mask=valid_mask)
    assert np.isclose(cross_val, 1.0 / 17.0)
    width_val = interval_width(perfect_q[..., 0] - 0.05, perfect_q[..., 2] + 0.05, valid_mask=valid_mask)
    assert np.isclose(width_val, 0.10)

    df = metrics_to_dataframe({"id": res_perf}, model_name="perfect")
    assert len(df) == 4  # 3 indices + OVERALL
    print("[+] Exact point and probabilistic metrics test PASSED.")


def test_inverse_transform_and_pinball_loss_torch():
    scaler_path = PROJECT_ROOT / "DATA" / "train_multi_output_clear_sky" / "scaler.json"
    if not scaler_path.exists():
        print("[!] scaler.json not found, skipping inverse_transform_quantiles test.")
    else:
        scaler = ArcsinhScaler(scaler_path)
        orig = np.random.RandomState(42).uniform(-0.2, 0.85, size=(5, 3, 3, 3)).astype(np.float32)
        norm = np.moveaxis(
            scaler.transform_targets(np.moveaxis(orig, -1, -2)),
            -2,
            -1,
        )
        rec_np = inverse_transform_quantiles(norm, scaler)
        rec_pt = inverse_transform_quantiles(torch.from_numpy(norm), scaler).numpy()
        np.testing.assert_allclose(rec_np, orig, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(rec_pt, orig, rtol=1e-5, atol=1e-5)

    # PyTorch PinballLoss consistency + gradient flow
    rng = np.random.RandomState(7)
    y_np = rng.randn(4, 3, 3).astype(np.float32)
    p_np = rng.randn(4, 3, 3, 3).astype(np.float32)
    mask_np = rng.rand(4, 3, 3) > 0.2

    loss_fn = PinballLoss(quantiles=(0.1, 0.5, 0.9))
    p_t = torch.tensor(p_np, requires_grad=True)
    y_t = torch.tensor(y_np)
    m_t = torch.tensor(mask_np)
    loss_t = loss_fn(p_t, y_t, valid_mask=m_t)
    loss_t.backward()
    assert p_t.grad is not None and torch.isfinite(p_t.grad).all()

    expected_pinball = np.mean(
        [pinball_loss(y_np, p_np[..., i], q, valid_mask=mask_np) for i, q in enumerate((0.1, 0.5, 0.9))]
    )
    assert np.isclose(loss_t.item(), expected_pinball, rtol=1e-5, atol=1e-6)
    print("[+] Inverse transform & PyTorch PinballLoss test PASSED.")


def test_persistence_baselines():
    rng = np.random.RandomState(123)
    # Create synthetic variable-length histories with 3 clear-sky observations each
    histories = []
    history_masks = []
    for _ in range(400):
        L = rng.randint(12, 22)
        h = np.full((L, 21), np.nan, dtype=np.float32)
        h[:, :18] = 1.0  # weather
        m = np.zeros((L, 21), dtype=bool)
        m[:, 18:21] = True
        obs_days = sorted(rng.choice(L, size=3, replace=False))
        base = rng.uniform(0.2, 0.7, size=3).astype(np.float32)
        for step_i, d_idx in enumerate(obs_days):
            h[d_idx, 18:21] = np.clip(base + step_i * 0.02, -1.0, 1.0)
            m[d_idx, 18:21] = False
        histories.append(h)
        history_masks.append(m)

    hist_obs = extract_clear_sky_history(histories, history_masks)
    last_obs = extract_last_clear_sky(histories, history_masks)
    assert hist_obs.shape == (400, 3, 3)
    assert last_obs.shape == (400, 3)
    np.testing.assert_allclose(last_obs, hist_obs[:, -1, :])

    scale = compute_naive_history_scale(hist_obs)
    assert scale.shape == (3,) and np.all(scale > 0.0)

    # Synthetic future targets = last_obs + small Gaussian noise
    noise = rng.normal(0.0, 0.05, size=(400, 3, 3)).astype(np.float32)
    targets = np.clip(last_obs[:, None, :] + noise, -1.0, 1.0)

    # Split train/test
    det_model = PersistenceBaseline().fit()
    det_preds = det_model.predict(last_obs[300:])
    assert det_preds.shape == (100, 3, 3, 3)
    np.testing.assert_allclose(det_preds[..., 0], det_preds[..., 1])
    np.testing.assert_allclose(det_preds[..., 1], det_preds[..., 2])

    prob_model = ProbabilisticPersistenceBaseline().fit(last_obs[:300], targets[:300])
    prob_preds = prob_model.predict(last_obs[300:])
    assert prob_preds.shape == (100, 3, 3, 3)
    assert np.all(prob_preds >= -1.0) and np.all(prob_preds <= 1.0)
    assert np.all(np.diff(prob_preds, axis=-1) >= -1e-7)

    eval_prob = evaluate_forecast(targets[300:], prob_preds, mase_scale=scale)
    assert 0.65 <= eval_prob["overall"]["Coverage"] <= 0.95
    assert eval_prob["overall"]["CrossingRate"] == 0.0
    print(f"[+] Persistence baselines test PASSED (synthetic coverage={eval_prob['overall']['Coverage']:.3f}).")


def test_import_order_no_circular_imports():
    """Each module must be importable as the FIRST import in a fresh interpreter."""
    import subprocess

    statements = [
        "import src.models",
        "from src.models.baselines import PersistenceBaseline",
        "from src.models.history_imputer import BiGRUHistoryImputer",
        "from src.metrics import evaluate_forecast",
        "import src.data",
        "from src.data.bigru_imputer import BiGRUHistoryImputer",
    ]
    for stmt in statements:
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Circular import for '{stmt}':\n{proc.stderr}"
    print("[+] Import order (no circular imports) test PASSED.")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING METRICS & BASELINES UNIT TESTS")
    print("=" * 60)
    test_exact_point_and_probabilistic_metrics()
    test_inverse_transform_and_pinball_loss_torch()
    test_persistence_baselines()
    test_import_order_no_circular_imports()
    print("\n[***] ALL METRICS & BASELINES TESTS PASSED! [***]")
