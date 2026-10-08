"""Tests for the optional probability-calibration layer (section 23)."""

import numpy as np
import pytest

from app.models.calibration import ProbabilityCalibrator, evaluate_calibration
from app.models.registry import ModelRegistry
from app.models.router import ModelRouter
from app.core.config import get_settings


def test_calibrator_shape_and_normalization():
    rng = np.random.default_rng(0)
    proba = rng.random((200, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 200)

    cal = ProbabilityCalibrator(method="platt").fit(proba, y)
    out = cal.transform(proba)
    assert out.shape == proba.shape
    assert np.all(out > 0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_calibrator_isotonic_roundtrip():
    rng = np.random.default_rng(1)
    proba = rng.random((150, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 150)

    cal = ProbabilityCalibrator(method="isotonic")
    out = cal.fit_transform(proba, y)
    assert out.shape == proba.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_calibrator_handles_single_class_column():
    # column c never the true class -> calibrator must not crash
    proba = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.9, 0.05, 0.05]])
    y = np.array([0, 0, 0])
    cal = ProbabilityCalibrator(method="isotonic").fit(proba, y)
    out = cal.transform(proba)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_calibration_reports_both_sides():
    rng = np.random.default_rng(2)
    proba = rng.random((120, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 120)
    metrics = evaluate_calibration(proba, y, proba * 0.9)
    for key in ("brier_score", "log_loss", "brier_score_calibrated",
                "log_loss_calibrated", "brier_improvement"):
        assert key in metrics


def test_training_writes_calibrators_when_enabled(tmp_storage, synthetic_candles, fast_xgb):
    from app.data import storage
    from app.training.trainer import train
    from app.training.walk_forward import WalkForwardConfig

    candles = synthetic_candles(n=6 * 1440, freq="1min", seed=9)
    storage.save_candles(candles, "crypto", "BTCUSDT", "1m")

    from dataclasses import replace as _replace

    settings = _replace(
        get_settings(),
        models={
            **get_settings().models,
            "calibration": {"enabled": True, "method": "platt"},
        },
    )

    report = train(
        "crypto", "BTCUSDT", "1m",
        settings=settings, activate=False, save_oos=False,
        walk_forward=WalkForwardConfig(
            train_days=2, validation_days=1, test_days=1, step_days=2,
            min_train_days=1, min_test_days=1,
        ),
    )
    assert "calibration" in report.out_of_sample_metrics
    assert f"h1" in report.out_of_sample_metrics["calibration"]

    registry = ModelRegistry()
    calibrators = registry.load_calibrators("crypto", "BTCUSDT", "1m")
    assert len(calibrators) == 10

    package = ModelRouter().get_model("crypto", "BTCUSDT", "1m")
    assert package.calibrators is not None
    assert len(package.calibrators) == 10
