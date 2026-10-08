"""Tests for the drift-monitoring module (section 38)."""

import numpy as np
import pandas as pd

from app.monitoring.drift import (
    build_monitoring_report,
    categorical_psi,
    feature_drift,
    psi,
    target_drift,
)


def test_psi_same_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 4000)
    score = psi(base, base.copy())
    assert score == 0.0 or abs(score) < 1e-9


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 4000)
    act = rng.normal(1.5, 1.0, 4000)
    score = psi(ref, act)
    assert score > 0.20  # a 1.5-sigma shift must trip the default threshold


def test_psi_too_few_samples_is_nan():
    assert np.isnan(psi([1.0, 2.0], [1.0, 2.0]))


def test_categorical_psi_on_direction():
    opposite = [2] * 80 + [1] * 10 + [0] * 10
    base = [0] * 80 + [1] * 10 + [2] * 10
    assert categorical_psi(base, base) < 1e-9
    assert categorical_psi(base, opposite) > 0.20


def test_feature_drift_flags_only_shifted_features():
    rng = np.random.default_rng(2)
    n = 3000
    reference = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, n),
            "shifting": rng.normal(0, 1, n),
        }
    )
    recent = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, n),
            "shifting": rng.normal(2.0, 1, n),  # regime changed
        }
    )
    out = feature_drift(reference, recent, ["stable", "shifting"], threshold=0.20)
    assert out["features_checked"] == 2
    assert out["n_drifted"] == 1
    assert out["drifted_features"] == ["shifting"]
    assert out["max_psi"] == out["psi_by_feature"]["shifting"]


def test_target_drift_reports_shares():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"target_direction_h1": rng.integers(0, 3, 2000)})
    recent = pd.DataFrame({"target_direction_h1": rng.integers(0, 3, 500)})
    out = target_drift(reference, recent)
    assert out["available"] is True
    assert set(out["recent_share"].keys()) <= {0, 1, 2}


def test_monitoring_report_insufficient_recent_data():
    reference = pd.DataFrame({"f": np.zeros(500)})
    recent = pd.DataFrame({"f": np.zeros(10)})
    report = build_monitoring_report(
        "crypto", "BTCUSDT", "1m", reference, recent, ["f"]
    )
    assert report["status"] == "insufficient_recent_data"
    assert "retrain_recommended" not in report


def test_monitoring_report_healthy_and_drifted():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame({"f": rng.normal(0, 1, 2000)})
    healthy = build_monitoring_report(
        "crypto", "BTCUSDT", "1m", reference,
        pd.DataFrame({"f": rng.normal(0, 1, 800)}), ["f"],
    )
    assert healthy["status"] == "healthy"
    assert healthy["retrain_recommended"] is False

    drifted = build_monitoring_report(
        "crypto", "BTCUSDT", "1m", reference,
        pd.DataFrame({"f": rng.normal(3.0, 1, 800)}), ["f"],
    )
    assert drifted["status"] == "drift_detected"
    assert drifted["retrain_recommended"] is True
    assert drifted["feature_drift"]["drifted_features"] == ["f"]


def test_monitor_cli_smoke(trained_once, capsys):
    """CLI runs against the session-trained sandbox model and prints JSON."""
    from scripts.monitor import main

    code = main(["--market", "crypto", "--symbol", "BTCUSDT", "--timeframe", "1m"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert '"status"' in out
