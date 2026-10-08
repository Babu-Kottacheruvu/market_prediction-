"""End-to-end pipeline: train -> predict -> backtest.

All data is synthetic test data; no performance claim about real markets.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.core.exceptions import ModelNotFoundError
from app.models.router import ModelRouter
from app.prediction.predictor import predict
from app.training.walk_forward import WalkForwardConfig


def test_training_report_structure(trained_system):
    report = trained_system.to_dict()
    assert report["market"] == "crypto"
    assert report["symbol"] == "BTCUSDT"
    assert report["timeframe"] == "1m"
    assert report["n_rows"] > 1000
    assert report["n_features"] > 50
    assert len(report["folds"]) >= 2
    agg = report["out_of_sample_metrics"]["aggregate"]
    assert agg["n_oos_rows"] >= 200
    assert "mean_accuracy" in agg and agg["mean_accuracy"] is not None
    assert "mean_brier" in agg
    assert "mean_accuracy_lift_vs_baseline" in agg
    assert "mean_mae_improvement_vs_baseline" in agg
    assert len(report.get("selected_estimators_by_horizon", {})) > 0
    per_h = report["out_of_sample_metrics"]["per_horizon"]
    assert len(per_h) == 10
    for h in range(1, 11):
        assert f"h{h}" in per_h
        assert "direction" in per_h[f"h{h}"] and "return" in per_h[f"h{h}"]
    assert "passed" in report["acceptance"]
    # model files exist for every horizon
    from app.models.registry import package_dir

    pkg = package_dir("crypto", "BTCUSDT", "1m")
    for h in range(1, 11):
        assert (pkg / f"horizon_{h}.joblib").exists()
    assert (pkg / "metadata.json").exists()


def test_prediction_output_contract(trained_system):
    result = predict("crypto", "BTCUSDT", "1m", live_data=False)
    assert result["symbol"] == "BTCUSDT"
    assert result["timeframe"] == "1m"
    assert result["current_price"] > 0
    assert result["overall_bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert result["regime"] in {
        "TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY",
    }
    assert result["signal"] in {"BUY", "SELL", "WAIT"}
    assert result["data_source"] == "stored"
    assert result["data_timestamp"]
    assert 0.0 <= result["confidence"] <= 1.0

    fc = result["forecast"]
    assert len(fc) == 10
    for i, row in enumerate(fc, start=1):
        assert row["horizon"] == i
        expected_time = pd.Timestamp(result["data_timestamp"]) + pd.Timedelta(minutes=i)
        assert pd.Timestamp(row["candle_time"]) == expected_time
        assert row["direction"] in {"UP", "DOWN", "FLAT"}
        total = row["probability_up"] + row["probability_flat"] + row["probability_down"]
        assert total == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["expected_low"] < row["expected_high"]
        assert row["expected_price"] == pytest.approx(
            result["current_price"] * (1.0 + row["expected_return"]), abs=1e-4
        )
        assert row["expected_volatility"] > 0
    # JSON-serializable
    json.dumps(result, default=str)
    # risk plan is advisory only
    assert result["risk"]["note"].endswith("does not execute orders") or not result["risk"]["actionable"]


def test_prediction_fetches_live_mt5_data(trained_system, monkeypatch):
    from app.data import storage
    from app.data.base import DataProviderFactory
    from app.data.mt5_provider import MT5Provider

    provider = DataProviderFactory.get("crypto")
    assert isinstance(provider, MT5Provider)
    calls = []
    monkeypatch.setattr(
        provider,
        "get_live_data",
        lambda symbol, timeframe: (
            calls.append((symbol, timeframe))
            or storage.load_candles("crypto", symbol, timeframe)
        ),
    )

    result = predict("crypto", "BTCUSDT", "1m")

    assert calls == [("BTCUSDT", "1m")]
    assert result["data_source"] == "mt5"


def test_backtest_includes_costs_and_metrics(trained_system):
    engine = BacktestEngine("crypto", "BTCUSDT", "1m", BacktestConfig(horizon=1, min_probability=0.30))
    result = engine.run()
    assert result["n_oos_rows"] >= 200
    assert result["costs"]["round_trip"] > 0
    tm = result["trading_metrics"]
    ntm = result["trading_metrics_without_costs"]
    assert "accuracy" in result["prediction_metrics"]["classification"]
    assert "mae" in result["prediction_metrics"]["regression"]
    if tm["n_trades"] > 0:
        # costs must reduce (or equal) expectancy versus the no-cost counterfactual
        assert tm["expectancy"] <= ntm["expectancy"] + 1e-12
        for key in ("win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct",
                    "expectancy", "total_return"):
            assert key in tm
    assert "profitable_after_costs" in result


def test_backtest_without_predictions_is_clear_error(tmp_storage, synthetic_candles):
    from app.data import storage

    storage.save_candles(synthetic_candles(n=2000), "crypto", "BTCUSDT", "1m")
    with pytest.raises(Exception) as exc:
        BacktestEngine("crypto", "BTCUSDT", "1m").run()
    assert "scripts.train" in str(exc.value)


def test_router_refuses_untrained_model(tmp_storage, synthetic_candles):
    with pytest.raises(ModelNotFoundError) as exc:
        ModelRouter().get_model("crypto", "ETHUSDT", "5m")
    assert "scripts.train" in str(exc.value)


def test_router_never_returns_wrong_timeframe(trained_system):
    router = ModelRouter()
    with pytest.raises(ModelNotFoundError):
        router.get_model("crypto", "BTCUSDT", "15m")  # only 1m was trained

    package = router.get_model("crypto", "BTCUSDT", "1m")
    assert package.timeframe == "1m"
    assert package.metadata.symbol == "BTCUSDT"
    assert len(package.models) == 10
    status = router.model_status("crypto", "BTCUSDT", "1m")
    assert status["trained"] is True


def test_missing_data_error_is_actionable(tmp_storage):
    from app.core.exceptions import PlatformError

    with pytest.raises(PlatformError) as exc:
        predict("crypto", "SOLUSDT", "1m")
    assert "scripts.train" in str(exc.value) or "download_data" in str(exc.value)


def test_walk_forward_config_overrides_are_honored(tmp_storage, synthetic_candles, fast_xgb):
    from app.data import storage
    from app.training.trainer import train

    candles = synthetic_candles(n=7 * 1440, freq="1min", seed=77)
    storage.save_candles(candles, "crypto", "BTCUSDT", "1m")
    wf = WalkForwardConfig(
        train_days=3, validation_days=1, test_days=1, step_days=1,
        min_train_days=1, min_test_days=1,
    )
    report = train("crypto", "BTCUSDT", "1m", walk_forward=wf, activate=False, save_oos=False)
    # 7 days of data -> test windows on days 4 and 5 are both reachable
    assert len(report.folds) >= 2
    for fold in report.folds:
        assert fold["train_rows"] > fold["test_rows"]
