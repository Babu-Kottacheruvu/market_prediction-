"""Shared fixtures.

IMPORTANT: the candle data below is *synthetic* and exists ONLY for unit
testing.  It is never used to claim predictive performance - real training
requires real historical data (see scripts/download_data.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.normalizer import normalize_ohlcv


@pytest.fixture
def synthetic_candles():
    """Generate a normalized synthetic OHLCV frame (unit tests only)."""

    def _make(
        n: int = 5000,
        freq: str = "1min",
        start: str = "2025-06-01",
        seed: int = 42,
        symbol: str = "BTCUSDT",
        market: str = "crypto",
        price: float = 60000.0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
        rets = rng.normal(0.0, 0.0006, n)
        close = price * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0.0, 0.0004, n)) * close
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        high = body_hi + wick
        low = body_lo - wick
        vol = rng.integers(100, 5000, n).astype(float)
        frame = pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )
        return normalize_ohlcv(frame, symbol=symbol, market=market)

    return _make


@pytest.fixture
def tmp_storage(monkeypatch, tmp_path):
    """Redirect all data/model outputs into an isolated temp directory.

    ``app.data.storage``, ``app.models.registry``, the trainer and the
    backtester all resolve their paths from ``app.core.config`` at call time,
    so patching the module attributes is enough.
    """
    import app.core.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "processed").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    yield tmp_path


@pytest.fixture
def fast_xgb(monkeypatch):
    """Shrink XGBoost hyperparameters so tests run quickly."""
    import app.models.xgboost_model as xgb_mod

    clf = {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.3,
        "objective": "multi:softprob",
        "n_jobs": 1,
        "tree_method": "hist",
        "random_state": 0,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }
    reg = {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.3,
        "objective": "reg:squarederror",
        "n_jobs": 1,
        "tree_method": "hist",
        "random_state": 0,
        "verbosity": 0,
    }
    monkeypatch.setattr(xgb_mod, "_clf_params", lambda: dict(clf))
    monkeypatch.setattr(xgb_mod, "_reg_params", lambda: dict(reg))


@pytest.fixture
def small_walk_forward():
    from app.training.walk_forward import WalkForwardConfig

    return WalkForwardConfig(
        train_days=2,
        validation_days=1,
        test_days=1,
        step_days=2,
        min_train_days=1,
        min_test_days=1,
    )


def _make_synthetic(n=8 * 1440, seed=123):
    """Module-level synthetic generator so session fixtures can use it too."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-06-01", periods=n, freq="1min", tz="UTC")
    rets = rng.normal(0.0, 0.0006, n)
    close = 60000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0.0, 0.0004, n)) * close
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": rng.integers(100, 5000, n).astype(float),
        }
    )
    from app.data.normalizer import normalize_ohlcv

    return normalize_ohlcv(frame, symbol="BTCUSDT", market="crypto")


@pytest.fixture(scope="session")
def trained_once():
    """Train one small model in a session-wide sandbox (runs a single time).

    Data is synthetic test data; its metrics carry no claim about real markets.
    """
    import shutil
    import tempfile
    from pathlib import Path

    import pytest

    import app.core.config as cfg
    import app.models.xgboost_model as xgb_mod

    tmp = Path(tempfile.mkdtemp(prefix="gmp_sandbox_"))
    mp = pytest.MonkeyPatch()
    mp.setattr(cfg, "DATA_DIR", tmp)
    mp.setattr(cfg, "RAW_DATA_DIR", tmp / "raw")
    mp.setattr(cfg, "PROCESSED_DATA_DIR", tmp / "processed")
    mp.setattr(cfg, "MODELS_DIR", tmp / "models")
    (tmp / "raw").mkdir(exist_ok=True)
    (tmp / "processed").mkdir(exist_ok=True)
    (tmp / "models").mkdir(exist_ok=True)

    small_clf = {
        "n_estimators": 40, "max_depth": 3, "learning_rate": 0.3,
        "objective": "multi:softprob", "n_jobs": 1, "tree_method": "hist",
        "random_state": 0, "eval_metric": "mlogloss", "verbosity": 0,
    }
    small_reg = {
        "n_estimators": 40, "max_depth": 3, "learning_rate": 0.3,
        "objective": "reg:squarederror", "n_jobs": 1, "tree_method": "hist",
        "random_state": 0, "verbosity": 0,
    }
    mp.setattr(xgb_mod, "_clf_params", lambda: dict(small_clf))
    mp.setattr(xgb_mod, "_reg_params", lambda: dict(small_reg))

    from app.data import storage
    from app.training.trainer import train
    from app.training.walk_forward import WalkForwardConfig

    storage.save_candles(_make_synthetic(), "crypto", "BTCUSDT", "1m")
    report = train(
        "crypto",
        "BTCUSDT",
        "1m",
        walk_forward=WalkForwardConfig(
            train_days=2, validation_days=1, test_days=1, step_days=2,
            min_train_days=1, min_test_days=1,
        ),
        activate=True,
    )
    yield report
    mp.undo()
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def trained_system(trained_once):
    """Function-scoped handle on the session-trained model package."""
    return trained_once
