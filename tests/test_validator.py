import numpy as np
import pandas as pd

from app.data.normalizer import normalize_ohlcv
from app.data.validator import DataQualityValidator


def _frame(n=200, freq="1min", market="crypto"):
    rng = np.random.default_rng(7)
    ts = pd.date_range("2025-06-01", periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": rng.integers(1, 100, n).astype(float),
            }
        ),
        symbol="BTCUSDT",
        market=market,
    )


def test_clean_data_passes():
    report = DataQualityValidator().validate(_frame(), symbol="BTCUSDT", timeframe="1m")
    assert report.passed, report.to_dict()


def test_duplicate_timestamps_are_fatal():
    df = _frame()
    dup = pd.concat([df, df.iloc[[5]]]).sort_values("timestamp")
    report = DataQualityValidator().validate(dup, symbol="BTCUSDT", timeframe="1m")
    assert not report.passed
    assert any(i.code == "duplicate_timestamps" for i in report.fatal)


def test_high_below_low_is_fatal():
    df = _frame()
    df.loc[3, "high"] = df.loc[3, "low"] - 1.0
    report = DataQualityValidator().validate(df, symbol="BTCUSDT", timeframe="1m")
    assert not report.passed
    assert any(i.code == "high_below_low" for i in report.fatal)


def test_close_outside_high_low_is_fatal():
    df = _frame()
    df.loc[2, "close"] = df.loc[2, "high"] + 5.0
    report = DataQualityValidator().validate(df, symbol="BTCUSDT", timeframe="1m")
    assert not report.passed
    assert any(i.code in {"close_above_high", "close_below_low"} for i in report.fatal)


def test_negative_volume_is_fatal():
    df = _frame()
    df.loc[10, "volume"] = -5.0
    report = DataQualityValidator().validate(df, symbol="BTCUSDT", timeframe="1m")
    assert not report.passed
    assert any(i.code == "negative_volume" for i in report.fatal)


def test_gaps_are_reported_as_warnings():
    df = _frame(n=200)
    df = df.drop(df.index[50:120])  # remove 70 candles
    report = DataQualityValidator().validate(df, symbol="BTCUSDT", timeframe="1m")
    assert any(i.code == "missing_candles" for i in report.issues)
    assert report.passed  # gaps warn, they do not silently corrupt


def test_weekend_gaps_ignored_for_forex():
    # Friday close -> Sunday open weekend gap should not be flagged for forex
    ts = pd.date_range("2025-06-06 00:00", periods=48, freq="h", tz="UTC")  # Fri
    more = pd.date_range("2025-06-08 22:00", periods=24, freq="h", tz="UTC")  # Sun 22:00+
    ts = pd.DatetimeIndex(ts.union(more))
    n = len(ts)
    close = np.full(n, 1.1)
    frame = normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.001,
                "low": close - 0.001,
                "close": close,
                "volume": np.ones(n),
            }
        ),
        symbol="EURUSD",
        market="forex",
    )
    report = DataQualityValidator().validate(frame, symbol="EURUSD", timeframe="1h")
    assert not any(i.code == "missing_candles" for i in report.issues), report.to_dict()
