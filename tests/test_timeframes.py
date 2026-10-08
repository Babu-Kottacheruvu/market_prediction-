import numpy as np
import pandas as pd
import pytest

from app.data.normalizer import normalize_ohlcv
from app.data.timeframes import resample_ohlcv, with_availability


def test_resample_1m_to_5m_matches_manual_aggregation(synthetic_candles):
    df = synthetic_candles(n=60, freq="1min")
    r = resample_ohlcv(df, "5m")
    assert len(r) == 12

    first = r.iloc[0]
    src = df.iloc[:5]
    assert first["open"] == pytest.approx(src["open"].iloc[0])
    assert first["high"] == pytest.approx(src["high"].max())
    assert first["low"] == pytest.approx(src["low"].min())
    assert first["close"] == pytest.approx(src["close"].iloc[-1])
    assert first["volume"] == pytest.approx(src["volume"].sum())
    assert first["timestamp"] == src["timestamp"].iloc[0]
    assert first["available_at"] == first["timestamp"] + pd.Timedelta(minutes=5)


def test_resample_never_creates_candles_for_empty_buckets(synthetic_candles):
    df = synthetic_candles(n=5, freq="1min")  # only 5 minutes of data
    r = resample_ohlcv(df, "1h")
    assert len(r) == 1  # a single bucket, not 24 fabricated ones


def test_resample_output_is_sorted_and_available_after_close(synthetic_candles):
    df = synthetic_candles(n=120, freq="1min")
    r = resample_ohlcv(df, "15m")
    assert r["timestamp"].is_monotonic_increasing
    assert (r["available_at"] > r["timestamp"]).all()
    # available_at equals close time of the last contributing 1m candle + 1 minute
    assert r["available_at"].iloc[0] == r["timestamp"].iloc[0] + pd.Timedelta(minutes=15)


def test_with_availability_adds_close_time(synthetic_candles):
    df = synthetic_candles(n=10, freq="1min")
    out = with_availability(df, "1m")
    assert (out["available_at"] - out["timestamp"] == pd.Timedelta(minutes=1)).all()


def test_derivation_from_smaller_tf_only(monkeypatch, tmp_storage, synthetic_candles):
    from app.data import storage

    df = synthetic_candles(n=300, freq="1min")
    storage.save_candles(df, "crypto", "BTCUSDT", "1m")

    # 1m -> 15m is legitimate aggregation
    derived = storage.derive_higher_timeframe("crypto", "BTCUSDT", "15m")
    assert derived is not None and len(derived) == 20

    # the reverse must never happen: no 1m data can be produced from 5m alone
    import shutil

    shutil.rmtree(tmp_storage / "raw" / "crypto" / "BTCUSDT")
    storage.save_candles(df.iloc[::5], "crypto", "BTCUSDT", "5m")  # only 5m exists
    assert storage.derive_higher_timeframe("crypto", "BTCUSDT", "1m", prefer_base=("5m", "15m")) is None
    import pytest as _pytest

    with _pytest.raises(Exception):
        storage.load_candles("crypto", "BTCUSDT", "1m")
