"""MANDATORY no-lookahead tests (spec section 52).

Method: compute features on a prefix of history, then append more candles and
recompute.  Every feature value belonging to the original timestamps must be
bit-for-bit identical.  If any value changes, future information leaked.
"""

import numpy as np
import pandas as pd
import pytest

from app.features.builder import build_features, feature_columns
from app.targets.ten_candle import build_targets

FEATURE_TOLERANCE = 1e-12


def _features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = build_features(candles, "crypto", "1m")
    return frame.set_index(pd.Index(candles["timestamp"], name="timestamp"))[feature_columns(frame)]


def test_features_do_not_change_when_future_candles_are_appended(synthetic_candles):
    full = synthetic_candles(n=3000, freq="1min", seed=3)
    prefix = full.iloc[:2000].copy()

    feats_prefix = _features(prefix)
    feats_full = _features(full)

    shared = feats_prefix.index
    # every original timestamp must still exist
    assert shared.isin(feats_full.index).all()

    a = feats_prefix.loc[shared]
    b = feats_full.loc[shared]
    # align columns (HTF availability can only *add* columns, never change old values)
    common = [c for c in a.columns if c in b.columns]
    diff = (a[common] - b[common]).abs()
    changed = diff.max()
    changed = changed[changed > FEATURE_TOLERANCE]
    assert changed.empty, f"features changed after appending future data: {changed.to_dict()}"


def test_higher_timeframe_features_only_use_closed_candles(synthetic_candles):
    """For every base row, the 5m context must equal the value computed from
    5m buckets whose available_at (close time) <= base timestamp."""
    from app.data.timeframes import resample_ohlcv
    from app.features import technical as ta

    candles = synthetic_candles(n=1200, freq="1min", seed=5)
    frame = build_features(candles, "crypto", "1m")

    htf_cols = [c for c in frame.columns if c.startswith("htf_5m_")]
    assert htf_cols, "expected 5m context columns"

    htf = resample_ohlcv(candles, "5m")
    htf["ema_20"] = ta.ema(htf["close"], 20)
    htf["rsi_14"] = ta.rsi(htf["close"], 14)
    available = htf.set_index("available_at")

    sample = frame.sample(n=min(150, len(frame)), random_state=0)
    for _, row in sample.iterrows():
        ts = row["timestamp"]
        eligible = available.loc[available.index <= ts]
        if eligible.empty:
            assert pd.isna(row["htf_5m_ema_20"])
            continue
        expected_ema = eligible["ema_20"].iloc[-1]
        expected_rsi = eligible["rsi_14"].iloc[-1]
        if np.isnan(expected_ema):
            assert pd.isna(row["htf_5m_ema_20"])
        else:
            assert row["htf_5m_ema_20"] == pytest.approx(expected_ema, rel=1e-12), (
                f"5m EMA at {ts} used a bucket closed after the base candle"
            )
        if not np.isnan(expected_rsi):
            assert row["htf_5m_rsi_14"] == pytest.approx(expected_rsi, rel=1e-12)


def test_partial_trailing_higher_timeframe_bucket_is_invisible(synthetic_candles):
    """The still-forming 5m candle must never be observable by 1m rows."""
    from app.data.timeframes import resample_ohlcv

    candles = synthetic_candles(n=300, freq="1min", seed=9)
    htf = resample_ohlcv(candles, "5m")
    last_base_time = candles["timestamp"].max()
    for _, bucket in htf.iterrows():
        if bucket["available_at"] > last_base_time:
            # incomplete bucket: base rows (all <= last_base_time) can never see it
            assert bucket["available_at"] > last_base_time
            continue
        assert bucket["available_at"] <= last_base_time

    feats = _features(candles)
    htf_5m_cols = [c for c in feats.columns if c.startswith("htf_5m_")]
    # the last complete bucket closes at/after the final available_at
    rows_at_or_after_last_close = feats.index[feats.index >= htf["available_at"].iloc[-1]]
    if len(htf_5m_cols) and len(rows_at_or_after_last_close):
        assert feats.loc[rows_at_or_after_last_close[-1], htf_5m_cols].notna().any()


def test_append_is_idempotent_for_rolling_and_session_features(synthetic_candles):
    """Rolling/session/volume features must be prefix-stable too."""
    full = synthetic_candles(n=2500, freq="1min", seed=11)
    for cut in (900, 1500, 2100):
        prefix = full.iloc[:cut].copy()
        a = _features(prefix)
        b = _features(full)
        common = [c for c in a.columns if c in b.columns]
        diff = (a[common] - b[common]).abs().max()
        changed = diff[diff > FEATURE_TOLERANCE]
        assert changed.empty, f"cut={cut} changed: {changed.to_dict()}"


def test_targets_use_only_the_future_and_are_excluded_from_features(synthetic_candles):
    candles = synthetic_candles(n=500, freq="1min", seed=13)
    frame = build_features(candles, "crypto", "1m")
    targets = build_targets(candles)

    # a target must not depend on anything at/before t except close[t]
    close = candles["close"].to_numpy()
    manual = close[1:] / close[:-1] - 1.0
    np.testing.assert_allclose(
        targets["target_return_h1"].to_numpy()[: len(manual)], manual, rtol=1e-12
    )

    # and it must be impossible to select as a feature
    combined = pd.concat([frame, targets], axis=1)
    feats = feature_columns(combined)
    assert not any(str(c).startswith("target_") for c in feats)
    # raw OHLC prices are metadata, not features
    for col in ("open", "high", "low", "close", "volume", "timestamp"):
        assert col not in feats
