import numpy as np
import pandas as pd
import pytest

from app.features.builder import build_features, feature_columns
from app.targets.ten_candle import DOWN, FLAT, UP, build_targets, direction_label, direction_threshold


@pytest.fixture
def feature_frame(synthetic_candles):
    candles = synthetic_candles(n=1500, freq="1min")
    return build_features(candles, "crypto", "1m"), candles


def test_feature_set_has_expected_groups(feature_frame):
    frame, _ = feature_frame
    cols = set(feature_columns(frame))
    expected = {
        "return_1", "return_20", "rsi_14", "macd", "macd_hist", "adx", "atr_pct",
        "bb_width", "stoch_k", "cci", "roc_5", "volume_zscore", "relative_volume",
        "realized_vol", "hl_volatility", "volatility_regime", "breakout", "breakdown",
        "trend_strength", "ema_alignment", "hour_of_day", "is_weekend",
        "htf_5m_ema_20", "htf_15m_rsi_14",
    }
    missing = expected - cols
    assert not missing, f"missing features: {sorted(missing)}"


def test_forex_session_features_present(synthetic_candles):
    candles = synthetic_candles(n=400, freq="1h", symbol="EURUSD", market="forex", price=1.1)
    frame = build_features(candles, "forex", "1h")
    cols = set(feature_columns(frame))
    assert {"session_london", "session_new_york", "session_return",
            "session_volatility", "session_range_position"} <= cols


def test_commodity_features_build(synthetic_candles):
    candles = synthetic_candles(
        n=400, freq="1min", symbol="XAUUSD", market="commodities", price=2300.0
    )
    frame = build_features(candles, "commodities", "1m")
    cols = set(feature_columns(frame))
    assert {
        "day_range_position",
        "day_range_so_far",
        "day_return",
        "trend_persistence",
        "mean_reversion_signal",
    } <= cols


def test_targets_shape_and_prefix(feature_frame):
    frame, candles = feature_frame
    targets = build_targets(candles)
    assert list(targets.columns[:6]) == [
        "target_return_h1",
        "target_direction_h1",
        "target_high_h1",
        "target_low_h1",
        "target_range_h1",
        "target_volatility_h1",
    ]
    assert all(c.startswith("target_") for c in targets.columns)
    assert "target_return_h10" in targets.columns
    # rows too close to the end cannot have a 10-candle target
    assert targets["target_return_h10"].isna().iloc[-1]
    assert targets["target_return_h1"].notna().iloc[-2]


def test_target_values_match_manual_computation(synthetic_candles):
    candles = synthetic_candles(n=400, freq="1min")
    targets = build_targets(candles)
    close = candles["close"].to_numpy()
    for h in (1, 2, 5, 10):
        manual = close[h:] / close[:-h] - 1.0
        col = targets[f"target_return_h{h}"].to_numpy()
        np.testing.assert_allclose(col[: len(manual)], manual, rtol=1e-12)


def test_future_high_low_windows(synthetic_candles):
    candles = synthetic_candles(n=300, freq="1min")
    targets = build_targets(candles)
    i, h = 10, 3
    expected_high = candles["high"].iloc[i + 1: i + h + 1].max()
    expected_low = candles["low"].iloc[i + 1: i + h + 1].min()
    assert targets["target_high_h3"].iloc[i] == pytest.approx(expected_high)
    assert targets["target_low_h3"].iloc[i] == pytest.approx(expected_low)
    assert np.isnan(targets["target_high_h3"].iloc[len(candles) - 2])


def test_direction_threshold_is_atr_scaled(synthetic_candles):
    candles = synthetic_candles(n=500, freq="1min")
    thr = direction_threshold(candles)
    assert (thr.iloc[30:] > 0).all()  # 0 during indicator warm-up, positive after
    hot = candles.copy()
    rng = np.random.default_rng(1)
    hot["high"] = candles["high"] + np.abs(rng.normal(0, 1.0, len(candles)))
    hot["low"] = candles["low"] - np.abs(rng.normal(0, 1.0, len(candles)))
    assert direction_threshold(hot).mean() > thr.mean()


def test_direction_labels_respect_threshold(synthetic_candles):
    candles = synthetic_candles(n=600, freq="1min")
    targets = build_targets(candles)
    thr = direction_threshold(candles)
    r = targets["target_return_h1"]
    d = targets["target_direction_h1"]
    up, down, flat = d == 1, d == -1, d == 0
    assert (r[up] > thr[up]).all()
    assert (r[down] < -thr[down]).all()
    assert (r[flat].abs() <= thr[flat] + 1e-12).all()


def test_no_target_column_can_be_selected_as_feature(feature_frame, synthetic_candles):
    frame, candles = feature_frame
    combined = pd.concat([frame, build_targets(candles)], axis=1)
    feats = feature_columns(combined)
    assert not any(c.startswith("target_") for c in feats)
    from app.features.builder import assert_no_target_leakage

    assert_no_target_leakage(combined)  # runtime guard does not raise


def test_direction_label_helpers():
    assert direction_label(0.5) == UP
    assert direction_label(-0.5) == DOWN
    assert direction_label(0.0) == FLAT
    assert direction_label(float("nan")) == FLAT
