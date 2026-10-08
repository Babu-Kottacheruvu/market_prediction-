import numpy as np
import pandas as pd
import pytest

from app.signals.engine import BUY, SELL, WAIT, generate_signal, round_trip_cost
from app.signals.regime import (
    HIGH_VOLATILITY,
    RANGING,
    TRENDING_UP,
    classify_regime,
)
from app.signals.risk import build_risk_plan


def _forecast(p_up=0.7, p_down=0.15, expected_return=0.002, confidence=0.7):
    p_flat = max(0.0, 1.0 - p_up - p_down)
    return [
        {
            "horizon": h,
            "direction": "UP" if p_up >= p_down else "DOWN",
            "probability": max(p_up, p_down),
            "probability_up": p_up,
            "probability_flat": p_flat,
            "probability_down": p_down,
            "expected_return": expected_return,
            "expected_high": 0.0,
            "expected_low": 0.0,
            "expected_volatility": 0.001,
            "confidence": confidence,
        }
        for h in range(1, 11)
    ]


def test_signal_is_buy_when_all_thresholds_met():
    sig = generate_signal(
        _forecast(), regime=TRENDING_UP, market="crypto", timeframe="1m",
        overall={"bias": "BULLISH"},
    )
    assert sig["signal"] == BUY
    assert sig["reasons"] == ["all thresholds met"]


def test_signal_waits_when_probability_below_threshold():
    sig = generate_signal(
        _forecast(p_up=0.55), regime=TRENDING_UP, market="crypto", timeframe="1m", overall={}
    )
    assert sig["signal"] == WAIT
    assert any("probability" in r for r in sig["reasons"])


def test_signal_requires_edge_over_transaction_costs():
    # tiny expected return vs real costs -> WAIT even with high probability
    sig = generate_signal(
        _forecast(expected_return=0.00001), regime=TRENDING_UP, market="crypto",
        timeframe="1m", overall={},
    )
    assert sig["signal"] == WAIT
    assert any("edge" in r or "cost" in r for r in sig["reasons"])


def test_signal_1m_needs_stricter_thresholds():
    # probability 0.67 passes the default 0.65 but fails the 1m override of 0.70
    sig = generate_signal(
        _forecast(p_up=0.67, expected_return=0.01), regime=TRENDING_UP,
        market="crypto", timeframe="1m", overall={},
    )
    assert sig["signal"] == WAIT

    sig_5m = generate_signal(
        _forecast(p_up=0.67, expected_return=0.01), regime=TRENDING_UP,
        market="crypto", timeframe="5m", overall={},
    )
    assert sig_5m["signal"] == BUY


def test_signal_can_be_sell():
    fc = _forecast(p_up=0.1, p_down=0.8, expected_return=-0.003, confidence=0.8)
    sig = generate_signal(fc, regime="TRENDING_DOWN", market="crypto", timeframe="15m", overall={})
    assert sig["signal"] == SELL


def test_scalp_entry_requires_validated_model_fresh_live_data_and_alignment():
    now = pd.Timestamp("2026-10-08T11:01:00Z").to_pydatetime()
    signal = generate_signal(
        _forecast(),
        regime=TRENDING_UP,
        market="crypto",
        timeframe="1m",
        overall={"bias": "BULLISH"},
        model_status="active",
        data_source="binance",
        data_timestamp="2026-10-08T11:00:00Z",
        now=now,
    )
    assert signal["signal"] == BUY
    assert signal["scalp"] is True
    assert signal["directional_agreement"] == 1.0

    candidate = generate_signal(
        _forecast(),
        regime=TRENDING_UP,
        market="crypto",
        timeframe="1m",
        overall={},
        model_status="candidate",
        data_source="binance",
        data_timestamp="2026-10-08T11:00:00Z",
        now=now,
    )
    assert candidate["signal"] == WAIT
    assert any("not validated" in reason for reason in candidate["reasons"])

    stored = generate_signal(
        _forecast(),
        regime=TRENDING_UP,
        market="crypto",
        timeframe="1m",
        overall={},
        model_status="active",
        data_source="stored",
        data_timestamp="2026-10-08T11:00:00Z",
        now=now,
    )
    assert stored["signal"] == WAIT
    assert any("live market data" in reason for reason in stored["reasons"])

    stale = generate_signal(
        _forecast(),
        regime=TRENDING_UP,
        market="crypto",
        timeframe="1m",
        overall={},
        model_status="active",
        data_source="binance",
        data_timestamp="2026-10-08T10:55:00Z",
        now=now,
    )
    assert stale["signal"] == WAIT
    assert any("stale" in reason for reason in stale["reasons"])


def test_scalp_risk_plan_uses_tighter_timeframe_levels():
    plan = build_risk_plan(
        "BUY", entry_price=100.0, atr=2.0, equity=10_000.0, timeframe="1m"
    )
    assert plan["stop_loss"] == pytest.approx(98.4)
    assert plan["take_profit"] == pytest.approx(102.8)
    assert plan["risk_reward"] == pytest.approx(1.75)
    assert plan["risk_amount"] == pytest.approx(50.0)


def test_round_trip_cost():
    assert round_trip_cost({"spread": 0.0002, "commission": 0.0004, "slippage": 0.0001}) == pytest.approx(0.0007)


def test_risk_plan_long():
    plan = build_risk_plan("BUY", entry_price=100.0, atr=2.0, equity=10_000.0)
    assert plan["stop_loss"] == pytest.approx(100.0 - 3.0)     # 1.5 * ATR
    assert plan["take_profit"] == pytest.approx(100.0 + 6.0)    # 3.0 * ATR
    assert plan["risk_reward"] == pytest.approx(2.0)
    assert plan["actionable"] is True
    # risk 1% of 10k = 100 -> 100/3 units, but capped at 25% of equity (25 units)
    assert plan["position_size_units"] == pytest.approx(25.0, rel=1e-6)
    assert plan["risk_amount"] == pytest.approx(100.0)


def test_risk_plan_short():
    plan = build_risk_plan("SELL", entry_price=100.0, atr=1.0, equity=10_000.0)
    assert plan["stop_loss"] == pytest.approx(101.5)
    assert plan["take_profit"] == pytest.approx(97.0)


def test_risk_plan_wait_is_not_actionable():
    plan = build_risk_plan("WAIT", entry_price=100.0, atr=1.0)
    assert plan["actionable"] is False
    assert "stop_loss" not in plan


def test_regime_trending_up():
    row = {"adx": 32.0, "trend_strength": 0.01, "volatility_percentile": 0.5, "ema_alignment": 1}
    out = classify_regime(row)
    assert out["regime"] == TRENDING_UP


def test_regime_high_volatility_dominates():
    row = {"adx": 45.0, "trend_strength": 0.02, "volatility_regime": 0.95, "ema_alignment": 1}
    out = classify_regime(row)
    assert out["regime"] == HIGH_VOLATILITY


def test_regime_ranging_when_no_trend():
    row = {"adx": 12.0, "trend_strength": 0.0001, "volatility_regime": 0.4, "ema_alignment": 1}
    out = classify_regime(row)
    assert out["regime"] == RANGING


def test_regime_from_real_feature_row(synthetic_candles):
    from app.features.builder import build_features

    candles = synthetic_candles(n=1500, freq="1min")
    frame = build_features(candles, "crypto", "1m")
    out = classify_regime(frame.iloc[-1])
    assert out["regime"] in {TRENDING_UP, "TRENDING_DOWN", RANGING, HIGH_VOLATILITY, "LOW_VOLATILITY"}
    assert isinstance(out["reason"], str)
