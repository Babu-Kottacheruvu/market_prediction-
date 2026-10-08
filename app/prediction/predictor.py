"""Prediction engine: candles + trained model -> full forecast response.

Prediction never trains or refits anything: load data, compute features,
load model, predict (fast path for 1-minute markets).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import InsufficientHistoricalDataError, ModelNotFoundError
from app.core.logging import get_logger
from app.data.base import DataProviderFactory, get_data
from app.data.mt5_provider import MT5Provider
from app.features.builder import build_features
from app.features.technical import atr as atr_fn
from app.models.router import ModelPackage, ModelRouter
from app.prediction.forecast import build_forecast, overall_bias
from app.prediction.confidence import overall_confidence
from app.signals.engine import generate_signal
from app.signals.regime import classify_regime

logger = get_logger(__name__)

MIN_FEATURE_ROWS = 300


def _latest_feature_row(
    candles: pd.DataFrame,
    market: str,
    timeframe: str,
    settings: Settings,
    required_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features for the series and return (frame, newest usable row).

    "Newest usable" = the most recent candle for which every feature the model
    was trained on is available (a warm-up gap would otherwise poison the row).
    """
    frame = build_features(candles, market, timeframe, settings=settings)
    if len(frame) < MIN_FEATURE_ROWS:
        raise InsufficientHistoricalDataError(
            f"Need at least {MIN_FEATURE_ROWS} candles to compute features, "
            f"got {len(frame)} for {market}/{timeframe}"
        )
    cols = [c for c in (required_columns or []) if c in frame.columns]
    mask = frame[cols].notna().all(axis=1) if cols else frame.notna().any(axis=1)
    usable = frame.loc[mask]
    if usable.empty:
        raise InsufficientHistoricalDataError(
            f"No candle has a complete feature row for {market}/{timeframe} "
            f"(required {len(cols)} columns)."
        )
    return frame, usable.tail(1)


def predict(
    market: str,
    symbol: str,
    timeframe: str,
    settings: Settings | None = None,
    router: ModelRouter | None = None,
    equity: float | None = None,
    include_risk: bool = True,
    live_data: bool = True,
) -> dict[str, Any]:
    """Full prediction request (see spec section 30)."""
    settings = settings or get_settings()
    router = router or ModelRouter()
    started = time.perf_counter()

    package: ModelPackage = router.get_model(market, symbol, timeframe)
    symbol_key = package.symbol
    tf = package.timeframe

    provider = DataProviderFactory.get(market)
    if live_data and isinstance(provider, MT5Provider):
        candles = provider.get_live_data(symbol_key, tf)
        data_source = "mt5"
    else:
        candles = get_data(market, symbol_key, tf)
        data_source = (
            "stored" if isinstance(provider, MT5Provider)
            else type(provider).__name__.removesuffix("Provider").lower()
        )
    features_frame, last_row = _latest_feature_row(
        candles, market, tf, settings, required_columns=package.feature_columns
    )

    current_price = float(candles["close"].iloc[-1])
    atr_value = float(atr_fn(candles, int(settings.features.get("atr_period", 14))).iloc[-1])

    forecast = build_forecast(package, last_row, current_price, atr_value, settings)
    last_candle_time = pd.Timestamp(candles["timestamp"].iloc[-1])
    candle_delta = pd.Timedelta(minutes=settings.timeframe(tf).minutes)
    for row in forecast:
        row["candle_time"] = (
            last_candle_time + candle_delta * row["horizon"]
        ).isoformat()
    bias = overall_bias(forecast)
    regime_info = classify_regime(last_row.iloc[0])
    conf = overall_confidence([row["confidence"] for row in forecast], [
        {"UP": 1, "FLAT": 0, "DOWN": -1}[row["direction"]] for row in forecast
    ])

    signal_info = generate_signal(
        forecast,
        regime=regime_info["regime"],
        market=market.lower(),
        timeframe=tf,
        overall=bias,
        settings=settings,
        model_status=package.status,
        data_source=data_source,
        data_timestamp=str(candles["timestamp"].iloc[-1]),
    )

    risk_plan = None
    if include_risk:
        from app.signals.risk import build_risk_plan

        risk_plan = build_risk_plan(
            signal=signal_info["signal"],
            entry_price=current_price,
            atr=atr_value,
            equity=equity,
            settings=settings,
            timeframe=tf,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "prediction %s/%s/%s | signal=%s | regime=%s | %.1f ms",
        market, symbol_key, tf, signal_info["signal"], regime_info["regime"], latency_ms,
    )

    return {
        "market": market.lower(),
        "symbol": symbol_key,
        "timeframe": tf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_timestamp": str(candles["timestamp"].iloc[-1]),
        "data_source": data_source,
        "current_price": current_price,
        "atr": round(atr_value, 6),
        "overall_bias": bias["bias"],
        "regime": regime_info["regime"],
        "regime_detail": regime_info,
        "confidence": conf["confidence"],
        "agreement": conf["agreement"],
        "signal": signal_info["signal"],
        "signal_detail": signal_info,
        "risk": risk_plan,
        "forecast": forecast,
        "model": {
            "model_id": package.model_id,
            "status": package.status,
            "model_version": package.metadata.model_version,
            "feature_version": package.metadata.feature_version,
            "trained_at": package.metadata.created_at,
            "n_features": len(package.feature_columns),
        },
        "latency_ms": round(latency_ms, 2),
    }


def model_status(market: str, symbol: str, timeframe: str) -> dict[str, Any]:
    return ModelRouter().model_status(market, symbol, timeframe)
