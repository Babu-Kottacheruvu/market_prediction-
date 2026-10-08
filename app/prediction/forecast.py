"""Forecast assembly: raw model outputs -> structured 10-horizon forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.models.router import ModelPackage
from app.prediction.confidence import horizon_confidence
from app.targets.ten_candle import DOWN, FLAT, UP

_PROBA_TO_LABEL = {0: DOWN, 1: FLAT, 2: UP}  # canonical column order


def build_forecast(
    package: ModelPackage,
    latest_features: pd.DataFrame,
    current_price: float,
    atr: float,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Produce the per-horizon forecast rows."""
    settings = settings or get_settings()
    horizon = settings.forecast_horizon
    # reindex: columns the model expects but the current frame lacks become NaN
    # (XGBoost handles missing values natively); extra columns are ignored.
    X = latest_features.reindex(columns=package.feature_columns)
    rows: list[dict[str, Any]] = []

    for h in range(1, horizon + 1):
        model_set = package.models[h]
        proba = model_set.predict_proba(X)[0]        # [P(DOWN), P(FLAT), P(UP)]
        calibrator = (package.calibrators or {}).get(h)
        if calibrator is not None:
            import numpy as _np

            proba = calibrator.transform(_np.asarray(proba).reshape(1, -1))[0]
        pred_return = float(model_set.predict_return(X)[0])
        expected_return = round(pred_return, 8)

        direction_idx = int(np.argmax(proba))
        direction = _PROBA_TO_LABEL[direction_idx]
        confidence = horizon_confidence(proba)

        vol = float(atr / current_price) * np.sqrt(h)
        pivot = current_price * (1.0 + expected_return)
        expected_high = pivot + 0.5 * atr * np.sqrt(h)
        expected_low = pivot - 0.5 * atr * np.sqrt(h)

        rows.append(
            {
                "horizon": h,
                "direction": direction,
                "probability": round(float(proba[direction_idx]), 6),
                "probability_up": round(float(proba[2]), 6),
                "probability_flat": round(float(proba[1]), 6),
                "probability_down": round(float(proba[0]), 6),
                "expected_return": expected_return,
                "expected_price": round(pivot, 6),
                "expected_high": round(expected_high, 6),
                "expected_low": round(expected_low, 6),
                "expected_volatility": round(vol, 8),
                "confidence": confidence,
            }
        )
    return rows


def overall_bias(forecast: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the 10 horizons into a single directional bias."""
    if not forecast:
        return {"bias": "NEUTRAL", "confidence": 0.0, "agreement": 0.0}
    weights = np.array([1.0 / row["horizon"] for row in forecast])
    weights /= weights.sum()
    net = sum(
        w * (row["probability_up"] - row["probability_down"])
        for w, row in zip(weights, forecast)
    )
    mean_conf = float(np.mean([row["confidence"] for row in forecast]))
    if net > 0.10:
        bias = "BULLISH"
    elif net < -0.10:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    return {
        "bias": bias,
        "net_probability": round(float(net), 6),
        "confidence": round(mean_conf, 6),
    }
