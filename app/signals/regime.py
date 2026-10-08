"""Market regime classification (simple, configurable, feature-driven)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings

TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
RANGING = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"


def classify_regime(features_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Classify the current regime from the latest feature row."""
    cfg = get_settings().risk.get("regime", {})
    adx_trending = float(cfg.get("adx_trending", 25.0))
    adx_strong = float(cfg.get("adx_strong", 40.0))
    vol_high_q = float(cfg.get("vol_high_quantile", 0.80))
    vol_low_q = float(cfg.get("vol_low_quantile", 0.20))

    row = features_row if isinstance(features_row, dict) else features_row.to_dict()
    adx = _f(row.get("adx"))
    trend = _f(row.get("trend_strength"))
    vol_rank = _f(row.get("volatility_regime"))
    ema_align = _f(row.get("ema_alignment"))

    if vol_rank is not None and vol_rank >= vol_high_q:
        regime = HIGH_VOLATILITY
        reason = f"volatility percentile {vol_rank:.2f} >= {vol_high_q}"
    elif adx is not None and adx >= adx_trending and trend is not None:
        if trend > 0 or ema_align == 1:
            regime = TRENDING_UP
            reason = f"ADX {adx:.1f} >= {adx_trending} with positive slope"
        else:
            regime = TRENDING_DOWN
            reason = f"ADX {adx:.1f} >= {adx_trending} with negative slope"
    elif vol_rank is not None and vol_rank <= vol_low_q:
        regime = LOW_VOLATILITY
        reason = f"volatility percentile {vol_rank:.2f} <= {vol_low_q}"
    elif adx is not None and adx >= adx_strong:
        regime = TRENDING_UP if (trend or 0) >= 0 else TRENDING_DOWN
        reason = f"ADX {adx:.1f} strong"
    else:
        regime = RANGING
        reason = f"ADX {adx if adx is not None else float('nan'):.1f} below trending threshold"

    return {
        "regime": regime,
        "adx": adx,
        "trend_strength": trend,
        "volatility_percentile": vol_rank,
        "reason": reason,
    }


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v
