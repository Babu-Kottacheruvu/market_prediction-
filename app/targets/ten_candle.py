"""10-horizon target generation.

For each horizon h in 1..10 and candle t:

    future_return_h     = close[t+h] / close[t] - 1
    future_direction    = UP / DOWN / FLAT using an ATR-adjusted threshold
    future_high_h       = max(high[t+1 .. t+h])
    future_low_h        = min(low[t+1  .. t+h])
    future_range_h      = future_high_h - future_low_h
    future_volatility_h = stdev of per-candle returns over t+1..t+h

All target columns are prefixed ``target_`` so the feature layer can never
mistake them for inputs (see ``app.features.builder.feature_columns``).
Rows within ``horizon`` candles of the end of the dataset get NaN targets and
are dropped at training time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.features.technical import atr as atr_fn

UP = "UP"
DOWN = "DOWN"
FLAT = "FLAT"

DIRECTION_INT = {UP: 1, DOWN: -1, FLAT: 0}


def direction_threshold(df: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    """Per-candle threshold used to classify UP/DOWN/FLAT.

    ATR mode: ``threshold = (ATR / close) * atr_multiplier`` so the bar moves
    with volatility, timeframe and instrument.  A volatility floor derived from
    the rolling std of past returns prevents the threshold from collapsing in
    dead markets.
    """
    settings = settings or get_settings()
    cfg = settings.risk.get("targets", {})
    mode = str(cfg.get("mode", "atr"))
    close = df["close"].astype(float)

    if mode == "fixed":
        threshold = pd.Series(float(cfg.get("fixed_threshold", 0.0005)), index=df.index)
    else:
        atr_period = int(settings.features.get("atr_period", 14))
        atr = atr_fn(df, atr_period)
        threshold = (atr / close) * float(cfg.get("atr_multiplier", 0.5))

    floor_period = int(cfg.get("vol_floor_period", 50))
    vol_floor = close.pct_change().rolling(floor_period, min_periods=10).std(ddof=0) * 0.25
    return np.maximum(threshold.fillna(0.0), vol_floor.fillna(0.0))


def _forward_roll(series: pd.Series, horizon: int, agg: str) -> pd.Series:
    """Rolling window over ``t+1 .. t+horizon`` (a *future* window).

    Implemented by shifting one step forward and rolling backwards on the
    reversed series, so no future value beyond the window is ever included and
    rows without a full future window stay NaN.
    """
    shifted = series.shift(-1)
    rev = shifted.iloc[::-1]
    rolling = rev.rolling(horizon, min_periods=horizon)
    if agg == "max":
        out = rolling.max()
    elif agg == "min":
        out = rolling.min()
    else:
        out = rolling.std(ddof=0)
    return out.iloc[::-1]


def build_targets(
    df: pd.DataFrame,
    settings: Settings | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Create ``target_*`` columns for horizons 1..horizon."""
    settings = settings or get_settings()
    targets_cfg = settings.risk.get("targets", {})
    horizon = int(horizon or targets_cfg.get("horizon") or settings.forecast_horizon)
    if len(df) == 0:
        return pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    thr = direction_threshold(df, settings).to_numpy()
    ret_1 = pd.Series(close.pct_change().to_numpy(), index=df.index)

    out = pd.DataFrame(index=df.index)
    for h in range(1, horizon + 1):
        future_return = close.shift(-h) / close - 1.0
        fut_high = _forward_roll(high, h, "max")
        fut_low = _forward_roll(low, h, "min")
        fut_vol = _forward_roll(ret_1, h, "std")
        fut_range = fut_high - fut_low

        r = future_return.to_numpy()
        with np.errstate(invalid="ignore"):
            direction = np.where(r > thr, 1.0, np.where(r < -thr, -1.0, 0.0))
        direction = np.where(np.isnan(r), np.nan, direction)

        out[f"target_return_h{h}"] = future_return
        out[f"target_direction_h{h}"] = direction
        out[f"target_high_h{h}"] = fut_high
        out[f"target_low_h{h}"] = fut_low
        out[f"target_range_h{h}"] = fut_range
        out[f"target_volatility_h{h}"] = fut_vol

    return out


def add_targets(
    df: pd.DataFrame,
    settings: Settings | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    return pd.concat([df, build_targets(df, settings, horizon)], axis=1)


def direction_label(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return FLAT
    if value > 0:
        return UP
    if value < 0:
        return DOWN
    return FLAT
