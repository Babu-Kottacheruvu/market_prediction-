"""Commodity specific features.

Optional macro inputs (DXY, treasury yields, VIX, weather ...) are only added
when the caller supplies real datasets - never invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.technical import ema, rsi


def build_commodity_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    session_id = df["timestamp"].dt.floor("D")

    # causal day-to-date range position (only candles seen so far today)
    day_high = df["high"].groupby(session_id, sort=False).cummax()
    day_low = df["low"].groupby(session_id, sort=False).cummin()
    span = (day_high - day_low).replace(0.0, np.nan)
    out["day_range_position"] = (df["close"] - day_low) / span
    out["day_range_so_far"] = (day_high - day_low) / df["close"]

    # distance from the opening price of the day (causal)
    day_open = df["open"].groupby(session_id, sort=False).transform("first")
    out["day_return"] = df["close"] / day_open - 1.0

    close = df["close"]
    out["trend_persistence"] = ema(close, 20).pct_change(5)
    out["mean_reversion_signal"] = (close - ema(close, 50)) / close
    out["rsi_divergence_proxy"] = rsi(close, 14) - rsi(close, 7)

    # gap between sessions (relevant for exchange-traded commodities)
    gap = df["open"] - df["close"].shift(1)
    out["session_gap"] = gap / df["close"].shift(1)
    out["session_gap_sign"] = (gap.fillna(0.0) > 0).astype(float)
    out["has_gap"] = (gap.notna() & (gap.abs() > 0)).astype(float)
    return out
