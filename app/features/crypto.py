"""Crypto specific features.

Optional datasets (funding rate, open interest, liquidations, BTC dominance,
ETH/BTC ...) are supplied only when the user provides real data.  Nothing is
fabricated: when the extras are absent no placeholder column is created.
"""

from __future__ import annotations

import pandas as pd

from app.data.timeframes import utc_ns
from app.features.technical import rolling_zscore, ema, rsi


def build_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ts = df["timestamp"]
    hour = ts.dt.hour
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    out["is_us_session"] = ((hour >= 13) & (hour <= 20)).astype(float)
    out["is_asia_session"] = ((hour >= 0) & (hour < 8)).astype(float)
    # liquidity proxy: where volume sits relative to the recent distribution
    out["volume_percentile"] = df["volume"].rolling(200, min_periods=20).rank(pct=True)
    # acceleration of the trend
    close = df["close"]
    fast = ema(close, 8)
    slow = ema(close, 21)
    out["ema_acceleration"] = ((fast - slow) / close).diff(3)
    out["rsi_extreme"] = (rsi(close, 14) - 50.0).abs() / 50.0
    out["returns_zscore"] = rolling_zscore(close.pct_change(), 50)
    return out


def attach_optional_extras(features: pd.DataFrame, base: pd.DataFrame, extras: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """Merge optional on-chain/derivative datasets (funding, OI, ...).

    Each extra must already be normalized with a ``timestamp`` column.  Merging
    is done backwards in time so only values known at or before the candle are
    used.  Missing extras are skipped silently - the model simply won't have
    those features.
    """
    if not extras:
        return features
    out = features.copy()
    left = base[["timestamp"]].reset_index(drop=True)
    for name, frame in extras.items():
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            continue
        right = frame.sort_values("timestamp").copy()
        value_cols = [c for c in right.columns if c not in {"timestamp", "symbol", "market", "available_at"}]
        if not value_cols:
            continue
        merged = pd.merge_asof(
            left.assign(timestamp=utc_ns(left["timestamp"])),
            right[["timestamp"] + value_cols].assign(
                timestamp=utc_ns(right["timestamp"])
            ),
            left_on="timestamp",
            right_on="timestamp",
            direction="backward",
        ).drop(columns=["timestamp"])
        merged.index = out.index
        out = pd.concat([out, merged.add_prefix(f"extra_{name}_")], axis=1)
    return out
