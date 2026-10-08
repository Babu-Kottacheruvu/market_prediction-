"""Common feature set shared by every market."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.features import technical as ta


def build_common_features(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Return a frame of common features aligned to ``df`` (same index)."""
    settings = settings or get_settings()
    cfg = settings.features
    out = pd.DataFrame(index=df.index)
    close, high, low, open_, vol = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # ---- returns -----------------------------------------------------------------
    for period in cfg.get("returns", [1, 2, 3, 5, 10, 20]):
        out[f"return_{period}"] = close.pct_change(period)

    # ---- candle structure --------------------------------------------------------
    body = close - open_
    rng = (high - low).replace(0.0, np.nan)
    out["body"] = body
    out["body_percent"] = body / close
    out["upper_wick"] = high - pd.concat([close, open_], axis=1).max(axis=1)
    out["lower_wick"] = pd.concat([close, open_], axis=1).min(axis=1) - low
    out["range"] = rng
    out["range_percent"] = rng / close
    out["bullish"] = (close > open_).astype(float)
    out["bearish"] = (close < open_).astype(float)

    # ---- moving averages ---------------------------------------------------------
    for period in cfg.get("ema_periods", [9, 15, 20, 50, 100, 200]):
        out[f"ema_{period}"] = ta.ema(close, period)
        out[f"ema_{period}_dist"] = (close - out[f"ema_{period}"]) / close
    for period in cfg.get("sma_periods", [20, 50, 200]):
        out[f"sma_{period}"] = ta.sma(close, period)
        out[f"sma_{period}_dist"] = (close - out[f"sma_{period}"]) / close

    # ---- indicators --------------------------------------------------------------
    out["rsi_14"] = ta.rsi(close, int(cfg.get("rsi_period", 14)))
    macd_cfg = cfg.get("macd", {})
    macd_df = ta.macd(
        close,
        int(macd_cfg.get("fast", 12)),
        int(macd_cfg.get("slow", 26)),
        int(macd_cfg.get("signal", 9)),
    )
    out = out.join(macd_df)
    out["macd_hist_norm"] = out["macd_hist"] / close
    out["adx"] = ta.adx(df, int(cfg.get("adx_period", 14)))
    out["atr"] = ta.atr(df, int(cfg.get("atr_period", 14)))
    out["atr_pct"] = out["atr"] / close

    bb_cfg = cfg.get("bollinger", {})
    out = out.join(ta.bollinger(close, int(bb_cfg.get("period", 20)), float(bb_cfg.get("std", 2.0))))

    st_cfg = cfg.get("stochastic", {})
    out = out.join(
        ta.stochastic(df, int(st_cfg.get("k_period", 14)), int(st_cfg.get("d_period", 3)), int(st_cfg.get("smooth", 3)))
    )
    out["cci"] = ta.cci(df, int(cfg.get("cci_period", 20)))
    for period in cfg.get("roc_periods", [5, 10, 20]):
        out[f"roc_{period}"] = ta.roc(close, int(period))
    for period in cfg.get("momentum_periods", [5, 10, 20]):
        out[f"momentum_{period}"] = ta.momentum(close, int(period)) / close

    # ---- VWAP --------------------------------------------------------------------
    out["vwap"] = ta.daily_vwap(df)
    out["vwap_dist"] = (close - out["vwap"]) / close

    # ---- volume ------------------------------------------------------------------
    vol_cfg = cfg.get("volume", {})
    ma_period = int(vol_cfg.get("ma_period", 20))
    z_period = int(vol_cfg.get("zscore_period", 20))
    out["volume"] = vol
    out["volume_change"] = vol.pct_change()
    vol_ma = ta.sma(vol, ma_period)
    out["volume_ma"] = vol_ma
    out["relative_volume"] = vol / vol_ma.replace(0.0, np.nan)
    out["volume_zscore"] = ta.rolling_zscore(vol, z_period)

    # ---- volatility --------------------------------------------------------------
    volb = int(cfg.get("volatility", {}).get("realized_period", 20))
    out["realized_vol"] = ta.rolling_volatility(close, volb)
    out["hl_volatility"] = (high - low) / close
    out["cc_volatility"] = close.pct_change().abs()
    out["volatility_ratio"] = out["hl_volatility"] / out["atr_pct"].replace(0.0, np.nan)
    regime_lb = int(cfg.get("volatility", {}).get("regime_lookback", 100))
    out["volatility_regime"] = (
        out["realized_vol"].rolling(regime_lb, min_periods=20).rank(pct=True)
    )

    # ---- market structure --------------------------------------------------------
    lb = int(cfg.get("market_structure", {}).get("lookback", 20))
    dist_lb = int(cfg.get("market_structure", {}).get("distance_lookback", 50))
    prev_high, prev_low = high.shift(1), low.shift(1)
    out["higher_high"] = (high > prev_high).astype(float)
    out["higher_low"] = (low > prev_low).astype(float)
    out["lower_high"] = (high < prev_high).astype(float)
    out["lower_low"] = (low < prev_low).astype(float)
    rolling_high = high.rolling(dist_lb, min_periods=lb).max()
    rolling_low = low.rolling(dist_lb, min_periods=lb).min()
    out["distance_from_recent_high"] = (close - rolling_high) / close
    out["distance_from_recent_low"] = (close - rolling_low) / close
    out["support_distance"] = (close - rolling_low) / close
    out["resistance_distance"] = (rolling_high - close) / close
    out["breakout"] = (close > rolling_high.shift(1)).astype(float)
    out["breakdown"] = (close < rolling_low.shift(1)).astype(float)
    ema20 = ta.ema(close, 20)
    ema50 = ta.ema(close, 50)
    out["trend_strength"] = (ema20 - ema50) / close
    out["ema_alignment"] = np.sign(ema20 - ema50)

    # ---- time features -----------------------------------------------------------
    ts = df["timestamp"]
    out["hour_of_day"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    return out
