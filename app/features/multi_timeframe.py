"""Multi-timeframe context features.

Leakage rule (tested by tests/test_no_lookahead.py): a higher-timeframe candle
is only joined onto a base candle when ``available_at <= base_timestamp``,
i.e. the higher-timeframe candle was already *closed* when the base candle
opened.  The trailing incomplete higher-timeframe bucket can therefore never
be observed.
"""

from __future__ import annotations

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.data.timeframes import resample_ohlcv, utc_ns
from app.features import technical as ta

logger = get_logger(__name__)


def _compact_htf_features(resampled: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Small causal feature set computed on the higher timeframe frame."""
    close = resampled["close"]
    out = pd.DataFrame(index=resampled.index)
    out["ema_20"] = ta.ema(close, 20)
    out["ema_50"] = ta.ema(close, 50)
    out["ema_20_dist"] = (close - out["ema_20"]) / close
    out["ema_50_dist"] = (close - out["ema_50"]) / close
    out["rsi_14"] = ta.rsi(close, 14)
    out["atr_pct"] = ta.atr(resampled, 14) / close
    out["return_5"] = close.pct_change(5)
    out["return_10"] = close.pct_change(10)
    out["adx"] = ta.adx(resampled, 14)
    out["trend_up"] = (out["ema_20"] > out["ema_50"]).astype(float)
    out["realized_vol"] = ta.rolling_volatility(close, 20)
    wanted = cfg.get("htf_features") or list(out.columns)
    keep = [c for c in wanted if c in out.columns]
    out = out[keep].copy()
    out["available_at"] = resampled["available_at"]
    return out


def build_multi_timeframe_features(
    df: pd.DataFrame,
    timeframe: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Join closed higher-timeframe context features onto base candles."""
    settings = settings or get_settings()
    context = settings.context_timeframes(timeframe)
    if not context:
        return pd.DataFrame(index=df.index)

    base = df[["timestamp"]].copy()
    base = base.sort_values("timestamp").reset_index(drop=True)
    result = base.copy()

    for ctx in context:
        try:
            resampled = resample_ohlcv(df, ctx, settings)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("multi_timeframe resample failed for %s: %s", ctx, exc)
            continue
        if len(resampled) < 60:
            logger.info(
                "multi_timeframe: skipping %s context (%s candles) - not enough history",
                ctx, len(resampled),
            )
            continue
        feats = _compact_htf_features(resampled, settings.features).sort_values("available_at")
        before = set(result.columns)
        merged = pd.merge_asof(
            result.assign(timestamp=utc_ns(result["timestamp"])),
            feats.assign(available_at=utc_ns(feats["available_at"])),
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["available_at"])
        new_cols = [c for c in merged.columns if c not in before and c != "timestamp"]
        merged = merged.rename(columns={c: f"htf_{ctx}_{c}" for c in new_cols})
        result = merged

    result = result.set_index(df.index)
    return result.drop(columns=["timestamp"], errors="ignore")
