"""Local candle storage (Parquet primary, CSV fallback).

Layout::

    data/raw/{market}/{symbol}/{timeframe}.parquet
    data/processed/{market}/{symbol}/{timeframe}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core import config as app_config
from app.core.config import get_settings
from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger
from app.data.normalizer import normalize_ohlcv
from app.data.timeframes import resample_ohlcv

logger = get_logger(__name__)


def raw_dir(market: str, symbol: str) -> Path:
    return app_config.RAW_DATA_DIR / market.lower() / symbol.upper()


def raw_path(market: str, symbol: str, timeframe: str) -> Path:
    return raw_dir(market, symbol) / f"{_tf(timeframe)}.parquet"


def processed_path(market: str, symbol: str, timeframe: str) -> Path:
    return app_config.PROCESSED_DATA_DIR / market.lower() / symbol.upper() / f"{_tf(timeframe)}.parquet"


def _tf(timeframe: str) -> str:
    return get_settings().timeframe(timeframe).name


def save_candles(
    df: pd.DataFrame,
    market: str,
    symbol: str,
    timeframe: str,
    kind: str = "raw",
) -> Path:
    base = (app_config.RAW_DATA_DIR if kind == "raw" else app_config.PROCESSED_DATA_DIR) / market.lower() / symbol.upper()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{_tf(timeframe)}.parquet"
    frame = normalize_ohlcv(df, symbol=symbol, market=market)
    frame.to_parquet(path, index=False)
    logger.info("saved candles %s/%s/%s rows=%s path=%s", market, symbol, _tf(timeframe), len(frame), path)
    return path


def load_candles(
    market: str,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load local candles.  Falls back to CSV when no Parquet exists.

    Never invents data: raises ``DataUnavailableError`` when nothing is stored.
    """
    market_key = market.lower()
    symbol_key = symbol.upper()
    tf = _tf(timeframe)
    folder = app_config.RAW_DATA_DIR / market_key / symbol_key
    parquet = folder / f"{tf}.parquet"
    csv = folder / f"{tf}.csv"

    if parquet.exists():
        frame = pd.read_parquet(parquet)
    elif csv.exists():
        frame = pd.read_csv(csv)
    else:
        raise DataUnavailableError(
            f"No local data for {market_key}/{symbol_key}/{tf}. "
            f"Expected {parquet} (or .csv). Run: python -m scripts.download_data "
            f"--market {market_key} --symbol {symbol_key} --timeframe {tf} "
            f"or drop a real historical export at that path."
        )

    frame = normalize_ohlcv(frame, symbol=symbol_key, market=market_key)
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if frame.empty:
        raise DataUnavailableError(
            f"No rows for {market_key}/{symbol_key}/{tf} in the requested range"
        )
    return frame.reset_index(drop=True)


def has_data(market: str, symbol: str, timeframe: str) -> bool:
    folder = app_config.RAW_DATA_DIR / market.lower() / symbol.upper()
    tf = _tf(timeframe)
    return (folder / f"{tf}.parquet").exists() or (folder / f"{tf}.csv").exists()


def list_stored_symbols() -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if not app_config.RAW_DATA_DIR.exists():
        return found
    for market_dir in sorted(p for p in app_config.RAW_DATA_DIR.iterdir() if p.is_dir()):
        for symbol_dir in sorted(p for p in market_dir.iterdir() if p.is_dir()):
            tfs = sorted(p.stem for p in symbol_dir.glob("*.parquet")) + sorted(
                p.stem for p in symbol_dir.glob("*.csv")
            )
            for tf in sorted(set(tfs)):
                found.append({"market": market_dir.name, "symbol": symbol_dir.name, "timeframe": tf})
    return found


def derive_higher_timeframe(
    market: str,
    symbol: str,
    timeframe: str,
    prefer_base: tuple[str, ...] = ("1m", "3m", "5m", "15m", "30m", "1h", "4h"),
) -> pd.DataFrame | None:
    """Build a higher timeframe from smaller *real* candles (aggregation only).

    Lowering resolution (1m -> 15m) is legitimate aggregation of real data.
    The reverse (5m -> 1m) is never done anywhere in this code base.
    The trailing incomplete bucket is dropped.
    """
    settings = get_settings()
    target_minutes = settings.timeframe(timeframe).minutes
    for base_tf in prefer_base:
        base_minutes = settings.timeframe(base_tf).minutes
        if base_minutes >= target_minutes or target_minutes % base_minutes != 0:
            continue
        if not has_data(market, symbol, base_tf):
            continue
        base = load_candles(market, symbol, base_tf)
        resampled = resample_ohlcv(base, timeframe)
        data_end = (base["timestamp"] + pd.Timedelta(minutes=base_minutes)).max()
        resampled = resampled[resampled["available_at"] <= data_end]
        if not resampled.empty:
            logger.info(
                "derived %s/%s %s from %s (%s rows, last incomplete bucket dropped)",
                market, symbol, timeframe, base_tf, len(resampled),
            )
            return resampled.reset_index(drop=True)
    return None
