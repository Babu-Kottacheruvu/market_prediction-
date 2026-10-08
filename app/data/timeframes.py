"""Centralized timeframe utilities.

Convention used across the whole platform:

* ``timestamp`` is the candle **open** time (tz-aware UTC).
* A candle is only "known" once it has closed, so every candle carries
  ``available_at = timestamp + timeframe``.  Higher-timeframe context is joined
  on ``available_at <= base_timestamp`` which guarantees that only *closed*
  higher-timeframe candles are ever used (no look-ahead).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import UnsupportedTimeframeError

OHLC_COLUMNS = ["open", "high", "low", "close", "volume"]


def utc_ns(series: pd.Series) -> pd.Series:
    """Cast a timestamp series to tz-aware nanosecond precision.

    Parquet round-trips can come back as ms/us while derived columns are ns;
    ``merge_asof`` refuses to join mismatched units, so both keys are cast to
    the same unit before any as-of join.
    """
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.astype("datetime64[ns, UTC]")


def normalize_timeframe(name: str, settings: Settings | None = None) -> str:
    """Canonicalize a timeframe label (``1D`` -> ``1d``)."""
    settings = settings or get_settings()
    return settings.timeframe(name).name


def timeframe_delta(name: str, settings: Settings | None = None) -> pd.Timedelta:
    settings = settings or get_settings()
    return pd.Timedelta(minutes=settings.timeframe(name).minutes)


def is_valid_timeframe(name: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    try:
        settings.timeframe(name)
        return True
    except UnsupportedTimeframeError:
        return False


def assert_known_timeframes(names: Iterable[str], settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for name in names:
        settings.timeframe(name)


def with_availability(df: pd.DataFrame, timeframe: str, settings: Settings | None = None) -> pd.DataFrame:
    """Add the ``available_at`` column (= candle close time) to a candle frame."""
    settings = settings or get_settings()
    out = df.copy()
    out["available_at"] = out["timestamp"] + timeframe_delta(timeframe, settings)
    return out


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Resample normalized candles to ``timeframe``.

    ``df`` must be sorted by ``timestamp`` (open time), tz-aware UTC.
    The output keeps the *open* time as ``timestamp`` and sets
    ``available_at`` to the close time of the resampled candle.
    Empty buckets are dropped (no synthetic candles are invented).
    """
    settings = settings or get_settings()
    spec = settings.timeframe(timeframe)
    if df.empty:
        out = df.copy()
        out["available_at"] = pd.to_datetime([], utc=True)
        return out

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    frame = frame.set_index("timestamp")

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    extra = {c: "last" for c in frame.columns if c not in agg}
    agg.update(extra)

    resampled = frame.resample(spec.offset, label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["open"])
    if resampled.empty:
        out = resampled.reset_index()
        out["available_at"] = pd.to_datetime([], utc=True)
        return out

    out = resampled.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["available_at"] = out["timestamp"] + pd.Timedelta(minutes=spec.minutes)
    return out
