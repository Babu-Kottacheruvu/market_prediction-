"""Column normalization for every incoming data source.

The normalized schema is always::

    timestamp (UTC, open time), open, high, low, close, volume, symbol, market
"""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import DataQualityError

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_COLUMN_ALIASES = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "open_time": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "vol": "volume",
    "tick_volume": "volume",
}


def normalize_ohlcv(
    df: pd.DataFrame,
    symbol: str | None = None,
    market: str | None = None,
) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the normalized schema."""
    if df is None or len(df) == 0:
        raise DataQualityError("Received an empty dataset")

    frame = df.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = frame.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in frame.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataQualityError(f"Dataset is missing required columns: {missing}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame["timestamp"] = _to_utc_datetime(frame["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if symbol is not None:
        frame["symbol"] = str(symbol).upper()
    elif "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()

    if market is not None:
        frame["market"] = str(market).lower()
    elif "market" in frame.columns:
        frame["market"] = frame["market"].astype(str).str.lower()

    frame = frame.sort_values("timestamp", kind="mergesort")
    frame = frame.reset_index(drop=True)
    return frame[["timestamp"] + [c for c in CANDLE_COLUMNS if c != "timestamp"] + [
        c for c in ("symbol", "market") if c in frame.columns
    ]]


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise DataQualityError(f"{bad} rows have unparseable timestamps")
    # one canonical unit for every timestamp in the platform (merge_asof-safe)
    return ts.astype("datetime64[ns, UTC]")
