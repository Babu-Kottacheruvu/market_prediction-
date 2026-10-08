"""Binance public REST adapter (crypto).

Uses the unauthenticated klines endpoint only - no API keys are required for
historical candles.  Keys are never written to logs or stored in the repo.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests

from app.core.config import get_settings
from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger
from app.data import storage
from app.data.base import MarketDataProvider
from app.data.normalizer import normalize_ohlcv

logger = get_logger(__name__)

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]

_BINANCE_INTERVALS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


class BinanceProvider(MarketDataProvider):
    market = "crypto"

    def __init__(self, base_url: str | None = None, timeout: int = 20):
        self.base_url = (base_url or os.getenv("BINANCE_BASE_URL") or "https://api.binance.com").rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    def get_historical_data(self, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
        settings = get_settings()
        symbol = settings.resolve_symbol(self.market, symbol)
        if storage.has_data(self.market, symbol, timeframe):
            return storage.load_candles(self.market, symbol, timeframe, start=start, end=end)
        derived = storage.derive_higher_timeframe(self.market, symbol, timeframe)
        if derived is not None:
            return normalize_ohlcv(derived, symbol=symbol, market=self.market)
        raise DataUnavailableError(
            f"No local data for {self.market}/{symbol}/{timeframe} and no network fetch "
            f"has been performed yet. Run: python -m scripts.download_data --market crypto "
            f"--symbol {symbol} --timeframe {timeframe}"
        )

    def download(self, symbol: str, timeframe: str, start: Any | None = None, end: Any | None = None) -> pd.DataFrame:
        settings = get_settings()
        symbol = settings.resolve_symbol(self.market, symbol)
        interval = _BINANCE_INTERVALS.get(settings.timeframe(timeframe).name)
        if interval is None:
            raise DataUnavailableError(f"Binance does not offer interval '{timeframe}'")

        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)

        rows: list[list[Any]] = []
        cursor = start_ms or (end_ms - 1000 * 60 * 1000)  # default: ~1000 candles back
        while cursor < end_ms:
            params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "endTime": end_ms, "limit": 1000}
            if start_ms is not None:
                params["startTime"] = cursor
            data = self._get("/api/v3/klines", params)
            if not data:
                break
            rows.extend(data)
            last_open = int(data[-1][0])
            if last_open <= cursor:
                break
            cursor = last_open + 1
            if len(data) < 1000:
                break
            time.sleep(0.15)

        if not rows:
            raise DataUnavailableError(f"Binance returned no klines for {symbol} {interval}")

        frame = pd.DataFrame(rows, columns=_KLINE_COLUMNS)
        frame = frame[["open_time", "open", "high", "low", "close", "volume"]]
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame.drop(columns=["open_time"])
        frame = normalize_ohlcv(frame, symbol=symbol, market=self.market)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        storage.save_candles(frame, self.market, symbol, timeframe)
        return frame

    def _get(self, path: str, params: dict[str, Any]) -> list:
        try:
            resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:  # network problems must be explicit
            raise DataUnavailableError(f"Binance request failed: {exc}") from exc
        if resp.status_code == 451:
            raise DataUnavailableError(
                "Binance is geo-restricted from this host. Set BINANCE_BASE_URL to a "
                "reachable mirror or supply local Parquet/CSV data."
            )
        if resp.status_code != 200:
            raise DataUnavailableError(f"Binance HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        return payload if isinstance(payload, list) else payload.get("data", [])
