"""Provider abstraction.

``MarketDataProvider`` is the only interface the ML pipeline talks to, so any
broker/exchange adapter (MT5, Binance, Bybit, OANDA, IB, FXCM...) can be
plugged in later without touching the feature/model code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger
from app.data import storage
from app.data.normalizer import normalize_ohlcv
from app.data.validator import DataQualityValidator

logger = get_logger(__name__)


class MarketDataProvider(ABC):
    """Reads historical OHLCV for a market."""

    market: str = ""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        """Return normalized candles: timestamp/open/high/low/close/volume/symbol/market."""

    @abstractmethod
    def download(
        self,
        symbol: str,
        timeframe: str,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        """Fetch fresh data from the upstream source and persist it locally."""

    def validate(self, df: pd.DataFrame, symbol: str, timeframe: str, strict: bool = True) -> pd.DataFrame:
        report = DataQualityValidator().validate(df, symbol=symbol, timeframe=timeframe)
        if strict:
            report.raise_if_failed()
        return df


class FileProvider(MarketDataProvider):
    """Local Parquet/CSV backed provider (development & manual datasets)."""

    def __init__(self, market: str):
        self.market = market.lower()

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        settings = get_settings()
        symbol = settings.resolve_symbol(self.market, symbol)
        try:
            return storage.load_candles(self.market, symbol, timeframe, start=start, end=end)
        except DataUnavailableError:
            derived = storage.derive_higher_timeframe(self.market, symbol, timeframe)
            if derived is None:
                raise
            frame = normalize_ohlcv(derived, symbol=symbol, market=self.market)
            if start is not None:
                frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
            if frame.empty:
                raise DataUnavailableError(
                    f"No rows for {self.market}/{symbol}/{timeframe} in the requested range"
                )
            return frame.reset_index(drop=True)

    def download(self, symbol: str, timeframe: str, start: Any | None = None, end: Any | None = None) -> pd.DataFrame:
        raise DataUnavailableError(
            f"No network download implemented for market '{self.market}'. "
            f"Provide real historical data as Parquet/CSV at "
            f"{storage.raw_path(self.market, symbol, timeframe)}"
        )


class DataProviderFactory:
    """Resolves the provider for a market from configuration."""

    _cache: dict[str, MarketDataProvider] = {}

    @classmethod
    def get(cls, market: str) -> MarketDataProvider:
        key = market.strip().lower()
        if key in cls._cache:
            return cls._cache[key]
        settings = get_settings()
        cfg = settings.market(key)
        # DATA_PROVIDER env wins over config so one variable can route every
        # market to a single source (e.g. DATA_PROVIDER=mt5).
        backend = (os.getenv("DATA_PROVIDER", "").strip() or cfg.get("provider", "csv")).lower()
        if backend == "binance":
            from app.data.crypto.binance import BinanceProvider

            provider: MarketDataProvider = BinanceProvider()
        elif backend == "mt5":
            from app.data.mt5_provider import MT5Provider

            provider = MT5Provider(key)
        else:
            provider = FileProvider(key)
        provider.market = key
        cls._cache[key] = provider
        return provider

    @classmethod
    def register(cls, market: str, provider: MarketDataProvider) -> None:
        cls._cache[market.strip().lower()] = provider

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


def get_data(market: str, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
    """Convenience entry point used by training/prediction code."""
    return DataProviderFactory.get(market).get_historical_data(symbol, timeframe, start, end)
