"""MetaTrader 5 data provider — the single market-data source for the platform.

All markets (forex / crypto / commodities) are pulled through MT5 when
``provider: mt5`` is configured.  The terminal must be installed and logged in
(the API attaches to the running terminal); credentials are only needed when
``MT5_LOGIN``/``MT5_PASSWORD``/``MT5_SERVER`` are supplied.

Notes
-----
* Bar times come from the broker as Unix epoch seconds and are treated as UTC,
  which is the standard convention for MT5 pipelines.  All candles in a dataset
  share the same clock, so walk-forward splits stay consistent.
* Volume is the broker's ``tick_volume`` (real_volume is 0 for most symbols).
* ``copy_rates_range``'s ``date_to`` is exclusive, so we request one day extra.
* Symbols are broker-dependent: forex majors and metals are usually present,
  crypto CFDs may not be.  Missing symbols produce an actionable error listing
  close candidates — never a fabricated substitute.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger
from app.data import storage
from app.data.base import MarketDataProvider
from app.data.normalizer import normalize_ohlcv
from app.data.timeframes import resample_ohlcv

logger = get_logger(__name__)

# our canonical timeframe -> MetaTrader5 constant name (3m has no MT5 bar type)
_MT5_TIMEFRAME = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}
_RESAMPLE_FROM = {"3m": "1m"}  # timeframe we must fetch to build it
_DEFAULT_LOOKBACK_DAYS = 90
_DEFAULT_MAX_BARS = 100_000
_MAX_BARS_PER_REQUEST = 10_000


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise DataUnavailableError(
            "The MetaTrader5 package is not installed (Windows only). "
            "Install it with: pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Provider(MarketDataProvider):
    """Reads OHLCV from a MetaTrader 5 terminal and stores it locally."""

    def __init__(self, market: str, terminal_path: str | None = None):
        self.market = market.lower()
        self.terminal_path = (
            terminal_path
            or os.getenv("MT5_TERMINAL_PATH")
            or r"C:\Program Files\MetaTrader 5\terminal64.exe"
        )

    # ------------------------------------------------------------- connect
    def _connect(self, mt5) -> None:
        kwargs: dict[str, Any] = {}
        if os.path.exists(self.terminal_path):
            kwargs["path"] = self.terminal_path
        login, password, server = (
            os.getenv("MT5_LOGIN"),
            os.getenv("MT5_PASSWORD"),
            os.getenv("MT5_SERVER"),
        )
        if login and password and server:
            kwargs.update(login=int(login), password=password, server=server)

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            raise DataUnavailableError(
                f"Could not initialize MetaTrader 5 (last_error={err}). "
                f"Start terminal '{self.terminal_path}', log in to an account, "
                "or set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER/MT5_TERMINAL_PATH."
            )
        info = mt5.terminal_info()
        if info is not None and not getattr(info, "connected", True):
            raise DataUnavailableError(
                "MetaTrader 5 terminal started but is not connected to a broker. "
                "Log in inside the terminal, then retry."
            )

    # ------------------------------------------------------------ symbols
    def _resolve_symbol(self, mt5, symbol: str) -> str:
        """Return the broker's symbol name for ``symbol`` (or raise)."""
        settings = get_settings()
        candidates: list[str] = [symbol]
        aliases = settings.market(self.market).get("aliases", {}) or {}
        # aliases map alias -> canonical; try every alias pointing at this one
        candidates += [a for a, canon in aliases.items() if canon == symbol]
        # and the canonical value of any alias we were given
        if symbol in aliases:
            candidates.append(aliases[symbol])

        seen: set[str] = set()
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            info = mt5.symbol_info(name)
            if info is not None:
                if not info.visible:
                    mt5.symbol_info_tick(name)  # nudge: forces symbol download
                    mt5.symbol_select(name, True)
                    info = mt5.symbol_info(name)
                if info is not None and (info.visible or mt5.symbol_select(name, True)):
                    return name

        # broker does not know it: offer close candidates instead of guessing
        base = symbol.upper().rstrip("0123456789T")  # BTCUSDT -> BTC
        near = [
            s.name
            for s in (mt5.symbols_get(f"{base[:3]}*") or [])
            if s.visible or mt5.symbol_select(s.name, True)
        ][:12]
        raise DataUnavailableError(
            f"Broker does not offer symbol '{symbol}' for market '{self.market}'. "
            f"Tried: {seen}. Similar symbols on this broker: {near or 'none found'}. "
            "Use one of the broker's symbols (python -c \"import MetaTrader5 as m; "
            "m.initialize(); print([s.name for s in m.symbols_get() if s.visible])\")."
        )

    # --------------------------------------------------------- conversions
    @staticmethod
    def _mt5_timeframe(mt5, timeframe: str) -> int:
        const = _MT5_TIMEFRAME.get(timeframe)
        if const is None:
            raise DataUnavailableError(
                f"Timeframe '{timeframe}' is not a native MT5 bar type. "
                f"Native: {sorted(_MT5_TIMEFRAME)} (3m is built by resampling 1m)."
            )
        return getattr(mt5, const)

    @staticmethod
    def _rates_frame(rates, symbol: str, market: str) -> pd.DataFrame:
        """Convert MT5's structured array into a normalized OHLCV frame."""
        if rates is None or len(rates) == 0:
            raise DataUnavailableError(
                f"MetaTrader 5 returned no candles for {symbol} (check symbol, "
                "date range and terminal connection: mt5.last_error())."
            )
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(rates["time"], unit="s", utc=True),
                "open": rates["open"].astype(float),
                "high": rates["high"].astype(float),
                "low": rates["low"].astype(float),
                "close": rates["close"].astype(float),
                "volume": rates["tick_volume"].astype(float),
            }
        )
        return normalize_ohlcv(frame, symbol=symbol, market=market)

    def _download_latest_bars(
        self,
        mt5,
        broker_symbol: str,
        symbol: str,
        timeframe: str,
        mt5_timeframe: int,
        max_bars: int,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch latest MT5 bars in bounded batches, walking back to ``start``."""
        copy_from_pos = getattr(mt5, "copy_rates_from_pos", None)
        if copy_from_pos is None:
            raise DataUnavailableError(
                "This MetaTrader5 package does not provide copy_rates_from_pos; "
                "update the MetaTrader5 Python package."
            )

        frames: list[pd.DataFrame] = []
        start_pos = 0
        bars_read = 0
        batch_size = min(max_bars, _MAX_BARS_PER_REQUEST)
        last_error = mt5.last_error()

        while bars_read < max_bars:
            count = min(batch_size, max_bars - bars_read)
            rates = copy_from_pos(broker_symbol, mt5_timeframe, start_pos, count)
            if rates is None or len(rates) == 0:
                last_error = mt5.last_error()
                if not frames and batch_size > 1:
                    batch_size = max(1, batch_size // 10)
                    continue
                logger.warning(
                    "mt5 latest-bars fetch stopped at position %s for %s/%s "
                    "(last_error=%s)",
                    start_pos, broker_symbol, timeframe, last_error,
                )
                break

            chunk = self._rates_frame(rates, symbol, self.market)
            oldest = chunk["timestamp"].min()
            mask = chunk["timestamp"] < end
            if start is not None:
                mask &= chunk["timestamp"] >= start
            chunk = chunk.loc[mask]
            if not chunk.empty:
                frames.append(chunk)

            returned = len(rates)
            bars_read += returned
            start_pos += returned
            if returned < count or (start is not None and oldest <= start):
                break

        if not frames:
            raise DataUnavailableError(
                f"MT5 returned no candles for {broker_symbol} {timeframe} "
                f"(latest-bars last_error={last_error}). Check symbol visibility, "
                "broker history for this timeframe, and the terminal's 'Max bars "
                "in chart' setting."
            )

        result = pd.concat(frames, ignore_index=True)
        return (
            result.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ----------------------------------------------------------- download
    def download(
        self,
        symbol: str,
        timeframe: str,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        mt5 = _import_mt5()
        self._connect(mt5)
        settings = get_settings()
        symbol = settings.resolve_symbol(self.market, symbol)
        broker_symbol = self._resolve_symbol(mt5, symbol)

        fetch_tf = _RESAMPLE_FROM.get(timeframe, timeframe)
        mt5_tf = self._mt5_timeframe(mt5, fetch_tf)

        # incremental: when no explicit start, continue from what we store
        local: pd.DataFrame | None = None
        if storage.has_data(self.market, symbol, timeframe):
            local = storage.load_candles(self.market, symbol, timeframe)
            default_start = local["timestamp"].iloc[-1]
        elif start is None:
            default_start = pd.Timestamp.now("UTC") - pd.Timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        else:
            default_start = None

        date_from = pd.Timestamp(start) if start is not None else default_start
        if date_from is not None:
            date_from = pd.Timestamp(date_from, tz="UTC") if pd.Timestamp(date_from).tz is None \
                else pd.Timestamp(date_from)
        date_to = pd.Timestamp(end) if end is not None else pd.Timestamp.now("UTC")
        if date_to.tz is None:
            date_to = date_to.tz_localize("UTC")
        date_to = date_to + pd.Timedelta(days=1)  # copy_rates_range excludes date_to

        logger.info(
            "mt5 download %s/%s/%s | broker symbol=%s | %s -> %s",
            self.market, symbol, timeframe, broker_symbol,
            date_from.date() if date_from is not None else "beginning",
            date_to.date(),
        )
        rates = mt5.copy_rates_range(
            broker_symbol,
            mt5_tf,
            date_from.to_pydatetime() if date_from is not None else None,
            date_to.to_pydatetime(),
        )
        if (rates is None or len(rates) == 0) and end is None:
            try:
                max_bars = int(os.getenv("MT5_MAX_BARS", str(_DEFAULT_MAX_BARS)))
            except ValueError as exc:
                raise DataUnavailableError("MT5_MAX_BARS must be a positive integer.") from exc
            if max_bars <= 0:
                raise DataUnavailableError("MT5_MAX_BARS must be a positive integer.")

            range_error = mt5.last_error()
            logger.warning(
                "mt5 range download returned no candles for %s/%s "
                "(last_error=%s); retrying in batches up to %s bars",
                broker_symbol, timeframe, range_error, max_bars,
            )
            try:
                fresh = self._download_latest_bars(
                    mt5,
                    broker_symbol,
                    symbol,
                    timeframe,
                    mt5_tf,
                    max_bars,
                    date_from,
                    date_to,
                )
            except DataUnavailableError as exc:
                raise DataUnavailableError(
                    f"MT5 range request failed (last_error={range_error}); "
                    f"latest-bars retry failed: {exc}"
                ) from exc
        else:
            fresh = self._rates_frame(rates, symbol, self.market)

        if fetch_tf != timeframe:  # e.g. 3m built from MT5 1m bars
            fresh = resample_ohlcv(fresh, timeframe, settings)
            fresh = fresh.drop(columns=["available_at"], errors="ignore")

        if local is not None:
            merged = pd.concat([local, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset="timestamp", keep="last")
            fresh = merged.sort_values("timestamp").reset_index(drop=True)

        if fresh.empty:
            raise DataUnavailableError(
                f"MT5 returned no candles for {broker_symbol} {timeframe} in the "
                f"requested range (last_error={mt5.last_error()}). Check that the "
                "symbol is visible in Market Watch, the broker provides history "
                "for this timeframe, and MT5's 'Max bars in chart' setting allows "
                "enough history. For larger training windows, increase "
                "MT5_MAX_BARS and the terminal's chart-history limit."
            )

        self.validate(fresh, symbol, timeframe, strict=True)
        storage.save_candles(fresh, self.market, symbol, timeframe)
        return fresh

    # --------------------------------------------------- serving predictions
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Any | None = None,
        end: Any | None = None,
    ) -> pd.DataFrame:
        """Local candles first, transparently refreshed from MT5 when stale.

        Freshness window: if the newest stored candle is more than
        ``MT5_STALE_CANDLES`` (default 3) candle durations old, only the gap is
        fetched from the terminal.  Set ``MT5_REFRESH=0`` to disable.
        """
        settings = get_settings()
        symbol = settings.resolve_symbol(self.market, symbol)
        refresh = os.getenv("MT5_REFRESH", "1").strip().lower() not in {"0", "false", "no"}

        if not storage.has_data(self.market, symbol, timeframe):
            frame = self.download(symbol, timeframe, start=start, end=end)
            return self._slice(frame, start, end)

        frame = storage.load_candles(self.market, symbol, timeframe, start=start, end=end)
        if refresh and start is None and end is None:
            stale_after = int(os.getenv("MT5_STALE_CANDLES", "3"))
            tf_minutes = settings.timeframe(timeframe).minutes
            last = frame["timestamp"].iloc[-1]
            age = pd.Timestamp.now("UTC") - last
            if age > pd.Timedelta(minutes=tf_minutes * stale_after):
                try:
                    fresh = self.download(symbol, timeframe)  # incremental tail
                    logger.info(
                        "mt5 refreshed %s/%s/%s (+%s rows, newest %s)",
                        self.market, symbol, timeframe,
                        len(fresh) - len(frame), fresh["timestamp"].iloc[-1],
                    )
                    frame = self._slice(fresh, start, end)
                except DataUnavailableError as exc:
                    # terminal unavailable: serving stored data is honest and safe
                    logger.warning("mt5 refresh skipped, using stored data: %s", exc)
        return frame

    def get_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch the latest candles from MT5 without falling back to local data."""
        return self.download(symbol, timeframe)

    @staticmethod
    def _slice(frame: pd.DataFrame, start: Any | None, end: Any | None) -> pd.DataFrame:
        out = frame
        if start is not None:
            out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC")]
        return out.reset_index(drop=True)
