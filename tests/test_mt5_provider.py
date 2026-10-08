"""Tests for the MetaTrader 5 provider (fake terminal, no live connection)."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DataUnavailableError
from app.data.mt5_provider import MT5Provider, _MT5_TIMEFRAME, _RESAMPLE_FROM
from app.data.base import DataProviderFactory

RATES_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("spread", "<i4"),
        ("tick_volume", "<i8"),
        ("volume", "<i8"),
        ("real_volume", "<i8"),
    ]
)


def make_rates(n: int = 400, start: dt.datetime | None = None, step_seconds: int = 3600):
    start = start or dt.datetime(2026, 1, 5, 0, 0)  # a Monday
    times = [int((start + dt.timedelta(seconds=step_seconds * i)).timestamp()) for i in range(n)]
    arr = np.zeros(n, dtype=RATES_DTYPE)
    arr["time"] = times
    base = np.linspace(1.10, 1.12, n)
    arr["open"] = base
    arr["high"] = base + 0.001
    arr["low"] = base - 0.001
    arr["close"] = base + 0.0005
    arr["tick_volume"] = np.arange(n) + 10
    return arr


class FakeMt5:
    """Minimal stand-in for the MetaTrader5 module."""

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408

    def __init__(self, rates=None, known_symbols=("EURUSD", "XAUUSD", "GBPUSD", "BTCUSD")):
        self.rates = rates if rates is not None else make_rates()
        self.known = set(known_symbols)
        self.calls: list[tuple] = []

    def initialize(self, **kwargs):
        self.calls.append(("initialize", kwargs))
        return True

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        return SimpleNamespace(connected=True)

    def symbol_info(self, name):
        if name in self.known:
            return SimpleNamespace(name=name, visible=True)
        return None

    def symbol_info_tick(self, name):
        return SimpleNamespace(time=0, bid=1.0, ask=1.0)

    def symbol_select(self, name, enable=True):
        return name in self.known

    def symbols_get(self, pattern=None):
        base = (pattern or "*").replace("*", "")
        return [
            SimpleNamespace(name=s, visible=True) for s in sorted(self.known) if s.startswith(base)
        ]

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        self.calls.append(("copy_rates_range", symbol, timeframe, date_from, date_to))
        assert symbol in self.known, f"unknown symbol {symbol}"

        def aware(x):
            if x is None:
                return None
            ts = pd.Timestamp(x)
            return ts.tz_localize("UTC") if ts.tz is None else ts

        lo, hi = aware(date_from), aware(date_to)
        # filter the canned rates to the requested window
        out = []
        for row in self.rates:
            t = pd.Timestamp(row["time"], unit="s", tz="UTC")
            if lo is not None and t < lo:
                continue
            if hi is not None and t >= hi:
                continue
            out.append(row)
        if not out:
            return np.array([], dtype=RATES_DTYPE)
        return np.array(out, dtype=RATES_DTYPE)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.calls.append(("copy_rates_from_pos", symbol, timeframe, start_pos, count))
        assert symbol in self.known, f"unknown symbol {symbol}"
        end_pos = len(self.rates) - start_pos
        start_index = max(0, end_pos - count)
        return self.rates[start_index:end_pos]


@pytest.fixture
def fake_mt5(monkeypatch):
    fake = FakeMt5()
    monkeypatch.setattr("app.data.mt5_provider._import_mt5", lambda: fake)
    return fake


def test_timeframe_mapping():
    assert "1m" in _MT5_TIMEFRAME and "1d" in _MT5_TIMEFRAME
    assert _RESAMPLE_FROM["3m"] == "1m"


def test_rates_frame_is_normalized_utc():
    frame = MT5Provider._rates_frame(make_rates(50), "EURUSD", "forex")
    assert list(frame.columns)[:6] == [
        "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert frame["volume"].iloc[0] == 10.0  # tick_volume
    assert frame["symbol"].iloc[0] == "EURUSD"
    assert frame["market"].iloc[0] == "forex"


def test_download_saves_to_storage(tmp_storage, fake_mt5):
    provider = MT5Provider("forex")
    frame = provider.download("EURUSD", "1h", start="2026-01-05")
    assert len(frame) > 0
    assert (tmp_storage / "raw" / "forex" / "EURUSD" / "1h.parquet").exists()
    # the fake was called with the broker symbol and an exclusive end date
    _, broker_symbol, mt5_tf, date_from, date_to = fake_mt5.calls[-1]
    assert broker_symbol == "EURUSD"
    assert mt5_tf == FakeMt5.TIMEFRAME_H1
    assert date_to > date_from


def test_download_uses_aliases_then_candidates(tmp_storage, monkeypatch):
    # broker knows BTCUSD but config alias maps BTCUSD -> BTCUSDT
    fake = FakeMt5(known_symbols=("BTCUSD",))
    monkeypatch.setattr("app.data.mt5_provider._import_mt5", lambda: fake)
    provider = MT5Provider("crypto")
    frame = provider.download("BTCUSDT", "1h", start="2026-01-05")
    assert len(frame) > 0
    assert fake.calls[-1][1] == "BTCUSD"  # resolved through the alias
    # stored under the platform's canonical symbol
    assert (tmp_storage / "raw" / "crypto" / "BTCUSDT" / "1h.parquet").exists()


def test_unconfigured_symbol_rejected_by_config(tmp_storage, fake_mt5):
    from app.core.exceptions import UnsupportedSymbolError

    provider = MT5Provider("commodities")
    with pytest.raises(UnsupportedSymbolError) as exc:
        provider.download("XAUUSDT", "1h", start="2026-01-05")  # not a configured asset
    assert "XAUUSD" in str(exc.value)  # config lists what is supported


def test_configured_symbol_missing_on_broker_lists_candidates(tmp_storage, monkeypatch):
    # configured as USOIL, but this broker only offers USOILX
    fake = FakeMt5(known_symbols=("EURUSD", "XAUUSD", "USOILX"))
    monkeypatch.setattr("app.data.mt5_provider._import_mt5", lambda: fake)
    provider = MT5Provider("commodities")
    with pytest.raises(DataUnavailableError) as exc:
        provider.download("USOIL", "1h", start="2026-01-05")
    msg = str(exc.value)
    assert "USOIL" in msg
    assert "USOILX" in msg  # close candidate from the broker


def test_download_3m_resamples_from_1m(tmp_storage, fake_mt5):
    fake_mt5.rates = make_rates(400, step_seconds=60)  # 1m bars
    provider = MT5Provider("forex")
    frame = provider.download("EURUSD", "3m", start="2026-01-05")
    deltas = frame["timestamp"].diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(minutes=3)]
    # asked the terminal for M1, not an unsupported 3m constant
    assert fake_mt5.calls[-1][2] == FakeMt5.TIMEFRAME_M1


def test_download_falls_back_to_latest_bars_in_batches(tmp_storage, fake_mt5, monkeypatch):
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=400)).replace(microsecond=0)
    fake_mt5.rates = make_rates(400, start=start, step_seconds=60)
    monkeypatch.setattr("app.data.mt5_provider._MAX_BARS_PER_REQUEST", 150)
    monkeypatch.setattr(
        fake_mt5,
        "copy_rates_range",
        lambda *_args, **_kwargs: np.array([], dtype=RATES_DTYPE),
    )

    frame = MT5Provider("forex").download("EURUSD", "1m", start=start)

    assert len(frame) == 400
    calls = [call for call in fake_mt5.calls if call[0] == "copy_rates_from_pos"]
    assert [call[3] for call in calls] == [0, 150, 300]


def test_download_empty_mt5_history_includes_last_error(tmp_storage, fake_mt5, monkeypatch):
    monkeypatch.setattr(
        fake_mt5,
        "copy_rates_range",
        lambda *_args, **_kwargs: np.array([], dtype=RATES_DTYPE),
    )
    monkeypatch.setattr(
        fake_mt5,
        "copy_rates_from_pos",
        lambda *_args, **_kwargs: np.array([], dtype=RATES_DTYPE),
    )

    with pytest.raises(
        DataUnavailableError,
        match="latest-bars last_error=\\(1, 'Success'\\)",
    ):
        MT5Provider("forex").download("EURUSD", "1m")


def test_incremental_download_appends_only_new(tmp_storage, fake_mt5):
    provider = MT5Provider("forex")
    first = provider.download("EURUSD", "1h", start="2026-01-05")
    n_first = len(first)
    # same data again -> merged/deduped, not duplicated
    second = provider.download("EURUSD", "1h")
    assert len(second) == n_first
    assert second["timestamp"].is_monotonic_increasing
    assert second["timestamp"].duplicated().sum() == 0


def test_get_historical_refreshes_when_stale(tmp_storage, fake_mt5, monkeypatch):
    provider = MT5Provider("forex")
    provider.download("EURUSD", "1h", start="2026-01-05")
    calls_before = len(fake_mt5.calls)

    monkeypatch.setenv("MT5_REFRESH", "1")
    monkeypatch.setenv("MT5_STALE_CANDLES", "3")
    frame = provider.get_historical_data("EURUSD", "1h")
    assert len(fake_mt5.calls) > calls_before  # stale -> refresh happened
    assert len(frame) > 0


def test_get_historical_no_refresh_when_disabled(tmp_storage, fake_mt5, monkeypatch):
    provider = MT5Provider("forex")
    provider.download("EURUSD", "1h", start="2026-01-05")
    calls_before = len(fake_mt5.calls)

    monkeypatch.setenv("MT5_REFRESH", "0")
    frame = provider.get_historical_data("EURUSD", "1h")
    assert len(fake_mt5.calls) == calls_before
    assert len(frame) > 0


def test_get_live_data_always_fetches_from_mt5(tmp_storage, fake_mt5):
    provider = MT5Provider("forex")
    provider.download("EURUSD", "1h", start="2026-01-05")
    calls_before = len(fake_mt5.calls)

    frame = provider.get_live_data("EURUSD", "1h")

    assert len(fake_mt5.calls) > calls_before
    assert len(frame) > 0


def test_get_live_data_propagates_mt5_failure(tmp_storage, fake_mt5, monkeypatch):
    provider = MT5Provider("forex")
    provider.download("EURUSD", "1h", start="2026-01-05")

    def boom(*_args, **_kwargs):
        raise DataUnavailableError("terminal gone")

    monkeypatch.setattr(provider, "download", boom)
    with pytest.raises(DataUnavailableError, match="terminal gone"):
        provider.get_live_data("EURUSD", "1h")


def test_terminal_failure_falls_back_to_stored(tmp_storage, fake_mt5, monkeypatch):
    provider = MT5Provider("forex")
    provider.download("EURUSD", "1h", start="2026-01-05")

    def boom(*_args, **_kwargs):
        raise DataUnavailableError("terminal gone")

    monkeypatch.setattr(provider, "download", boom)
    monkeypatch.setenv("MT5_REFRESH", "1")
    frame = provider.get_historical_data("EURUSD", "1h")
    assert len(frame) > 0  # served stored data instead of crashing


def test_factory_routes_all_markets_to_mt5():
    DataProviderFactory.clear()
    for market in ("forex", "crypto", "commodities"):
        provider = DataProviderFactory.get(market)
        assert isinstance(provider, MT5Provider), f"{market} -> {type(provider)}"
        DataProviderFactory.clear()
