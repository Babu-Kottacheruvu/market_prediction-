import pytest

from app.core.config import get_settings
from app.core.exceptions import (
    UnsupportedMarketError,
    UnsupportedSymbolError,
    UnsupportedTimeframeError,
)


def test_timeframes_configured():
    s = get_settings()
    assert set(s.timeframes) == {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}
    assert s.timeframe("1D").name == "1d"
    assert s.timeframe("5m").minutes == 5
    assert s.timeframe("1h").minutes == 60


def test_unknown_timeframe_raises():
    with pytest.raises(UnsupportedTimeframeError):
        get_settings().timeframe("7m")


def test_markets_and_symbols():
    s = get_settings()
    assert s.resolve_symbol("crypto", "btcusdt") == "BTCUSDT"
    assert s.resolve_symbol("crypto", "BTCUSD") == "BTCUSDT"      # alias
    assert s.resolve_symbol("commodities", "GOLD") == "XAUUSD"    # alias
    with pytest.raises(UnsupportedSymbolError):
        s.resolve_symbol("crypto", "NOTAREALPAIR")
    with pytest.raises(UnsupportedMarketError):
        s.market("stocks")


def test_multi_timeframe_context_is_ordered_and_valid():
    s = get_settings()
    ctx = s.context_timeframes("1m")
    assert ctx == ["5m", "15m", "1h"]
    assert all(c != "1m" for c in ctx)
    assert s.context_timeframes("4h") == ["1d"]


def test_costs_per_market_and_timeframe():
    s = get_settings()
    crypto_1m = s.costs("crypto", "1m")
    assert crypto_1m["spread"] > 0
    assert crypto_1m["commission"] > 0
    fx = s.costs("forex", "5m")
    assert fx["commission"] == 0.0  # spread-based pricing
