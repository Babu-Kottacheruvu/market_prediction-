"""Application configuration.

All YAML files under ``config/`` are loaded once and exposed through
``get_settings()``.  No magic numbers live in the rest of the code base.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Load .env (if present) so credentials/paths stay out of the repo.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass


class ConfigError(Exception):
    """Raised when configuration files are missing or malformed."""


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping")
    return data


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    offset: str
    minutes: int


@dataclass(frozen=True)
class Settings:
    timeframes: dict[str, TimeframeSpec]
    markets: dict[str, Any]
    multi_timeframe_context: dict[str, list[str]]
    features: dict[str, Any]
    models: dict[str, Any]
    risk: dict[str, Any]
    assets: dict[str, Any]

    @property
    def forecast_horizon(self) -> int:
        return int(self.models.get("forecast_horizon", 10))

    @property
    def enabled_timeframes(self) -> list[str]:
        return list(self.assets.get("enabled_timeframes", list(self.timeframes)))

    def timeframe(self, name: str) -> TimeframeSpec:
        from app.core.exceptions import UnsupportedTimeframeError

        key = _normalize_timeframe(name)
        if key not in self.timeframes:
            raise UnsupportedTimeframeError(
                f"Timeframe '{name}' is not configured. Supported: {sorted(self.timeframes)}"
            )
        return self.timeframes[key]

    def market(self, market: str) -> dict[str, Any]:
        from app.core.exceptions import UnsupportedMarketError

        key = market.strip().lower()
        if key not in self.markets:
            raise UnsupportedMarketError(
                f"Market '{market}' is not configured. Supported: {sorted(self.markets)}"
            )
        return self.markets[key]

    def resolve_symbol(self, market: str, symbol: str) -> str:
        """Resolve aliases (GOLD -> XAUUSD, BTCUSD -> BTCUSDT)."""
        from app.core.exceptions import UnsupportedSymbolError

        cfg = self.market(market)
        sym = symbol.strip().upper()
        aliases = {str(k).upper(): str(v).upper() for k, v in (cfg.get("aliases") or {}).items()}
        sym = aliases.get(sym, sym)
        assets = [str(a).upper() for a in cfg.get("assets", [])]
        if sym not in assets:
            raise UnsupportedSymbolError(
                f"Symbol '{symbol}' is not configured for market '{market}'. "
                f"Supported: {assets}"
            )
        return sym

    def context_timeframes(self, timeframe: str) -> list[str]:
        key = _normalize_timeframe(timeframe)
        return [t for t in self.multi_timeframe_context.get(key, []) if t in self.timeframes]

    def costs(self, market: str, timeframe: str) -> dict[str, float]:
        cfg = self.market(market)
        costs = cfg.get("costs", {}) or {}
        key = _normalize_timeframe(timeframe)
        base = dict(costs.get("default", {}) or {})
        base.update(costs.get(key, {}) or {})
        return {
            "spread": float(base.get("spread", 0.0)),
            "commission": float(base.get("commission", 0.0)),
            "slippage": float(base.get("slippage", 0.0)),
        }


def _normalize_timeframe(name: str) -> str:
    key = str(name).strip().lower()
    if key == "1d":
        return "1d"
    if key in {"1h", "4h"}:
        return key
    if key.endswith("d"):
        return key
    return key


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    tf_raw = _load_yaml("timeframes.yaml").get("timeframes", {})
    timeframes = {
        str(name): TimeframeSpec(
            name=str(name),
            offset=str(spec["offset"]),
            minutes=int(spec["minutes"]),
        )
        for name, spec in tf_raw.items()
    }
    markets_yaml = _load_yaml("markets.yaml")
    return Settings(
        timeframes=timeframes,
        markets=markets_yaml.get("markets", {}),
        multi_timeframe_context={
            str(k): [str(x) for x in (v or [])]
            for k, v in (markets_yaml.get("multi_timeframe_context") or {}).items()
        },
        features=_load_yaml("features.yaml"),
        models=_load_yaml("models.yaml"),
        risk=_load_yaml("risk.yaml"),
        assets=_load_yaml("assets.yaml"),
    )


def reset_settings_cache() -> None:
    """Testing helper: forget the cached settings."""
    get_settings.cache_clear()
