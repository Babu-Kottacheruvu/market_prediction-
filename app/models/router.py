"""Model router: market + symbol + timeframe -> the correct model package.

Never falls back to a different timeframe or symbol; fails clearly instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ModelNotFoundError
from app.core.logging import get_logger
from app.models.registry import ModelMetadata, ModelRegistry, package_dir

logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _load_cached_package(
    registry_path: str,
    market: str,
    symbol: str,
    timeframe: str,
    metadata_mtime_ns: int,
) -> tuple[dict[int, Any], ModelMetadata, dict[int, Any]]:
    registry = ModelRegistry(Path(registry_path))
    models, metadata = registry.load_package(market, symbol, timeframe)
    calibrators = registry.load_calibrators(market, symbol, timeframe)
    return models, metadata, calibrators


@dataclass
class ModelPackage:
    market: str
    symbol: str
    timeframe: str
    models: dict[int, Any]
    metadata: ModelMetadata
    calibrators: dict[int, Any] | None = None

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.feature_columns)

    @property
    def status(self) -> str:
        return self.metadata.status

    @property
    def model_id(self) -> str:
        return self.metadata.model_id


class ModelRouter:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()

    def get_model(self, market: str, symbol: str, timeframe: str) -> ModelPackage:
        settings = get_settings()
        market_key = market.strip().lower()
        symbol_key = settings.resolve_symbol(market_key, symbol)
        tf = settings.timeframe(timeframe).name

        model_dir = package_dir(market_key, symbol_key, tf)
        metadata_path = model_dir / "metadata.json"
        if not model_dir.exists() or not metadata_path.exists():
            raise ModelNotFoundError(
                f"No model directory for {market_key}/{symbol_key}/{tf}. "
                f"Train first: python -m scripts.train --market {market_key} "
                f"--symbol {symbol_key} --timeframe {tf}"
            )

        models, metadata, calibrators = _load_cached_package(
            str(self.registry.registry_path.resolve()),
            market_key,
            symbol_key,
            tf,
            metadata_path.stat().st_mtime_ns,
        )
        if metadata.timeframe != tf:
            raise ModelNotFoundError(
                f"Model {metadata.model_id} belongs to timeframe {metadata.timeframe}, "
                f"not {tf}. Refusing to use the wrong timeframe."
            )
        if metadata.symbol != symbol_key or metadata.market != market_key:
            raise ModelNotFoundError(
                f"Model {metadata.model_id} belongs to {metadata.market}/{metadata.symbol}."
            )
        logger.info(
            "routed %s/%s/%s -> model %s (status=%s)",
            market_key, symbol_key, tf, metadata.model_id, metadata.status,
        )
        return ModelPackage(
            market=market_key,
            symbol=symbol_key,
            timeframe=tf,
            models=models,
            metadata=metadata,
            calibrators=calibrators or None,
        )

    def model_status(self, market: str, symbol: str, timeframe: str) -> dict[str, Any]:
        settings = get_settings()
        market_key = market.strip().lower()
        symbol_key = settings.resolve_symbol(market_key, symbol)
        tf = settings.timeframe(timeframe).name
        if not package_dir(market_key, symbol_key, tf).exists():
            return {"trained": False, "market": market_key, "symbol": symbol_key, "timeframe": tf}
        _, metadata = self.registry.load_package(market_key, symbol_key, tf)
        return {
            "trained": True,
            "market": market_key,
            "symbol": symbol_key,
            "timeframe": tf,
            "model_id": metadata.model_id,
            "status": metadata.status,
            "created_at": metadata.created_at,
            "training_start": metadata.training_start,
            "training_end": metadata.training_end,
            "n_rows": metadata.n_rows,
            "n_features": len(metadata.feature_columns),
        }
