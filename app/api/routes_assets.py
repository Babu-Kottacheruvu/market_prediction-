"""Asset / timeframe / health discovery endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.data import storage

router = APIRouter(tags=["assets"])


@router.get("/health")
def health() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def _v(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "versions": {"xgboost": _v("xgboost"), "pandas": _v("pandas"), "fastapi": _v("fastapi")},
    }


@router.get("/assets")
def assets() -> dict:
    settings = get_settings()
    out: dict[str, list[str]] = {}
    for market, cfg in settings.markets.items():
        out[market] = [str(a) for a in cfg.get("assets", [])]
    stored = storage.list_stored_symbols()
    return {"assets": out, "stored_data": stored, "mvp": settings.assets.get("mvp", [])}


@router.get("/timeframes")
def timeframes() -> dict:
    settings = get_settings()
    return {
        "timeframes": [
            {"name": tf.name, "minutes": tf.minutes, "offset": tf.offset}
            for tf in settings.timeframes.values()
        ],
        "enabled": settings.enabled_timeframes,
        "multi_timeframe_context": settings.multi_timeframe_context,
    }
