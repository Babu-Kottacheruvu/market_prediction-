"""Model registry.

A model package lives at::

    models/{market}/{symbol}/{timeframe}/
        direction_h1.pkl ... direction_h10.pkl
        return_h1.pkl    ... return_h10.pkl
        calibrator_h1.pkl (optional)
        metadata.json

The registry index (``models/registry.json``) tracks versions so a validated
model can be *activated* without silently overwriting production models.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.core import config as app_config
from app.core.config import get_settings
from app.core.exceptions import ModelNotFoundError, ModelValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()


@dataclass
class ModelMetadata:
    model_id: str
    market: str
    symbol: str
    timeframe: str
    horizon: int
    feature_version: str
    model_version: str
    training_start: str
    training_end: str
    n_rows: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward: list[dict[str, Any]] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    status: str = "candidate"          # candidate | active | inactive
    created_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def package_dir(market: str, symbol: str, timeframe: str) -> Path:
    tf = get_settings().timeframe(timeframe).name
    return app_config.MODELS_DIR / market.lower() / symbol.upper() / tf


def make_model_id(market: str, symbol: str, timeframe: str, suffix: str = "xgb") -> str:
    tf = get_settings().timeframe(timeframe).name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{symbol.upper()}_{tf}_{suffix}_{stamp}"


class ModelRegistry:
    """Filesystem-backed registry (no database required)."""

    def __init__(self, registry_path: Path | None = None):
        self.registry_path = Path(registry_path or (app_config.MODELS_DIR / "registry.json"))

    # ---------------------------------------------------------------- index
    def _load_index(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"models": {}}
        with self.registry_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_index(self, index: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, default=str)
        tmp.replace(self.registry_path)

    # ---------------------------------------------------------------- save
    def save_package(
        self,
        models: dict[int, Any],
        metadata: ModelMetadata,
        calibrators: dict[int, Any] | None = None,
    ) -> Path:
        """Persist a package of horizon models plus its metadata."""
        target = package_dir(metadata.market, metadata.symbol, metadata.timeframe)
        target.mkdir(parents=True, exist_ok=True)

        for horizon, model_set in models.items():
            joblib.dump(model_set, target / f"horizon_{horizon}.joblib")
        for horizon, calibrator in (calibrators or {}).items():
            joblib.dump(calibrator, target / f"calibrator_h{horizon}.joblib")
        metadata.created_at = metadata.created_at or datetime.now(timezone.utc).isoformat()
        with (target / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, indent=2, default=str)

        with _LOCK:
            index = self._load_index()
            entry = metadata.to_dict()
            entry["path"] = str(target)
            entry["calibrated"] = bool(calibrators)
            index["models"][metadata.model_id] = entry
            self._save_index(index)
        logger.info("registered model %s at %s", metadata.model_id, target)
        return target

    def load_calibrators(
        self,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> dict[int, Any]:
        """Load optional per-horizon calibrators (empty dict when none exist)."""
        target = package_dir(market, symbol, timeframe)
        settings = get_settings()
        out: dict[int, Any] = {}
        for h in range(1, settings.forecast_horizon + 1):
            path = target / f"calibrator_h{h}.joblib"
            if path.exists():
                out[h] = joblib.load(path)
        return out

    # ---------------------------------------------------------------- load
    def load_package(
        self,
        market: str,
        symbol: str,
        timeframe: str,
        require_active: bool = False,
    ) -> tuple[dict[int, Any], ModelMetadata]:
        target = package_dir(market, symbol, timeframe)
        meta_path = target / "metadata.json"
        if not meta_path.exists():
            raise ModelNotFoundError(
                f"No trained model for {market}/{symbol}/{get_settings().timeframe(timeframe).name}. "
                f"Expected {meta_path}. Train one with: python -m scripts.train "
                f"--market {market} --symbol {symbol} --timeframe {timeframe}"
            )
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = ModelMetadata.from_dict(json.load(fh))

        if require_active and metadata.status != "active":
            raise ModelValidationError(
                f"Model {metadata.model_id} is not active (status={metadata.status})"
            )

        settings = get_settings()
        horizon = settings.forecast_horizon
        models: dict[int, Any] = {}
        missing = []
        for h in range(1, horizon + 1):
            path = target / f"horizon_{h}.joblib"
            if not path.exists():
                missing.append(h)
                continue
            models[h] = joblib.load(path)
        if missing:
            raise ModelNotFoundError(
                f"Model package for {market}/{symbol}/{timeframe} is incomplete; "
                f"missing horizons {missing}"
            )
        return models, metadata

    # ---------------------------------------------------------------- list
    def list(self, market: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]:
        index = self._load_index()
        rows = list(index.get("models", {}).values())
        if market:
            rows = [r for r in rows if r.get("market") == market.lower()]
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol.upper()]
        return sorted(rows, key=lambda r: r.get("created_at", ""))

    def get(self, model_id: str) -> dict[str, Any]:
        index = self._load_index()
        if model_id not in index.get("models", {}):
            raise ModelNotFoundError(f"Unknown model_id '{model_id}'")
        return index["models"][model_id]

    # ---------------------------------------------------------------- status
    def _set_status(self, model_id: str, status: str) -> None:
        with _LOCK:
            index = self._load_index()
            if model_id not in index.get("models", {}):
                raise ModelNotFoundError(f"Unknown model_id '{model_id}'")
            entry = index["models"][model_id]
            entry["status"] = status
            meta_path = Path(entry["path"]) / "metadata.json"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                meta["status"] = status
                with meta_path.open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2, default=str)
            self._save_index(index)
        logger.info("model %s status -> %s", model_id, status)

    def activate(self, model_id: str) -> None:
        """Activate a model; demotes the previously active package for the same slot."""
        entry = self.get(model_id)
        with _LOCK:
            index = self._load_index()
            for mid, other in index.get("models", {}).items():
                same_slot = (
                    other.get("market") == entry.get("market")
                    and other.get("symbol") == entry.get("symbol")
                    and other.get("timeframe") == entry.get("timeframe")
                )
                if same_slot and mid != model_id and other.get("status") == "active":
                    other["status"] = "inactive"
            index["models"][model_id]["status"] = "active"
            self._save_index(index)
        self._sync_package_status(model_id, "active")

    def deactivate(self, model_id: str) -> None:
        self._set_status(model_id, "inactive")

    def _sync_package_status(self, model_id: str, status: str) -> None:
        entry = self.get(model_id)
        meta_path = Path(entry["path"]) / "metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            meta["status"] = status
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, default=str)

    def active_for(self, market: str, symbol: str, timeframe: str) -> dict[str, Any] | None:
        tf = get_settings().timeframe(timeframe).name
        for row in self.list(market, symbol):
            if row.get("timeframe") == tf and row.get("status") == "active":
                return row
        return None
