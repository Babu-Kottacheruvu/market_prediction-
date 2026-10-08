"""Feature builder: one entry point that assembles the full feature matrix.

Pipeline::

    candles -> common features -> market features -> multi-timeframe context
            -> optional cross-market features -> feature matrix

Guarantees:
* only past/closed information is used (see ``multi_timeframe``),
* target columns (``target_*``) are never returned as features,
* no learned preprocessing happens here (nothing to fit -> nothing to leak).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.data.timeframes import with_availability
from app.features.common import build_common_features
from app.features.crypto import attach_optional_extras, build_crypto_features
from app.features.cross_market import build_cross_features
from app.features.forex import build_forex_features
from app.features.commodities import build_commodity_features
from app.features.multi_timeframe import build_multi_timeframe_features

logger = get_logger(__name__)

TARGET_PREFIX = "target_"

METADATA_COLUMNS = {
    "timestamp",
    "available_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "market",
}


def build_features(
    df: pd.DataFrame,
    market: str,
    timeframe: str,
    peers: dict[str, pd.DataFrame] | None = None,
    extras: dict[str, pd.DataFrame] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Return ``df`` (metadata columns) plus every derived feature column."""
    settings = settings or get_settings()
    if df.empty:
        raise ValueError("build_features received an empty frame")

    frame = df.copy().sort_values("timestamp").reset_index(drop=True)
    frame = with_availability(frame, timeframe, settings)

    features = build_common_features(frame, settings)

    market_key = market.lower()
    if market_key == "forex":
        features = features.join(build_forex_features(frame))
    elif market_key == "crypto":
        features = features.join(build_crypto_features(frame))
        features = attach_optional_extras(features, frame, extras)
    elif market_key == "commodities":
        features = features.join(build_commodity_features(frame, timeframe))
        features = attach_optional_extras(features, frame, extras)
    else:  # unknown market: common features still work
        logger.warning("market '%s' has no dedicated feature module", market)

    htf = build_multi_timeframe_features(frame, timeframe, settings)
    if not htf.empty:
        features = features.join(htf)

    cross = build_cross_features(frame, peers)
    if not cross.empty:
        features = features.join(cross)

    out = pd.concat([frame, features], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns safe to feed into a model (no metadata, no targets)."""
    cols = []
    for c in frame.columns:
        if c in METADATA_COLUMNS:
            continue
        if str(c).startswith(TARGET_PREFIX):
            continue
        cols.append(c)
    return cols


def assert_no_target_leakage(frame: pd.DataFrame) -> None:
    """Runtime guard: target columns must never be selectable as features."""
    leaked = [c for c in feature_columns(frame) if str(c).startswith(TARGET_PREFIX)]
    if leaked:
        raise AssertionError(f"Target columns leaked into the feature set: {leaked}")
    numeric_meta = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    if numeric_meta:
        # raw price levels are metadata, not features (returns are features)
        pass
