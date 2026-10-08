"""Cross-market features (optional).

Peers are joined backwards in time: a peer candle is only used when it was
available at or before the base candle timestamp.
"""

from __future__ import annotations

import pandas as pd

from app.core.logging import get_logger
from app.data.timeframes import utc_ns
from app.features.technical import ema, rsi

logger = get_logger(__name__)


def build_cross_features(
    base: pd.DataFrame,
    peers: dict[str, pd.DataFrame] | None,
    periods: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Relative-strength features versus available peer instruments.

    ``peers`` maps e.g. ``{"GBPUSD": frame, "EURJPY": frame}`` where each frame
    is normalized OHLCV at the *same* timeframe as ``base``.
    """
    out = pd.DataFrame(index=base.index)
    if not peers:
        return out

    left = base[["timestamp", "close"]].reset_index(drop=True)
    for name, frame in peers.items():
        if frame is None or frame.empty:
            continue
        right = frame[["timestamp", "close"]].sort_values("timestamp")
        right = right.rename(columns={"close": f"peer_close"})
        merged = pd.merge_asof(
            left.assign(timestamp=utc_ns(left["timestamp"])),
            right.assign(timestamp=utc_ns(right["timestamp"])),
            left_on="timestamp",
            right_on="timestamp",
            direction="backward",
        )
        peer_close = merged["peer_close"]
        for p in periods:
            out[f"peer_{name}_ret_{p}"] = peer_close.pct_change(p)
        out[f"peer_{name}_ema20_dist"] = (peer_close - ema(peer_close, 20)) / peer_close
        out[f"peer_{name}_rsi"] = rsi(peer_close, 14)
        out[f"rel_strength_{name}"] = merged["close"].reset_index(drop=True).pct_change(20) - peer_close.pct_change(20)
        out.index = base.index
    logger.info("cross features built for peers: %s", list(peers))
    return out
