"""Build and persist the feature matrix for a symbol/timeframe.

Example::

    python -m scripts.build_features --market crypto --symbol BTCUSDT --timeframe 1m
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from app.core import config as app_config
from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.data.base import get_data
from app.data.validator import DataQualityValidator
from app.features.builder import build_features, feature_columns
from app.targets.ten_candle import build_targets

logger = get_logger("scripts.build_features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the feature matrix")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    args = parser.parse_args(argv)

    try:
        candles = get_data(args.market, args.symbol, args.timeframe)
        report = DataQualityValidator().validate(candles, symbol=args.symbol.upper(), timeframe=args.timeframe)
        report.raise_if_failed()
        frame = build_features(candles, args.market, args.timeframe)
        frame = pd.concat([frame, build_targets(frame)], axis=1)
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    out_dir = app_config.PROCESSED_DATA_DIR / args.market.lower() / args.symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.timeframe}_features.parquet"
    frame.to_parquet(path, index=False)
    logger.info("wrote %s rows / %s cols -> %s", len(frame), len(frame.columns), path)
    print(f"OK: {len(frame)} rows, {len(feature_columns(frame))} features -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
