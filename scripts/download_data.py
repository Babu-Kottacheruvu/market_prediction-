"""Download real historical candles (MetaTrader 5 is the default source).

Examples::

    python -m scripts.download_data --market forex --symbol EURUSD --timeframe 1h --bars 8760
    python -m scripts.download_data --market commodities --symbol XAUUSD --timeframe 1h --bars 8760
    python -m scripts.download_data --market crypto --symbol BTCUSDT --timeframe 1h --bars 1000

All markets route through the provider configured in ``config/markets.yaml``
(``provider: mt5`` by default).  The MT5 terminal must be installed, running and
logged in; the API attaches to it.  Symbols are broker-dependent — a symbol the
broker does not offer raises an actionable error (no data is ever fabricated).
"""

from __future__ import annotations

import argparse
import sys

from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.data.base import DataProviderFactory

logger = get_logger("scripts.download_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download real market data")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--start", default=None, help="ISO date, e.g. 2025-01-01")
    parser.add_argument("--end", default=None, help="ISO date")
    parser.add_argument("--bars", type=int, default=None, help="approximate number of candles back")
    args = parser.parse_args(argv)

    try:
        provider = DataProviderFactory.get(args.market)
        if args.bars and not args.start:
            from app.core.config import get_settings

            minutes = get_settings().timeframe(args.timeframe).minutes
            import pandas as pd

            args.start = str(
                pd.Timestamp.now("UTC") - pd.Timedelta(minutes=minutes * args.bars)
            )
        df = provider.download(args.symbol, args.timeframe, start=args.start, end=args.end)
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("downloaded %s rows for %s/%s/%s", len(df), args.market, args.symbol, args.timeframe)
    print(f"OK: {len(df)} candles -> data/raw/{args.market.lower()}/{args.symbol.upper()}/{args.timeframe}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
