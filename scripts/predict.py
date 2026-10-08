"""Produce a 10-candle forecast.

Example::

    python -m scripts.predict --market crypto --symbol BTCUSDT --timeframe 1m
"""

from __future__ import annotations

import argparse
import json
import sys

from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.prediction.predictor import predict

logger = get_logger("scripts.predict")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the next 10 candles")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--equity", type=float, default=None, help="equity for position sizing")
    args = parser.parse_args(argv)

    try:
        result = predict(args.market, args.symbol, args.timeframe, equity=args.equity)
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    print(json.dumps(result, indent=2, default=str))
    m = result["model"]
    print(
        f"\nOK: {result['symbol']} {result['timeframe']} | price={result['current_price']} | "
        f"bias={result['overall_bias']} | signal={result['signal']} | regime={result['regime']} | "
        f"model={m['model_id']} ({m['status']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
