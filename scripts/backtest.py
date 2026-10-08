"""Run a backtest on out-of-sample predictions.

Example::

    python -m scripts.backtest --market crypto --symbol BTCUSDT --timeframe 1m --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.core.exceptions import PlatformError
from app.core.logging import get_logger

logger = get_logger("scripts.backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest out-of-sample predictions")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-probability", type=float, default=0.65)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--cost-multiplier", type=float, default=1.0)
    args = parser.parse_args(argv)

    config = BacktestConfig(
        horizon=args.horizon,
        min_probability=args.min_probability,
        min_confidence=args.min_confidence,
        cost_multiplier=args.cost_multiplier,
    )
    try:
        engine = BacktestEngine(args.market, args.symbol, args.timeframe, config)
        result = engine.run()
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    print(json.dumps(result, indent=2, default=str))
    tm = result["trading_metrics"]
    print(
        f"\nOK: {result['n_trades']} trades | expectancy={tm.get('expectancy')} | "
        f"profitable_after_costs={result['profitable_after_costs']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
