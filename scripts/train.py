"""Train walk-forward XGBoost models.

Example::

    python -m scripts.train --market crypto --symbol BTCUSDT --timeframe 1m
"""

from __future__ import annotations

import argparse
import json
import sys

from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.training.trainer import train
from app.training.walk_forward import WalkForwardConfig

logger = get_logger("scripts.train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train 10-horizon XGBoost models")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--train-days", type=int, default=None)
    parser.add_argument("--val-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--no-activate", action="store_true", help="do not activate even if accepted")
    args = parser.parse_args(argv)

    wf = WalkForwardConfig.from_settings()
    if args.train_days:
        wf = WalkForwardConfig(**{**wf.__dict__, "train_days": args.train_days})
    if args.val_days:
        wf = WalkForwardConfig(**{**wf.__dict__, "validation_days": args.val_days})
    if args.test_days:
        wf = WalkForwardConfig(**{**wf.__dict__, "test_days": args.test_days})
    if args.step_days:
        wf = WalkForwardConfig(**{**wf.__dict__, "step_days": args.step_days})

    try:
        report = train(
            args.market,
            args.symbol,
            args.timeframe,
            walk_forward=wf,
            activate=not args.no_activate,
        )
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    payload = report.to_dict()
    print(json.dumps(payload, indent=2, default=str))
    print(
        f"\nOK: model {payload['model_id']} status={payload['status']} "
        f"OOS accuracy={payload['out_of_sample_metrics']['aggregate'].get('mean_accuracy')} "
        f"brier={payload['out_of_sample_metrics']['aggregate'].get('mean_brier')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
