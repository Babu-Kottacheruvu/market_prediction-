"""Monitor a trained model for data drift since its training window.

Example::

    python -m scripts.monitor --market crypto --symbol BTCUSDT --timeframe 1m
    python -m scripts.monitor --market crypto --symbol BTCUSDT --timeframe 1m --fail-on-drift

Exit codes: 0 healthy (or not checking), 1 drift detected with --fail-on-drift,
2 error (no model / no data).
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.data import storage
from app.features.builder import build_features, feature_columns
from app.models.router import ModelRouter
from app.monitoring.drift import build_monitoring_report
from app.targets.ten_candle import add_targets

logger = get_logger("scripts.monitor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a trained model for data drift")
    parser.add_argument("--market", required=True, choices=["crypto", "forex", "commodities"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--lookback-days", type=int, default=None,
                        help="window of recent data to compare (default: monitoring.lookback_days)")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="exit with code 1 when drift is detected")
    args = parser.parse_args(argv)

    router = ModelRouter()
    status = router.model_status(args.market, args.symbol, args.timeframe)
    if not status.get("trained"):
        logger.error(
            "no trained model for %s/%s/%s - train first: python -m scripts.train "
            "--market %s --symbol %s --timeframe %s",
            args.market, args.symbol, args.timeframe,
            args.market, args.symbol, args.timeframe,
        )
        return 2

    try:
        candles = storage.load_candles(args.market, args.symbol, args.timeframe)
        frame = add_targets(build_features(candles, args.market, args.timeframe))
    except PlatformError as exc:
        logger.error("%s", exc)
        return 2

    settings = get_settings()
    train_start = pd.Timestamp(status.get("training_start"))
    train_end = pd.Timestamp(status.get("training_end"))
    ts = frame["timestamp"]
    reference = frame[(ts >= train_start) & (ts <= train_end)]

    mon = settings.models.get("monitoring", {}) or {}
    lookback = args.lookback_days or int(mon.get("lookback_days", 7))
    after = frame[ts > train_end]
    if after.empty:
        recent = after
    else:
        cutoff = after["timestamp"].max() - pd.Timedelta(days=lookback)
        recent = after[after["timestamp"] >= cutoff]

    report = build_monitoring_report(
        args.market,
        args.symbol,
        args.timeframe,
        reference=reference,
        recent=recent,
        feature_cols=feature_columns(frame),
        settings=settings,
        model_id=status.get("model_id"),
    )
    print(json.dumps(report, indent=2, default=str))

    label = report.get("status")
    drift_info = report.get("feature_drift", {})
    print(
        f"\nOK: {args.symbol} {args.timeframe} status={label} "
        f"drifted_features={drift_info.get('n_drifted', 0)} "
        f"max_psi={drift_info.get('max_psi')}",
        file=sys.stderr,
    )
    if args.fail_on_drift and label == "drift_detected":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
