"""Backtest metrics: prediction quality AND trade economics.

Everything is computed from real prediction/outcome pairs; nothing is assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.training.evaluation import classification_metrics, regression_metrics


def trading_metrics(pnl: np.ndarray, costs_paid: np.ndarray | None = None) -> dict[str, Any]:
    """Metrics of a trade-return stream (one value per trade, net of costs)."""
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    n = int(len(pnl))
    if n == 0:
        return {"n_trades": 0, "note": "no trades generated under current thresholds"}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    expectancy = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0

    # equity curve / drawdown
    equity = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min()) if n else 0.0

    sharpe = float(pnl.mean() / std * np.sqrt(252)) if std > 0 else None
    downside = pnl[pnl < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(pnl.mean() / dstd * np.sqrt(252)) if dstd > 0 else None

    return {
        "n_trades": n,
        "total_return": round(float((equity[-1] - 1.0) * 100), 4),
        "win_rate": round(len(wins) / n, 6),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "average_win": round(float(wins.mean()), 8) if len(wins) else 0.0,
        "average_loss": round(float(losses.mean()), 8) if len(losses) else 0.0,
        "expectancy": round(expectancy, 8),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 4),
        "std": round(std, 8),
        "gross_win": round(gross_win, 8),
        "gross_loss": round(gross_loss, 8),
        "total_costs_paid": round(float(costs_paid.sum()), 8) if costs_paid is not None else None,
    }


def prediction_metrics(
    y_true_dir: np.ndarray,
    y_pred_dir: np.ndarray,
    y_proba: np.ndarray | None,
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
) -> dict[str, Any]:
    return {
        "classification": classification_metrics(y_true_dir, y_pred_dir, y_proba),
        "regression": regression_metrics(y_true_ret, y_pred_ret),
    }


def performance_summary(equity_curve: pd.Series) -> dict[str, Any]:
    if equity_curve.empty:
        return {}
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return {
        "final_equity": round(float(equity_curve.iloc[-1]), 6),
        "max_drawdown_pct": round(float(dd.min() * 100), 4),
        "n_points": int(len(equity_curve)),
    }
