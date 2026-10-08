"""Backtest engine.

Consumes *out-of-sample* walk-forward predictions (produced by training) so
that the engine can never be fed in-sample numbers.  Costs, spread and
slippage are always applied - especially important on 1-minute data where a
51% accurate model can still lose money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.costs import TransactionCosts
from app.backtesting.metrics import prediction_metrics, trading_metrics
from app.core import config as app_config
from app.core.config import Settings, get_settings
from app.core.exceptions import BacktestError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    horizon: int = 1
    min_probability: float = 0.65
    min_confidence: float = 0.55
    cost_multiplier: float = 1.0     # stress costs (e.g. 2.0 = double)
    allow_flat: bool = False         # take FLAT predictions as trades
    extra: dict[str, Any] = field(default_factory=dict)


def load_oos_predictions(market: str, symbol: str, timeframe: str) -> pd.DataFrame:
    tf = get_settings().timeframe(timeframe).name
    path = app_config.PROCESSED_DATA_DIR / market.lower() / symbol.upper() / f"{tf}_oos_predictions.parquet"
    if not path.exists():
        raise BacktestError(
            f"No out-of-sample predictions at {path}. "
            f"Run: python -m scripts.train --market {market} --symbol {symbol} --timeframe {tf}"
        )
    return pd.read_parquet(path)


class BacktestEngine:
    def __init__(
        self,
        market: str,
        symbol: str,
        timeframe: str,
        config: BacktestConfig | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.market = market.lower()
        self.symbol = self.settings.resolve_symbol(market, symbol)
        self.timeframe = self.settings.timeframe(timeframe).name
        self.config = config or BacktestConfig()
        self.costs = TransactionCosts.for_market(self.market, self.timeframe)
        if self.config.cost_multiplier != 1.0:
            self.costs = TransactionCosts(
                spread=self.costs.spread * self.config.cost_multiplier,
                commission=self.costs.commission * self.config.cost_multiplier,
                slippage=self.costs.slippage * self.config.cost_multiplier,
            )

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        oos = load_oos_predictions(self.market, self.symbol, self.timeframe)
        h = self.config.horizon
        required = [f"dir_h{h}", f"ret_h{h}", f"p_up_h{h}", f"p_down_h{h}",
                    f"p_flat_h{h}", f"target_direction_h{h}", f"target_return_h{h}"]
        missing = [c for c in required if c not in oos.columns]
        if missing:
            raise BacktestError(f"OOS predictions are missing columns: {missing}")
        if len(oos) < 50:
            raise BacktestError(f"Only {len(oos)} out-of-sample rows; backtest would be meaningless")

        pred_dir = oos[f"dir_h{h}"].to_numpy()          # 0/1/2 -> DOWN/FLAT/UP
        true_dir = oos[f"target_direction_h{h}"].to_numpy() + 1
        proba = np.column_stack([
            oos[f"p_down_h{h}"].to_numpy(),
            oos[f"p_flat_h{h}"].to_numpy(),
            oos[f"p_up_h{h}"].to_numpy(),
        ])
        true_ret = oos[f"target_return_h{h}"].to_numpy()
        pred_ret = oos[f"ret_h{h}"].to_numpy()

        # ---------------- signal filtering ----------------
        max_p = proba.max(axis=1)
        confidence = max_p
        direction_side = np.where(pred_dir == 2, 1, np.where(pred_dir == 0, -1, 0))
        if self.config.allow_flat:
            active = (max_p >= self.config.min_probability) | (direction_side == 0)
        else:
            active = (max_p >= self.config.min_probability) & (direction_side != 0)
        active &= confidence >= self.config.min_confidence

        edge_ok = np.abs(pred_ret) >= self.costs.round_trip * self.config.cost_multiplier
        active &= edge_ok

        cost = self.costs.round_trip * self.config.cost_multiplier
        gross = direction_side.astype(float) * true_ret
        pnl = np.where(active, gross - cost, 0.0)
        costs_paid = np.where(active, cost, 0.0)

        # ---------------- metrics ----------------
        pred_metrics = prediction_metrics(true_dir, pred_dir, proba, true_ret, pred_ret)
        trade_metrics = trading_metrics(pnl[active], costs_paid[active])

        # counterfactual: same trades without any costs
        pnl_no_cost = np.where(active, gross, 0.0)
        no_cost = trading_metrics(pnl_no_cost[active])

        by_regime: dict[str, Any] = {}
        if "regime" in oos.columns:
            for regime, grp in oos.groupby("regime"):
                idx = grp.index.to_numpy()
                sub = pnl[idx]
                by_regime[str(regime)] = trading_metrics(sub[sub != 0])

        result = {
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": h,
            "n_oos_rows": int(len(oos)),
            "n_trades": int(active.sum()),
            "costs": self.costs.to_dict(),
            "config": {
                "min_probability": self.config.min_probability,
                "min_confidence": self.config.min_confidence,
                "cost_multiplier": self.config.cost_multiplier,
            },
            "prediction_metrics": pred_metrics,
            "trading_metrics": trade_metrics,
            "trading_metrics_without_costs": no_cost,
            "by_regime": by_regime,
            "oos_window": {
                "start": str(pd.to_datetime(oos["timestamp"]).min()),
                "end": str(pd.to_datetime(oos["timestamp"]).max()),
            },
        }

        edge = (no_cost.get("expectancy") or 0.0) - (trade_metrics.get("expectancy") or 0.0)
        result["edge_after_costs"] = round(float(edge), 8)
        result["profitable_after_costs"] = bool(
            (trade_metrics.get("expectancy") or 0.0) > 0
            and (trade_metrics.get("n_trades") or 0) > 0
        )
        logger.info(
            "backtest %s/%s/%s h=%s | trades=%s | expectancy=%s | profitable=%s",
            self.market, self.symbol, self.timeframe, h,
            trade_metrics.get("n_trades"), trade_metrics.get("expectancy"),
            result["profitable_after_costs"],
        )
        return result
