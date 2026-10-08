"""Risk engine: entry / stop-loss / take-profit / position sizing.

ATR-based, fully configurable, and strictly non-executing - it only *suggests*
levels for a manual or downstream workflow.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.signals.engine import BUY, SELL


def build_risk_plan(
    signal: str,
    entry_price: float,
    atr: float,
    equity: float | None = None,
    settings: Settings | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    cfg = dict(settings.risk.get("risk", {}))
    if timeframe:
        cfg.update((cfg.get("timeframe_overrides") or {}).get(timeframe, {}))
    sl_mult = float(cfg.get("stop_loss_atr", 1.5))
    tp_mult = float(cfg.get("take_profit_atr", 3.0))
    risk_pct = float(cfg.get("risk_per_trade", 0.01))
    max_pos = float(cfg.get("max_position_pct", 0.25))
    rr_min = float(cfg.get("reward_risk_min", 1.0))
    equity = float(equity if equity is not None else 10_000.0)

    if signal not in {"BUY", "SELL"} or atr <= 0:
        return {
            "signal": signal,
            "actionable": False,
            "entry": entry_price,
            "note": "no trade planned (WAIT or missing ATR)",
        }

    direction = 1 if signal == BUY else -1
    stop_loss = entry_price - direction * sl_mult * atr
    take_profit = entry_price + direction * tp_mult * atr
    risk_per_unit = abs(entry_price - stop_loss)
    reward_per_unit = abs(take_profit - entry_price)
    reward_risk = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

    risk_amount = equity * risk_pct
    position_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
    max_units = (equity * max_pos) / entry_price if entry_price > 0 else 0.0
    position_units = min(position_units, max_units)

    return {
        "signal": signal,
        "actionable": reward_risk >= rr_min,
        "entry": round(entry_price, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "risk_reward": round(reward_risk, 4),
        "atr": round(atr, 6),
        "atr_multipliers": {"stop_loss": sl_mult, "take_profit": tp_mult},
        "position_size_units": round(position_units, 8),
        "position_size_notional": round(position_units * entry_price, 2),
        "risk_amount": round(risk_amount, 2),
        "equity": equity,
        "note": "suggestion only - this system does not execute orders",
    }
