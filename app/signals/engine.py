"""Signal engine: forecast -> BUY / SELL / WAIT.

All thresholds come from ``config/risk.yaml`` - nothing is hardcoded.
The signal is advisory only; this platform never executes orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings

BUY = "BUY"
SELL = "SELL"
WAIT = "WAIT"


def _signal_config(settings: Settings, timeframe: str) -> dict[str, Any]:
    cfg = dict(settings.risk.get("signal", {}))
    override = (cfg.get("timeframe_overrides") or {}).get(timeframe, {})
    cfg.update(override)
    return cfg


def round_trip_cost(costs: dict[str, float]) -> float:
    """Total round-trip cost as a fraction of price."""
    return float(costs.get("spread", 0.0)) + float(costs.get("commission", 0.0)) + float(
        costs.get("slippage", 0.0)
    )


def generate_signal(
    forecast: list[dict[str, Any]],
    regime: str,
    market: str,
    timeframe: str,
    overall: dict[str, Any],
    settings: Settings | None = None,
    *,
    model_status: str | None = None,
    data_source: str | None = None,
    data_timestamp: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Convert a forecast into an advisory signal, failing closed for scalp setups."""
    settings = settings or get_settings()
    cfg = _signal_config(settings, timeframe)
    costs = settings.costs(market, timeframe)
    cost = round_trip_cost(costs)

    min_prob = float(cfg.get("min_probability", 0.65))
    min_conf = float(cfg.get("min_confidence", 0.60))
    edge_factor = float(cfg.get("min_expected_return_over_cost", 1.0))
    allowed_regimes = set(cfg.get("allowed_regimes", []))
    scalp_enabled = bool(cfg.get("scalp_enabled", False))

    if not forecast:
        return {
            "signal": WAIT,
            "reasons": ["no forecast"],
            "cost": cost,
            "probability": 0.0,
            "min_probability": min_prob,
            "scalp": scalp_enabled,
        }

    near = forecast[0]
    p_up = near["probability_up"]
    p_down = near["probability_down"]
    exp_ret = near["expected_return"]
    confidence = near["confidence"]

    reasons: list[str] = []
    if regime not in allowed_regimes:
        reasons.append(f"regime {regime} not allowed")
    if confidence < min_conf:
        reasons.append(f"confidence {confidence:.3f} < {min_conf}")
    if max(p_up, p_down) < min_prob:
        reasons.append(f"max probability {max(p_up, p_down):.3f} < {min_prob}")
    required_edge = cost * edge_factor
    if abs(exp_ret) < required_edge:
        reasons.append(
            f"expected return {exp_ret:.6f} below required edge {required_edge:.6f} "
            f"(round-trip cost {cost:.6f})"
        )

    if scalp_enabled:
        scalp_horizons = int(cfg.get("scalp_horizons", 3))
        min_agreement = float(cfg.get("scalp_min_directional_agreement", 0.67))
        sample = forecast[:scalp_horizons]
        directional_side = "UP" if p_up >= p_down else "DOWN"
        agreed = sum(row["direction"] == directional_side for row in sample)
        agreement = agreed / len(sample) if sample else 0.0
        if agreement < min_agreement:
            reasons.append(
                f"near-term forecast agreement {agreement:.0%} < {min_agreement:.0%}"
            )

        if model_status is not None and model_status != "active":
            reasons.append(f"model status {model_status} is not validated for scalp entries")
        if data_source is not None and data_source == "stored":
            reasons.append("live market data is required for scalp entries")
        if data_timestamp is not None:
            candle_time = datetime.fromisoformat(data_timestamp.replace("Z", "+00:00"))
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            current_time = now or datetime.now(timezone.utc)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            max_age = float(cfg.get("scalp_max_candle_age_bars", 2.5))
            candle_minutes = settings.timeframe(timeframe).minutes
            age_minutes = (current_time - candle_time).total_seconds() / 60.0
            if age_minutes > candle_minutes * max_age:
                reasons.append(
                    f"latest candle is stale ({age_minutes:.1f} minutes old; "
                    f"limit {candle_minutes * max_age:.1f})"
                )

    if reasons:
        return {
            "signal": WAIT,
            "reasons": reasons,
            "cost": cost,
            "required_edge": required_edge,
            "probability_up": p_up,
            "probability_down": p_down,
            "expected_return": exp_ret,
            "probability": max(p_up, p_down),
            "min_probability": min_prob,
            "scalp": scalp_enabled,
        }

    signal = BUY if p_up >= p_down else SELL
    return {
        "signal": signal,
        "reasons": ["all thresholds met"],
        "cost": cost,
        "required_edge": required_edge,
        "probability_up": p_up,
        "probability_down": p_down,
        "expected_return": exp_ret,
        "probability": max(p_up, p_down),
        "min_probability": min_prob,
        "scalp": scalp_enabled,
        "directional_agreement": agreement if scalp_enabled else None,
        "thresholds": {
            "min_probability": min_prob,
            "min_confidence": min_conf,
            "min_expected_return_over_cost": edge_factor,
            **(
                {
                    "scalp_horizons": scalp_horizons,
                    "scalp_min_directional_agreement": min_agreement,
                }
                if scalp_enabled
                else {}
            ),
        },
    }
