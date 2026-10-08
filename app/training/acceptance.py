"""Model acceptance gates: a model is only promoted if it passes these."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings


def check_acceptance(oos_metrics: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Evaluate out-of-sample metrics against the configured gates."""
    settings = settings or get_settings()
    gates = settings.models.get("acceptance", {})
    agg = oos_metrics.get("aggregate", {})
    reasons: list[str] = []

    n = agg.get("n_oos_rows") or 0
    min_n = int(gates.get("min_samples", 200))
    if n < min_n:
        reasons.append(f"out-of-sample rows {n} < required {min_n}")

    min_acc = float(gates.get("min_directional_accuracy", 0.5))
    acc = agg.get("mean_accuracy")
    if acc is None:
        reasons.append("accuracy unavailable")
    elif acc < min_acc:
        reasons.append(f"mean accuracy {acc} < required {min_acc}")

    if gates.get("require_baseline_accuracy_improvement", True):
        accuracy_lift = agg.get("mean_accuracy_lift_vs_baseline")
        if accuracy_lift is None:
            reasons.append("baseline direction comparison unavailable")
        elif accuracy_lift <= 0:
            reasons.append(
                f"mean accuracy did not beat persistence baseline "
                f"(lift={accuracy_lift})"
            )

    if gates.get("require_baseline_price_improvement", True):
        mae_improvement = agg.get("mean_mae_improvement_vs_baseline")
        if mae_improvement is None:
            reasons.append("baseline price comparison unavailable")
        elif mae_improvement <= 0:
            reasons.append(
                f"mean price MAE did not beat zero-return baseline "
                f"(improvement={mae_improvement})"
            )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "gates": gates,
        "aggregate": agg,
    }
