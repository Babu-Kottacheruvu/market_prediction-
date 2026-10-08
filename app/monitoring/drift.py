"""Post-training monitoring: feature drift, target drift, prediction stability.

Compares recent data against the model's training window using the
Population Stability Index (PSI). A PSI above the configured threshold
flags a feature as drifted, which means the model's inputs no longer
match what it was trained on.

Drift is a *warning*, not a failure: it never changes predictions and
never fabricates metrics. It tells you when it is time to retrain.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

CLASSES = (0, 1, 2)  # DOWN, FLAT, UP (model space)


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.

    Bins are the quantiles of `expected`, so every bin carries an equal
    share of the reference distribution (the standard PSI convention).
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) < 10 or len(act) < 10:
        return float("nan")

    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        # degenerate (near-constant) feature: fall back to a single mid split
        edges = np.array([exp.min() - 1e-9, np.median(exp), exp.max() + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts = np.histogram(exp, bins=edges)[0].astype(float)
    act_counts = np.histogram(act, bins=edges)[0].astype(float)
    exp_p = exp_counts / max(exp_counts.sum(), 1.0)
    act_p = act_counts / max(act_counts.sum(), 1.0)

    eps = 1e-6
    exp_p = np.clip(exp_p, eps, None)
    act_p = np.clip(act_p, eps, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def categorical_psi(
    expected: Sequence[Any],
    actual: Sequence[Any],
    categories: Iterable[Any] | None = None,
) -> float:
    """PSI between two discrete distributions (e.g. predicted direction)."""
    exp = pd.Series(expected).astype(str)
    act = pd.Series(actual).astype(str)
    if len(exp) < 10 or len(act) < 10:
        return float("nan")
    cats = list(dict.fromkeys(categories)) if categories is not None else sorted(
        set(exp.unique()) | set(act.unique())
    )
    exp_p = exp.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy(float)
    act_p = act.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy(float)
    eps = 1e-6
    exp_p = np.clip(exp_p, eps, None)
    act_p = np.clip(act_p, eps, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def feature_drift(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    feature_cols: Sequence[str],
    bins: int = 10,
    threshold: float = 0.20,
) -> dict[str, Any]:
    """PSI for every feature; `drifted` lists those above `threshold`."""
    values: dict[str, float] = {}
    for col in feature_cols:
        if col not in reference.columns or col not in recent.columns:
            continue
        score = psi(reference[col].to_numpy(), recent[col].to_numpy(), bins=bins)
        if math.isfinite(score):
            values[col] = round(score, 4)
    drifted = sorted(
        (c for c, s in values.items() if s >= threshold),
        key=lambda c: values[c],
        reverse=True,
    )
    top = sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "threshold": threshold,
        "features_checked": len(values),
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "max_psi": round(max(values.values()), 4) if values else None,
        "top_drift": [{"feature": f, "psi": s} for f, s in top],
        "psi_by_feature": values,
    }


def target_drift(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    horizon: int = 1,
) -> dict[str, Any]:
    """Realized direction distribution shift (only rows with a known target)."""
    col = f"target_direction_h{horizon}"
    if col not in reference.columns or col not in recent.columns:
        return {"available": False}
    exp = reference[col].dropna()
    act = recent[col].dropna()
    if len(act) < 10:
        return {"available": False, "reason": "not enough labeled recent rows"}
    score = categorical_psi(exp.to_numpy(), act.to_numpy(), categories=list(CLASSES))
    return {
        "available": True,
        "psi": round(score, 4) if math.isfinite(score) else None,
        "reference_share": exp.value_counts(normalize=True).to_dict(),
        "recent_share": act.value_counts(normalize=True).to_dict(),
        "drifted": bool(math.isfinite(score) and score >= 0.20),
    }


def build_monitoring_report(
    market: str,
    symbol: str,
    timeframe: str,
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    feature_cols: Sequence[str],
    settings: Settings | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the full drift report for one model."""
    settings = settings or get_settings()
    mon = settings.models.get("monitoring", {}) or {}
    bins = int(mon.get("psi_bins", 10))
    threshold = float(mon.get("psi_threshold", 0.20))
    min_rows = int(mon.get("min_recent_rows", 100))

    report: dict[str, Any] = {
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "model_id": model_id,
        "n_reference": int(len(reference)),
        "n_recent": int(len(recent)),
    }
    if len(recent) < min_rows:
        report["status"] = "insufficient_recent_data"
        report["reason"] = (
            f"{len(recent)} recent rows < min_recent_rows={min_rows}; "
            "collect more data before monitoring drift"
        )
        return report

    drift = feature_drift(reference, recent, feature_cols, bins=bins, threshold=threshold)
    targets = target_drift(reference, recent, horizon=1)
    report["status"] = "drift_detected" if drift["n_drifted"] else "healthy"
    report["feature_drift"] = drift
    report["target_drift"] = targets
    report["retrain_recommended"] = drift["n_drifted"] > 0
    return report
