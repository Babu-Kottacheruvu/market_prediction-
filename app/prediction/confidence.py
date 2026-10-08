"""Confidence scoring for forecasts.

Confidence is never asserted to be a real-world accuracy until calibration has
been evaluated (see app/models/calibration.py).  Without a calibrator the
confidence is the raw maximum class probability scaled by cross-horizon
agreement.
"""

from __future__ import annotations

import numpy as np


def horizon_confidence(proba: np.ndarray, calibrated: np.ndarray | None = None) -> float:
    """Confidence for a single horizon: max class probability."""
    p = np.asarray(calibrated if calibrated is not None else proba, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    total = p.sum()
    if total <= 0:
        return 0.0
    return round(float(p.max() / total), 6)


def overall_confidence(horizon_confidences: list[float], directions: list[int]) -> dict[str, float]:
    """Aggregate horizon confidences, penalizing disagreement across horizons."""
    if not horizon_confidences:
        return {"confidence": 0.0, "agreement": 0.0}
    conf = np.asarray(horizon_confidences, dtype=float)
    dirs = np.asarray(directions, dtype=int)
    dominant = 1
    counts = {c: int((dirs == c).sum()) for c in set(dirs.tolist())}
    dominant = max(counts, key=counts.get)
    agreement = counts[dominant] / len(dirs)
    # weight: horizons closer to the present carry more actionable weight
    weights = 1.0 / np.arange(1, len(conf) + 1)
    weights /= weights.sum()
    base = float((conf * weights).sum())
    return {
        "confidence": round(base * (0.5 + 0.5 * agreement), 6),
        "agreement": round(agreement, 6),
        "dominant_direction": int(dominant),
    }
