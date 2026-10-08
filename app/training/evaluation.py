"""Out-of-sample evaluation metrics.

Classification metrics are computed on the 3-class direction problem
(UP / FLAT / DOWN); regression metrics on the future-return predictions.
No metric is reported unless it is computed from real predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.targets.ten_candle import DOWN, FLAT, UP

CLASS_LABELS = [DOWN, FLAT, UP]          # index order used by probabilities
CLASS_TO_INT = {DOWN: 0, FLAT: 1, UP: 2}


def _safe(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics for direction labels given as ints 0/1/2 (DOWN/FLAT/UP)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~pd.isna(y_true)
    y_true, y_pred = y_true[mask].astype(int), y_pred[mask].astype(int)
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if len(y_true) == 0:
        return {"n": 0}

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": _safe(accuracy_score(y_true, y_pred)),
        "precision_macro": _safe(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": _safe(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": _safe(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
    }
    # directional accuracy over rows where truth or prediction is a real direction
    directional = (y_true != 1) | (y_pred != 1)
    if directional.any():
        out["directional_accuracy"] = _safe(
            accuracy_score(y_true[directional], y_pred[directional])
        )

    if y_proba is not None and len(y_true) >= 20:
        proba = np.asarray(y_proba, dtype=float)[mask]
        proba = np.clip(proba, 1e-9, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        try:
            out["log_loss"] = _safe(log_loss(y_true, proba, labels=[0, 1, 2]))
        except ValueError:
            out["log_loss"] = None
        try:
            brier = 0.0
            for cls in (0, 1, 2):
                truth = (y_true == cls).astype(float)
                brier += np.mean((proba[:, cls] - truth) ** 2)
            out["brier_score"] = _safe(brier / 3.0)
        except Exception:
            out["brier_score"] = None
        try:
            if len(np.unique(y_true)) > 1:
                out["roc_auc_ovr"] = _safe(roc_auc_score(y_true, proba, multi_class="ovr", labels=[0, 1, 2]))
        except ValueError:
            out["roc_auc_ovr"] = None
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0}
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "n": int(len(y_true)),
        "mae": _safe(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe(np.sqrt(mse)),
        "mean_actual_return": _safe(np.mean(y_true)),
        "mean_predicted_return": _safe(np.mean(y_pred)),
    }
    # hit-rate of the sign of the predicted return
    nonzero = (y_true != 0) & (y_pred != 0)
    if nonzero.any():
        out["sign_accuracy"] = _safe(np.mean(np.sign(y_true[nonzero]) == np.sign(y_pred[nonzero])))
    return out


def evaluate_horizon(
    y_true_dir: np.ndarray,
    y_pred_dir: np.ndarray,
    y_proba: np.ndarray | None,
    y_true_ret: np.ndarray,
    y_pred_ret: np.ndarray,
) -> dict[str, Any]:
    return {
        "direction": classification_metrics(y_true_dir, y_pred_dir, y_proba),
        "return": regression_metrics(y_true_ret, y_pred_ret),
    }
