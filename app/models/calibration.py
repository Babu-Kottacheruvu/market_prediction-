"""Optional probability calibration (Platt scaling / isotonic regression).

Raw classifier probabilities are NOT assumed to be real-world probabilities.
When calibration is enabled the calibrator is fitted on the *first half* of the
out-of-sample predictions and evaluated on the *second half*, so the reported
calibration metrics stay out-of-sample for the calibrator as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

Method = Literal["platt", "isotonic"]


@dataclass
class ProbabilityCalibrator:
    method: Method = "isotonic"
    n_classes: int = 3
    _models: list[Any] | None = None

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit one calibrator per class (one-vs-rest), then renormalize."""
        proba = np.asarray(proba, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        proba = np.clip(proba, 1e-6, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        self.n_classes = proba.shape[1]
        models = []
        for c in range(self.n_classes):
            binary = (y_true == c).astype(int)
            if len(np.unique(binary)) < 2:
                models.append(None)
                continue
            if self.method == "platt":
                # log-odds input for Platt scaling
                logits = np.log(proba[:, c] / (1.0 - proba[:, c] + 1e-9) + 1e-9).reshape(-1, 1)
                m = LogisticRegression(C=1e6, solver="lbfgs")
                m.fit(logits, binary)
                models.append(("platt", m))
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
                iso.fit(proba[:, c], binary)
                models.append(("isotonic", iso))
        self._models = models
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if not self._models:
            return proba
        proba = np.asarray(proba, dtype=float)
        proba = np.clip(proba, 1e-6, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        out = np.zeros_like(proba)
        for c, model in enumerate(self._models):
            if model is None:
                out[:, c] = proba[:, c]
                continue
            kind, m = model
            if kind == "platt":
                logits = np.log(proba[:, c] / (1.0 - proba[:, c] + 1e-9) + 1e-9).reshape(-1, 1)
                out[:, c] = m.predict_proba(logits)[:, 1]
            else:
                out[:, c] = m.predict(proba[:, c])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)

    def fit_transform(self, proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(proba, y_true).transform(proba)


def evaluate_calibration(
    proba: np.ndarray,
    y_true: np.ndarray,
    calibrated: np.ndarray | None = None,
) -> dict[str, Any]:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_true = np.asarray(y_true, dtype=int)
    out: dict[str, Any] = {
        "brier_score": round(float(_multiclass_brier(y_true, proba)), 6),
        "log_loss": round(float(log_loss(y_true, proba, labels=[0, 1, 2])), 6),
    }
    if calibrated is not None:
        cal = np.clip(np.asarray(calibrated, dtype=float), 1e-9, 1.0)
        cal = cal / cal.sum(axis=1, keepdims=True)
        out["brier_score_calibrated"] = round(float(_multiclass_brier(y_true, cal)), 6)
        out["log_loss_calibrated"] = round(float(log_loss(y_true, cal, labels=[0, 1, 2])), 6)
        out["brier_improvement"] = round(
            out["brier_score"] - out["brier_score_calibrated"], 6
        )
    return out


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    total = 0.0
    for c in range(proba.shape[1]):
        total += float(np.mean((proba[:, c] - (y_true == c).astype(float)) ** 2))
    return total / proba.shape[1]
