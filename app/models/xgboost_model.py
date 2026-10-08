"""XGBoost wrappers: one classifier + one regressor per forecast horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CANONICAL_CLASSES = [0, 1, 2]  # DOWN, FLAT, UP


def _clf_params() -> dict[str, Any]:
    p = dict(get_settings().models.get("xgboost", {}).get("classifier", {}))
    p.setdefault("eval_metric", "mlogloss")
    p.setdefault("verbosity", 0)
    return p


def _reg_params() -> dict[str, Any]:
    p = dict(get_settings().models.get("xgboost", {}).get("regressor", {}))
    p.setdefault("verbosity", 0)
    return p


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (len(counts) * counts)
    return w[y]


@dataclass
class HorizonModelSet:
    """Model pair for a single horizon."""

    horizon: int
    classifier: XGBClassifier | None = None
    regressor: XGBRegressor | None = None
    classes_: list[int] = field(default_factory=lambda: list(CANONICAL_CLASSES))

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y_direction: np.ndarray,
        y_return: np.ndarray,
        eval_set: tuple[pd.DataFrame | np.ndarray, np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        estimator_counts: tuple[int | None, int | None] = (None, None),
    ) -> "HorizonModelSet":
        y_direction = np.asarray(y_direction, dtype=int)
        y_return = np.asarray(y_return, dtype=float)
        valid = ~(np.isnan(y_return))
        Xd, yd = X[valid] if isinstance(X, pd.DataFrame) else X[valid], y_direction[valid]
        classifier_rounds, regressor_rounds = estimator_counts

        classes = sorted(set(np.unique(yd).tolist()))
        self.classes_ = classes
        if len(classes) >= 2:
            clf_params = _clf_params()
            if classifier_rounds is not None:
                clf_params["n_estimators"] = classifier_rounds
            clf_eval_set = None
            if eval_set is not None and early_stopping_rounds and len(classes) == 3:
                X_val, y_val_direction, y_val_return = eval_set
                y_val_class = np.asarray(y_val_direction, dtype=float) + 1
                valid_val = ~np.isnan(y_val_class)
                y_val_class = y_val_class[valid_val].astype(int)
                if len(y_val_class) >= 20 and set(np.unique(y_val_class)) <= set(classes):
                    X_val_classifier = X_val.loc[valid_val] if isinstance(X_val, pd.DataFrame) else X_val[valid_val]
                    clf_params["early_stopping_rounds"] = early_stopping_rounds
                    clf_eval_set = [(X_val_classifier, y_val_class)]
            clf = XGBClassifier(**clf_params)
            fit_kwargs: dict[str, Any] = {"sample_weight": balanced_class_weights(yd)}
            if clf_eval_set is not None:
                fit_kwargs.update(eval_set=clf_eval_set, verbose=False)
            clf.fit(Xd, yd, **fit_kwargs)
            self.classifier = clf
        else:
            logger.warning(
                "horizon %s: only one direction class present (%s) - classifier skipped",
                self.horizon, classes,
            )
            self.classifier = None

        reg_params = _reg_params()
        if regressor_rounds is not None:
            reg_params["n_estimators"] = regressor_rounds
        reg_eval_set = None
        if eval_set is not None and early_stopping_rounds:
            X_val, _, y_val_return = eval_set
            y_val_return = np.asarray(y_val_return, dtype=float)
            valid_val = ~np.isnan(y_val_return)
            if valid_val.sum() >= 20:
                X_val_regressor = X_val.loc[valid_val] if isinstance(X_val, pd.DataFrame) else X_val[valid_val]
                reg_params["early_stopping_rounds"] = early_stopping_rounds
                reg_eval_set = [(X_val_regressor, y_val_return[valid_val])]
        reg = XGBRegressor(**reg_params)
        reg_fit_kwargs: dict[str, Any] = {}
        if reg_eval_set is not None:
            reg_fit_kwargs.update(eval_set=reg_eval_set, verbose=False)
        reg.fit(X[valid], y_return[valid], **reg_fit_kwargs)
        self.regressor = reg
        return self

    @property
    def selected_estimator_counts(self) -> tuple[int | None, int | None]:
        """Return best classifier/regressor tree counts after early stopping."""
        classifier_best = getattr(self.classifier, "best_iteration", None)
        regressor_best = getattr(self.regressor, "best_iteration", None)
        return (
            int(classifier_best) + 1 if classifier_best is not None else None,
            int(regressor_best) + 1 if regressor_best is not None else None,
        )

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probabilities in canonical order [P(DOWN), P(FLAT), P(UP)]."""
        n = len(X)
        out = np.zeros((n, 3), dtype=float)
        if self.classifier is None:
            if 1 in self.classes_:
                out[:, 1] = 1.0
            else:
                out[:, 1] = 1.0
            return out
        proba = self.classifier.predict_proba(X)
        for src_idx, cls in enumerate(self.classifier.classes_.tolist()):
            out[:, int(cls)] = proba[:, src_idx]
        totals = out.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return out / totals

    def predict_direction(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_return(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.regressor is None:
            return np.zeros(len(X))
        return self.regressor.predict(X).astype(float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        return {
            "proba": self.predict_proba(X),
            "direction": self.predict_direction(X),
            "return": self.predict_return(X),
        }
