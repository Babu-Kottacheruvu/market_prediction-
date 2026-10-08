"""Walk-forward (rolling) time-series validation.

No random splitting anywhere.  Folds are produced as::

    [---- TRAIN ----][-- VAL --][-- TEST --]
                      [-- VAL --][-- TEST --]   <- step forward

with an *embargo* so that no training row's future-target window overlaps the
validation/test period (label leakage guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import TrainingError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 180
    validation_days: int = 30
    test_days: int = 30
    step_days: int = 30
    min_train_days: int = 30
    min_test_days: int = 5

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "WalkForwardConfig":
        settings = settings or get_settings()
        wf = settings.models.get("walk_forward", {})
        return cls(
            train_days=int(wf.get("train_days", 180)),
            validation_days=int(wf.get("validation_days", 30)),
            test_days=int(wf.get("test_days", 30)),
            step_days=int(wf.get("step_days", 30)),
            min_train_days=int(wf.get("min_train_days", 30)),
            min_test_days=int(wf.get("min_test_days", 5)),
        )


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def mask(self, ts: pd.Series, side: str) -> pd.Series:
        if side == "train":
            return (ts >= self.train_start) & (ts < self.train_end)
        if side == "val":
            return (ts >= self.val_start) & (ts < self.val_end)
        if side == "test":
            return (ts >= self.test_start) & (ts < self.test_end)
        raise ValueError(side)

    def to_dict(self) -> dict:
        return {
            "fold": self.fold_index,
            "train": [str(self.train_start), str(self.train_end)],
            "val": [str(self.val_start), str(self.val_end)],
            "test": [str(self.test_start), str(self.test_end)],
        }


def generate_walk_forward_splits(
    timestamps: Sequence[pd.Timestamp] | pd.Series,
    config: WalkForwardConfig | None = None,
    embargo: pd.Timedelta | None = None,
    settings: Settings | None = None,
) -> list[WalkForwardFold]:
    """Build walk-forward folds over ``timestamps`` (sorted candle times).

    ``embargo`` should be ``horizon * timeframe`` so targets of a training row
    can never reach into the next split.
    """
    config = config or WalkForwardConfig.from_settings(settings)
    ts = pd.Series(pd.to_datetime(list(timestamps), utc=True)).drop_duplicates().sort_values()
    if ts.empty:
        raise TrainingError("No timestamps available for walk-forward splitting")

    start, end = ts.iloc[0], ts.iloc[-1]
    span = end - start
    day = pd.Timedelta(days=1)
    need = (config.train_days + config.validation_days + config.test_days) * day
    embargo = embargo or pd.Timedelta(0)

    folds: list[WalkForwardFold] = []
    if span >= need:
        cursor = start
        while True:
            tr_s = cursor
            tr_e = tr_s + config.train_days * day
            va_s = tr_e
            va_e = va_s + config.validation_days * day
            te_s = va_e
            te_e = te_s + config.test_days * day
            if te_e > end:
                break
            folds.append(
                WalkForwardFold(
                    fold_index=len(folds),
                    train_start=tr_s,
                    train_end=tr_e - embargo,   # targets of last train rows must finish before val
                    val_start=va_s,
                    val_end=va_e - embargo,
                    test_start=te_s,
                    test_end=te_e,
                )
            )
            cursor += config.step_days * day
            if config.step_days <= 0:
                raise TrainingError("step_days must be positive")
    else:
        logger.warning(
            "dataset span %s is shorter than train+val+test (%s days) - using a single "
            "proportional split",
            span, config.train_days + config.validation_days + config.test_days,
        )
        total_days = span / day
        if total_days < config.min_train_days + config.min_test_days:
            raise TrainingError(
                f"Insufficient history for walk-forward validation: {total_days:.1f} days "
                f"available, need at least {config.min_train_days + config.min_test_days}"
            )
        train_frac = 0.70
        val_frac = 0.15
        tr_s = start
        tr_e = start + span * train_frac
        va_s = tr_e
        va_e = start + span * (train_frac + val_frac)
        te_s = va_e
        te_e = end + pd.Timedelta(seconds=1)
        folds.append(
            WalkForwardFold(
                fold_index=0,
                train_start=tr_s,
                train_end=tr_e - embargo,
                val_start=va_s,
                val_end=va_e - embargo,
                test_start=te_s,
                test_end=te_e,
            )
        )

    if not folds:
        raise TrainingError("Walk-forward configuration produced no folds")

    for fold in folds:
        logger.info("walk_forward fold %s | train %s..%s | val %s..%s | test %s..%s",
                    fold.fold_index, fold.train_start.date(), fold.train_end.date(),
                    fold.val_start.date(), fold.val_end.date(),
                    fold.test_start.date(), fold.test_end.date())
    return folds


def split_frame(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    side: str,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    mask = fold.mask(df[time_col], side)
    return df.loc[mask].copy()
