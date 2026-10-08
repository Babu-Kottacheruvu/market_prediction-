import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import TrainingError
from app.training.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
    split_frame,
)


def _ts(n=1000, freq="1min", start="2025-01-01"):
    return pd.Series(pd.date_range(start, periods=n, freq=freq, tz="UTC"))


def test_splits_are_contiguous_and_disjoint():
    ts = pd.Series(pd.date_range("2025-01-01", periods=10 * 1440, freq="min", tz="UTC"))
    cfg = WalkForwardConfig(train_days=2, validation_days=1, test_days=1, step_days=1,
                            min_train_days=1, min_test_days=1)
    folds = generate_walk_forward_splits(ts, cfg, settings=None)
    assert len(folds) >= 3
    for f in folds:
        assert f.train_start <= f.train_end <= f.val_start <= f.val_end <= f.test_start <= f.test_end
        assert f.train_end <= f.val_start
        assert f.val_end <= f.test_start


def test_no_test_period_overlaps_training():
    ts = pd.Series(pd.date_range("2025-01-01", periods=8 * 1440, freq="min", tz="UTC"))
    cfg = WalkForwardConfig(train_days=2, validation_days=1, test_days=1, step_days=1,
                            min_train_days=1, min_test_days=1)
    for f in generate_walk_forward_splits(ts, cfg):
        assert f.train_end <= f.test_start
        assert f.val_end <= f.test_start


def test_embargo_shortens_training_window():
    ts = pd.Series(pd.date_range("2025-01-01", periods=8 * 1440, freq="min", tz="UTC"))
    cfg = WalkForwardConfig(train_days=2, validation_days=1, test_days=1, step_days=2,
                            min_train_days=1, min_test_days=1)
    embargo = pd.Timedelta(minutes=10)
    folds = generate_walk_forward_splits(ts, cfg, embargo=embargo)
    for f in folds:
        # training window ends 10 minutes before the validation window starts
        assert f.train_end == f.val_start - embargo
        assert f.val_end <= f.test_start


def test_split_frame_uses_correct_rows():
    ts = pd.Series(pd.date_range("2025-01-01", periods=6 * 1440, freq="min", tz="UTC"))
    cfg = WalkForwardConfig(train_days=1, validation_days=1, test_days=1, step_days=1,
                            min_train_days=1, min_test_days=1)
    fold = generate_walk_forward_splits(ts, cfg)[0]
    frame = pd.DataFrame({"timestamp": ts, "value": np.arange(len(ts))})
    train = split_frame(frame, fold, "train")
    test = split_frame(frame, fold, "test")
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) <= len(frame)


def test_raises_when_data_too_short():
    ts = _ts(n=200, freq="1h")  # ~8 days
    cfg = WalkForwardConfig(train_days=30, validation_days=30, test_days=30, step_days=30,
                            min_train_days=30, min_test_days=5)
    with pytest.raises(TrainingError):
        generate_walk_forward_splits(ts, cfg)


def test_fallback_single_split_for_medium_history():
    ts = _ts(n=2000, freq="1min", start="2025-06-01")  # ~1.4 days? no: 2000 min
    # 40 days of hourly data:
    ts = pd.Series(pd.date_range("2025-01-01", periods=40 * 24, freq="h", tz="UTC"))
    cfg = WalkForwardConfig(train_days=180, validation_days=30, test_days=30, step_days=30,
                            min_train_days=30, min_test_days=5)
    folds = generate_walk_forward_splits(ts, cfg)
    assert len(folds) == 1
    f = folds[0]
    assert f.train_start < f.test_start < f.test_end
