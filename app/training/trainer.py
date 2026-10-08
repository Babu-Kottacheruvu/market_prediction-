"""Training pipeline: data -> features -> targets -> walk-forward -> models.

Run with::

    python -m scripts.train --market crypto --symbol BTCUSDT --timeframe 1m

The pipeline never randomizes rows, never fits preprocessing on the full
dataset, and evaluates exclusively on out-of-sample walk-forward test folds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import InsufficientHistoricalDataError, TrainingError
from app.core.logging import get_logger
from app.data.base import get_data
from app.data.validator import DataQualityValidator
from app.features.builder import build_features, feature_columns
from app.models.registry import ModelMetadata, ModelRegistry, make_model_id
from app.models.xgboost_model import HorizonModelSet
from app.targets.ten_candle import build_targets
from app.training.acceptance import check_acceptance
from app.training.evaluation import classification_metrics, regression_metrics
from app.training.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    generate_walk_forward_splits,
    split_frame,
)

logger = get_logger(__name__)

MIN_TRAIN_ROWS = 300
TrainingProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    callback: TrainingProgressCallback | None,
    percentage: float,
    phase: str,
    message: str,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    if callback is None:
        return
    payload: dict[str, Any] = {
        "percentage": round(percentage, 1),
        "phase": phase,
        "message": message,
    }
    if completed is not None:
        payload["completed"] = completed
    if total is not None:
        payload["total"] = total
    callback(payload)


@dataclass
class TrainingReport:
    market: str
    symbol: str
    timeframe: str
    model_id: str
    n_rows: int
    n_features: int
    folds: list[dict[str, Any]]
    out_of_sample_metrics: dict[str, Any]
    acceptance: dict[str, Any]
    status: str
    duration_seconds: float
    saved_to: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_id": self.model_id,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "folds": self.folds,
            "out_of_sample_metrics": self.out_of_sample_metrics,
            "acceptance": self.acceptance,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
            "saved_to": self.saved_to,
            **self.extra,
        }


# ----------------------------------------------------------------------
def prepare_dataset(
    market: str,
    symbol: str,
    timeframe: str,
    settings: Settings | None = None,
    validate: bool = True,
    progress_callback: TrainingProgressCallback | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load candles, validate, build features and targets."""
    settings = settings or get_settings()
    symbol = settings.resolve_symbol(market, symbol)
    tf = settings.timeframe(timeframe).name

    _emit_progress(progress_callback, 2, "data", "Fetching candle history")
    candles = get_data(market, symbol, tf)
    _emit_progress(progress_callback, 6, "data", f"Loaded {len(candles):,} candles")
    if validate:
        _emit_progress(progress_callback, 8, "validation", "Checking candle quality")
        report = DataQualityValidator().validate(candles, symbol=symbol, timeframe=tf)
        report.raise_if_failed()

    _emit_progress(progress_callback, 11, "features", "Building technical and market features")
    frame = build_features(candles, market, tf, settings=settings)
    _emit_progress(progress_callback, 13, "targets", "Preparing 10-candle training targets")
    frame = pd.concat([frame, build_targets(frame, settings)], axis=1)

    feats = feature_columns(frame)
    if not feats:
        raise TrainingError("Feature engine produced no columns")

    # drop warm-up rows where every feature is still NaN
    valid_mask = frame[feats].notna().any(axis=1)
    # rows must also have their h1 target available
    valid_mask &= frame["target_return_h1"].notna()
    frame = frame.loc[valid_mask].reset_index(drop=True)
    if len(frame) < MIN_TRAIN_ROWS:
        raise InsufficientHistoricalDataError(
            f"Only {len(frame)} usable rows for {market}/{symbol}/{tf}; "
            f"need at least {MIN_TRAIN_ROWS}. Download more history."
        )
    _emit_progress(
        progress_callback,
        15,
        "ready",
        f"Prepared {len(frame):,} training rows and {len(feats)} features",
    )
    return frame, feats


def _fit_horizons(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    validation_df: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
    estimator_counts: dict[int, tuple[int | None, int | None]] | None = None,
    on_horizon_complete: Callable[[int], None] | None = None,
) -> dict[int, HorizonModelSet]:
    """Fit the classifier/regressor pair for every horizon.

    Direction targets are stored as -1/0/1 (DOWN/FLAT/UP) but modelled in
    canonical space 0/1/2 (+1 shift), which is the order used for probability
    columns everywhere downstream: [P(DOWN), P(FLAT), P(UP)].
    """
    models: dict[int, HorizonModelSet] = {}
    X = train_df[feature_cols]
    X_validation = validation_df[feature_cols] if validation_df is not None else None
    for h in range(1, horizon + 1):
        y_dir = train_df[f"target_direction_h{h}"].to_numpy()
        y_ret = train_df[f"target_return_h{h}"].to_numpy()
        keep = ~np.isnan(y_dir) & ~np.isnan(y_ret)
        if int(keep.sum()) < MIN_TRAIN_ROWS:
            raise TrainingError(f"Not enough labeled rows for horizon {h}: {int(keep.sum())}")
        y_class = y_dir[keep].astype(int) + 1  # -1/0/1 -> 0/1/2
        eval_set = None
        if validation_df is not None and X_validation is not None:
            eval_set = (
                X_validation,
                validation_df[f"target_direction_h{h}"].to_numpy(),
                validation_df[f"target_return_h{h}"].to_numpy(),
            )
        model_set = HorizonModelSet(horizon=h).fit(
            X.loc[keep],
            y_class,
            y_ret[keep],
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            estimator_counts=(estimator_counts or {}).get(h, (None, None)),
        )
        models[h] = model_set
        if on_horizon_complete is not None:
            on_horizon_complete(h)
    return models


def _predict_horizon(model_set: HorizonModelSet, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return model_set.predict(X)


# ----------------------------------------------------------------------
def train(
    market: str,
    symbol: str,
    timeframe: str,
    walk_forward: WalkForwardConfig | None = None,
    settings: Settings | None = None,
    activate: bool = True,
    registry: ModelRegistry | None = None,
    save_oos: bool = True,
    progress_callback: TrainingProgressCallback | None = None,
) -> TrainingReport:
    settings = settings or get_settings()
    registry = registry or ModelRegistry()
    started = time.time()
    symbol = settings.resolve_symbol(market, symbol)
    tf = settings.timeframe(timeframe).name
    horizon = settings.forecast_horizon
    wf_cfg = walk_forward or WalkForwardConfig.from_settings(settings)

    logger.info("training start | %s | %s | %s", market, symbol, tf)
    _emit_progress(progress_callback, 1, "starting", f"Starting {market}/{symbol} {tf} training")
    frame, feature_cols = prepare_dataset(
        market, symbol, tf, settings, progress_callback=progress_callback
    )

    _emit_progress(progress_callback, 16, "validation", "Preparing walk-forward splits")
    embargo = pd.Timedelta(minutes=settings.timeframe(tf).minutes * horizon)
    folds = generate_walk_forward_splits(frame["timestamp"], wf_cfg, embargo=embargo, settings=settings)

    oos_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    selected_rounds: dict[int, list[tuple[int, int]]] = {h: [] for h in range(1, horizon + 1)}
    early_stopping_rounds = int(
        settings.models.get("xgboost", {}).get("early_stopping_rounds", 30)
    )

    usable_folds: list[tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for fold in folds:
        train_df = split_frame(frame, fold, "train")
        validation_df = split_frame(frame, fold, "val")
        test_df = split_frame(frame, fold, "test")
        if len(train_df) < MIN_TRAIN_ROWS or len(test_df) < 20:
            logger.warning(
                "fold %s skipped (train=%s test=%s)", fold.fold_index, len(train_df), len(test_df)
            )
            continue
        usable_folds.append((fold, train_df, validation_df, test_df))

    if not usable_folds:
        raise TrainingError(
            "No walk-forward fold produced usable train/test splits. "
            "Check data volume and walk_forward configuration."
        )

    model_pairs_total = len(usable_folds) * horizon
    model_pairs_completed = 0
    folds_evaluated = 0
    _emit_progress(
        progress_callback,
        18,
        "walk_forward",
        f"Evaluating {len(usable_folds)} walk-forward folds",
        completed=0,
        total=len(usable_folds),
    )
    for fold, train_df, validation_df, test_df in usable_folds:
        def horizon_fitted(_h: int) -> None:
            nonlocal model_pairs_completed
            model_pairs_completed += 1
            pct = 18 + 52 * model_pairs_completed / model_pairs_total
            _emit_progress(
                progress_callback,
                pct,
                "walk_forward",
                f"Fold {fold.fold_index + 1}/{len(usable_folds)}: fitted horizon "
                f"{_h}/{horizon}",
                completed=model_pairs_completed,
                total=model_pairs_total,
            )

        models = _fit_horizons(
            train_df,
            feature_cols,
            horizon,
            validation_df=validation_df,
            early_stopping_rounds=early_stopping_rounds,
            on_horizon_complete=horizon_fitted,
        )
        X_test = test_df[feature_cols]
        fold_metrics: dict[str, Any] = {}
        pred_frame = pd.DataFrame({"timestamp": test_df["timestamp"].to_numpy()})

        for h in range(1, horizon + 1):
            preds = _predict_horizon(models[h], X_test)
            y_true_dir = test_df[f"target_direction_h{h}"].to_numpy() + 1  # -> 0/1/2
            y_true_ret = test_df[f"target_return_h{h}"].to_numpy()
            fold_metrics[f"h{h}"] = {
                "direction": classification_metrics(y_true_dir, preds["direction"], preds["proba"]),
                "return": regression_metrics(y_true_ret, preds["return"]),
            }
            pred_frame[f"dir_h{h}"] = preds["direction"]
            pred_frame[f"ret_h{h}"] = preds["return"]
            pred_frame[f"p_up_h{h}"] = preds["proba"][:, 2]
            pred_frame[f"p_flat_h{h}"] = preds["proba"][:, 1]
            pred_frame[f"p_down_h{h}"] = preds["proba"][:, 0]
            train_directions = train_df[f"target_direction_h{h}"].dropna().to_numpy().astype(int) + 1
            class_counts = np.bincount(train_directions, minlength=3)
            majority_class = int(np.argmax(class_counts))
            pred_frame[f"baseline_dir_h{h}"] = majority_class
            pred_frame[f"baseline_ret_h{h}"] = 0.0
            for col in ("target_direction", "target_return"):
                pred_frame[f"{col}_h{h}"] = test_df[f"{col}_h{h}"].to_numpy()
            classifier_rounds, regressor_rounds = models[h].selected_estimator_counts
            if classifier_rounds is not None and regressor_rounds is not None:
                selected_rounds[h].append((classifier_rounds, regressor_rounds))

        oos_rows.append(pred_frame)
        folds_evaluated += 1
        _emit_progress(
            progress_callback,
            70 + 8 * folds_evaluated / len(usable_folds),
            "evaluation",
            f"Completed fold {folds_evaluated}/{len(usable_folds)} out-of-sample evaluation",
            completed=folds_evaluated,
            total=len(usable_folds),
        )
        acc_1 = fold_metrics["h1"]["direction"].get("accuracy")
        fold_summaries.append(
            {
                "fold": fold.fold_index,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "h1_accuracy": acc_1,
            }
        )

    oos = pd.concat(oos_rows, ignore_index=True).sort_values("timestamp")
    oos_metrics = _aggregate_oos_metrics(oos, horizon)

    cal_cfg = settings.models.get("calibration", {}) or {}
    calibrators: dict[int, Any] = {}
    if cal_cfg.get("enabled") and len(oos) >= 40:
        calibrators, cal_metrics = _fit_calibrators(oos, horizon, cal_cfg)
        oos_metrics["calibration"] = cal_metrics
        logger.info("fitted %d calibrators (method=%s)", len(calibrators), cal_cfg.get("method"))

    acceptance = check_acceptance(oos_metrics, settings)

    # deployment model: retrain on everything (evaluation above stayed out-of-sample)
    _emit_progress(progress_callback, 79, "final_fit", "Fitting final models on all eligible history")
    final_horizons_done = 0

    def final_horizon_fitted(_h: int) -> None:
        nonlocal final_horizons_done
        final_horizons_done += 1
        _emit_progress(
            progress_callback,
            79 + 18 * final_horizons_done / horizon,
            "final_fit",
            f"Final model horizons fitted: {final_horizons_done}/{horizon}",
            completed=final_horizons_done,
            total=horizon,
        )

    estimator_counts: dict[int, tuple[int | None, int | None]] = {}
    for h, rounds in selected_rounds.items():
        if rounds:
            classifier_values, regressor_values = zip(*rounds)
            estimator_counts[h] = (
                max(1, int(np.median(classifier_values))),
                max(1, int(np.median(regressor_values))),
            )
    final_models = _fit_horizons(
        frame,
        feature_cols,
        horizon,
        estimator_counts=estimator_counts,
        on_horizon_complete=final_horizon_fitted,
    )
    model_id = make_model_id(market, symbol, tf)
    metadata = ModelMetadata(
        model_id=model_id,
        market=market.lower(),
        symbol=symbol,
        timeframe=tf,
        horizon=horizon,
        feature_version=str(settings.features.get("feature_version", "v1")),
        model_version="xgb_v002",
        training_start=str(frame["timestamp"].min()),
        training_end=str(frame["timestamp"].max()),
        n_rows=int(len(frame)),
        hyperparameters={
            **settings.models.get("xgboost", {}),
            "selected_estimators_by_horizon": {
                str(h): list(counts) for h, counts in estimator_counts.items()
            },
        },
        validation_metrics=oos_metrics.get("aggregate", {}),
        test_metrics=oos_metrics,
        walk_forward=fold_summaries,
        feature_columns=feature_cols,
        status="candidate",
        notes=("Walk-forward OOS evaluation. Acceptance: "
               f"{'PASSED' if acceptance['passed'] else 'FAILED'}"),
    )
    _emit_progress(progress_callback, 98, "saving", "Saving validated model package")
    path = registry.save_package(final_models, metadata, calibrators or None)

    if save_oos:
        _save_oos_predictions(oos, market, symbol, tf)

    if activate and acceptance["passed"]:
        try:
            registry.activate(model_id)
            metadata.status = "active"
        except Exception as exc:  # pragma: no cover
            logger.warning("could not activate model %s: %s", model_id, exc)
    elif activate:
        logger.warning(
            "model %s saved as candidate (acceptance failed): %s",
            model_id, acceptance.get("reasons"),
        )

    report = TrainingReport(
        market=market.lower(),
        symbol=symbol,
        timeframe=tf,
        model_id=model_id,
        n_rows=int(len(frame)),
        n_features=len(feature_cols),
        folds=fold_summaries,
        out_of_sample_metrics=oos_metrics,
        acceptance=acceptance,
        status=metadata.status,
        duration_seconds=time.time() - started,
        saved_to=str(path),
        extra={"selected_estimators_by_horizon": {
            str(h): list(counts) for h, counts in estimator_counts.items()
        }},
    )
    _emit_progress(progress_callback, 99, "complete", "Training report is ready")
    logger.info(
        "training done | model=%s | status=%s | folds=%s | duration=%.1fs",
        model_id, report.status, len(fold_summaries), report.duration_seconds,
    )
    return report


def _aggregate_oos_metrics(oos: pd.DataFrame, horizon: int) -> dict[str, Any]:
    result: dict[str, Any] = {"aggregate": {}, "per_horizon": {}}
    accs, briers, maes, signs = [], [], [], []
    baseline_accs, baseline_maes, accuracy_lifts, mae_improvements = [], [], [], []
    for h in range(1, horizon + 1):
        y_dir = oos[f"target_direction_h{h}"].to_numpy() + 1  # -> 0/1/2
        p_dir = oos[f"dir_h{h}"].to_numpy()
        proba = np.column_stack([
            oos[f"p_down_h{h}"].to_numpy(),
            oos[f"p_flat_h{h}"].to_numpy(),
            oos[f"p_up_h{h}"].to_numpy(),
        ])
        y_ret = oos[f"target_return_h{h}"].to_numpy()
        p_ret = oos[f"ret_h{h}"].to_numpy()
        d = classification_metrics(y_dir, p_dir, proba)
        r = regression_metrics(y_ret, p_ret)
        baseline_direction = classification_metrics(
            y_dir, oos[f"baseline_dir_h{h}"].to_numpy()
        )
        baseline_return = regression_metrics(
            y_ret, oos[f"baseline_ret_h{h}"].to_numpy()
        )
        if d.get("accuracy") is not None and baseline_direction.get("accuracy") is not None:
            d["baseline_accuracy"] = baseline_direction["accuracy"]
            d["accuracy_lift_vs_baseline"] = round(
                d["accuracy"] - baseline_direction["accuracy"], 6
            )
            baseline_accs.append(baseline_direction["accuracy"])
            accuracy_lifts.append(d["accuracy_lift_vs_baseline"])
        if r.get("mae") is not None and baseline_return.get("mae") is not None:
            r["baseline_mae"] = baseline_return["mae"]
            r["mae_improvement_vs_baseline"] = round(
                baseline_return["mae"] - r["mae"], 10
            )
            baseline_maes.append(baseline_return["mae"])
            mae_improvements.append(r["mae_improvement_vs_baseline"])
        result["per_horizon"][f"h{h}"] = {
            "direction": d,
            "return": r,
            "baseline_direction": baseline_direction,
            "baseline_return": baseline_return,
        }
        if d.get("accuracy") is not None:
            accs.append(d["accuracy"])
        if d.get("brier_score") is not None:
            briers.append(d["brier_score"])
        if r.get("mae") is not None:
            maes.append(r["mae"])
        if r.get("sign_accuracy") is not None:
            signs.append(r["sign_accuracy"])

    result["aggregate"] = {
        "n_oos_rows": int(len(oos)),
        "mean_accuracy": round(float(np.mean(accs)), 6) if accs else None,
        "mean_brier": round(float(np.mean(briers)), 6) if briers else None,
        "mean_mae": round(float(np.mean(maes)), 10) if maes else None,
        "mean_sign_accuracy": round(float(np.mean(signs)), 6) if signs else None,
        "mean_baseline_accuracy": round(float(np.mean(baseline_accs)), 6) if baseline_accs else None,
        "mean_accuracy_lift_vs_baseline": (
            round(float(np.mean(accuracy_lifts)), 6) if accuracy_lifts else None
        ),
        "mean_baseline_mae": round(float(np.mean(baseline_maes)), 10) if baseline_maes else None,
        "mean_mae_improvement_vs_baseline": (
            round(float(np.mean(mae_improvements)), 10) if mae_improvements else None
        ),
    }
    return result


def _fit_calibrators(
    oos: pd.DataFrame,
    horizon: int,
    cal_cfg: dict[str, Any],
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Fit one calibrator per horizon on the first half of OOS, score on the second.

    Both halves stay out-of-sample with respect to model fitting, so the
    reported calibration metrics are honest.
    """
    from app.models.calibration import ProbabilityCalibrator, evaluate_calibration

    method = cal_cfg.get("method", "isotonic")
    half = len(oos) // 2
    fit_part, eval_part = oos.iloc[:half], oos.iloc[half:]
    calibrators: dict[int, Any] = {}
    metrics: dict[str, Any] = {}

    for h in range(1, horizon + 1):
        cols = [f"p_down_h{h}", f"p_flat_h{h}", f"p_up_h{h}"]
        proba_fit = fit_part[cols].to_numpy(dtype=float)
        proba_eval = eval_part[cols].to_numpy(dtype=float)
        y_fit = fit_part[f"target_direction_h{h}"].to_numpy() + 1  # -> 0/1/2
        y_eval = eval_part[f"target_direction_h{h}"].to_numpy() + 1
        cal = ProbabilityCalibrator(method=method).fit(proba_fit, y_fit)
        calibrators[h] = cal
        metrics[f"h{h}"] = evaluate_calibration(
            proba_eval, y_eval, cal.transform(proba_eval)
        )
    return calibrators, metrics


def _save_oos_predictions(oos: pd.DataFrame, market: str, symbol: str, timeframe: str) -> None:
    from app.core import config as app_config

    target = app_config.PROCESSED_DATA_DIR / market.lower() / symbol.upper()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{timeframe}_oos_predictions.parquet"
    oos.to_parquet(path, index=False)
    logger.info("saved out-of-sample predictions -> %s", path)
