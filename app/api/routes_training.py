"""Training and model-management endpoints."""

from __future__ import annotations

import threading
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.deps import rate_limit, require_api_key
from app.api.schemas import TrainRequest
from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.models.registry import ModelRegistry
from app.models.router import ModelRouter
from app.training.trainer import train
from app.training.walk_forward import WalkForwardConfig

logger = get_logger(__name__)
router = APIRouter(tags=["training"])
_training_jobs: dict[str, dict[str, Any]] = {}
_training_jobs_lock = threading.Lock()


@router.post("/train", status_code=202)
def post_train(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    __: None = Depends(rate_limit),
):
    wf = WalkForwardConfig.from_settings()
    overrides = {
        "train_days": body.train_days,
        "validation_days": body.val_days,
        "test_days": body.test_days,
        "step_days": body.step_days,
    }
    wf = WalkForwardConfig(**{k: (v if v is not None else getattr(wf, k)) for k, v in overrides.items()})
    training_id = uuid.uuid4().hex
    with _training_jobs_lock:
        _training_jobs[training_id] = {
            "training_id": training_id,
            "status": "queued",
            "progress": {
                "percentage": 0,
                "phase": "queued",
                "message": "Waiting for a training worker",
            },
            "report": None,
            "error": None,
        }
    background_tasks.add_task(
        _run_training_job,
        training_id,
        body.market,
        body.symbol,
        body.timeframe,
        wf,
        body.activate,
    )
    return {"training_id": training_id, "status": "queued"}


def _run_training_job(
    training_id: str,
    market: str,
    symbol: str,
    timeframe: str,
    walk_forward: WalkForwardConfig,
    activate: bool,
) -> None:
    _update_training_job(training_id, status="running")
    try:
        report = train(
            market,
            symbol,
            timeframe,
            walk_forward=walk_forward,
            activate=activate,
            progress_callback=lambda progress: _update_training_job(
                training_id, progress=progress
            ),
        )
    except PlatformError as exc:
        logger.warning("training failed for %s/%s/%s: %s", market, symbol, timeframe, exc)
        _update_training_job(training_id, status="failed", error=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected trainer failures
        logger.exception("training failed for %s/%s/%s", market, symbol, timeframe)
        _update_training_job(training_id, status="failed", error=f"Training failed: {exc}")
    else:
        _update_training_job(
            training_id,
            status="succeeded",
            progress={
                "percentage": 100,
                "phase": "complete",
                "message": "Training completed successfully",
            },
            report=report.to_dict(),
        )


def _update_training_job(training_id: str, **changes: Any) -> None:
    with _training_jobs_lock:
        job = _training_jobs.get(training_id)
        if job is not None:
            job.update(changes)


@router.get("/train/{training_id}")
def get_training_progress(training_id: str, _: None = Depends(require_api_key)):
    with _training_jobs_lock:
        job = _training_jobs.get(training_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return dict(job)


@router.get("/models")
def list_models(market: str | None = None, symbol: str | None = None):
    return {"models": ModelRegistry().list(market=market, symbol=symbol)}


@router.get("/model-status")
def model_status(market: str, symbol: str, timeframe: str):
    try:
        return ModelRouter().model_status(market, symbol, timeframe)
    except PlatformError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str, _: None = Depends(require_api_key)):
    try:
        ModelRegistry().activate(model_id)
    except PlatformError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"model_id": model_id, "status": "active"}


@router.post("/models/{model_id}/deactivate")
def deactivate_model(model_id: str, _: None = Depends(require_api_key)):
    try:
        ModelRegistry().deactivate(model_id)
    except PlatformError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"model_id": model_id, "status": "inactive"}
