"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import rate_limit, require_api_key
from app.api.schemas import PredictRequest
from app.core.exceptions import PlatformError
from app.core.logging import get_logger
from app.prediction.predictor import predict

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["predict"])


@router.post("/predict")
def post_predict(body: PredictRequest, _: None = Depends(require_api_key), __: None = Depends(rate_limit)):
    try:
        result = predict(body.market, body.symbol, body.timeframe, equity=body.equity)
    except PlatformError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    _persist(result)
    return result


def _persist(result: dict) -> None:
    try:
        from app.database.models import PredictionRecord
        from app.database.session import get_session_factory

        factory = get_session_factory()
        with factory() as session:
            session.add(
                PredictionRecord(
                    market=result["market"],
                    symbol=result["symbol"],
                    timeframe=result["timeframe"],
                    model_id=result["model"]["model_id"],
                    signal=result["signal"],
                    regime=result["regime"],
                    confidence=float(result["confidence"]),
                    payload=result,
                )
            )
            session.commit()
    except Exception as exc:  # DB must never break prediction
        logger.warning("could not persist prediction: %s", exc)
