"""Backtest endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import rate_limit, require_api_key
from app.api.schemas import BacktestRequest
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.core.exceptions import PlatformError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["backtest"])


@router.post("/backtest")
def post_backtest(body: BacktestRequest, _: None = Depends(require_api_key), __: None = Depends(rate_limit)):
    config = BacktestConfig(
        horizon=body.horizon,
        min_probability=body.min_probability,
        min_confidence=body.min_confidence,
        cost_multiplier=body.cost_multiplier,
    )
    try:
        result = BacktestEngine(body.market, body.symbol, body.timeframe, config).run()
    except PlatformError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("backtest failed")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
    _persist(result)
    return result


def _persist(result: dict) -> None:
    try:
        from app.database.models import BacktestRecord
        from app.database.session import get_session_factory

        with get_session_factory()() as session:
            session.add(
                BacktestRecord(
                    market=result["market"],
                    symbol=result["symbol"],
                    timeframe=result["timeframe"],
                    horizon=int(result["horizon"]),
                    result=result,
                )
            )
            session.commit()
    except Exception as exc:
        logger.warning("could not persist backtest: %s", exc)
