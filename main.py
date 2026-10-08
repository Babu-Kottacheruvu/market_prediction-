"""FastAPI application entry point.

Run::

    uvicorn main:app --reload
    python main.py
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_assets, routes_backtest, routes_predict, routes_training
from app.core.exceptions import PlatformError
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


def create_app() -> FastAPI:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    app = FastAPI(
        title="Global Market Prediction Platform",
        description="Predict the next 10 candles (direction / probability / return / volatility) "
        "for forex, crypto and commodities. Prediction and signals only - no trade execution.",
        version="0.1.0",
    )

    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(PlatformError)
    async def platform_error_handler(_: Request, exc: PlatformError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})

    app.include_router(routes_assets.router)
    app.include_router(routes_predict.router)
    app.include_router(routes_training.router)
    app.include_router(routes_backtest.router)

    try:
        from app.database.session import init_db

        init_db()
    except Exception as exc:  # DB is optional for the ML path
        logger.warning("database init skipped: %s", exc)

    if FRONTEND_DIST.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/",
            app=StaticFiles(directory=str(FRONTEND_DIST), html=True),
            name="frontend",
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
