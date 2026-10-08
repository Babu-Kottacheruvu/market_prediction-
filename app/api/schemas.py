"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Market = Literal["crypto", "forex", "commodities"]


class PredictRequest(BaseModel):
    market: Market
    symbol: str = Field(..., min_length=1, max_length=32)
    timeframe: str = Field(..., min_length=2, max_length=8)
    equity: float | None = Field(None, gt=0, description="optional equity for position sizing")


class TrainRequest(BaseModel):
    market: Market
    symbol: str = Field(..., min_length=1, max_length=32)
    timeframe: str = Field(..., min_length=2, max_length=8)
    train_days: int | None = Field(None, gt=0)
    val_days: int | None = Field(None, gt=0)
    test_days: int | None = Field(None, gt=0)
    step_days: int | None = Field(None, gt=0)
    activate: bool = True


class BacktestRequest(BaseModel):
    market: Market
    symbol: str = Field(..., min_length=1, max_length=32)
    timeframe: str = Field(..., min_length=2, max_length=8)
    horizon: int = Field(1, ge=1, le=10)
    min_probability: float = Field(0.65, ge=0.0, le=1.0)
    min_confidence: float = Field(0.55, ge=0.0, le=1.0)
    cost_multiplier: float = Field(1.0, gt=0.0, le=10.0)


class ModelStatusRequest(BaseModel):
    market: Market
    symbol: str
    timeframe: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
