"""Transaction cost model (spread + commission + slippage)."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class TransactionCosts:
    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def round_trip(self) -> float:
        """Cost of entering and exiting, as a fraction of price."""
        return self.spread + self.commission + self.slippage

    @property
    def one_way(self) -> float:
        return self.round_trip / 2.0

    @classmethod
    def for_market(cls, market: str, timeframe: str) -> "TransactionCosts":
        c = get_settings().costs(market, timeframe)
        return cls(spread=c["spread"], commission=c["commission"], slippage=c["slippage"])

    def to_dict(self) -> dict[str, float]:
        return {
            "spread": self.spread,
            "commission": self.commission,
            "slippage": self.slippage,
            "round_trip": self.round_trip,
        }
