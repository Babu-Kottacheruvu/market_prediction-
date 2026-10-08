"""Data quality validation.

The validator *reports* problems instead of silently fixing them.  The
training pipeline refuses to continue on fatal issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

WEEKEND_START = pd.Timedelta(hours=0)  # placeholder, weekend logic below uses weekdays


@dataclass
class QualityIssue:
    code: str
    severity: str  # "fatal" | "warning"
    message: str
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
        }


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    rows: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def fatal(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "fatal"]

    @property
    def passed(self) -> bool:
        return not self.fatal

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
        }

    def raise_if_failed(self) -> None:
        from app.core.exceptions import DataQualityError

        if not self.passed:
            details = "; ".join(f"{i.code}: {i.message}" for i in self.fatal)
            raise DataQualityError(f"Data quality check failed for {self.symbol}/{self.timeframe}: {details}")


class DataQualityValidator:
    """Validates normalized OHLCV frames."""

    def validate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> DataQualityReport:
        settings = get_settings()
        spec = settings.timeframe(timeframe)
        report = DataQualityReport(symbol=symbol, timeframe=spec.name, rows=len(df))
        if len(df) == 0:
            report.issues.append(QualityIssue("empty", "fatal", "Dataset contains no rows"))
            return report

        self._check_timestamps(df, report, spec.minutes)
        self._check_ohlc(df, report)
        self._check_volume(df, report)
        self._check_gaps(df, report, spec.minutes, market=_market_of(df))

        for issue in report.issues:
            log = logger.warning if issue.severity == "fatal" else logger.info
            log("data_quality %s | %s/%s | %s | count=%s",
                issue.code, symbol, spec.name, issue.message, issue.count)
        return report

    # ------------------------------------------------------------------
    def _check_timestamps(self, df: pd.DataFrame, report: DataQualityReport, minutes: int) -> None:
        ts = df["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            report.issues.append(QualityIssue("timestamp_type", "fatal", "timestamp is not datetime"))
            return
        if ts.isna().any():
            report.issues.append(QualityIssue("timestamp_null", "fatal", "null timestamps present", int(ts.isna().sum())))
        if ts.dt.tz is None:
            report.issues.append(QualityIssue("timestamp_tz", "fatal", "timestamps are not timezone-aware"))
        dupes = int(ts.duplicated().sum())
        if dupes:
            report.issues.append(QualityIssue("duplicate_timestamps", "fatal", "duplicate candle timestamps", dupes))
        diffs = ts.diff().dropna()
        backwards = int((diffs < pd.Timedelta(0)).sum())
        if backwards:
            report.issues.append(QualityIssue("out_of_order", "fatal", "candles out of order", backwards))

    def _check_ohlc(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        ohlc = df[["open", "high", "low", "close"]]
        nan_count = int(ohlc.isna().sum().sum())
        if nan_count:
            report.issues.append(QualityIssue("nan_ohlc", "fatal", "NaN values in OHLC", nan_count))

        high, low = df["high"], df["low"]
        bad_range = int((high < low).sum())
        if bad_range:
            report.issues.append(QualityIssue("high_below_low", "fatal", "high < low", bad_range))

        bad_high = int(((df["open"] > high) | (df["close"] > high)).sum())
        if bad_high:
            report.issues.append(QualityIssue("close_above_high", "fatal", "open/close above high", bad_high))

        bad_low = int(((df["open"] < low) | (df["close"] < low)).sum())
        if bad_low:
            report.issues.append(QualityIssue("close_below_low", "fatal", "open/close below low", bad_low))

        nonpositive = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
        if nonpositive:
            report.issues.append(QualityIssue("nonpositive_price", "fatal", "price <= 0", nonpositive))

    def _check_volume(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        if "volume" not in df.columns:
            report.issues.append(QualityIssue("volume_missing", "warning", "volume column absent"))
            return
        neg = int((df["volume"] < 0).sum())
        if neg:
            report.issues.append(QualityIssue("negative_volume", "fatal", "negative volume", neg))
        if bool((df["volume"] == 0).all()):
            report.issues.append(QualityIssue("volume_all_zero", "warning", "volume is zero for every row"))

    def _check_gaps(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        minutes: int,
        market: str,
    ) -> None:
        ts = df["timestamp"].sort_values().reset_index(drop=True)
        expected = pd.Timedelta(minutes=minutes)
        diffs = ts.diff().dropna()
        gaps = diffs[diffs > expected * 1.5]
        if gaps.empty:
            return
        if market in {"forex", "commodities"}:
            starts = [ts.iloc[int(pos)] for pos in gaps.index]
            gaps = gaps[[not _spans_weekend(start, gap) for start, gap in zip(starts, gaps)]]
            if gaps.empty:
                return
        total_missing = int(((gaps / expected) - 1).round().sum())
        report.issues.append(
            QualityIssue(
                "missing_candles",
                "warning",
                f"{len(gaps)} gaps in candle sequence (~{total_missing} missing candles); "
                "largest gap "
                f"{gaps.max()}",
                count=int(len(gaps)),
            )
        )


def _market_of(df: pd.DataFrame) -> str:
    if "market" in df.columns and len(df) and pd.notna(df["market"].iloc[0]):
        return str(df["market"].iloc[0]).lower()
    return ""


def _spans_weekend(start: pd.Timestamp, gap: pd.Timedelta) -> bool:
    end = start + gap
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    return any(d.dayofweek >= 5 for d in days)
