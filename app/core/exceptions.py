"""Typed exceptions used across the platform."""

from __future__ import annotations


class PlatformError(Exception):
    """Base class for all platform errors."""


class UnsupportedMarketError(PlatformError):
    pass


class UnsupportedSymbolError(PlatformError):
    pass


class UnsupportedTimeframeError(PlatformError):
    pass


class DataUnavailableError(PlatformError):
    pass


class InsufficientHistoricalDataError(PlatformError):
    pass


class DataQualityError(PlatformError):
    pass


class ModelNotFoundError(PlatformError):
    pass


class ModelValidationError(PlatformError):
    pass


class TrainingError(PlatformError):
    pass


class BacktestError(PlatformError):
    pass


class ConfigurationError(PlatformError):
    pass


class AuthenticationError(PlatformError):
    pass
