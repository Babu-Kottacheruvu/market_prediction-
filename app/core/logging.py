"""Structured logging setup.

Rule: never log secrets.  ``redact`` strips common credential keys from any
context dict before it reaches a log record.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "database_url",
    "binance_api_key",
    "binance_secret",
    "oanda_api_key",
    "mt5_password",
}

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def redact(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        k: ("***" if str(k).lower() in _SENSITIVE_KEYS else v)
        for k, v in payload.items()
    }


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
