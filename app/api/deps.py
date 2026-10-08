"""API dependencies: optional API-key auth and a small in-memory rate limiter."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

_RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
_rate_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests when an API key is configured but not supplied/matching."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return  # auth disabled in development
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def rate_limit(request: Request) -> None:
    if _RATE_LIMIT_PER_MINUTE <= 0:
        return
    now = time.time()
    bucket_key = request.client.host if request.client else "unknown"
    with _rate_lock:
        q = _hits[bucket_key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= _RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)
