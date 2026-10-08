"""SQLAlchemy engine/session. SQLite for dev, PostgreSQL for production.

The ML pipeline never depends on the database - it is an audit trail only.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import DATA_DIR
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DATA_DIR / 'platform.db'}"


@lru_cache(maxsize=1)
def get_engine():
    url = get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, echo=False, future=True, connect_args=connect_args)
    logger.info("database engine created for %s", _redact_url(url))
    return engine


def _redact_url(url: str) -> str:
    if "@" in url and "://" in url:
        head, _, tail = url.partition("://")
        _, _, host = tail.partition("@")
        return f"{head}://***@{host}"
    return url


def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    from app.database import models  # noqa: F401  (register tables)

    Base.metadata.create_all(get_engine())
    logger.info("database tables ensured")
