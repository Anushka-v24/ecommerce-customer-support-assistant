"""Lazy PostgreSQL engine and SQLAlchemy session lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import Config


class DatabaseNotConfiguredError(RuntimeError):
    """Raised only when database access is requested without a DATABASE_URL."""


_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[Engine, sessionmaker[Session]] = {}
_ENGINE_LOCK = RLock()


def normalize_database_url(database_url: str | None) -> str:
    """Validate a PostgreSQL URL and select the installed psycopg 3 driver.

    Common provider URLs using ``postgres://`` or ``postgresql://`` are accepted.
    SQLite and other dialects are rejected so they cannot silently stand in for
    PostgreSQL-specific behavior.
    """
    if not isinstance(database_url, str) or not database_url.strip():
        raise DatabaseNotConfiguredError(
            "Database access requires DATABASE_URL to be configured."
        )
    try:
        url = make_url(database_url.strip())
    except Exception as exc:
        raise ValueError("DATABASE_URL is not a valid SQLAlchemy URL.") from exc

    if url.get_backend_name() not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must use PostgreSQL; SQLite is not supported here.")
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    elif url.drivername == "postgresql+psycopg2":
        # Use the explicitly installed driver instead of requiring psycopg2.
        url = url.set(drivername="postgresql+psycopg")
    elif url.drivername != "postgresql+psycopg":
        raise ValueError("DATABASE_URL must use the PostgreSQL psycopg driver.")
    return url.render_as_string(hide_password=False)


def _configured_url(database_url: str | None) -> str:
    return normalize_database_url(database_url if database_url is not None else Config.DATABASE_URL)


def get_engine(database_url: str | None = None) -> Engine:
    """Return a cached PostgreSQL engine without connecting during construction."""
    normalized_url = _configured_url(database_url)
    if Config.DB_POOL_SIZE < 1:
        raise ValueError("DB_POOL_SIZE must be at least 1.")
    if Config.DB_MAX_OVERFLOW < 0:
        raise ValueError("DB_MAX_OVERFLOW cannot be negative.")
    if min(
        Config.DB_POOL_TIMEOUT_SECONDS,
        Config.DB_POOL_RECYCLE_SECONDS,
        Config.DB_CONNECT_TIMEOUT_SECONDS,
    ) < 1:
        raise ValueError("Database timeout settings must be positive integers.")
    with _ENGINE_LOCK:
        engine = _ENGINES.get(normalized_url)
        if engine is None:
            engine = create_engine(
                normalized_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=Config.DB_POOL_SIZE,
                max_overflow=Config.DB_MAX_OVERFLOW,
                pool_timeout=Config.DB_POOL_TIMEOUT_SECONDS,
                pool_recycle=Config.DB_POOL_RECYCLE_SECONDS,
                connect_args={"connect_timeout": Config.DB_CONNECT_TIMEOUT_SECONDS},
            )
            _ENGINES[normalized_url] = engine
        return engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a cached factory with explicit transaction behavior defaults."""
    bound_engine = engine or get_engine()
    if bound_engine.dialect.name != "postgresql":
        raise ValueError("The application database layer only supports PostgreSQL engines.")
    with _ENGINE_LOCK:
        factory = _SESSION_FACTORIES.get(bound_engine)
        if factory is None:
            factory = sessionmaker(
                bind=bound_engine,
                autoflush=False,
                expire_on_commit=False,
                class_=Session,
            )
            _SESSION_FACTORIES[bound_engine] = factory
        return factory


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a session, rollback on failure, and always close it.

    This helper deliberately does not commit automatically. Repository code must
    make transaction boundaries explicit with ``session.commit()``.
    """
    session = get_session_factory(engine)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engines(engine: Engine | None = None) -> None:
    """Dispose one engine or all cached engines and clear their session factories."""
    with _ENGINE_LOCK:
        selected = [engine] if engine is not None else list(_ENGINES.values())
        for item in selected:
            item.dispose()
            _SESSION_FACTORIES.pop(item, None)
            for url, cached in list(_ENGINES.items()):
                if cached is item:
                    del _ENGINES[url]

