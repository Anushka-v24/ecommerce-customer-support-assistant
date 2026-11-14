"""Process-lifetime LangGraph checkpointer selection and PostgreSQL pool owner."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url

from app.config import Config
from app.db.session import DatabaseNotConfiguredError, normalize_database_url


class CheckpointerConfigurationError(RuntimeError):
    """Raised when CHECKPOINT_BACKEND or its required configuration is invalid."""


class CheckpointerInitializationError(RuntimeError):
    """Raised when the configured PostgreSQL checkpoint store cannot start."""


@dataclass
class _PostgresRuntime:
    database_url: str
    pool: ConnectionPool
    saver: PostgresSaver


_RUNTIME: _PostgresRuntime | None = None
_RUNTIME_LOCK = RLock()


def _psycopg_conninfo(database_url: str) -> str:
    """Translate SQLAlchemy's driver-qualified URL to a psycopg PostgreSQL URI."""
    return make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)


def _postgres_checkpointer() -> PostgresSaver:
    global _RUNTIME

    if not Config.DATABASE_URL:
        raise CheckpointerConfigurationError(
            "CHECKPOINT_BACKEND=postgres requires DATABASE_URL to be configured."
        )
    try:
        database_url = normalize_database_url(Config.DATABASE_URL)
    except (DatabaseNotConfiguredError, ValueError) as exc:
        raise CheckpointerConfigurationError(
            "CHECKPOINT_BACKEND=postgres requires a valid PostgreSQL DATABASE_URL using psycopg."
        ) from exc

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            if _RUNTIME.database_url != database_url:
                raise CheckpointerConfigurationError(
                    "PostgreSQL checkpoint configuration changed after initialization; restart the process."
                )
            return _RUNTIME.saver

        if Config.DB_POOL_SIZE < 1 or Config.DB_MAX_OVERFLOW < 0:
            raise CheckpointerConfigurationError("Database pool size settings are invalid.")
        try:
            pool = ConnectionPool(
                conninfo=_psycopg_conninfo(database_url),
                min_size=1,
                max_size=Config.DB_POOL_SIZE + Config.DB_MAX_OVERFLOW,
                timeout=Config.DB_POOL_TIMEOUT_SECONDS,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "connect_timeout": Config.DB_CONNECT_TIMEOUT_SECONDS,
                },
                open=False,
            )
            pool.open(wait=True, timeout=Config.DB_CONNECT_TIMEOUT_SECONDS)
            saver = PostgresSaver(
                pool,
                serde=JsonPlusSerializer(allowed_msgpack_modules=None),
            )
            # The checkpointer package owns this separate schema and tracks its
            # own migrations. setup() is safe to rerun on process restart.
            saver.setup()
        except Exception as exc:
            if "pool" in locals():
                try:
                    pool.close()
                except Exception:
                    pass
            raise CheckpointerInitializationError(
                "PostgreSQL checkpoint initialization failed. Verify database availability and schema privileges."
            ) from exc

        _RUNTIME = _PostgresRuntime(database_url=database_url, pool=pool, saver=saver)
        return saver


def get_configured_checkpointer():
    """Construct the configured saver; PostgreSQL failures never fall back to memory."""
    backend = Config.CHECKPOINT_BACKEND
    if backend == "memory":
        return InMemorySaver()
    if backend == "postgres":
        return _postgres_checkpointer()
    raise CheckpointerConfigurationError(
        "CHECKPOINT_BACKEND must be either 'memory' or 'postgres'."
    )


def close_postgres_checkpointer() -> None:
    """Close the shared psycopg pool during process shutdown or explicit reset."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        runtime, _RUNTIME = _RUNTIME, None
        if runtime is not None:
            runtime.pool.close()
