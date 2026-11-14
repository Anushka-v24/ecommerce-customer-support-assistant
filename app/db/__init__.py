"""Optional SQLAlchemy database foundation; importing it opens no connection."""

from app.db.base import Base
from app.db import models as models
from app.db.session import (
    DatabaseNotConfiguredError,
    dispose_engines,
    get_engine,
    get_session_factory,
    normalize_database_url,
    session_scope,
)

__all__ = [
    "Base",
    "models",
    "DatabaseNotConfiguredError",
    "dispose_engines",
    "get_engine",
    "get_session_factory",
    "normalize_database_url",
    "session_scope",
]
