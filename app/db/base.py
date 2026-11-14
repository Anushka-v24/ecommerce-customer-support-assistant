"""SQLAlchemy declarative base for future application ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Common metadata base for relational application records."""

