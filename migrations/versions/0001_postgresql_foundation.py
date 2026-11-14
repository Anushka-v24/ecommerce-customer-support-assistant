"""Establish the Alembic revision baseline; application tables are deferred.

Revision ID: 0001_postgresql_foundation
Revises:
Create Date: 2026-11-14
"""

from typing import Sequence, Union

revision: str = "0001_postgresql_foundation"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No application tables are introduced in the PostgreSQL foundation phase."""


def downgrade() -> None:
    """The baseline has no application objects to remove."""

