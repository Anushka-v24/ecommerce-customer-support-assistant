"""Create the customer-facing support ticket table.

Revision ID: 0003_support_tickets
Revises: 0002_business_data
Create Date: 2026-11-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_support_tickets"
down_revision: Union[str, None] = "0002_business_data"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "support_tickets",
        sa.Column("ticket_id", sa.String(32), primary_key=True),
        sa.Column("customer_id", sa.String(64)),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("category", sa.String(80), nullable=False),
        sa.Column("subject", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("order_id", sa.String(32)),
        sa.Column("case_reference", sa.String(80)),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("priority", sa.String(20), nullable=False),
        sa.Column("details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_support_tickets_customer_created", "support_tickets", ["customer_id", "created_at"])
    op.create_index("ix_support_tickets_session_created", "support_tickets", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_support_tickets_session_created", table_name="support_tickets")
    op.drop_index("ix_support_tickets_customer_created", table_name="support_tickets")
    op.drop_table("support_tickets")
