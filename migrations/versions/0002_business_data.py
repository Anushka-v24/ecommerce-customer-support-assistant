"""Create relational tables for fictional demo business data.

Revision ID: 0002_business_data
Revises: 0001_postgresql_foundation
Create Date: 2026-11-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_business_data"
down_revision: Union[str, None] = "0001_postgresql_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("product_id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("stock", sa.Integer(), nullable=False),
        sa.Column("restock_date", sa.Date()),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("stock >= 0", name="ck_products_stock_nonnegative"),
    )
    op.create_index("ix_products_name", "products", ["name"])

    op.create_table(
        "orders",
        sa.Column("order_id", sa.String(32), primary_key=True),
        sa.Column("status", sa.String(80), nullable=False),
        sa.Column("eta", sa.Date()),
        sa.Column("product_id", sa.String(32), sa.ForeignKey("products.product_id", ondelete="RESTRICT")),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_product_id", "orders", ["product_id"])

    op.create_table(
        "support_cases",
        sa.Column("reference", sa.String(80), primary_key=True),
        sa.Column("record_type", sa.String(80), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("order_id", sa.String(32), sa.ForeignKey("orders.order_id", ondelete="RESTRICT")),
        sa.Column("reason", sa.Text()),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(100), nullable=False),
        sa.Column("details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_support_cases_session_created", "support_cases", ["session_id", "created_at"])
    op.create_index("ix_support_cases_order_id", "support_cases", ["order_id"])

    op.create_table(
        "return_request_drafts",
        sa.Column("reference", sa.String(80), primary_key=True),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("order_id", sa.String(32), sa.ForeignKey("orders.order_id", ondelete="RESTRICT"), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(100), nullable=False),
        sa.Column("details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_return_drafts_session_created", "return_request_drafts", ["session_id", "created_at"])

    op.create_table(
        "refund_records",
        sa.Column("order_id", sa.String(32), sa.ForeignKey("orders.order_id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("status", sa.String(100), nullable=False),
        sa.Column("refund_id", sa.String(100), unique=True),
        sa.Column("details", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("refund_records")
    op.drop_index("ix_return_drafts_session_created", table_name="return_request_drafts")
    op.drop_table("return_request_drafts")
    op.drop_index("ix_support_cases_order_id", table_name="support_cases")
    op.drop_index("ix_support_cases_session_created", table_name="support_cases")
    op.drop_table("support_cases")
    op.drop_index("ix_orders_product_id", table_name="orders")
    op.drop_index("ix_orders_status", table_name="orders")
    op.drop_table("orders")
    op.drop_index("ix_products_name", table_name="products")
    op.drop_table("products")
