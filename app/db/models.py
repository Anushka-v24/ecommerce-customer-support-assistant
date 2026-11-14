"""PostgreSQL ORM schema for the application's fictional demo records."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("stock >= 0", name="ck_products_stock_nonnegative"),
        Index("ix_products_name", "name"),
    )

    product_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False)
    restock_date: Mapped[date | None] = mapped_column(Date)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    orders: Mapped[list["Order"]] = relationship(back_populates="product")


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (Index("ix_orders_status", "status"), Index("ix_orders_product_id", "product_id"))

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(80), nullable=False)
    eta: Mapped[date | None] = mapped_column(Date)
    product_id: Mapped[str | None] = mapped_column(ForeignKey("products.product_id", ondelete="RESTRICT"))
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    product: Mapped[Product | None] = relationship(back_populates="orders")
    support_cases: Mapped[list["SupportCase"]] = relationship(back_populates="order")
    return_drafts: Mapped[list["ReturnRequestDraft"]] = relationship(back_populates="order")
    refunds: Mapped[list["RefundRecord"]] = relationship(back_populates="order")


class SupportCase(Base):
    __tablename__ = "support_cases"
    __table_args__ = (
        Index("ix_support_cases_session_created", "session_id", "created_at"),
        Index("ix_support_cases_order_id", "order_id"),
    )

    reference: Mapped[str] = mapped_column(String(80), primary_key=True)
    record_type: Mapped[str] = mapped_column(String(80), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.order_id", ondelete="RESTRICT"))
    reason: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    order: Mapped[Order | None] = relationship(back_populates="support_cases")


class ReturnRequestDraft(Base):
    __tablename__ = "return_request_drafts"
    __table_args__ = (Index("ix_return_drafts_session_created", "session_id", "created_at"),)

    reference: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id", ondelete="RESTRICT"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    order: Mapped[Order] = relationship(back_populates="return_drafts")


class RefundRecord(Base):
    """A recorded refund status; no refund operation is performed by this model."""

    __tablename__ = "refund_records"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id", ondelete="RESTRICT"), primary_key=True)
    status: Mapped[str] = mapped_column(String(100), nullable=False)
    refund_id: Mapped[str | None] = mapped_column(String(100), unique=True)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    order: Mapped[Order] = relationship(back_populates="refunds")
