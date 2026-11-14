"""Small SQLAlchemy repository operations with explicit transaction boundaries."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import select

from app.db.models import Order, Product, RefundRecord, ReturnRequestDraft, SupportCase
from app.db.session import session_scope


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _product_dict(row: Product) -> dict[str, Any]:
    return {"product_id": row.product_id, "name": row.name, "stock": row.stock,
            "restock_date": _date_text(row.restock_date)}


def _order_dict(row: Order) -> dict[str, Any]:
    return {"order_id": row.order_id, "status": row.status, "eta": _date_text(row.eta),
            "product_id": row.product_id}


def _record_dict(row: SupportCase | ReturnRequestDraft, record_type: str) -> dict[str, Any]:
    return {"reference": row.reference, "record_type": record_type,
            "session_id": row.session_id, "order_id": row.order_id,
            "reason": row.reason, "summary": row.summary, "status": row.status,
            "details": row.details or {},
            "created_at": row.created_at.isoformat() if row.created_at else None}


def get_order_by_id(order_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(Order, order_id.strip().upper())
        return _order_dict(row) if row else None


def list_orders() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [_order_dict(row) for row in session.scalars(select(Order).order_by(Order.order_id)).all()]


def get_product_by_id(product_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(Product, product_id.strip().upper())
        return _product_dict(row) if row else None


def list_products() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [_product_dict(row) for row in session.scalars(select(Product).order_by(Product.product_id)).all()]


def search_products(query: str) -> list[dict[str, Any]]:
    """Parameter-bound substring search; SQL wildcard characters are escaped."""
    term = query.strip()
    if not term:
        return []
    with session_scope() as session:
        rows = session.scalars(select(Product).where(Product.name.icontains(term, autoescape=True)).order_by(Product.product_id)).all()
        return [_product_dict(row) for row in rows]


def get_support_case(reference: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(SupportCase, reference)
        return _record_dict(row, row.record_type) if row else None


def list_support_cases(record_type: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
    statement = select(SupportCase).order_by(SupportCase.created_at, SupportCase.reference)
    if record_type:
        statement = statement.where(SupportCase.record_type == record_type)
    if session_id:
        statement = statement.where(SupportCase.session_id == session_id)
    with session_scope() as session:
        return [_record_dict(row, row.record_type) for row in session.scalars(statement).all()]


def list_return_request_drafts(session_id: str | None = None) -> list[dict[str, Any]]:
    statement = select(ReturnRequestDraft).order_by(ReturnRequestDraft.created_at, ReturnRequestDraft.reference)
    if session_id:
        statement = statement.where(ReturnRequestDraft.session_id == session_id)
    with session_scope() as session:
        return [_record_dict(row, "return_request") for row in session.scalars(statement).all()]


def create_support_case(*, reference: str, record_type: str, session_id: str,
                        status: str, summary: str, order_id: str | None = None,
                        reason: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    with session_scope() as session:
        row = SupportCase(reference=reference, record_type=record_type, session_id=session_id,
                          status=status, summary=summary, order_id=order_id, reason=reason,
                          details=details or {}, is_demo=True)
        session.add(row)
        session.commit()
        session.refresh(row)
        return _record_dict(row, row.record_type)


def create_return_request_draft(*, reference: str, session_id: str, status: str,
                                summary: str, order_id: str, reason: str,
                                details: dict[str, Any] | None = None) -> dict[str, Any]:
    with session_scope() as session:
        row = ReturnRequestDraft(reference=reference, session_id=session_id, status=status,
                                 summary=summary, order_id=order_id, reason=reason,
                                 details=details or {}, is_demo=True)
        session.add(row)
        session.commit()
        session.refresh(row)
        return _record_dict(row, "return_request")


def get_return_request_draft(reference: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(ReturnRequestDraft, reference)
        return _record_dict(row, "return_request") if row else None


def get_refund_record(order_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(RefundRecord, order_id.strip().upper())
        if not row:
            return None
        return {"order_id": row.order_id, "status": row.status, "refund_id": row.refund_id,
                **(row.details or {})}


def list_refunds() -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(select(RefundRecord).order_by(RefundRecord.order_id)).all()
        return [{"order_id": row.order_id, "status": row.status, "refund_id": row.refund_id,
                 **(row.details or {})} for row in rows]


def import_product(record: dict[str, Any]) -> bool:
    """Insert a fixture only if absent; return False for an existing primary key."""
    with session_scope() as session:
        if session.get(Product, record["product_id"]):
            return False
        session.add(Product(product_id=record["product_id"], name=record["name"],
                            stock=record["stock"], restock_date=date.fromisoformat(record["restock_date"]) if record.get("restock_date") else None,
                            is_demo=True))
        session.commit()
        return True


def import_order(record: dict[str, Any]) -> bool:
    with session_scope() as session:
        if session.get(Order, record["order_id"]):
            return False
        session.add(Order(order_id=record["order_id"], status=record["status"],
                          eta=date.fromisoformat(record["eta"]) if record.get("eta") else None,
                          product_id=record.get("product_id"), is_demo=True))
        session.commit()
        return True


def import_refund(record: dict[str, Any]) -> bool:
    with session_scope() as session:
        if session.get(RefundRecord, record["order_id"]):
            return False
        # A demo refund must never attach to a pre-existing non-demo order that
        # happens to share an identifier. Existing orders are never modified.
        order = session.get(Order, record["order_id"])
        if order is None or not order.is_demo:
            return False
        known = {"order_id", "status", "refund_id"}
        session.add(RefundRecord(order_id=record["order_id"], status=record["status"],
                                 refund_id=record.get("refund_id"),
                                 details={key: value for key, value in record.items() if key not in known},
                                 is_demo=True))
        session.commit()
        return True
