"""Read-only and demo-record tools for issue-support workflows."""

from __future__ import annotations

from typing import Any

from app.services.refund_service import get_refund_status
from app.services.return_service import prepare_demo_return_request
from app.services.support_case_service import investigate_missing_order, report_damaged_order


def refund_status_tool(order_id: str | None) -> dict[str, Any]:
    return get_refund_status(order_id)


def prepare_return_tool(
    order_id: str | None, reason: str | None, session_id: str | None
) -> dict[str, Any]:
    return prepare_demo_return_request(order_id, reason, session_id)


def damaged_order_tool(
    order_id: str | None, description: str | None, session_id: str | None,
    product_id: str | None = None,
) -> dict[str, Any]:
    return report_damaged_order(order_id, description, session_id, product_id)


def missing_order_tool(
    order_id: str | None, session_id: str | None, details: str | None = None
) -> dict[str, Any]:
    return investigate_missing_order(order_id, session_id, details)
