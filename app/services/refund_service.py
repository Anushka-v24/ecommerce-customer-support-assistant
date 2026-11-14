"""Read-only refund status lookup using explicit demonstration records."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError, load_refunds
from app.services.order_service import get_order_by_id


def get_refund_status(order_id: str | None) -> dict[str, Any]:
    """Return a stored refund status only when a refund record actually exists."""
    order = get_order_by_id(order_id)
    if order["outcome"] != "found":
        outcome = "unknown_order" if order["outcome"] == "not_found" else order["outcome"]
        return {"outcome": outcome, "order_id": order.get("order_id"), "refund": None, "error": order.get("error")}

    try:
        record = load_refunds().get(order["order"]["order_id"])
    except DataRepositoryError:
        return {
            "outcome": "data_unavailable", "order_id": order["order"]["order_id"],
            "refund": None, "error": "Demonstration refund records are unavailable.",
        }

    if record is None:
        return {
            "outcome": "no_refund_record", "order_id": order["order"]["order_id"],
            "refund": None,
            "error": "No refund status record exists in the demonstration data; I cannot confirm a refund was processed.",
        }
    status = record.get("status")
    if not isinstance(status, str) or not status.strip():
        return {
            "outcome": "data_unavailable", "order_id": order["order"]["order_id"],
            "refund": None, "error": "The demonstration refund record has no status.",
        }
    return {
        "outcome": "found", "order_id": order["order"]["order_id"],
        "refund": {key: value for key, value in record.items() if key != "order_id"},
        "error": None,
    }

