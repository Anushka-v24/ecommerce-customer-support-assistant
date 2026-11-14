"""Read-only order lookups and ETA-based delivery assessment."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.services.data_repository import load_orders


ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$", re.IGNORECASE)


def get_order_by_id(order_id: str | None) -> dict[str, Any]:
    """Return a structured lookup outcome without exposing customer fields."""
    if order_id is None or (isinstance(order_id, str) and not order_id.strip()):
        return {"outcome": "missing_order_id", "order": None, "error": "Please provide an order ID."}
    if not isinstance(order_id, str):
        return {"outcome": "invalid_order_id", "order": None, "error": "Order ID must be text."}

    normalized_id = order_id.strip().upper()
    if not ORDER_ID_PATTERN.fullmatch(normalized_id):
        return {
            "outcome": "invalid_order_id", "order_id": normalized_id,
            "order": None, "error": "Order ID must match the format ORD-1234.",
        }

    record = load_orders().get(normalized_id)
    if record is None:
        return {
            "outcome": "not_found", "order_id": normalized_id,
            "order": None, "error": "No demonstration order matches that ID.",
        }

    status = record.get("status")
    if not isinstance(status, str) or not status.strip():
        return {
            "outcome": "data_unavailable", "order_id": normalized_id,
            "order": None, "error": "The order record has no supported status.",
        }

    safe_order = {
        "order_id": normalized_id,
        "status": status,
        "eta": record.get("eta"),
        "product_id": record.get("product_id"),
    }
    return {"outcome": "found", "order": safe_order, "error": None}


def assess_delivery_progress(
    status: str, eta: str | None, as_of: date | None = None
) -> dict[str, Any]:
    """Assess only the recorded status/ETA; no carrier events are inferred."""
    today = as_of or date.today()
    normalized_status = status.strip().casefold()
    if normalized_status == "delivered":
        return {
            "stage": "delivered", "is_delayed": False,
            "explanation": "The demonstration record marks this order delivered; no delivery event details are stored.",
        }
    if normalized_status == "processing":
        return {
            "stage": "processing", "is_delayed": False,
            "explanation": "The demonstration record says processing; it contains no carrier scans.",
        }
    if normalized_status == "delayed":
        return {
            "stage": "delayed", "is_delayed": True,
            "explanation": "The demonstration record explicitly marks this order delayed; no carrier scans are stored.",
        }
    if normalized_status in {"shipped", "in transit", "in_transit"}:
        if isinstance(eta, str):
            try:
                eta_date = date.fromisoformat(eta)
            except ValueError:
                eta_date = None
            if eta_date and eta_date < today:
                return {
                    "stage": "delayed", "is_delayed": True,
                    "explanation": "The recorded ETA has passed; the sample suggests a delay, but contains no carrier scans.",
                }
        return {
            "stage": "in_transit", "is_delayed": False,
            "explanation": "The demonstration status says shipped; no carrier event history is available.",
        }
    return {
        "stage": "unknown", "is_delayed": False,
        "explanation": "The demonstration record does not contain a recognized delivery status.",
    }


def get_order_status(
    order_id: str | None, as_of: date | None = None
) -> dict[str, Any]:
    """Return recorded order status and a cautious status/ETA progress assessment."""
    result = get_order_by_id(order_id)
    order = result.get("order")
    if result["outcome"] != "found" or order is None:
        return {key: value for key, value in result.items() if key != "order"}
    progress = assess_delivery_progress(order["status"], order["eta"], as_of)
    return {
        "outcome": "found",
        "order_id": order["order_id"],
        "order_status": order["status"],
        "eta": order["eta"],
        "product_id": order["product_id"],
        "delivery_progress": progress,
        "error": None,
    }
