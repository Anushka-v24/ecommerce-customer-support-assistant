"""Local simulation services for damaged and missing delivery support cases."""

from __future__ import annotations

from typing import Any

from app.services.case_repository import create_demo_record
from app.services.order_service import get_order_by_id, get_order_status
from app.services.product_service import get_product_by_id


def _order_failure(result: dict[str, Any]) -> dict[str, Any] | None:
    if result["outcome"] == "found":
        return None
    return {
        "outcome": result["outcome"], "case": None,
        "error": result.get("error"),
    }


def report_damaged_order(
    order_id: str | None,
    description: str | None,
    session_id: str | None,
    product_id: str | None = None,
) -> dict[str, Any]:
    """Record a local damaged-item simulation after validating the sample order."""
    if not isinstance(description, str) or not description.strip():
        return {"outcome": "missing_damage_details", "case": None, "error": "Please describe what is damaged."}
    if not isinstance(session_id, str) or not session_id.strip():
        return {"outcome": "missing_session_id", "case": None, "error": "A conversation session is required."}

    order_result = get_order_by_id(order_id)
    failure = _order_failure(order_result)
    if failure:
        return failure
    order = order_result["order"]
    associated_product_id = order.get("product_id")
    if product_id and product_id.strip().upper() != associated_product_id:
        return {
            "outcome": "product_order_mismatch", "case": None,
            "error": "That product ID does not match the product in the demonstration order.",
        }
    product = get_product_by_id(associated_product_id)
    product_name = product.get("product", {}).get("name") if product["outcome"] == "found" else None
    details = {
        "product_id": associated_product_id,
        "product_name": product_name,
        "description": description.strip()[:2000],
        "retailer_or_courier_claim_submitted": False,
    }
    record = create_demo_record(
        record_type="damaged_order",
        session_id=session_id,
        order_id=order["order_id"],
        reason=description.strip()[:2000],
        summary=f"Demonstration damaged-item report for {order['order_id']}.",
        status="demo_case_recorded_pending_human_review",
        details=details,
    )
    return {
        "outcome": "demo_case_recorded", "case": record,
        "next_step": "A support representative would review the description in a real integration.",
        "retailer_or_courier_claim_submitted": False, "error": None,
    }


def investigate_missing_order(
    order_id: str | None, session_id: str | None, details: str | None = None
) -> dict[str, Any]:
    """Record a delivery investigation without declaring a parcel lost."""
    if not isinstance(session_id, str) or not session_id.strip():
        return {"outcome": "missing_session_id", "case": None, "error": "A conversation session is required."}
    if not isinstance(details, str) or not details.strip():
        return {"outcome": "missing_delivery_details", "case": None, "error": "Please provide details about the missing delivery."}
    status_result = get_order_status(order_id)
    if status_result["outcome"] != "found":
        return {
            "outcome": status_result["outcome"], "case": None,
            "order": None, "error": status_result.get("error"),
        }
    progress = status_result["delivery_progress"]
    if progress["stage"] == "unknown":
        return {
            "outcome": "delivery_status_unknown", "case": None,
            "order": status_result,
            "error": "The demonstration record has no recognized delivery status; no conclusion can be drawn.",
        }

    note = progress["explanation"]
    record = create_demo_record(
        record_type="missing_order_investigation",
        session_id=session_id,
        order_id=status_result["order_id"],
        reason=details.strip()[:2000],
        summary=f"Demonstration delivery investigation for {status_result['order_id']}: {note}",
        status="demo_case_recorded_pending_human_review",
        details={
            "recorded_order_status": status_result["order_status"],
            "eta": status_result["eta"],
            "delivery_progress": progress,
            "parcel_declared_lost": False,
            "carrier_contacted": False,
            "customer_reported_details": details.strip()[:2000],
        },
    )
    return {
        "outcome": "demo_case_recorded", "case": record,
        "order": status_result,
        "parcel_declared_lost": False, "carrier_contacted": False, "error": None,
    }
