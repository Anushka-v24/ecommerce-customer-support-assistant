"""Read-only return policy and explicitly simulated return-request preparation."""

from __future__ import annotations

from typing import Any

from app.services.case_repository import create_demo_record
from app.services.data_repository import DataRepositoryError, load_policies
from app.services.order_service import get_order_by_id


def get_return_policy(product_id: str | None = None) -> dict[str, Any]:
    """Return policy conditions without exposing dummy form links."""
    try:
        document = load_policies()
    except DataRepositoryError:
        return {"outcome": "data_unavailable", "policy": None, "error": "Demonstration policy data is unavailable."}
    base = document.get("return")
    if not isinstance(base, dict):
        return {"outcome": "data_unavailable", "policy": None, "error": "Return policy is not configured in demonstration data."}

    policy = {key: value for key, value in base.items() if key != "form_url"}
    specific = False
    if product_id:
        normalized = product_id.strip().upper() if isinstance(product_id, str) else ""
        product_policies = document.get("products", {})
        if not isinstance(product_policies, dict) or normalized not in product_policies:
            return {"outcome": "unknown_product", "product_id": normalized, "policy": None, "error": "No product-specific return policy is available."}
        product_return = product_policies[normalized].get("return", {})
        if isinstance(product_return, dict):
            policy.update(product_return)
            specific = True
    return {
        "outcome": "found", "product_id": product_id.strip().upper() if product_id else None,
        "product_specific": specific, "policy": policy,
        "data_notice": document.get("notice"), "error": None,
    }


def check_return_eligibility(order_id: str | None) -> dict[str, Any]:
    """Assess available policy data without claiming eligibility absent purchase facts."""
    order_result = get_order_by_id(order_id)
    if order_result["outcome"] != "found":
        return {
            "outcome": order_result["outcome"], "order_id": order_result.get("order_id"),
            "eligibility": "not_determined", "policy": None,
            "error": order_result.get("error"),
        }
    order = order_result["order"]
    policy_result = get_return_policy(order.get("product_id"))
    if policy_result["outcome"] != "found":
        return {
            "outcome": policy_result["outcome"], "order_id": order["order_id"],
            "eligibility": "not_determined", "policy": None,
            "error": policy_result.get("error"),
        }
    return {
        "outcome": "manual_review_required", "order_id": order["order_id"],
        "product_id": order.get("product_id"), "eligibility": "not_determined",
        "missing_evidence": ["purchase date", "item condition"],
        "policy": policy_result["policy"], "data_notice": policy_result["data_notice"],
        "error": None,
    }


def prepare_demo_return_request(
    order_id: str | None, reason: str | None, session_id: str | None
) -> dict[str, Any]:
    """Create a local draft record; no real return, pickup, or refund is initiated."""
    if not isinstance(reason, str) or not reason.strip():
        return {"outcome": "missing_reason", "request": None, "error": "Please provide a return reason."}
    if not isinstance(session_id, str) or not session_id.strip():
        return {"outcome": "missing_session_id", "request": None, "error": "A conversation session is required."}

    eligibility = check_return_eligibility(order_id)
    if eligibility["outcome"] != "manual_review_required":
        return {"outcome": eligibility["outcome"], "request": None, "error": eligibility.get("error")}

    summary = f"Demonstration return request for {eligibility['order_id']}; manual eligibility review is still required."
    record = create_demo_record(
        record_type="return_request",
        session_id=session_id,
        order_id=eligibility["order_id"],
        reason=reason.strip(),
        summary=summary,
        status="demo_prepared_pending_manual_review",
        details={
            "eligibility": "not_determined",
            "missing_evidence": eligibility["missing_evidence"],
            "policy": eligibility["policy"],
            "real_return_submitted": False,
        },
    )
    return {
        "outcome": "demo_request_prepared", "request": record,
        "eligibility": eligibility, "real_return_submitted": False, "error": None,
    }
