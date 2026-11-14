"""LangGraph-callable order tools; these wrappers do not mutate graph state."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError
from app.services.order_service import get_order_status


def lookup_order_status(order_id: str | None) -> dict[str, Any]:
    """Validate and return a read-only order status result."""
    try:
        result = get_order_status(order_id)
    except DataRepositoryError:
        return {
            "outcome": "data_unavailable", "order_id": None,
            "error": "Demonstration order data is currently unavailable.",
        }
    result["lookup_performed"] = result["outcome"] in {"found", "not_found"}
    return result

