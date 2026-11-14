"""LangGraph-callable product tools; these wrappers do not mutate graph state."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError
from app.services.product_service import get_product_stock, search_products


def check_product_stock(product_query: str | None) -> dict[str, Any]:
    """Resolve one product safely and return stock only for one clear match."""
    try:
        search_result = search_products(product_query)
        if search_result["outcome"] != "found":
            return {
                **search_result,
                "lookup_performed": False,
            }
        product = search_result["matches"][0]
        result = get_product_stock(product["product_id"])
    except DataRepositoryError:
        return {
            "outcome": "data_unavailable", "matches": [],
            "lookup_performed": False,
            "error": "Demonstration product data is currently unavailable.",
        }

    result["lookup_performed"] = result["outcome"] == "found"
    return result
