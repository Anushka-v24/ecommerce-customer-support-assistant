"""Structured UI blocks derived from a finished graph turn.

The text answer stays the source of truth; blocks only re-present data the
workflow already looked up (an order, products, a ticket, document sources) so
the frontend can draw cards. Unknown or partial results produce no block.
"""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError, load_products
from app.services.ticket_service import public_ticket

ORDER_STAGES = ["Placed", "Processing", "Shipped", "Out for delivery", "Delivered"]


def _products() -> dict[str, dict[str, Any]]:
    try:
        return load_products()
    except (DataRepositoryError, OSError):
        return {}


def _timeline(status: str) -> list[dict[str, str]]:
    normalized = status.casefold()
    current = next((i for i, stage in enumerate(ORDER_STAGES) if stage.casefold() == normalized), None)
    if current is None:
        return []
    return [
        {"label": stage, "state": "done" if i < current else "current" if i == current else "upcoming"}
        for i, stage in enumerate(ORDER_STAGES)
    ]


def _product_card(product: dict[str, Any]) -> dict[str, Any]:
    stock = product.get("stock")
    return {
        "type": "product_card", "product_id": product.get("product_id"), "name": product.get("name"),
        "stock": stock if isinstance(stock, int) else None,
        "in_stock": isinstance(stock, int) and stock > 0, "restock_date": product.get("restock_date"),
    }


def build_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    route = result.get("selected_route") or ""
    tool = result.get("tool_result") if isinstance(result.get("tool_result"), dict) else {}
    blocks: list[dict[str, Any]] = []

    if route == "order_tracking" and tool.get("outcome") == "found":
        product = _products().get(tool.get("product_id") or "", {})
        blocks.append({
            "type": "order_card", "order_id": tool["order_id"], "status": tool.get("order_status"),
            "eta": tool.get("eta"), "product_name": product.get("name"),
            "note": (tool.get("delivery_progress") or {}).get("explanation"),
            "timeline": _timeline(tool.get("order_status") or ""),
        })
    elif route == "product_inquiry" and tool.get("outcome") == "found" and tool.get("name"):
        blocks.append(_product_card(tool))
    elif route == "product_recommendation" and tool.get("products"):
        by_name = {p.get("name"): p for p in _products().values()}
        blocks.extend(_product_card(by_name[name]) for name in tool["products"][:3] if name in by_name)

    if isinstance(result.get("ticket"), dict):
        blocks.append({"type": "ticket_card", **public_ticket(result["ticket"])})
    sources = result.get("sources")
    if isinstance(sources, list) and sources:
        blocks.append({"type": "sources", "items": [s for s in sources if isinstance(s, (str, dict))][:4]})
    return blocks
