"""Product recommendations and warranty answers over the demonstration catalogue."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError, load_products, load_warranty
from app.services.product_service import search_products

DEMO_NOTE = "These are demonstration catalogue details, not live inventory or a real warranty."


def _mentioned_products(query: str | None) -> list[dict[str, Any]]:
    result = search_products(query)
    return result.get("matches", []) if result.get("outcome") in {"found", "ambiguous"} else []


def recommend_products(query: str | None) -> dict[str, Any]:
    """Suggest in-stock catalogue items, focusing on any product or category mentioned."""
    try:
        products = load_products()
    except DataRepositoryError:
        return {"outcome": "data_unavailable", "response": "Product suggestions are unavailable right now."}
    mentioned = {item["product_id"] for item in _mentioned_products(query)}
    pool = [p for pid, p in products.items() if pid in mentioned] or list(products.values())
    in_stock = [p for p in pool if isinstance(p.get("stock"), int) and p["stock"] > 0]
    if not in_stock:
        names = ", ".join(p["name"] for p in pool)
        response = f"{names} is currently out of stock in the demo catalogue. I can tell you when it is expected back."
    else:
        options = "; ".join(f"{p['name']} ({p['stock']} in stock)" for p in in_stock)
        lead = "Here is what I'd suggest" if len(in_stock) > 1 else "I'd suggest"
        response = f"{lead}: {options}. Tell me your budget or what you'll use it for and I can narrow it down."
    return {"outcome": "answered", "products": [p["name"] for p in in_stock], "response": f"{response} {DEMO_NOTE}"}


def answer_warranty_question(query: str | None) -> dict[str, Any]:
    """Return warranty length and coverage for a mentioned product, or the general terms."""
    try:
        warranty = load_warranty()
    except DataRepositoryError:
        return {"outcome": "data_unavailable", "response": "Warranty information is unavailable right now."}
    products = _mentioned_products(query)
    lines = []
    for product in products:
        terms = warranty["products"].get(product["product_id"], {})
        months = terms.get("months", warranty["default_months"])
        covers = terms.get("covers", "manufacturing defects")
        lines.append(f"{product['name']} has a {months}-month warranty covering {covers}.")
    if not lines:
        lines.append(f"Products come with a {warranty['default_months']}-month warranty against manufacturing defects unless stated otherwise.")
    response = " ".join(lines) + f" To claim: {warranty['claim_process']}"
    return {"outcome": "answered", "products": [p["name"] for p in products], "response": f"{response} {DEMO_NOTE}"}
