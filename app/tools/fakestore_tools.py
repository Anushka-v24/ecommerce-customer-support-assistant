"""Small wrappers for the optional FakeStore demonstration catalog."""

from __future__ import annotations

import re
from typing import Any

from app.config import Config
from app.integrations.fakestore_client import (
    FakeStoreError,
    FakeStoreNotFound,
    get_fakestore_client,
)

_SEARCH_STOP_WORDS = {
    "a", "an", "and", "are", "can", "check", "do", "find", "for", "have", "i",
    "in", "is", "item", "me", "of", "please", "product", "products", "search",
    "show", "stock", "the", "to", "you", "available", "availability", "inventory",
}


def _public_product(product: dict[str, Any]) -> dict[str, Any]:
    return {key: product[key] for key in ("id", "title", "price", "description", "category")}


def list_catalog_products(limit: int = 20) -> dict[str, Any]:
    if not Config.FAKESTORE_ENABLED:
        return {"outcome": "disabled", "products": []}
    try:
        products = get_fakestore_client().list_products(limit)
        return {"outcome": "found", "products": [_public_product(p) for p in products]}
    except (FakeStoreError, ValueError):
        return {"outcome": "unavailable", "products": []}


def search_catalog_products(query: str | None, limit: int = 20) -> dict[str, Any]:
    if not Config.FAKESTORE_ENABLED:
        return {"outcome": "disabled", "matches": []}
    if not isinstance(query, str) or not query.strip():
        return {"outcome": "missing_query", "matches": []}
    terms = {
        term for term in re.findall(r"[\w]+", query.casefold())
        if len(term) > 1 and term not in _SEARCH_STOP_WORDS
    }
    if not terms:
        return {"outcome": "missing_query", "matches": []}
    try:
        products = get_fakestore_client().list_products(limit)
    except (FakeStoreError, ValueError):
        return {"outcome": "unavailable", "matches": []}
    matches = []
    for product in products:
        searchable = f"{product['title']} {product['description']} {product['category']}".casefold()
        if terms.intersection(re.findall(r"[\w]+", searchable)):
            matches.append(_public_product(product))
    return {"outcome": "found" if matches else "not_found", "matches": matches}


def get_catalog_product(product_id: int) -> dict[str, Any]:
    if not Config.FAKESTORE_ENABLED:
        return {"outcome": "disabled", "product": None}
    try:
        product = get_fakestore_client().get_product(product_id)
        return {"outcome": "found", "product": _public_product(product)}
    except FakeStoreNotFound:
        return {"outcome": "not_found", "product": None}
    except (FakeStoreError, ValueError):
        return {"outcome": "unavailable", "product": None}


def check_catalog_stock(query: str | None) -> dict[str, Any]:
    """Resolve catalog products but explicitly report that quantity is not exposed."""
    result = search_catalog_products(query)
    if result["outcome"] != "found":
        return {**result, "stock_quantity": None}
    return {"outcome": "inventory_unavailable", "matches": result["matches"], "stock_quantity": None}


def resolve_explicit_catalog_id(query: str) -> int | None:
    """Only accept a numeric catalog ID explicitly written by the customer."""
    match = re.fullmatch(r"\s*(?:product|item)\s*(?:id\s*)?#?\s*(\d+)\s*", query, re.IGNORECASE)
    return int(match.group(1)) if match else None
