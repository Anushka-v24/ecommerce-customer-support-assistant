"""Read-only product search and stock lookups over demonstration JSON."""

from __future__ import annotations

import difflib
import re
from typing import Any

from app.services.data_repository import load_products
from app.services.llm_extraction import match_catalog_product


PRODUCT_ID_PATTERN = re.compile(r"^PROD-\d{3}$", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _close_token_match(query_words: set[str], name_words: set[str]) -> bool:
    """Match plurals ("laptops") and small typos ("headphnes") on meaningful words only."""
    query_stems = {_stem(word) for word in query_words if len(word) >= 4}
    for name_word in (_stem(word) for word in name_words if len(word) >= 4):
        if name_word in query_stems:
            return True
        if difflib.get_close_matches(name_word, query_stems, n=1, cutoff=0.85):
            return True
    return False


def _normalise_product_id(product_id: str) -> str:
    return product_id.strip().upper()


def _public_product(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": record["product_id"].strip().upper(),
        "name": record.get("name"),
        "stock": record.get("stock"),
        "restock_date": record.get("restock_date"),
    }


def get_product_by_id(product_id: str | None) -> dict[str, Any]:
    """Find a demonstration product by its canonical product ID."""
    if product_id is None or (isinstance(product_id, str) and not product_id.strip()):
        return {"outcome": "missing_product_id", "product": None, "error": "Please provide a product ID."}
    if not isinstance(product_id, str):
        return {"outcome": "invalid_product_id", "product": None, "error": "Product ID must be text."}
    normalized_id = _normalise_product_id(product_id)
    if not PRODUCT_ID_PATTERN.fullmatch(normalized_id):
        return {
            "outcome": "invalid_product_id", "product_id": normalized_id,
            "product": None, "error": "Product ID must match the format PROD-001.",
        }

    record = load_products().get(normalized_id)
    if record is None:
        return {
            "outcome": "not_found", "product_id": normalized_id,
            "product": None, "error": "No demonstration product matches that ID.",
        }
    return {"outcome": "found", "product": _public_product(record), "error": None}


def search_products(query: str | None) -> dict[str, Any]:
    """Resolve a name/ID or natural-language product mention.

    Token overlap is intentionally used only to identify candidates. Multiple
    candidates are returned as ambiguous and never resolved by guessing.
    """
    if query is None or (isinstance(query, str) and not query.strip()):
        return {"outcome": "missing_query", "matches": [], "error": "Please provide a product name or ID."}
    if not isinstance(query, str):
        return {"outcome": "invalid_query", "matches": [], "error": "Product query must be text."}

    normalized_query = " ".join(_WORD_PATTERN.findall(query.casefold()))
    query_words = set(normalized_query.split())
    products = load_products()
    candidates: list[dict[str, Any]] = []

    for product_id, record in products.items():
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized_name = " ".join(_WORD_PATTERN.findall(name.casefold()))
        name_words = set(normalized_name.split())
        exact_id = normalized_query == " ".join(_WORD_PATTERN.findall(product_id.casefold()))
        exact_name = normalized_query == normalized_name
        name_in_query = bool(normalized_name and normalized_name in normalized_query)
        token_match = bool(query_words & name_words) or _close_token_match(query_words, name_words)
        if exact_id or exact_name or name_in_query or token_match:
            candidates.append(_public_product(record))

    if not candidates:
        # Category words ("phone", "earphones", "laptop wala") need meaning, not tokens.
        names = tuple(sorted(r["name"] for r in products.values() if isinstance(r.get("name"), str)))
        matched_name = match_catalog_product(query, names)
        candidates = [_public_product(r) for r in products.values() if r.get("name") == matched_name]
    if not candidates:
        return {"outcome": "not_found", "matches": [], "error": "No matching demonstration product was found."}
    if len(candidates) > 1:
        choices = [{"product_id": item["product_id"], "name": item["name"]} for item in candidates]
        return {"outcome": "ambiguous", "matches": choices, "error": "More than one product matches; please clarify."}
    return {"outcome": "found", "matches": candidates, "error": None}


def get_product_stock(product_id: str | None) -> dict[str, Any]:
    """Return stock only when the selected demonstration record supports it."""
    result = get_product_by_id(product_id)
    if result["outcome"] != "found":
        return {key: value for key, value in result.items() if key != "product"}

    product = result["product"]
    stock = product.get("stock")
    if isinstance(stock, bool) or not isinstance(stock, int) or stock < 0:
        return {
            "outcome": "stock_unavailable", "product_id": product["product_id"],
            "name": product["name"], "error": "Stock is not available in the product record.",
        }
    return {
        "outcome": "found", "product_id": product["product_id"],
        "name": product["name"], "stock": stock,
        "restock_date": product.get("restock_date"), "error": None,
    }


