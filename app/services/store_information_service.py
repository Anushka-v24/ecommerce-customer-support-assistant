"""Read-only structured store-information lookup over demonstration JSON."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError, load_store_information


TOPIC_CUES = {
    "sales": ("sale", "sales", "deal", "deals"),
    "promotions": ("promotion", "promotions", "offer", "offers"),
    "discount_information": ("discount", "discounts", "coupon", "coupons", "promo code"),
    "store_details": ("store", "hours", "contact", "address", "about"),
}


def get_store_information(query: str | None = None) -> dict[str, Any]:
    """Return relevant structured sections, explicitly marked demonstration-only."""
    try:
        document = load_store_information()
    except DataRepositoryError:
        return {"outcome": "data_unavailable", "sections": {}, "error": "Demonstration store information is unavailable."}
    if not isinstance(query, str) or not query.strip():
        selected = list(TOPIC_CUES)
    else:
        normalized = query.casefold()
        selected = [
            section for section, cues in TOPIC_CUES.items()
            if any(cue in normalized for cue in cues)
        ]
        if not selected:
            selected = ["store_details"]
    sections = {key: document[key] for key in selected if key in document}
    return {
        "outcome": "found" if sections else "data_unavailable",
        "sections": sections,
        "data_notice": document.get("notice"),
        "error": None if sections else "No requested store information is configured.",
    }
