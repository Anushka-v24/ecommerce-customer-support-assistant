"""Answer delivery-option, shipping-cost, and payment questions from demonstration JSON."""

from __future__ import annotations

from typing import Any

from app.services.data_repository import DataRepositoryError, load_shipping_payment

DEMO_NOTE = "These are demonstration terms, not a real store's policy."

# Romanised and Devanagari spellings customers use for the demo cities.
CITY_ALIASES = {
    "Delhi": ("delhi", "dilli", "दिल्ली"),
    "Mumbai": ("mumbai", "bombay", "मुंबई"),
    "Bengaluru": ("bengaluru", "bangalore", "बैंगलोर", "बेंगलुरु"),
    "Chennai": ("chennai", "चेन्नई"),
    "Hyderabad": ("hyderabad", "हैदराबाद"),
    "Kolkata": ("kolkata", "calcutta", "कोलकाता"),
    "Pune": ("pune", "पुणे"),
    "Ahmedabad": ("ahmedabad", "अहमदाबाद"),
    "Jaipur": ("jaipur", "जयपुर"),
    "Lucknow": ("lucknow", "लखनऊ"),
    "Chandigarh": ("chandigarh", "चंडीगढ़"),
    "Noida": ("noida", "नोएडा"),
    "Gurugram": ("gurugram", "gurgaon", "गुड़गांव", "गुरुग्राम"),
    "Patna": ("patna", "पटना"),
    "Indore": ("indore", "इंदौर"),
}
_INTERNATIONAL_CUES = ("international", "abroad", "overseas", "outside india", "other countr", "usa", "uk", "dubai", "videsh", "विदेश")


def _document() -> dict[str, Any] | None:
    try:
        return load_shipping_payment()
    except DataRepositoryError:
        return None


def _mentioned_city(text: str) -> str | None:
    for city, aliases in CITY_ALIASES.items():
        if any(alias in text for alias in aliases):
            return city
    return None


def _method_line(method: dict[str, Any], with_fee: bool) -> str:
    line = f"{method['name']} ({method['days']}"
    if with_fee:
        line += f", {method['fee']}"
        if method.get("free_above"):
            line += f", free on orders above {method['free_above']}"
    if method.get("cities"):
        line += f", only in {', '.join(method['cities'])}"
    return line + ")"


def answer_shipping_question(query: str | None, intent: str | None) -> dict[str, Any]:
    """Return a delivery-options or shipping-cost answer for the demo store."""
    document = _document()
    if not document:
        return {"outcome": "data_unavailable", "response": "Shipping information is unavailable right now."}
    shipping = document["shipping"]
    text = (query or "").casefold()
    with_fee = intent == "shipping_cost"
    city = _mentioned_city(text)

    if any(cue in text for cue in _INTERNATIONAL_CUES) and not shipping.get("international"):
        response = "We currently deliver only within India; international shipping is not available."
    elif city and city not in shipping["serviceable_cities"]:
        response = f"Sorry, {city} is not in our delivery area yet. We deliver to {', '.join(shipping['serviceable_cities'])}."
    else:
        methods = [m for m in shipping["methods"] if not city or not m.get("cities") or city in m["cities"]]
        options = "; ".join(_method_line(m, with_fee) for m in methods)
        prefix = f"Yes, we deliver to {city}. " if city else ""
        lead = "Shipping charges" if with_fee else "Delivery options"
        response = f"{prefix}{lead}: {options}."
    return {"outcome": "answered", "city": city, "response": f"{response} {DEMO_NOTE}"}


def answer_payment_question(query: str | None) -> dict[str, Any]:
    """Return payment-method or invoice information for the demo store."""
    document = _document()
    if not document:
        return {"outcome": "data_unavailable", "response": "Payment information is unavailable right now."}
    payment = document["payment"]
    text = (query or "").casefold()
    if any(cue in text for cue in ("invoice", "bill", "receipt", "बिल", "इनवॉइस", "रसीद")):
        response = payment["invoice"]
    else:
        response = (
            f"We accept {', '.join(payment['methods'])}. "
            f"Cash on delivery is available for orders up to {payment['cash_on_delivery_limit']}."
        )
    return {"outcome": "answered", "response": f"{response} {DEMO_NOTE}"}
