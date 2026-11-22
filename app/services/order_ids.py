"""Read order numbers the way customers type them: 1010, #1010, "track 1010", "refund for 1010".

Order IDs look like ORD-1010. A bare four-digit number is treated as an order number
when the whole message is just that number or the message is about an order, so the
assistant looks the order up instead of asking for an ID the customer already gave.
"""

from __future__ import annotations

import re

_HAS_ORDER_ID = re.compile(r"\bORD[\s_-]*\d{4}\b", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"^\s*#?\s*(\d{4})\s*[.!?]?\s*$")
# Not part of a longer number, price, date or percentage.
_NUMBER = re.compile(r"(?<![\w₹$.,/:-])#?(\d{4})(?![\w%/:]|[.,]\d|\s*(?:rs\b|rupees\b|inr\b|₹))", re.IGNORECASE)
_CURRENCY_BEFORE = re.compile(r"(?:rs\.?|inr|usd|rupees?|₹|\$)\s*$", re.IGNORECASE)
_ORDER_WORDS = re.compile(
    r"\b(?:order|orders|ord|track|tracking|status|where|parcel|package|delivery|deliver|shipment|shipped|"
    r"refund|return|cancel|exchange|damaged|missing|late|delayed|arrive|arrived|bheja|mera)\b",
    re.IGNORECASE,
)
_STRICT_ORDER_WORDS = re.compile(r"\b(?:order|orders|ord|track|tracking|status)\b", re.IGNORECASE)


def with_order_prefix(query: str | None) -> str | None:
    """The query with a bare order number written as ORD-####, or None when nothing needs changing."""
    if not isinstance(query, str) or not query.strip() or _HAS_ORDER_ID.search(query):
        return None
    bare = _BARE_NUMBER.match(query)
    if bare:
        return f"track order ORD-{bare.group(1)}"
    matches = [m for m in _NUMBER.finditer(query) if not _CURRENCY_BEFORE.search(query[: m.start()])]
    if len({m.group(1) for m in matches}) != 1 or not _ORDER_WORDS.search(query):
        return None
    number = matches[0].group(1)
    # A number that looks like a year needs a clearer order word before it counts as an order number.
    if 1900 <= int(number) <= 2100 and not _STRICT_ORDER_WORDS.search(query):
        return None
    first = matches[0]
    return f"{query[: first.start()]}ORD-{number}{query[first.end():]}"
