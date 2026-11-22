"""Split one message that carries two separate requests.

"Track ORD-1001 and cancel ORD-1002" becomes two turns. A split is used only
when both halves classify confidently (>= MULTI_INTENT_MIN_CONFIDENCE) as
different, actionable intents, so "black and white shirt" or "return and
refund my order" stay one request. At most two parts; the second is queued
and answered after the first finishes.
"""

from __future__ import annotations

import re
from typing import Callable

_SPLIT = re.compile(
    r"\s*[,;]?\s*\b(?:and\s+also|and\s+then|also|then|and|aur|plus|uske\s+baad)\b\s*|\s*(?:और|फिर)\s*",
    re.IGNORECASE,
)
_ORDER_ID = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
# Small talk next to a request isn't a second request.
_NOT_A_REQUEST = {"greeting", "goodbye_thanks", "out_of_scope"}


def split_request(text: str) -> list[str] | None:
    parts = [part.strip(" ,.;!?") for part in _SPLIT.split(text.strip(), maxsplit=1)]
    if len(parts) != 2 or any(len(part.split()) < 2 for part in parts):
        return None
    # "track ORD-1001 and cancel it": the second half refers to the same order.
    first_ids, second_ids = _ORDER_ID.findall(parts[0]), _ORDER_ID.findall(parts[1])
    if first_ids and not second_ids:
        parts[1] = f"{parts[1]} {first_ids[-1].upper()}"
    return parts


def plan_requests(text: str, predict: Callable[[str], tuple[str, float]], min_confidence: float) -> list[str] | None:
    """Return [first, second] when the message holds two distinct requests, else None."""
    parts = split_request(text)
    if not parts:
        return None
    predictions = [predict(part) for part in parts]
    intents = [intent for intent, _ in predictions]
    if any(confidence < min_confidence for _, confidence in predictions):
        return None
    if intents[0] == intents[1] or _NOT_A_REQUEST & set(intents):
        return None
    return parts
