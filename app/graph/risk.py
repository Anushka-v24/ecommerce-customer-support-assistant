"""Risk tiers for workflow actions and the human-approval step for risky ones.

READ actions only look data up. CHANGE actions create a record the customer can
see (ticket, feedback). HIGH_RISK actions change an order and need an explicit
yes: the node calls ``request_approval``, which pauses the graph with LangGraph's
``interrupt()``. The next chat message resumes it via ``Command(resume=...)``.
Because a resumed node runs again from its start, everything before the
approval (including the order-status lookup) is re-checked before acting.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

READ, CHANGE, HIGH_RISK = "READ", "CHANGE", "HIGH_RISK"

RISK_TIERS = {
    "order_tracking": READ, "product_inquiry": READ, "payment": READ, "shipping": READ,
    "store_information": READ, "product_recommendation": READ, "warranty": READ,
    "greeting": READ, "goodbye": READ,
    "issue_support": CHANGE, "customer_care": CHANGE, "feedback": CHANGE,
    "order_cancellation": HIGH_RISK, "exchange": HIGH_RISK,
}

ABANDON = "__abandon__"
_YES = {
    "yes", "y", "yes please", "yes, cancel it", "yes cancel it", "yes, go ahead", "go ahead", "confirm", "ok", "okay", "sure",
    "haan", "han", "ha", "haan ji", "ji haan", "ji", "kar do", "haan kar do",
    "हाँ", "हां", "जी", "जी हाँ", "हाँ कर दो", "हां कर दो",
}
_NO = {
    "no", "n", "no, keep it", "no keep it", "keep it", "don't", "dont", "stop", "not now",
    "nahi", "nhi", "nahin", "na", "mat karo", "rehne do",
    "नहीं", "नही", "ना", "रहने दो", "मत करो",
}


def approval_answer(text: Any) -> str:
    """Classify a reply to an approval question as 'yes', 'no' or 'other'."""
    if not isinstance(text, str):
        return "other"
    normalized = text.casefold().strip(" .!\t")
    if normalized in _YES:
        return "yes"
    if normalized in _NO:
        return "no"
    return "other"


def request_approval(action: str, prompt: str, quick_replies: list[dict[str, str]], **details: Any) -> bool:
    """Pause until the customer answers; True only for an explicit yes."""
    reply = interrupt({
        "type": "approval", "risk": RISK_TIERS.get(action, HIGH_RISK), "action": action,
        "prompt": prompt, "quick_replies": quick_replies, **details,
    })
    return approval_answer(reply) == "yes"


def idempotency_key(action: str, order_id: str) -> str:
    return f"{action}:{order_id}"
