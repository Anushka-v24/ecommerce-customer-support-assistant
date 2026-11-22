"""Customer feedback written in the chat (opinions, reviews, suggestions).

This is separate from the thumbs up/down ratings on answers: these records are
what the customer chose to tell the store. Text is PII-masked before storage
and kept under ``data/runtime/`` with the other runtime logs.
"""

from __future__ import annotations

import re
from uuid import uuid4

from app.config import Config
from app.services.pii import mask_pii
from app.services.turn_log import append_record

# Words that only say "I want to give feedback" without giving any.
_REQUEST_WORDS = {
    "i", "want", "wanna", "would", "like", "to", "give", "leave", "share", "submit", "write", "post", "send",
    "a", "an", "some", "my", "the", "feedback", "review", "reviews", "suggestion", "suggestions", "rating",
    "rate", "about", "on", "for", "your", "you", "app", "store", "service", "please", "pls", "can", "how",
    "do", "where", "mujhe", "dena", "deni", "dene", "karna", "karni", "likhna", "hai", "h", "chahta",
    "chahti", "hu", "hoon", "kaise", "kahan", "apna", "apni", "ek", "de", "du", "doon",
    "मुझे", "देना", "है", "फ़ीडबैक", "फीडबैक", "रिव्यू", "सुझाव", "चाहता", "चाहती", "हूँ", "हूं",
    "कैसे", "दूँ", "दूं", "अपना", "एक", "लिखना",
}
_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


def has_feedback_content(text: str) -> bool:
    """True when the message carries an opinion, not only a request to give one."""
    words = [w.casefold() for w in _WORD.findall(text or "")]
    return sum(1 for w in words if w not in _REQUEST_WORDS) >= 2


def record_customer_feedback(text: str, session_id: str | None, customer_id: str | None = None) -> str:
    """Store masked feedback and return a short reference for the customer."""
    reference = f"FB-{uuid4().hex[:8].upper()}"
    if Config.TURN_LOG_ENABLED:
        append_record(Config.TURN_LOG_DIR, "customer_feedback.jsonl", {
            "reference": reference, "session_id": session_id, "customer_id": customer_id,
            "text": mask_pii(text.strip())[:2000],
        })
    return reference
