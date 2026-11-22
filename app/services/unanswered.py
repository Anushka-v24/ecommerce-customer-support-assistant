"""Customer-facing replies for questions the chat can't answer.

The reply depends on why there is no answer: offer and sale updates live on the
cart page, a plain miss gets a polite pointer to what the chat can do, and a
lookup failure asks the customer to try again. None of them mention documents,
searches or other internals.
"""

from __future__ import annotations

import re

# Offer, sale and coupon questions are redirected even when a document mentions them.
_OFFER_TOPIC = re.compile(
    r"\b(?:offers?|sales?|discounts?|coupons?|promo(?:s|tions?)?|promo\s*codes?|vouchers?|cashbacks?|clearance|festive|black\s*friday)\b",
    re.IGNORECASE,
)
# A refund, return or order action that happens to mention a coupon is a support request, not an offer question.
_SUPPORT_ACTION = re.compile(r"\b(?:refund|return|cancel|exchange|damaged|missing)\b|\bORD-\d{4}\b", re.IGNORECASE)

OFFERS_REPLY = (
    "Thanks for asking! This chat isn't set up for offer, sale or coupon updates. "
    "To stay up to date on offers, please follow the cart page."
)
NO_ANSWER_REPLY = (
    "I'm sorry, I don't have details on that one, but I'm happy to help with your orders, returns, refunds, delivery and products."
)
LOOKUP_TROUBLE_REPLY = "I'm having a little trouble looking that up right now. Please try again in a moment."
FOLLOW_UP_PROMPT = "Would you like to ask another question, or shall we end the conversation?"
FOLLOW_UP_REPLIES = [
    {"label": "Ask another question", "value": "I have another question"},
    {"label": "End conversation", "value": "end conversation"},
]
CLOSING_REPLIES = [
    {"label": "Thanks", "value": "thanks"},
    {"label": "End conversation", "value": "end conversation"},
]
FOLLOW_UP_ANSWER = "Of course! Please go ahead, what would you like to know?"

# What the two follow-up buttons send, and the same things typed in other words.
_END_CONVERSATION = re.compile(
    r"(?:please\s+)?(?:end|close|finish)\s+(?:the\s+|this\s+)?(?:conversation|chat)|that'?s\s+all|nothing\s+else|no,?\s+thanks?|i'?m\s+done",
    re.IGNORECASE,
)
_THANKS = re.compile(r"(?:thanks?|thank\s+you|thx|ty)(?:\s+(?:a\s+lot|so\s+much|very\s+much))?", re.IGNORECASE)
_ANOTHER_QUESTION = re.compile(
    r"(?:yes,?\s*)?(?:i\s+have\s+)?(?:another|one\s+more|a\s+new)\s+(?:question|doubt|query)|ask\s+(?:another|something\s+else)",
    re.IGNORECASE,
)


def follow_up_choice(query: str | None) -> str | None:
    """'end', 'thanks' or 'another' when the message is one of the follow-up answers."""
    text = (query or "").strip(" .!?")
    if _END_CONVERSATION.fullmatch(text):
        return "end"
    if _THANKS.fullmatch(text):
        return "thanks"
    if _ANOTHER_QUESTION.fullmatch(text):
        return "another"
    return None


def is_offer_question(query: str | None) -> bool:
    return isinstance(query, str) and bool(_OFFER_TOPIC.search(query)) and not _SUPPORT_ACTION.search(query)


def unanswered_reply(query: str | None, *, lookup_failed: bool = False) -> str:
    """The polite reply for a question with no answer, chosen by the kind of miss."""
    if is_offer_question(query):
        base = OFFERS_REPLY
    else:
        base = LOOKUP_TROUBLE_REPLY if lookup_failed else NO_ANSWER_REPLY
    return f"{base}\n\n{FOLLOW_UP_PROMPT}"
