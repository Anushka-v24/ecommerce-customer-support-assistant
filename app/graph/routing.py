"""Safe conditional routing based only on the model's trained labels."""

import json
import re
from pathlib import Path

from app.config import Config
from app.graph.state import AgentState
from app.services.unanswered import follow_up_choice, is_offer_question


MAPPING_PATH = Path(__file__).resolve().parents[1] / "classifier" / "intent_mapping.json"
with MAPPING_PATH.open("r", encoding="utf-8-sig") as mapping_file:
    _mapping = json.load(mapping_file)
if "id2label" in _mapping:
    INTENT_LABELS = [
        _mapping["id2label"][str(index)]
        for index in range(len(_mapping["id2label"]))
    ]
else:
    INTENT_LABELS = [
        label for label, _ in sorted(_mapping["label2id"].items(), key=lambda item: item[1])
    ]

INTENT_TO_ROUTE = {
    "order_tracking": "order_tracking",
    "product_inquiry": "product_inquiry",
    "billing_and_payment": "payment",
    "issue_support": "issue_support",
    "customer_care_support": "customer_care",
    "delivery_options": "shipping",
    "shipping_cost": "shipping",
    "store_information": "store_information",
    "greeting": "greeting",
    "goodbye_thanks": "goodbye",
    "product_recommendation": "product_recommendation",
    "warranty_inquiry": "warranty",
    "exchange_request": "exchange",
    "submit_feedback": "feedback",
}
UNSUPPORTED_INTENTS = {
    "account_management", "add_to_cart", "order_management",
    "place_order", "remove_from_cart",
}

if set(INTENT_LABELS) != set(INTENT_TO_ROUTE) | UNSUPPORTED_INTENTS | {"out_of_scope"}:
    raise RuntimeError("Routing table must cover the actual classifier labels exactly.")

ROUTE_TO_NODE = {
    "order_tracking": "order_tracking_node",
    "product_inquiry": "product_inquiry_node",
    "payment": "payment_node",
    "issue_support": "issue_resolver_node",
    "customer_care": "customer_care_node",
    "shipping": "shipping_node",
    "store_information": "store_information_node",
    "unsupported": "unsupported_node",
    "unknown_intent": "unknown_intent_node",
    "out_of_scope": "out_of_scope_node",
    "verification_error": "verification_error_node",
    "order_cancellation": "order_cancellation_node",
    "greeting": "greeting_node",
    "goodbye": "goodbye_node",
    "product_recommendation": "product_recommendation_node",
    "warranty": "warranty_node",
    "exchange": "exchange_node",
    "feedback": "feedback_node",
    "escalation": "escalation_node",
    "follow_up": "follow_up_node",
    "cancel_followup": "cancel_followup_node",
}
_VERIFIED_STATUSES = {"ACCEPTED_BY_CLASSIFIER", "CORRECT_INTENT", "CORRECTED_INTENT"}

# Refine only explicit general-policy questions that the broad issue_support
# label would otherwise send to transactional issue resolution. Keep questions
# about initiating actions and existing refund status on their business routes.
_POLICY_ACTION_REQUEST = re.compile(
    r"\bwhere\s+(?:is|are)\s+(?:my\s+)?refund\b|"
    r"\b(?:i\s+(?:want|need)\s+to|i\s+would\s+like\s+to|"
    r"i'd\s+like\s+to)\s+(?:return|refund)\b|"
    r"\b(?:please|can\s+you)\s+(?:initiate|start|create|submit|process|request|issue)\s+"
    r"(?:a|my|this|the)?\s*(?:return|refund)\b",
    re.IGNORECASE,
)
_GENERAL_POLICY_QUESTION = re.compile(
    r"\b(?:return|refund)\s+polic(?:y|ies)\b|"
    r"\bhow\s+many\s+days?\b.{0,80}\breturn\b|"
    r"\bhow\s+(?:do|can)\s+i\s+return\b|"
    r"\bhow\s+long\b.{0,80}\b(?:return|refund)s?\b|"
    r"\bhow\s+do\s+(?:your|the)?\s*refunds?\s+(?:work|operate)\b|"
    r"\b(?:return|refund)\s+(?:window|period|timeframe|conditions|eligibility)\b",
    re.IGNORECASE,
)


# Cancelling a specific order is supported; fee/policy questions and item
# changes stay on the unsupported order_management reply.
_CANCEL_REQUEST = re.compile(r"cancel|कैंसिल|कैंसल|रद्द|\bradd\b", re.IGNORECASE)
_CANCEL_POLICY_QUESTION = re.compile(r"\b(?:fee|fees|charge|charges|penalty|policy)\b|शुल्क|चार्ज", re.IGNORECASE)


# "How do I…" questions about things the chat can't do itself are answered from the help documents.
_HOW_TO_QUESTION = re.compile(
    r"\bhow\b|\bkaise\b|कैसे|\bwhere\s+(?:do|can)\s+i\b|\bsteps?\b|\bprocess\b|\bguide\b|\bpolicy\b|\bpolicies\b",
    re.IGNORECASE,
)
HOW_TO_INTENTS = {"place_order", "account_management", "add_to_cart", "remove_from_cart", "order_management", "submit_feedback"}


# Asking for the delivery person, or for an order to arrive sooner, is a delivery speed-up
# request: there is no live courier channel, so it ends in a priority ticket for the delivery team.
_DELIVERY_PERSON = re.compile(r"\b(?:delivery\s+(?:agent|partner|boy|person|executive|guy|man)|courier|rider)\b", re.IGNORECASE)
_SPEED_WORDS = re.compile(r"\b(?:sooner|faster|quicker|quickly|earlier|asap|urgent(?:ly)?|expedite|speed[\s-]?up|hurry|rush)\b", re.IGNORECASE)
_DELIVERY_CONTEXT = re.compile(r"\b(?:deliver(?:y|ed)?|send|ship(?:ped|ping)?|arrive|reach|package|parcel|dispatch)\b", re.IGNORECASE)
_CONTACT_VERB = re.compile(r"\b(?:talk|speak|call|contact|message)\b", re.IGNORECASE)
_NOT_DELIVERY = re.compile(r"\b(?:refund|return|cancel|exchange)\b", re.IGNORECASE)
_SPEEDUP_INTENTS = {"customer_care_support", "issue_support", "order_tracking", "delivery_options"}


def is_delivery_speedup_request(query: str | None) -> bool:
    if not isinstance(query, str) or _NOT_DELIVERY.search(query):
        return False
    if _DELIVERY_PERSON.search(query) and (_SPEED_WORDS.search(query) or _CONTACT_VERB.search(query)):
        return True
    return bool(_SPEED_WORDS.search(query) and _DELIVERY_CONTEXT.search(query))


def _is_how_to_question(query: str | None) -> bool:
    return isinstance(query, str) and bool(_HOW_TO_QUESTION.search(query))


# "How do I cancel an order?" and "can I cancel after it ships?" ask how cancelling works; they don't
# ask to cancel one. With an order ID in the message it is a request to cancel that order.
_CANCEL_INFO_QUESTION = re.compile(
    r"\bhow\s+(?:to|do|can|could|should|would)\b|\b(?:steps?|procedure|process)\b|\bkaise\b|कैसे|"
    r"\b(?:can|could)\s+i\s+cancel\b|\bis\s+it\s+possible\b|\bam\s+i\s+able\b",
    re.IGNORECASE,
)
# "but I have not received the order", said right after a cancellation question, asks whether
# an order that hasn't arrived can still be cancelled; it isn't a missing-delivery report.
_NOT_RECEIVED = re.compile(
    r"\b(?:not|never|haven'?t|hasn'?t|didn'?t|isn'?t)\b.{0,25}\b(?:rec(?:ei|ie)ve[d]?|deliver(?:ed)?|arriv(?:e|ed)|got|reach(?:ed)?)\b|"
    r"\bnahi\s+(?:aaya|aayi|mila|mili|pahuncha|pahunchi)\b|नहीं\s+(?:आया|मिला|पहुँचा)",
    re.IGNORECASE,
)
_ORDER_ID_MENTION = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)


def _is_cancel_information_question(query: str | None) -> bool:
    return isinstance(query, str) and bool(_CANCEL_INFO_QUESTION.search(query)) and not _ORDER_ID_MENTION.search(query)


def _is_order_cancellation_request(query: str | None) -> bool:
    return (
        isinstance(query, str) and bool(_CANCEL_REQUEST.search(query))
        and not _CANCEL_POLICY_QUESTION.search(query) and not _is_cancel_information_question(query)
    )


def _is_general_return_refund_policy_question(query: str | None) -> bool:
    if not isinstance(query, str) or _POLICY_ACTION_REQUEST.search(query):
        return False
    return bool(_GENERAL_POLICY_QUESTION.search(query))


def route_classifier_result(state: AgentState) -> str:
    """Return a conditional-edge key; fail closed for untrusted outcomes."""
    status = state.get("verifier_status")
    if status in {"UNVERIFIED", "CLASSIFICATION_ERROR"}:
        return "verification_error"
    if status == "UNKNOWN_ECOMMERCE_INTENT":
        return "unknown_intent"
    if status == "OUT_OF_SCOPE":
        return "store_information" if is_offer_question(state.get("query")) else "out_of_scope"
    if status not in _VERIFIED_STATUSES:
        return "unknown_intent"
    if follow_up_choice(state.get("query")) == "another":
        return "follow_up"
    if (
        state.get("last_topic") == "cancellation" and _NOT_RECEIVED.search(state.get("query") or "")
        and not _ORDER_ID_MENTION.search(state.get("query") or "")
    ):
        return "cancel_followup"
    if is_offer_question(state.get("query")):
        return "store_information"

    final_intent = state.get("final_intent")
    if not final_intent or final_intent not in INTENT_LABELS:
        return "unknown_intent"
    if final_intent == "out_of_scope":
        return "out_of_scope"
    if final_intent in _SPEEDUP_INTENTS and is_delivery_speedup_request(state.get("query")):
        return "customer_care"
    if final_intent == "order_management" and _is_order_cancellation_request(state.get("query")):
        return "order_cancellation"
    if final_intent in HOW_TO_INTENTS and (
        _is_how_to_question(state.get("query")) or _is_cancel_information_question(state.get("query"))
    ):
        return "store_information"
    if final_intent in UNSUPPORTED_INTENTS:
        return "unsupported"
    if final_intent == "issue_support" and _is_general_return_refund_policy_question(
        state.get("query")
    ):
        # Keep final_intent/predicted_intent intact; this only refines the
        # destination for an explicit policy question to the existing RAG node.
        return "store_information"
    return INTENT_TO_ROUTE.get(final_intent, "unknown_intent")


# Routes that end in a ticket ask the customer to confirm the understood request first.
CONFIRMATION_ROUTES = {"issue_support", "customer_care", "exchange"}


def route_with_intent_confirmation(state: AgentState) -> str:
    """Send ticket-raising routes through the confirmation node; others go straight on."""
    route = route_classifier_result(state)
    if (state.get("frustration") or {}).get("escalate") and not state.get("escalated"):
        return "escalation"
    if Config.INTENT_CONFIRMATION_ENABLED and route in CONFIRMATION_ROUTES:
        return "confirm_intent"
    return route
