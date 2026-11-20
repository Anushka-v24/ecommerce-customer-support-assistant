"""Safe conditional routing based only on the model's trained labels."""

import json
import re
from pathlib import Path

from app.graph.state import AgentState


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
}
UNSUPPORTED_INTENTS = {
    "account_management", "add_to_cart", "order_management",
    "place_order", "remove_from_cart", "submit_feedback",
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


def _is_order_cancellation_request(query: str | None) -> bool:
    return isinstance(query, str) and bool(_CANCEL_REQUEST.search(query)) and not _CANCEL_POLICY_QUESTION.search(query)


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
        return "out_of_scope"
    if status not in _VERIFIED_STATUSES:
        return "unknown_intent"

    final_intent = state.get("final_intent")
    if not final_intent or final_intent not in INTENT_LABELS:
        return "unknown_intent"
    if final_intent == "out_of_scope":
        return "out_of_scope"
    if final_intent == "order_management" and _is_order_cancellation_request(state.get("query")):
        return "order_cancellation"
    if final_intent in UNSUPPORTED_INTENTS:
        return "unsupported"
    if final_intent == "issue_support" and _is_general_return_refund_policy_question(
        state.get("query")
    ):
        # Keep final_intent/predicted_intent intact; this only refines the
        # destination for an explicit policy question to the existing RAG node.
        return "store_information"
    return INTENT_TO_ROUTE.get(final_intent, "unknown_intent")
