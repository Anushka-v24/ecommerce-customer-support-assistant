"""Classifier and workflow nodes for the multilingual ecommerce support graph."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from app.classifier.classifier_service import classify_and_verify
from app.config import Config
from app.graph.state import AgentState
from app.graph.risk import idempotency_key, request_approval
from app.graph.routing import INTENT_LABELS, is_delivery_speedup_request, route_classifier_result
from app.rag.service import (
    RAGConfigurationError,
    RAGProviderError,
    get_rag_service,
)
from app.rag import local_search
from app.rag.workflow import run_agentic_rag
from app.services.feedback_service import has_feedback_content, record_customer_feedback
from app.services.frustration import conversation_frustration, handoff_summary
from app.services.issue_resolver import resolve_issue_type
from app.services.product_service import search_products
from app.services.reply_translation import translate_reply
from app.services.catalog_help_service import answer_warranty_question, recommend_products
from app.services.shipping_payment_service import answer_payment_question, answer_shipping_question
from app.services.store_response import format_store_information
from app.services.ticket_service import create_ticket, public_ticket
from app.tools.customer_care_tools import customer_handoff_tool
from app.tools.issue_tools import (
    damaged_order_tool,
    missing_order_tool,
    prepare_return_tool,
    refund_status_tool,
)
from app.tools.fakestore_tools import (
    check_catalog_stock, get_catalog_product, list_catalog_products, resolve_explicit_catalog_id,
    search_catalog_products,
)
from app.tools.order_tools import lookup_order_status
from app.tools.product_tools import check_product_stock
from app.tools.store_tools import store_information_tool


logger = logging.getLogger(__name__)
_ORDER_ID_IN_TEXT = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
# What a customer may type when asked for an order ID: 1002, #1002, ORD 1002, order id is ord1002.
_ORDER_SLOT_REPLY = re.compile(
    r"(?:(?:my\s+)?order(?:\s*(?:id|number|no\.?))?\s*(?:is|:|#)?\s*)?(?:ORD[\s_-]*)?#?(\d{4})",
    re.IGNORECASE,
)
_RETURN_REASON = re.compile(r"\b(?:because|reason is|since)\s+(.+)$", re.IGNORECASE)
_GENERIC_DAMAGE = re.compile(
    r"\b(?:my\s+)?(?:order|item|product|package)\s+(?:arrived|came|was\s+delivered)\s+"
    r"(?:damaged|broken|cracked|defective)\b|\b(?:my\s+)?(?:item|product)\s+is\s+damaged\b",
    re.IGNORECASE,
)
_GENERIC_MISSING = re.compile(
    r"\b(?:my\s+)?(?:order|package|parcel|delivery)\s+(?:has\s+not\s+arrived|hasn't\s+arrived|"
    r"never\s+arrived|is\s+missing|not\s+arrived)\b|\b(?:i\s+have\s+not|i\s+haven't|"
    r"never)\s+received\s+(?:my\s+)?(?:order|package|parcel)\b",
    re.IGNORECASE,
)
_REQUEST_SHAPE = re.compile(
    r"^(?:where\b|when\b|how\b|can\s+you\b|could\s+you\b|"
    r"please\s+(?:check|tell|find|track|show|refund|return)\b|"
    r"i\s+(?:want|need|would\s+like)\s+(?:to\s+)?(?:track|check|return|refund|cancel|speak|contact)\b|"
    r"मेरा\s+(?:ऑर्डर|रिफंड)\b|मुझे\s+(?:रिफंड|मदद|बात)\b|क्या\b|कहाँ\b)",
    re.IGNORECASE,
)
_YES_ANSWERS = {"yes", "yes, confirm", "yes confirm", "confirm", "y", "\u0939\u093e\u0901", "\u0939\u093e\u0902", "\u091c\u0940", "\u0939\u093e\u0901 \u092a\u0941\u0937\u094d\u091f\u093f \u0915\u0930\u0947\u0902", "\u0939\u093e\u0902 \u092a\u0941\u0937\u094d\u091f\u093f \u0915\u0930\u0947\u0902"}
_NO_ANSWERS = {"no", "no, cancel", "cancel", "cancel it", "do not proceed", "n", "\u0928\u0939\u0940\u0902", "\u0928\u0939\u0940", "\u0928\u093e", "\u0930\u0926\u094d\u0926 \u0915\u0930\u0947\u0902", "\u0930\u0926\u094d\u0926"}
_VERIFIED_CLASSIFIER_STATUSES = {"ACCEPTED_BY_CLASSIFIER", "CORRECT_INTENT", "CORRECTED_INTENT"}
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Buttons shown under an assistant message; each value is sent back as the next message.
_CONFIRM_REPLIES = [{"label": "Yes, that's right", "value": "yes"}, {"label": "No, it's something else", "value": "no"}]
_CANCEL_REPLIES = [{"label": "Yes, cancel it", "value": "yes"}, {"label": "No, keep it", "value": "no"}]
_EXCHANGE_REPLIES = [{"label": "Yes, arrange it", "value": "yes"}, {"label": "No, not now", "value": "no"}]
_ISSUE_CHOICE_REPLIES = [
    {"label": "Refund status", "value": "refund status"},
    {"label": "Return an item", "value": "return an item"},
    {"label": "Damaged item", "value": "damaged item"},
    {"label": "Missing delivery", "value": "missing delivery"},
]
_CORRECTION_REPLIES = _ISSUE_CHOICE_REPLIES + [
    {"label": "Exchange an item", "value": "exchange an item"},
    {"label": "Talk to a person", "value": "talk to a person"},
    {"label": "Something else", "value": "something else"},
]
# Confirmation and correction prompts are dropped, not suspended, when the
# customer interrupts them; reminding them later would be confusing.
_DISCARDABLE_WORKFLOWS = {"intent_confirmation", "intent_correction"}
# Requests that end in a ticket are confirmed first; refund status is a lookup.
_CONFIRMED_ISSUE_TYPES = {"damaged_order", "missing_order", "return_order"}
# Phrases the existing issue rules resolve deterministically, used when the
# customer picks an issue from the correction menu.
_ISSUE_CHOICE_QUERIES = {
    "refund_status": "where is my refund",
    "return_order": "i want to return my order",
    "damaged_order": "my item arrived damaged",
    "missing_order": "my order has not arrived",
}
_CONFIRMATION_PROMPTS = {
    "en": {
        "damaged_order": "I'm really sorry to hear your order arrived damaged. Just to make sure I've understood: you'd like to report a damaged item{order}. Is that right?",
        "missing_order": "I'm sorry your order hasn't reached you yet. Just to make sure I've understood: you'd like us to look into a missing delivery{order}. Is that right?",
        "return_order": "Happy to help with that. Just to make sure I've understood: you'd like to return an item{order}. Is that right?",
        "customer_care": "I understand you'd like to talk to someone from our support team. Shall I raise a ticket and pass your conversation on to them?",
        "delivery_speedup": "I can't reach the delivery agent live, but I can raise a priority ticket asking our delivery team to speed up your order{order}. Shall I do that?",
        "exchange": "Sure, I can help with that. Just to make sure I've understood: you'd like to exchange an item{order}. Is that right?",
        "order": " for order {order_id}",
        "correction": "No problem, thanks for letting me know. Which of these is closest to what you need?",
        "describe": "No problem. Please tell me in your own words what you need help with, and I'll take it from there.",
    },
    "hi": {
        "damaged_order": "हमें खेद है कि आपका ऑर्डर खराब हालत में पहुँचा। क्या मैंने सही समझा कि आप{order} खराब सामान की शिकायत दर्ज करना चाहते हैं?",
        "missing_order": "हमें खेद है कि आपका ऑर्डर अभी तक नहीं पहुँचा। क्या मैंने सही समझा कि आप चाहते हैं कि हम{order} गुम डिलीवरी की जाँच करें?",
        "return_order": "ज़रूर, मैं मदद करूँगा। क्या मैंने सही समझा कि आप{order} सामान वापस (रिटर्न) करना चाहते हैं?",
        "customer_care": "मैं समझता हूँ कि आप हमारी सपोर्ट टीम से बात करना चाहते हैं। क्या मैं टिकट बनाकर आपकी बातचीत उन्हें भेज दूँ?",
        "delivery_speedup": "मैं डिलीवरी एजेंट से सीधे बात नहीं करा सकता, लेकिन आपका ऑर्डर{order} जल्दी भेजने के लिए डिलीवरी टीम को प्राथमिकता वाला टिकट बना सकता हूँ। क्या मैं बना दूँ?",
        "exchange": "ज़रूर, मैं मदद करूँगा। क्या मैंने सही समझा कि आप{order} सामान बदलना (एक्सचेंज) चाहते हैं?",
        "order": " ऑर्डर {order_id} के लिए",
        "correction": "कोई बात नहीं, बताने के लिए धन्यवाद। इनमें से कौन-सा विकल्प आपकी ज़रूरत के सबसे करीब है?",
        "describe": "कोई बात नहीं। कृपया अपने शब्दों में बताइए कि आपको किस चीज़ में मदद चाहिए।",
    },
}


def _verified_new_request(result: dict[str, Any], pending: dict[str, Any]) -> bool:
    status = result.get("verifier_status")
    if status == "OUT_OF_SCOPE":
        return True
    intent = result.get("final_intent")
    if status not in _VERIFIED_CLASSIFIER_STATUSES or intent not in INTENT_LABELS:
        return False
    parent_intent = pending.get("classifier_intent")
    if intent == parent_intent:
        if intent == "order_tracking":
            # A non-ID message that XLM-R verifies as another order query is a
            # new order request, not an order-ID slot value.
            return True
        # issue_support covers several sub-workflows; use the existing issue resolver
        # only after the primary classifier verifies the parent intent.
        if intent == "issue_support":
            issue = resolve_issue_type(result.get("query") or "")
            return issue.get("outcome") == "resolved" and issue.get("issue_type") != pending.get("issue_type")
        return False
    return True


def _slot_answer_is_valid(slot: str | None, query: str) -> bool:
    text = " ".join(query.split())
    if len(text) < 2 or text.endswith("?") or _REQUEST_SHAPE.search(text):
        return False
    if slot in {"damage_description", "missing_delivery_details", "return_reason"}:
        minimum_words = 2 if slot == "return_reason" else 3
        return len(text) >= 4 and len(text.split()) >= minimum_words
    return False


def _classify(query: str, state: AgentState) -> dict[str, Any]:
    """Classify the latest message; earlier turns only help the Groq verifier resolve references."""
    history = list(state.get("messages") or [])
    return classify_and_verify(query, history=history) if history else classify_and_verify(query)


def _classify_pending_message(state: AgentState, pending: dict[str, Any]) -> dict[str, Any]:
    """Classify only when the current text does not validate against its pending slot."""
    query = (state.get("query") or "").strip()
    try:
        result = _classify(query, state)
    except Exception as exc:
        result = {"verifier_status": "CLASSIFICATION_ERROR", "final_intent": None, "error": str(exc)}
    result = {**result, "query": query}
    if _verified_new_request(result, pending):
        if pending.get("workflow") in _DISCARDABLE_WORKFLOWS:
            return {
                "continuity_action": "classify", "resuming_pending": False,
                "pending_workflow": None, "workflow_status": None, "pending_clarification": None,
                "confirmation_required": False, "confirmation_status": None,
                "preclassified_result": result,
            }
        stack = list(state.get("suspended_workflows") or [])
        stack.append({**pending, "status": "suspended"})
        return {
            "continuity_action": "classify", "resuming_pending": False,
            "pending_workflow": None, "suspended_workflows": stack,
            "workflow_status": "suspended", "pending_clarification": pending.get("prompt"),
            "preclassified_result": result,
        }
    if pending.get("workflow") in _DISCARDABLE_WORKFLOWS:
        response = "Sorry, I didn't quite catch that. " + (pending.get("prompt") or "Please choose one of the options.")
    elif state.get("confirmation_required") and state.get("confirmation_status") != "confirmed":
        response = "This action still requires explicit confirmation. " + (pending.get("prompt") or "Reply 'confirm' to proceed or 'cancel' to stop.")
    else:
        response = "I couldn't use that as the requested information. " + (pending.get("prompt") or "Please clarify your reply.")
    return _continuity_clarification(state, response, pending)


def _pending(workflow: str, awaiting_slot: str, collected: dict[str, Any], prompt: str, **extra: Any) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "status": "awaiting_input",
        "awaiting_slot": awaiting_slot,
        "collected": dict(collected),
        "prompt": prompt,
        **extra,
    }


def _fakestore_terminal_result(
    state: AgentState, result: dict[str, Any], *, list_request: bool = False,
    product_task: str | None = None,
) -> dict[str, Any]:
    """Finish a FakeStore lookup without switching to the local product source."""
    outcome = result.get("outcome")
    if outcome == "missing_query":
        response = "Please provide a product name or FakeStore catalog ID so I can search that catalog."
        pending = _pending(
            "product_inquiry", "product_identifier", {}, response,
            required_slots=["product_identifier"], classifier_intent="product_inquiry",
            product_task=product_task,
        )
        workflow_status = "awaiting_input"
    elif outcome == "not_found":
        response = (
            "I could not find a matching product in the FakeStore demonstration catalog."
            if not list_request else "The FakeStore demonstration catalog returned no matching products."
        )
        pending = None
        workflow_status = "completed"
    else:
        response = "The FakeStore demonstration catalog is unavailable, so I could not verify that product."
        pending = None
        workflow_status = "completed"

    return {
        "selected_route": "product_inquiry",
        "tool_result": result,
        "response": response,
        "error": None,
        "messages": _append_message(state, "assistant", response),
        "pending_workflow": pending,
        "workflow_status": workflow_status,
        "pending_clarification": response if pending else None,
        "resuming_pending": False,
    }


def _issue_type_answer(query: str) -> str | None:
    answers = {
        "refund": "refund_status", "refund status": "refund_status", "where is my refund": "refund_status",
        "return": "return_order", "return order": "return_order",
        "damaged": "damaged_order", "damaged item": "damaged_order", "damage": "damaged_order",
        "missing": "missing_order", "missing order": "missing_order", "late delivery": "missing_order",
    }
    return answers.get(" ".join(query.casefold().strip(" .!?\t").split()))


def _intent_choice(query: str) -> str | None:
    """Map a correction-menu answer to an issue type, a human handoff, or 'other'."""
    normalized = " ".join(query.casefold().strip(" .!?\t").split())
    choices = {
        "return an item": "return_order", "missing delivery": "missing_order",
        "exchange an item": "exchange", "exchange": "exchange",
        "talk to a person": "customer_care", "talk to someone": "customer_care",
        "human": "customer_care", "agent": "customer_care", "customer care": "customer_care",
        "something else": "other", "other": "other", "none of these": "other",
    }
    return choices.get(normalized) or _issue_type_answer(query)


def _language(query: str | None) -> str:
    return "hi" if _DEVANAGARI.search(query or "") else "en"


def _free_text_detail(query: str, issue_type: str) -> str | None:
    text = _ORDER_ID_IN_TEXT.sub(" ", query).strip(" .,!?\t")
    if issue_type == "damaged_order":
        text = _GENERIC_DAMAGE.sub(" ", text)
    elif issue_type == "missing_order":
        text = _GENERIC_MISSING.sub(" ", text)
    text = " ".join(text.split()).strip(" .,!?\t")
    return text if len(text) >= 4 else None


def _continuity_clarification(state: AgentState, response: str, pending: dict[str, Any]) -> dict[str, Any]:
    messages = _append_message(state, "user", state.get("query") or "")
    response_message = {"role": "assistant", "content": response}
    if not messages or messages[-1] != response_message:
        messages.append(response_message)
    return {
        "continuity_action": "clarify_pending",
        "pending_workflow": pending,
        "workflow_status": pending.get("status", "awaiting_input"),
        "pending_clarification": response,
        "resuming_pending": False,
        "quick_replies": pending.get("quick_replies") or [],
        "selected_route": f"pending:{pending['workflow']}",
        "tool_result": {"status": "awaiting_input", "workflow": pending["workflow"], "awaiting_slot": pending["awaiting_slot"], "tools_executed": False},
        "response": response,
        "error": None,
        "messages": messages,
    }


def continuity_node(state: AgentState) -> dict[str, Any]:
    """Validate pending slot replies first; classify only incomplete/interrupting messages."""
    pending = state.get("pending_workflow")
    query = (state.get("query") or "").strip()
    if state.get("confirmation_required") and state.get("confirmation_status") != "confirmed":
        normalized = query.casefold().strip(" .!\t")
        if normalized in _YES_ANSWERS:
            ready = {**(pending or {}), "status": "ready", "awaiting_slot": None}
            return {
                "continuity_action": "resume_pending", "pending_workflow": ready,
                "workflow_status": "active", "confirmation_status": "confirmed",
                "resuming_pending": True, "messages": _append_message(state, "user", query),
            }
        if normalized in _NO_ANSWERS and (pending or {}).get("workflow") == "intent_confirmation":
            collected = dict(pending.get("collected") or {})
            prompt = _CONFIRMATION_PROMPTS[_language(collected.get("original_query"))]["correction"]
            correction = _pending(
                "intent_correction", "intent_choice", collected, prompt,
                required_slots=["intent_choice"], classifier_intent=pending.get("classifier_intent"),
                quick_replies=_CORRECTION_REPLIES,
            )
            clarification = _continuity_clarification(state, prompt, correction)
            return {**clarification, "confirmation_required": False, "confirmation_status": "rejected"}
        if normalized in _NO_ANSWERS:
            response = "The pending action was cancelled. No operation was performed."
            messages = _append_message(state, "user", query)
            messages.append({"role": "assistant", "content": response})
            return {
                "continuity_action": "clarify_pending", "pending_workflow": None,
                "workflow_status": "cancelled", "confirmation_status": "cancelled",
                "confirmation_required": False, "selected_route": "pending:cancelled",
                "tool_result": {"status": "cancelled", "executed": False},
                "response": response, "messages": messages, "resuming_pending": False,
            }
        return _classify_pending_message(
            state,
            pending or {"workflow": "confirmation", "awaiting_slot": "confirmation", "prompt": "Reply 'confirm' to proceed or 'cancel' to stop."},
        )
    if not isinstance(pending, dict) or pending.get("status") not in {"awaiting_input", "suspended"}:
        return {"continuity_action": "classify", "resuming_pending": False, "pending_clarification": None}

    slot = pending.get("awaiting_slot")
    collected = dict(pending.get("collected") or {})
    accepted: Any = None
    if slot == "order_id":
        match = _ORDER_SLOT_REPLY.fullmatch(query.strip(" .!"))
        if match:
            accepted = f"ORD-{match.group(1)}"
        elif query.casefold().startswith("ord-") or query.casefold().startswith("order id"):
            pending = {**pending, "prompt": "That order ID format is not recognized. Please use a format such as ORD-1001."}
    elif slot == "product_identifier":
        product_query = query
        id_match = re.search(r"\bPROD-\d{3}\b", query, re.IGNORECASE)
        if id_match:
            product_query = id_match.group(0)
        if Config.FAKESTORE_ENABLED:
            catalog_id = resolve_explicit_catalog_id(query)
            try:
                search = (
                    get_catalog_product(catalog_id)
                    if catalog_id is not None
                    else search_catalog_products(product_query)
                )
            except Exception:
                search = {"outcome": "unavailable", "matches": []}
            outcome = search.get("outcome")
            if outcome == "found":
                product = search.get("product")
                matches = [product] if product else search.get("matches", [])
                if len(matches) == 1:
                    accepted = {"product_name": matches[0]["title"]}
                elif len(matches) > 1:
                    choices = ", ".join(
                        f"{item['title']} (catalog ID {item['id']})" for item in matches[:5]
                    )
                    pending = {**pending, "prompt": f"I found multiple FakeStore demo products: {choices}. Which one did you mean?"}
            elif outcome == "not_found":
                response = "I could not find that product in the FakeStore demonstration catalog. Please provide another catalog name or ID."
                return _continuity_clarification(state, response, pending)
            else:
                response = "The FakeStore demonstration catalog is unavailable. Please try the product lookup again later."
                return _continuity_clarification(state, response, pending)
        else:
            try:
                search = search_products(product_query)
            except Exception:
                search = {"outcome": "data_unavailable", "matches": []}
            if search.get("outcome") == "found":
                product = search["matches"][0]
                accepted = {"product_id": product["product_id"], "product_name": product["name"]}
            elif search.get("outcome") == "ambiguous":
                names = ", ".join(f"{item['name']} ({item['product_id']})" for item in search.get("matches", []))
                pending = {**pending, "prompt": f"I found multiple sample products: {names}. Which one did you mean?"}
    elif slot == "issue_type":
        accepted = _issue_type_answer(query)
        if not accepted:
            resolved = resolve_issue_type(query)
            if resolved.get("outcome") == "resolved":
                accepted = resolved["issue_type"]
    elif slot in {"return_reason", "damage_description", "missing_delivery_details"}:
        if _slot_answer_is_valid(slot, query):
            accepted = query[:1000]
    elif slot == "intent_choice":
        accepted = _intent_choice(query)
    elif slot == "feedback_text":
        if len(query) >= 3:
            accepted = query[:2000]

    if accepted is None:
        return _classify_pending_message(state, pending)

    if isinstance(accepted, dict):
        collected.update(accepted)
    else:
        collected[slot] = accepted
    remaining = [item for item in pending.get("required_slots", []) if item != slot]
    if remaining:
        next_slot = remaining[0]
        prompts = {
            "order_id": "Please provide your order ID (for example, ORD-1001).",
            "product_identifier": "Please provide one exact product name or product ID.",
            "return_reason": "Please share why you would like to return the order.",
            "damage_description": "Please describe what is damaged (for example, a cracked screen).",
            "missing_delivery_details": "Please tell me what you observed about the missing delivery or its tracking.",
            "issue_type": "Please choose: refund status, return, damaged item, or missing order.",
        }
        updated = _pending(pending["workflow"], next_slot, collected, prompts[next_slot], required_slots=remaining, classifier_intent=pending.get("classifier_intent"), issue_type=pending.get("issue_type"), candidates=pending.get("candidates"))
        return _continuity_clarification(state, updated["prompt"], updated)

    ready = {**pending, "status": "ready", "collected": collected, "awaiting_slot": None, "required_slots": []}
    return {
        "continuity_action": "resume_pending", "pending_workflow": ready,
        "workflow_status": "active", "pending_clarification": None,
        "resuming_pending": True, "preclassified_result": None,
        "order_id": collected.get("order_id"),
        "product_id": collected.get("product_id"), "product_name": collected.get("product_name"),
        "issue_type": pending.get("issue_type") or collected.get("issue_type"), "issue_details": collected,
        "messages": _append_message(state, "user", query),
    }


def route_continuity_action(state: AgentState) -> str:
    action = state.get("continuity_action")
    return action if action in {"resume_pending", "clarify_pending"} else "classify"


def pending_clarification_node(state: AgentState) -> dict[str, Any]:
    """End safely after the continuity gate has asked for a required value."""
    return {key: state.get(key) for key in (
        "continuity_action", "pending_workflow", "workflow_status", "pending_clarification",
        "selected_route", "tool_result", "response", "error", "messages", "resuming_pending",
    )}


def resume_pending_node(state: AgentState) -> dict[str, Any]:
    """Dispatch validated slot values to the already selected workflow node."""
    pending = state.get("pending_workflow") or {}
    if state.get("confirmation_required") and state.get("confirmation_status") != "confirmed":
        response = "This action requires your explicit confirmation before it can continue. Reply 'confirm' to proceed or 'cancel' to stop."
        return _continuity_clarification(state, response, pending)
    workflow = pending.get("workflow")
    if workflow == "order_tracking":
        return order_tracking_node(state)
    if workflow == "product_inquiry":
        return product_inquiry_node(state)
    if workflow == "issue_support":
        return issue_resolver_node(state)
    if workflow == "order_cancellation":
        return order_cancellation_node(state)
    if workflow == "exchange":
        return exchange_node(state)
    if workflow == "feedback":
        return feedback_node(state)
    if workflow == "intent_confirmation":
        collected = pending.get("collected") or {}
        return _run_confirmed_request(
            state, collected.get("original_route"), collected.get("original_query") or "", collected.get("order_id"),
        )
    if workflow == "intent_correction":
        collected = pending.get("collected") or {}
        choice = collected.get("intent_choice")
        if choice == "customer_care":
            return _run_confirmed_request(state, "customer_care", collected.get("original_query") or "", collected.get("order_id"))
        if choice == "exchange":
            # The original message described something else, so it is not used as the requested change.
            return _run_confirmed_request(state, "exchange", "", collected.get("order_id"))
        if choice in _ISSUE_CHOICE_QUERIES:
            return _run_confirmed_request(state, "issue_support", _ISSUE_CHOICE_QUERIES[choice], collected.get("order_id"))
        response = _CONFIRMATION_PROMPTS[_language(collected.get("original_query"))]["describe"]
        return {
            "selected_route": "pending:intent_correction", "response": response,
            "tool_result": {"status": "awaiting_description", "executed": False},
            "pending_workflow": None, "workflow_status": "idle", "pending_clarification": None,
            "resuming_pending": False, "messages": _append_message(state, "assistant", response),
        }
    response = "I could not safely resume that pending request. Please describe what you need again."
    return {"selected_route": "pending:unsupported", "response": response, "tool_result": {"status": "not_executed"}, "pending_workflow": None, "workflow_status": "idle", "resuming_pending": False, "messages": _append_message(state, "assistant", response)}


def _run_confirmed_request(state: AgentState, route: str | None, query: str, order_id: str | None) -> dict[str, Any]:
    """Run the workflow the customer just confirmed as if it had been routed directly."""
    sub_state: AgentState = {
        **state, "query": query, "order_id": order_id, "issue_type": None,
        "issue_details": {"order_id": order_id} if order_id else {},
        "pending_workflow": None, "resuming_pending": False,
        "confirmation_required": False, "confirmation_status": None,
        # The customer just said yes to this exact request; don't ask twice.
        "action_approved": True,
    }
    node = {"customer_care": customer_care_node, "exchange": exchange_node}.get(route or "", issue_resolver_node)
    return {
        "pending_workflow": None, "workflow_status": "completed", "pending_clarification": None,
        "resuming_pending": False, "confirmation_required": False, "confirmation_status": None,
        **node(sub_state),
    }


def confirm_intent_node(state: AgentState) -> dict[str, Any]:
    """Restate the understood request politely and wait for the customer's yes or no."""
    route = route_classifier_result(state)
    query = state.get("query") or ""
    issue_type = None
    order_id = _order_id(state)
    if route == "issue_support":
        resolution = resolve_issue_type(query)
        issue_type = resolution.get("issue_type") if resolution.get("outcome") == "resolved" else None
        if issue_type not in _CONFIRMED_ISSUE_TYPES:
            # Ambiguous issues already get a choice question; lookups need no confirmation.
            return issue_resolver_node(state)
        order_id = order_id or resolution.get("order_id")
    prompts = _CONFIRMATION_PROMPTS[_language(query)]
    order_text = prompts["order"].format(order_id=order_id) if order_id else ""
    speedup = route == "customer_care" and is_delivery_speedup_request(query)
    prompt = prompts["delivery_speedup" if speedup else (issue_type or route)].format(order=order_text)
    if speedup and not order_id:
        prompt += " If you tell me the order ID (for example, ORD-1001), I'll attach it." if _language(query) == "en" else ""
    collected = {"original_route": route, "original_issue_type": issue_type, "original_query": query, "order_id": order_id}
    pending = _pending(
        "intent_confirmation", "confirmation", collected, prompt,
        classifier_intent=state.get("final_intent"), quick_replies=_CONFIRM_REPLIES,
    )
    pending["status"] = "awaiting_confirmation"
    return {
        "selected_route": f"confirm:{route}",
        "tool_result": {"outcome": "awaiting_intent_confirmation", "route": route, "issue_type": issue_type, "executed": False},
        "response": prompt, "error": None,
        "messages": _append_message(state, "assistant", prompt),
        "pending_workflow": pending, "workflow_status": "awaiting_confirmation",
        "pending_clarification": prompt, "resuming_pending": False,
        "confirmation_required": True, "confirmation_status": "requested",
        "quick_replies": _CONFIRM_REPLIES,
    }


def _raise_ticket(
    state: AgentState, category: str, order_id: str | None, description: str,
    case_reference: str | None, details: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Open a customer-facing ticket; a storage failure never fails the conversation."""
    try:
        ticket = create_ticket(
            session_id=state.get("session_id") or "", category=category, description=description,
            customer_id=state.get("customer_id"), order_id=order_id,
            case_reference=case_reference, details=details,
        )
    except Exception:
        logger.exception("Ticket creation failed (%s).", category)
        return None
    return public_ticket(ticket)


def _with_ticket(output: dict[str, Any], ticket: dict[str, Any] | None) -> dict[str, Any]:
    if not ticket:
        return output
    response = (
        f"{output['response']}\n\nI've raised ticket {ticket['ticket_id']} for you. "
        "You can follow its progress in the My Tickets tab."
    )
    messages = list(output.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = {"role": "assistant", "content": response}
    return {**output, "response": response, "messages": messages, "ticket": ticket}


def _remember_turn(state: AgentState) -> dict[str, Any]:
    """Keep the last few customer turns for frustration scoring and hand-off summaries."""
    query = state.get("query") or ""
    match = _ORDER_ID_IN_TEXT.search(query)
    resolved = (state.get("tool_result") or {}).get("outcome") in {"found", "cancelled", "exchange_prepared", "recorded"} \
        if isinstance(state.get("tool_result"), dict) else False
    turn = {"role": "user", "text": query, "intent": state.get("final_intent"),
            "order_id": match.group(0).upper() if match else None, "resolved": resolved}
    return {"turn_history": (list(state.get("turn_history") or []) + [turn])[-5:]}


def finalize_turn_node(state: AgentState) -> dict[str, Any]:
    """Restore the most recently interrupted workflow after the current task finishes."""
    remembered = _remember_turn(state)
    active = state.get("pending_workflow")
    if isinstance(active, dict) and active.get("status") in {"awaiting_input", "suspended", "awaiting_confirmation"}:
        return remembered
    suspended = list(state.get("suspended_workflows") or [])
    if not suspended:
        return remembered
    restored = dict(suspended.pop())
    restored["status"] = "awaiting_input"
    prompt = restored.get("prompt") or "Please provide the requested information to continue."
    reminder = f"To continue your earlier request: {prompt}"
    response = state.get("response") or "Your request was handled."
    response = response if reminder in response else f"{response}\n\n{reminder}"
    messages = list(state.get("messages") or [])
    assistant_message = {"role": "assistant", "content": response}
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = assistant_message
    else:
        messages.append(assistant_message)
    return {
        **remembered,
        "pending_workflow": restored,
        "suspended_workflows": suspended,
        "workflow_status": "awaiting_input",
        "pending_clarification": prompt,
        "response": response,
        "messages": messages,
        "resuming_pending": False,
    }


def localize_reply_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the finished reply in the customer's script; keep English on any failure."""
    translated = translate_reply(state.get("query") or "", state.get("response") or "", state.get("quick_replies") or [])
    if not translated:
        return {"reply_language": "en" if state.get("response") else None}
    messages = list(state.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages[-1] = {**messages[-1], "content": translated["response"]}
    return {
        "response": translated["response"], "quick_replies": translated["quick_replies"],
        "reply_language": translated["lang"], "messages": messages,
    }


def _append_message(state: AgentState, role: str, content: str) -> list[dict[str, Any]]:
    messages = list(state.get("messages") or [])
    new_message = {"role": role, "content": content}
    if not messages or messages[-1] != new_message:
        messages.append(new_message)
    return messages


def _order_id(state: AgentState) -> str | None:
    value = state.get("order_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    match = _ORDER_ID_IN_TEXT.search(state.get("query") or "")
    return match.group(0) if match else None


def _return_reason(query: str) -> str | None:
    match = _RETURN_REASON.search(query)
    if not match:
        return None
    reason = _ORDER_ID_IN_TEXT.sub("", match.group(1)).strip(" .,!?")
    return reason or None


def _issue_result(
    state: AgentState,
    issue_type: str,
    outcome: str,
    step: str,
    response: str,
    tool_result: dict[str, Any],
    error: str | None = None,
    pending_workflow: dict[str, Any] | None = None,
    issue_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = dict(state.get("issue_details") or {})
    details.update(issue_details or {})
    waiting = pending_workflow is not None
    return {
        "selected_route": f"issue_support:{issue_type}",
        "issue_type": issue_type,
        "workflow_step": step,
        "order_id": _order_id(state),
        "issue_details": details,
        "support_case_reference": (tool_result.get("case") or tool_result.get("request") or {}).get("reference"),
        "tool_result": {"issue_type": issue_type, **tool_result},
        "response": response,
        "error": error,
        "messages": _append_message(state, "assistant", response),
        "pending_workflow": pending_workflow if waiting else (None if state.get("resuming_pending") else state.get("pending_workflow")),
        "workflow_status": "awaiting_input" if waiting else (
            "completed" if state.get("resuming_pending") or not state.get("pending_workflow")
            else state.get("workflow_status", "suspended")
        ),
        "pending_clarification": pending_workflow.get("prompt") if waiting else (None if state.get("resuming_pending") else state.get("pending_clarification")),
        "resuming_pending": False,
    }


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    """Call the existing classifier once and clear per-turn workflow outputs."""
    query = state.get("query") or ""
    messages = _append_message(state, "user", query)
    try:
        result = state.get("preclassified_result") or _classify(query, state)
    except Exception as exc:
        result = {
            "predicted_intent": None, "confidence_score": None, "final_intent": None,
            "verifier_status": "CLASSIFICATION_ERROR", "routing": "ERROR",
            "latency_ms": None, "error": str(exc),
        }
    frustration = None
    if Config.ESCALATION_ENABLED:
        turns = list(state.get("turn_history") or []) + [{"text": query, "intent": result.get("final_intent"), "role": "user"}]
        try:
            frustration = conversation_frustration(turns)
        except Exception:
            logger.exception("Frustration scoring failed; continuing without escalation.")
    return {
        "frustration": frustration,
        "predicted_intent": result.get("predicted_intent"),
        "confidence_score": result.get("confidence_score"),
        "final_intent": result.get("final_intent"),
        "verifier_status": result.get("verifier_status"),
        "classifier_routing": result.get("routing"),
        "selected_route": None,
        "tool_result": None,
        "response": None,
        "error": result.get("error"),
        "order_id": None,
        "product_id": None,
        "product_name": None,
        "issue_type": None,
        "workflow_step": None,
        "issue_details": {},
        "support_case_reference": None,
        "handoff_status": None,
        "messages": messages,
        "preclassified_result": None,
        "continuity_action": "classify",
        "resuming_pending": False,
    }


def _workflow_pending(state: AgentState, route: str, response: str) -> dict[str, Any]:
    """Status response for integrations not included in this phase."""
    return {
        "selected_route": route,
        "tool_result": {"status": "pending", "workflow": route, "executed": False},
        "response": response,
        "error": state.get("error"),
        "messages": _append_message(state, "assistant", response),
    }


def order_tracking_node(state: AgentState) -> dict[str, Any]:
    """Look up status and explain progress using only stored status and ETA."""
    try:
        result = lookup_order_status(_order_id(state))
    except Exception:
        logger.exception("Order tracking tool failed.")
        result = {"outcome": "data_unavailable", "lookup_performed": False}

    outcome = result["outcome"]
    if outcome == "found":
        progress = result["delivery_progress"]
        eta_text = f" The recorded ETA is {result['eta']}." if result.get("eta") else " No ETA is recorded."
        response = (
            f"Demonstration data: order {result['order_id']} has status {result['order_status']}."
            f"{eta_text} {progress['explanation']} This is not a live carrier lookup."
        )
    elif outcome == "missing_order_id":
        response = "Please provide your order ID (for example, ORD-1001) to check the demonstration order data."
    elif outcome == "invalid_order_id":
        response = "That order ID format is not recognized. Please use a format such as ORD-1001."
    elif outcome == "not_found":
        response = f"I could not find order {result.get('order_id')} in the demonstration data. Please check the ID."
    else:
        response = "The demonstration order data is currently unavailable, so I could not check that order."
    output = {
        "selected_route": "order_tracking", "tool_result": result, "response": response,
        "error": result.get("error") if outcome == "data_unavailable" else None,
        "messages": _append_message(state, "assistant", response),
    }
    if outcome == "missing_order_id":
        prompt = "Please provide your order ID (for example, ORD-1001)."
        output.update({
            "pending_workflow": _pending("order_tracking", "order_id", {}, prompt, required_slots=["order_id"], classifier_intent="order_tracking"),
            "workflow_status": "awaiting_input", "pending_clarification": prompt,
        })
    elif state.get("resuming_pending"):
        output.update({"pending_workflow": None, "workflow_status": "completed", "pending_clarification": None, "resuming_pending": False})
    elif not state.get("pending_workflow"):
        output["workflow_status"] = "completed"
    return output


_CANCELLABLE_STATUSES = {"processing"}


def order_cancellation_node(state: AgentState) -> dict[str, Any]:
    """Cancel a not-yet-shipped demo order, but only after an explicit yes."""
    pending = state.get("pending_workflow") or {}
    collected = dict(pending.get("collected") or {})
    order_id = _order_id(state) or collected.get("order_id")
    base = {"selected_route": "order_cancellation", "error": None}

    def reply(response: str, tool_result: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {**base, "tool_result": tool_result, "response": response,
                "messages": _append_message(state, "assistant", response), **extra}

    if not order_id:
        prompt = "Which order would you like to cancel? Please share the order ID (for example, ORD-1002)."
        return reply(prompt, {"outcome": "missing_order_id", "executed": False},
                     pending_workflow=_pending("order_cancellation", "order_id", {}, prompt, required_slots=["order_id"], classifier_intent="order_management"),
                     workflow_status="awaiting_input", pending_clarification=prompt, resuming_pending=False)

    done = {"pending_workflow": None, "workflow_status": "completed", "pending_clarification": None,
            "resuming_pending": False, "confirmation_required": False, "confirmation_status": None}
    completed = dict(state.get("completed_actions") or {})
    key = idempotency_key("cancel_order", order_id)
    if key in completed:
        response = f"Order {order_id} was already cancelled in this conversation (reference {completed[key]}). Nothing else was changed."
        return reply(response, {"outcome": "already_cancelled", "order_id": order_id, "reference": completed[key], "executed": False}, **done)

    # On resume after approval this lookup runs again, so the status is re-checked before acting.
    try:
        result = lookup_order_status(order_id)
    except Exception:
        logger.exception("Order lookup for cancellation failed.")
        result = {"outcome": "data_unavailable"}
    if result["outcome"] != "found":
        response = f"I could not find order {order_id} in the demonstration data, so nothing was cancelled. Please check the ID."
        return reply(response, {"outcome": result["outcome"], "executed": False}, **done)

    status = result["order_status"]
    if status.casefold() not in _CANCELLABLE_STATUSES:
        response = (
            f"Order {order_id} is already {status.lower()}, so it can no longer be cancelled. "
            "Once it arrives, I can help you start a return instead."
        )
        return reply(response, {"outcome": "not_cancellable", "order_id": order_id, "order_status": status, "executed": False}, **done)

    prompt = f"Order {order_id} is still {status.lower()}. Do you want me to cancel it? Reply 'yes' to cancel or 'no' to keep it."
    if not request_approval("order_cancellation", prompt, _CANCEL_REPLIES, order_id=order_id):
        response = f"Okay, I've kept order {order_id} as it is. No operation was performed."
        return reply(response, {"outcome": "kept", "order_id": order_id, "executed": False}, **done)
    reference = f"DEMO-CANCEL-{uuid4().hex[:10].upper()}"
    response = (
        f"Order {order_id} has been cancelled in this demonstration (reference {reference}). "
        "Any payment would be refunded to the original method. No real order was changed."
    )
    return reply(response, {"outcome": "cancelled", "order_id": order_id, "reference": reference, "executed": True},
                 completed_actions={**completed, key: reference}, **done)


def product_inquiry_node(state: AgentState) -> dict[str, Any]:
    """Use optional FakeStore catalog data, otherwise retain sample stock behavior."""
    product_query = state.get("product_id") or state.get("product_name") or state.get("query")
    query = state.get("query") or ""
    if Config.FAKESTORE_ENABLED and isinstance(product_query, str):
        catalog_id = resolve_explicit_catalog_id(query)
        pending_task = (state.get("pending_workflow") or {}).get("product_task")
        stock_question = pending_task == "availability" or bool(re.search(r"\b(stock|available|availability|inventory|in stock)\b|在庫|有货|有貨|उपलब्ध", query, re.IGNORECASE))
        list_request = bool(re.search(r"\b(list|browse|show)\b.*\b(products|catalog|items)\b", query, re.IGNORECASE))
        if list_request:
            catalog_result = list_catalog_products()
            listed = catalog_result.get("products", [])
            if catalog_result.get("outcome") == "found":
                response = "FakeStore demo catalog: " + "; ".join(
                    f"{item['title']} ({item['category']}, ${item['price']:.2f})" for item in listed[:10]
                )
                return {
                    "selected_route": "product_inquiry", "tool_result": catalog_result, "response": response,
                    "error": None, "messages": _append_message(state, "assistant", response),
                    "pending_workflow": None, "workflow_status": "completed",
                    "pending_clarification": None, "resuming_pending": False,
                }
            return _fakestore_terminal_result(state, catalog_result, list_request=True)
        elif catalog_id is not None:
            catalog_result = get_catalog_product(catalog_id)
        elif stock_question:
            catalog_result = check_catalog_stock(product_query)
        else:
            catalog_result = search_catalog_products(product_query)
        if catalog_result.get("outcome") in {"found", "inventory_unavailable"}:
            product = catalog_result.get("product")
            matches = [product] if product else catalog_result.get("matches", [])
            if len(matches) == 1:
                selected = matches[0]
                price = f"${selected['price']:.2f}"
                response = f"FakeStore demo catalog lists {selected['title']} ({selected['category']}) at {price}. "
                response += "The catalog provides no inventory quantity, so I cannot confirm current stock." if stock_question else "This is a demo catalog and does not represent a real retailer."
                result = {"outcome": "inventory_unavailable" if stock_question else "found", "product": selected, "stock_quantity": None, "source": "fakestore_demo"}
                return {
                    "selected_route": "product_inquiry", "tool_result": result, "response": response,
                    "error": None, "messages": _append_message(state, "assistant", response),
                    "pending_workflow": None, "workflow_status": "completed",
                    "pending_clarification": None, "resuming_pending": False,
                }
            if len(matches) > 1:
                choices = ", ".join(f"{item['title']} (catalog ID {item['id']})" for item in matches[:5])
                response = f"I found several matching FakeStore demo products: {choices}. Which one did you mean?"
                return {
                    "selected_route": "product_inquiry", "tool_result": {"outcome": "ambiguous", "matches": matches},
                    "response": response, "error": None, "messages": _append_message(state, "assistant", response),
                    "pending_workflow": _pending("product_inquiry", "product_identifier", {}, response, required_slots=["product_identifier"], classifier_intent="product_inquiry", candidates=matches),
                    "workflow_status": "awaiting_input", "pending_clarification": response,
                }
        # FakeStore is the selected product source for this request. Its miss or
        # failure is terminal; never silently switch to PostgreSQL/JSON fixtures.
        return _fakestore_terminal_result(
            state, catalog_result,
            product_task="availability" if stock_question else "catalog_search",
        )
    try:
        result = check_product_stock(product_query)
    except Exception:
        logger.exception("Product inquiry tool failed.")
        result = {"outcome": "data_unavailable", "lookup_performed": False}

    outcome = result["outcome"]
    if outcome == "found":
        stock = result["stock"]
        availability = "in stock" if stock > 0 else "out of stock"
        restock_date = result.get("restock_date")
        restock_text = f" Sample restock date: {restock_date}." if restock_date else ""
        response = (
            f"Demonstration catalog: {result['name']} ({result['product_id']}) is {availability} "
            f"with {stock} unit(s) recorded in sample stock.{restock_text} This is not live inventory."
        )
    elif outcome == "ambiguous":
        names = ", ".join(f"{item['name']} ({item['product_id']})" for item in result.get("matches", []))
        response = f"I found multiple matching products in the demonstration catalog: {names}. Which one did you mean?"
    elif outcome in {"missing_query", "invalid_query"}:
        response = "Please provide a product name or product ID so I can check the demonstration catalog."
    elif outcome == "not_found":
        response = "I could not identify that product in the demonstration catalog. Please provide its exact name or ID."
    else:
        response = "Stock information is unavailable in the demonstration data for that product."
    output = {
        "selected_route": "product_inquiry", "tool_result": result, "response": response,
        "error": result.get("error") if outcome == "data_unavailable" else None,
        "messages": _append_message(state, "assistant", response),
    }
    if outcome in {"missing_query", "invalid_query", "not_found", "ambiguous"}:
        prompt = response
        output.update({
            "pending_workflow": _pending(
                "product_inquiry", "product_identifier", {}, prompt,
                required_slots=["product_identifier"], classifier_intent="product_inquiry", candidates=result.get("matches", []),
            ),
            "workflow_status": "awaiting_input", "pending_clarification": prompt,
        })
    elif state.get("resuming_pending"):
        output.update({"pending_workflow": None, "workflow_status": "completed", "pending_clarification": None, "resuming_pending": False})
    elif not state.get("pending_workflow"):
        output["workflow_status"] = "completed"
    return output


def issue_resolver_node(state: AgentState) -> dict[str, Any]:
    """Resolve broad issue_support into a supported issue flow or ask clearly."""
    query = state.get("query") or ""
    prior_pending = state.get("pending_workflow") or {}
    collected = dict(prior_pending.get("collected") or {})
    collected.update(state.get("issue_details") or {})
    prior_issue = state.get("issue_type") or prior_pending.get("issue_type")
    resolution = resolve_issue_type(query, {"issue_type": prior_issue})
    if resolution["outcome"] != "resolved":
        response = (
            "I can help with a refund status, a return, a damaged item, or an order that has not arrived. "
            "Which issue are you contacting us about?"
        )
        pending = _pending(
            "issue_support", "issue_type", collected, response,
            required_slots=["issue_type"], classifier_intent="issue_support",
            candidates=resolution.get("candidates", []), quick_replies=_ISSUE_CHOICE_REPLIES,
        )
        return {
            "selected_route": "issue_support:clarification", "issue_type": None,
            "workflow_step": "awaiting_issue_clarification",
            "tool_result": {"outcome": "clarification_required", "candidates": resolution.get("candidates", []), "tools_executed": False},
            "response": response, "error": None,
            "messages": _append_message(state, "assistant", response),
            "pending_workflow": pending, "workflow_status": "awaiting_input",
            "pending_clarification": response, "resuming_pending": False,
            "quick_replies": _ISSUE_CHOICE_REPLIES,
        }

    issue_type = resolution["issue_type"]
    order_id = _order_id(state) or collected.get("order_id") or resolution.get("order_id")
    if order_id:
        collected["order_id"] = order_id
    if issue_type == "return_order":
        reason = _return_reason(query) or collected.get("return_reason")
        if reason:
            collected["return_reason"] = reason
    elif issue_type == "damaged_order":
        description = _free_text_detail(query, issue_type) or collected.get("damage_description")
        if description:
            collected["damage_description"] = description
    elif issue_type == "missing_order":
        details = _free_text_detail(query, issue_type) or collected.get("missing_delivery_details")
        if details:
            collected["missing_delivery_details"] = details

    def awaiting(slot: str, prompt: str, required: list[str]) -> dict[str, Any]:
        return _pending(
            "issue_support", slot, collected, prompt,
            required_slots=required, classifier_intent="issue_support", issue_type=issue_type,
        )

    session_id = state.get("session_id")
    result: dict[str, Any]
    try:
        if issue_type == "refund_status":
            result = refund_status_tool(order_id)
            if result["outcome"] == "missing_order_id":
                response = "Please provide the order ID so I can check for a refund record in the demonstration data."
            elif result["outcome"] == "found":
                refund = result["refund"]
                status_label = str(refund["status"]).replace("_", " ")
                response = f"Demonstration data records refund status '{status_label}' for order {result['order_id']}. This is not a live payment-system check."
            elif result["outcome"] == "no_refund_record":
                response = f"There is no refund status record for order {result['order_id']} in the demonstration data, so I cannot confirm that a refund was processed."
            elif result["outcome"] == "unknown_order":
                response = f"I could not find order {result.get('order_id')} in the demonstration orders, so I cannot check a refund status."
            else:
                response = "I could not check refund status from the demonstration records."
            if result["outcome"] == "missing_order_id":
                pending = awaiting("order_id", response, ["order_id"])
                return _issue_result(state, issue_type, result["outcome"], "awaiting_order_id", response, result, pending_workflow=pending, issue_details=collected)
            return _issue_result(state, issue_type, result["outcome"], "refund_status_checked", response, result, issue_details=collected)

        if issue_type == "return_order":
            if not order_id:
                result = {"outcome": "missing_order_id", "request": None, "error": "Please provide your order ID."}
                response = "Please provide your order ID to review the demonstration return policy."
                pending = awaiting("order_id", response, ["order_id"] + ([] if collected.get("return_reason") else ["return_reason"]))
                return _issue_result(state, issue_type, result["outcome"], "awaiting_order_id", response, result, pending_workflow=pending, issue_details=collected)
            reason = collected.get("return_reason")
            if not reason:
                result = {"outcome": "missing_reason", "request": None, "error": "Please provide a return reason."}
                response = "Please share why you would like to return the order so I can prepare a demonstration request."
                pending = awaiting("return_reason", response, ["return_reason"])
                return _issue_result(state, issue_type, result["outcome"], "awaiting_return_reason", response, result, pending_workflow=pending, issue_details=collected)
            result = prepare_return_tool(order_id, reason, session_id)
            if result["outcome"] == "demo_request_prepared":
                policy = result["eligibility"]["policy"]
                response = (
                    f"A demonstration return request {result['request']['reference']} was prepared for order {result['request']['order_id']}. "
                    f"Sample policy: return within {policy.get('days')} days; condition: {policy.get('condition', 'not specified')}. "
                    "Eligibility still needs manual review because purchase date/item condition are not in the sample data. "
                    "No real return, pickup, or refund has been arranged."
                )
            else:
                response = f"I could not prepare a demonstration return request ({result['outcome']}). Please check the order ID."
            output = _issue_result(state, issue_type, result["outcome"], "return_request_prepared", response, result, issue_details=collected)
            if result["outcome"] == "demo_request_prepared":
                request = result["request"]
                output = _with_ticket(output, _raise_ticket(
                    state, issue_type, request["order_id"], f"Return requested. Reason: {reason}",
                    request["reference"], {"return_reason": reason},
                ))
            return output

        if issue_type == "damaged_order":
            description = collected.get("damage_description")
            if not order_id:
                result = {"outcome": "missing_order_id", "case": None, "error": "Please provide your order ID."}
                response = "Please provide your order ID so I can prepare a demonstration damaged-item report."
                pending = awaiting("order_id", response, ["order_id"] + ([] if description else ["damage_description"]))
                return _issue_result(state, issue_type, result["outcome"], "awaiting_order_id", response, result, pending_workflow=pending, issue_details=collected)
            if not description:
                result = {"outcome": "missing_damage_details", "case": None, "error": "Please describe what is damaged."}
                response = "Please describe what is damaged so I can prepare a demonstration support case."
                pending = awaiting("damage_description", response, ["damage_description"])
                return _issue_result(state, issue_type, result["outcome"], "awaiting_damage_description", response, result, pending_workflow=pending, issue_details=collected)
            result = damaged_order_tool(order_id, description, session_id)
            if result["outcome"] == "demo_case_recorded":
                case = result["case"]
                response = f"I recorded demonstration case {case['reference']} for order {case['order_id']}. A support representative would review it; no retailer or courier claim was submitted."
            elif result["outcome"] == "missing_damage_details":
                response = "Please describe what is damaged so I can prepare a demonstration support case."
            else:
                response = "I could not prepare a demonstration damaged-item case. Please check the order ID and product details."
            output = _issue_result(state, issue_type, result["outcome"], "damaged_case_recorded", response, result, issue_details=collected)
            if result["outcome"] == "demo_case_recorded":
                case = result["case"]
                output = _with_ticket(output, _raise_ticket(
                    state, issue_type, case["order_id"], f"Item arrived damaged: {description}",
                    case["reference"], {"damage_description": description},
                ))
            return output

        if issue_type == "missing_order":
            details = collected.get("missing_delivery_details")
            if not order_id:
                result = {"outcome": "missing_order_id", "case": None, "error": "Please provide your order ID."}
                response = "Please provide your order ID so I can review its demonstration status."
                pending = awaiting("order_id", response, ["order_id"] + ([] if details else ["missing_delivery_details"]))
                return _issue_result(state, issue_type, result["outcome"], "awaiting_order_id", response, result, pending_workflow=pending, issue_details=collected)
            if not details:
                result = {"outcome": "missing_delivery_details", "case": None, "error": "Please provide delivery details."}
                response = "Please tell me what you observed about the missing delivery or its tracking so I can prepare a demonstration investigation."
                pending = awaiting("missing_delivery_details", response, ["missing_delivery_details"])
                return _issue_result(state, issue_type, result["outcome"], "awaiting_delivery_details", response, result, pending_workflow=pending, issue_details=collected)
            result = missing_order_tool(order_id, session_id, details)
            if result["outcome"] == "demo_case_recorded":
                order = result["order"]
                progress = order["delivery_progress"]
                case_ref = result["case"]["reference"]
                if progress["stage"] == "delayed":
                    detail = "The sample ETA has passed, which suggests a delay; no carrier scans are available."
                elif progress["stage"] == "delivered":
                    detail = "The sample record says delivered, but that does not prove receipt; the case requests review."
                else:
                    detail = progress["explanation"]
                response = f"{detail} I recorded demonstration investigation {case_ref}. The parcel has not been declared lost and no carrier was contacted."
            elif result["outcome"] == "delivery_status_unknown":
                response = "The demonstration order has no recognized delivery status, so I cannot determine progress or declare it lost."
            else:
                response = "I could not find that order in the demonstration data, so no investigation case was created."
            output = _issue_result(state, issue_type, result["outcome"], "missing_delivery_investigated", response, result, issue_details=collected)
            if result["outcome"] == "demo_case_recorded":
                output = _with_ticket(output, _raise_ticket(
                    state, issue_type, order_id, f"Delivery not received: {details}",
                    result["case"]["reference"], {"missing_delivery_details": details},
                ))
            return output
    except Exception:
        logger.exception("Issue support workflow failed (%s).", issue_type)
        result = {"outcome": "service_unavailable", "error": "Demonstration support service is unavailable."}
        return _issue_result(
            state, issue_type, "service_unavailable", "service_error",
            "I could not complete the demonstration support step right now. Please try again later.",
            result, error=result["error"],
        )

    response = "I could not resolve that support request. Please clarify what happened."
    return _issue_result(state, issue_type, "unsupported_issue", "clarification", response, {"outcome": "unsupported_issue"})


def customer_care_node(state: AgentState) -> dict[str, Any]:
    """Save a simulated handoff request and concise recent conversation summary."""
    messages = state.get("messages") or []
    summary = "\n".join(
        f"{item.get('role', 'message')}: {str(item.get('content', ''))[:300]}"
        for item in messages[-6:]
    )[:1800]
    reason = (state.get("query") or "Customer requested a human representative.").strip()
    speedup = is_delivery_speedup_request(reason)
    try:
        result = customer_handoff_tool(state.get("session_id"), reason, summary or reason)
        if result["outcome"] == "simulated_handoff_ready" and speedup:
            response = (
                "I've raised a priority ticket asking our delivery team to speed up your order. "
                f"Reference: {result['handoff']['reference']}. I can't reach the delivery agent live, so the team will follow up. "
                "(Demonstration: no live agent has been contacted.)"
            )
            handoff_status = "simulated_handoff_ready"
        elif result["outcome"] == "simulated_handoff_ready":
            response = f"Your request is ready for human assistance in this demonstration. Reference: {result['handoff']['reference']}. No live agent has been contacted."
            handoff_status = "simulated_handoff_ready"
        else:
            response = "I could not record the demonstration handoff request. No agent was contacted."
            handoff_status = result["outcome"]
    except Exception:
        logger.exception("Customer-care handoff simulation failed.")
        result = {"outcome": "service_unavailable", "live_agent_contacted": False}
        response = "The demonstration handoff service is unavailable. No agent was contacted."
        handoff_status = "service_unavailable"
    output = {
        "selected_route": "customer_care",
        "handoff_status": handoff_status,
        "support_case_reference": (result.get("handoff") or {}).get("reference"),
        "tool_result": result, "response": response,
        "error": None if handoff_status == "simulated_handoff_ready" else result.get("error"),
        "messages": _append_message(state, "assistant", response),
    }
    if handoff_status == "simulated_handoff_ready":
        output = _with_ticket(output, _raise_ticket(
            state, "delivery_speedup" if speedup else "customer_care", _order_id(state), reason, result["handoff"]["reference"],
            {"conversation_summary": summary},
        ))
    return output


def store_information_node(state: AgentState) -> dict[str, Any]:
    """Run the bounded document RAG subgraph for the existing store-information route."""
    query = state.get("query") or ""
    try:
        rag_state = run_agentic_rag(query, state.get("messages") or [], service=get_rag_service())
        status = rag_state.get("rag_status")
        sources = rag_state.get("sources") or []
        result = {
            "status": status,
            "answer": rag_state.get("rag_answer"),
            "sources": sources,
            "documents": rag_state.get("rag_documents") or [],
            "retry_count": rag_state.get("rag_retry_count", 0),
            "evidence": rag_state.get("rag_evidence") or {},
            "search_query": rag_state.get("rag_search_query"),
        }
        response = result["answer"] or "The document knowledge service is temporarily unavailable. I could not verify that information."
        if status == "answered" and sources:
            response += "\n\nSources: " + ", ".join(sources)
        error = rag_state.get("rag_error")
        error_kind = rag_state.get("rag_error_kind")
        if error:
            workflow_step = "rag_unavailable"
        elif status == "answered":
            workflow_step = "rag_answered"
        else:
            workflow_step = "rag_no_evidence"
        if error and error_kind == "configuration":
            raise RAGConfigurationError(error)
        if error:
            raise RAGProviderError(error)
    except RAGConfigurationError as exc:
        logger.info("RAG configuration is incomplete: %s", exc)
        result = {"status": "configuration_missing", "error": str(exc), "sources": [], "documents": []}
        response = "I couldn't find that in our help documents. Could you rephrase it, or pick one of these?"
        error = str(exc)
        workflow_step = "rag_unavailable"
        # Preserve the existing, explicitly fictional sample lookup for known
        # structured topics in local development only. Never use it after a
        # retrieval miss/provider failure or in a production environment.
        from app.config import Config
        normalized_query = query.casefold()
        explicit_demo_cues = (
            "sale", "sales", "deal", "deals", "promotion", "promotions",
            "offer", "offers", "discount", "coupon", "promo code",
            "store hours", "opening hours", "store name", "store details",
            "store address", "store contact", "about the store", "about your store",
        )
        if Config.APP_ENV.lower() == "development" and any(
            cue in normalized_query for cue in explicit_demo_cues
        ):
            demo = store_information_tool(query)
            if demo.get("outcome") == "found":
                result = {
                    **demo,
                    "rag_status": "configuration_missing",
                    "rag_error": result["error"],
                    "sources": [],
                    "documents": [],
                }
                response = "RAG is not configured. Existing development sample data follows and is fictional:\n"
                response += format_store_information(demo)
        if result.get("status") == "configuration_missing" and not result.get("rag_status"):
            offline = local_search.answer(f"{query} {_SEARCH_HINTS.get(state.get('final_intent') or '', '')}")
            if offline:
                # Providers aren't configured: answer from the documents directly, quoting and citing them.
                titles = ", ".join(source["title"] for source in offline["sources"])
                return {
                    "selected_route": "store_information", "workflow_step": "rag_offline_answered",
                    "tool_result": {"status": "answered_offline", "sources": offline["sources"], "documents": offline["documents"]},
                    "response": f"{offline['answer']}\n\nSource: {titles}", "error": None,
                    "rag_context": offline["documents"], "sources": offline["sources"],
                    "messages": _append_message(state, "assistant", f"{offline['answer']}\n\nSource: {titles}"),
                }
    except RAGProviderError as exc:
        logger.warning("RAG provider unavailable: %s", exc)
        result = {"status": "provider_unavailable", "error": str(exc), "sources": [], "documents": []}
        response = "The document knowledge service is temporarily unavailable. I could not verify that information."
        error = str(exc)
        workflow_step = "rag_provider_error"
    except Exception:
        logger.exception("Unexpected store-information RAG failure")
        result = {"status": "provider_unavailable", "error": "unexpected RAG failure", "sources": [], "documents": []}
        response = "The document knowledge service is temporarily unavailable. I could not verify that information."
        error = "RAG service unavailable"
        workflow_step = "rag_provider_error"
    return {
        "selected_route": "store_information",
        "workflow_step": workflow_step,
        "tool_result": result, "response": response,
        "error": error,
        "quick_replies": _HELP_MENU_REPLIES if workflow_step != "rag_answered" and not result.get("rag_status") else [],
        "rag_context": result.get("documents", []),
        "sources": result.get("sources", []),
        "messages": _append_message(state, "assistant", response),
    }


# Words added to the document search for how-to questions routed here from other intents.
_SEARCH_HINTS = {
    "place_order": "place order checkout", "account_management": "account", "add_to_cart": "cart add",
    "remove_from_cart": "cart remove", "order_management": "order change", "submit_feedback": "feedback review",
}


def _information_result(state: AgentState, route: str, result: dict[str, Any]) -> dict[str, Any]:
    response = result["response"]
    return {
        "selected_route": route,
        "tool_result": {**result, "executed": result["outcome"] == "answered"},
        "response": response, "error": None,
        "workflow_status": "completed",
        "messages": _append_message(state, "assistant", response),
    }


def payment_node(state: AgentState) -> dict[str, Any]:
    if state.get("final_intent") == "issue_support":
        return _workflow_pending(
            state, "payment_support",
            "The payment support workflow was selected, but its business integration is pending. No payment operation was performed.",
        )
    return _information_result(state, "payment", answer_payment_question(state.get("query")))


def shipping_node(state: AgentState) -> dict[str, Any]:
    result = answer_shipping_question(state.get("query"), state.get("final_intent"))
    return _information_result(state, "shipping", result)


def rag_node(state: AgentState) -> dict[str, Any]:
    return _workflow_pending(
        state, "rag",
        "The knowledge workflow was selected, but retrieval is pending. No documents or external models were queried.",
    )


SUPPORTED_ALTERNATIVES = (
    "I can check an order's status, help with a return, refund, damaged or missing item, "
    "look up product availability, or connect you with a support agent."
)
_FEEDBACK_TEXT = {
    "en": {
        "ask": "I'd love to hear it. What would you like to tell us? You can share what you liked, what went wrong, or a suggestion.",
        "thanks": "Thank you for your feedback. I've passed it on to our team (reference {reference}).",
        "problem": " If something needs fixing, such as a damaged, missing or wrong item, tell me and I'll raise it as an issue.",
    },
    "hi": {
        "ask": "ज़रूर, हम सुनना चाहेंगे। आप क्या बताना चाहते हैं? आपको क्या अच्छा लगा, क्या गलत हुआ, या कोई सुझाव।",
        "thanks": "आपके फ़ीडबैक के लिए धन्यवाद। मैंने इसे हमारी टीम तक पहुँचा दिया है (संदर्भ {reference})।",
        "problem": " अगर कुछ ठीक करवाना है, जैसे खराब, गुम या गलत सामान, तो बताइए, मैं उसकी शिकायत दर्ज कर दूँगा।",
    },
}


def feedback_node(state: AgentState) -> dict[str, Any]:
    """Record an opinion or suggestion; ask for it first if the message only announces it."""
    pending = state.get("pending_workflow") or {}
    collected = (pending.get("collected") or {}) if pending.get("workflow") == "feedback" else {}
    query = state.get("query") or ""
    text = collected.get("feedback_text") or (query if has_feedback_content(query) else None)
    lang = _language(collected.get("original_query") or query)
    copy = _FEEDBACK_TEXT[lang]
    if not text:
        prompt = copy["ask"]
        return {
            **_workflow_pending(state, "feedback", prompt),
            "pending_workflow": _pending("feedback", "feedback_text", {"original_query": query}, prompt,
                                         required_slots=["feedback_text"]),
        }
    try:
        reference = record_customer_feedback(text, state.get("session_id"), state.get("customer_id"))
    except OSError as exc:
        logger.warning("Could not store customer feedback (%s)", type(exc).__name__)
        reference = "not saved"
    response = copy["thanks"].format(reference=reference) + copy["problem"]
    return {
        "selected_route": "feedback",
        "tool_result": {"status": "recorded", "reference": reference, "executed": True},
        "response": response, "error": None,
        "pending_workflow": None, "workflow_status": "completed", "pending_clarification": None,
        "resuming_pending": False,
        "messages": _append_message(state, "assistant", response),
    }


def escalation_node(state: AgentState) -> dict[str, Any]:
    """Hand a frustrated customer to a person with a summary, instead of another bot answer."""
    frustration = state.get("frustration") or {}
    turns = list(state.get("turn_history") or []) + [{"role": "user", "text": state.get("query") or "", "intent": state.get("final_intent")}]
    intents = [t["intent"] for t in turns if t.get("intent")]
    order_ids = sorted({m.upper() for t in turns for m in _ORDER_ID_IN_TEXT.findall(t.get("text") or "")})
    summary = handoff_summary(turns, intents, order_ids, frustration)
    reference = f"DEMO-ESCALATION-{uuid4().hex[:10].upper()}"
    response = (
        "I'm really sorry this has been so frustrating. I've passed your conversation to a senior member of our "
        "support team with a summary, so you won't need to repeat yourself. They'll pick it up first, as a "
        "high-priority request, during support hours (Mon-Fri, 09:00-17:00)."
    )
    output = {
        "selected_route": "escalation",
        "tool_result": {"outcome": "escalated", "reference": reference, "score": frustration.get("score"),
                        "reasons": frustration.get("reasons"), "executed": True},
        "response": response, "error": None, "escalated": True, "handoff_status": "escalated",
        "messages": _append_message(state, "assistant", response),
    }
    return _with_ticket(output, _raise_ticket(
        state, "escalation", order_ids[-1] if order_ids else None, summary, reference,
        {"frustration_score": frustration.get("score"), "reasons": frustration.get("reasons")},
    ))


# Offered whenever the assistant can't act on a message, so the customer is never stuck.
_HELP_MENU_REPLIES = [
    {"label": "Track my order", "value": "where is my order"},
    {"label": "Return or refund", "value": "I want to return an item"},
    {"label": "Damaged item", "value": "my item arrived damaged"},
    {"label": "Talk to a person", "value": "I want to talk to a human"},
    {"label": "Give feedback", "value": "I want to give feedback"},
]

UNSUPPORTED_RESPONSES = {
    "account_management": "I can't make changes to your account here yet.",
    "add_to_cart": "I can't add items to your cart from this chat yet.",
    "order_management": "I can cancel an order that hasn't shipped yet, but I can't change the items in an order from this chat.",
    "place_order": "I can't place orders from this chat yet.",
    "remove_from_cart": "I can't change your cart from this chat yet.",
}


def unsupported_node(state: AgentState) -> dict[str, Any]:
    intent = state.get("final_intent")
    route = f"unsupported:{intent}" if intent else "unsupported"
    # Never show internal labels to customers; offer what the assistant can do instead.
    lead = UNSUPPORTED_RESPONSES.get(intent, "I can't help with that request from this chat yet.")
    response = f"{lead} {SUPPORTED_ALTERNATIVES}"
    return {
        "selected_route": route,
        "tool_result": {"status": "unsupported", "intent": intent, "executed": False},
        "response": response, "error": None, "quick_replies": _HELP_MENU_REPLIES,
        "messages": _append_message(state, "assistant", response),
    }


def unknown_intent_node(state: AgentState) -> dict[str, Any]:
    response = "I'm not sure I understood that. Could you tell me a bit more, or pick one of these?"
    return {
        "selected_route": "unknown_intent",
        "tool_result": {"status": "not_executed", "reason": "unknown_intent"},
        "response": response, "error": state.get("error"), "quick_replies": _HELP_MENU_REPLIES,
        "messages": _append_message(state, "assistant", response),
    }


def out_of_scope_node(state: AgentState) -> dict[str, Any]:
    response = (
        "I can help with ecommerce customer service, including order status or delays, return/refund guidance, "
        "damaged or missing deliveries, store information, product availability, or preparing a simulated human handoff. "
        "I cannot help with unrelated topics. No business tools were run."
    )
    return {
        "selected_route": "out_of_scope",
        "tool_result": {"status": "not_executed", "reason": "out_of_scope", "tools_executed": False},
        "response": response, "error": None, "quick_replies": _HELP_MENU_REPLIES,
        "messages": _append_message(state, "assistant", response),
    }


def verification_error_node(state: AgentState) -> dict[str, Any]:
    response = "I could not verify the intent, so I did not run a business workflow. Please try again later."
    return {
        "selected_route": "verification_error",
        "tool_result": {"status": "not_executed", "reason": "verification_error"},
        "response": response,
        "error": state.get("error") or "Intent classification or verification failed.",
        "messages": _append_message(state, "assistant", response),
    }


def greeting_node(state: AgentState) -> dict[str, Any]:
    response = f"Hello! How can I help you today? {SUPPORTED_ALTERNATIVES}"
    return _information_result(state, "greeting", {"outcome": "answered", "response": response})


def goodbye_node(state: AgentState) -> dict[str, Any]:
    response = "You're welcome! If anything else comes up with your order, just message me here. Have a great day!"
    return _information_result(state, "goodbye", {"outcome": "answered", "response": response})


def product_recommendation_node(state: AgentState) -> dict[str, Any]:
    return _information_result(state, "product_recommendation", recommend_products(state.get("query")))


def warranty_node(state: AgentState) -> dict[str, Any]:
    return _information_result(state, "warranty", answer_warranty_question(state.get("query")))


def exchange_node(state: AgentState) -> dict[str, Any]:
    """Prepare a demo exchange for a delivered order; earlier stages get the right alternative."""
    pending = state.get("pending_workflow") or {}
    collected = dict(pending.get("collected") or {})
    order_id = _order_id(state) or collected.get("order_id")
    request_text = collected.get("exchange_details") or _ORDER_ID_IN_TEXT.sub("", state.get("query") or "").strip(" .,!?")
    base = {"selected_route": "exchange", "error": None}

    def reply(response: str, tool_result: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {**base, "tool_result": tool_result, "response": response,
                "messages": _append_message(state, "assistant", response), **extra}

    if not order_id:
        prompt = "I can help with an exchange. Which order is it for? Please share the order ID (for example, ORD-1003)."
        return reply(prompt, {"outcome": "missing_order_id", "executed": False},
                     pending_workflow=_pending("exchange", "order_id", {"exchange_details": request_text}, prompt,
                                               required_slots=["order_id"], classifier_intent="exchange_request"),
                     workflow_status="awaiting_input", pending_clarification=prompt, resuming_pending=False)

    done = {"pending_workflow": None, "workflow_status": "completed", "pending_clarification": None, "resuming_pending": False}
    completed = dict(state.get("completed_actions") or {})
    key = idempotency_key("exchange", order_id)
    if key in completed:
        response = f"An exchange for order {order_id} is already in progress (reference {completed[key]}). Nothing new was created."
        return reply(response, {"outcome": "already_prepared", "order_id": order_id, "reference": completed[key], "executed": False}, **done)
    try:
        result = lookup_order_status(order_id)
    except Exception:
        logger.exception("Order lookup for exchange failed.")
        result = {"outcome": "data_unavailable"}
    if result["outcome"] != "found":
        return reply(f"I could not find order {order_id} in the demonstration data. Please check the ID.",
                     {"outcome": result["outcome"], "executed": False}, **done)
    status = result["order_status"]
    if status.casefold() != "delivered":
        alternative = "cancel it" if status.casefold() == "processing" else "wait for it to arrive"
        response = (f"Order {order_id} is {status.lower()}, so it can't be exchanged yet. "
                    f"You can {alternative}, and once it is delivered I can arrange an exchange.")
        return reply(response, {"outcome": "not_delivered", "order_id": order_id, "order_status": status, "executed": False}, **done)

    detail = f" Requested change: {request_text}." if request_text else ""
    if not state.get("action_approved"):
        prompt = f"Order {order_id} was delivered. Shall I arrange an exchange?{detail} A pickup of the original item will be scheduled."
        if not request_approval("exchange", prompt, _EXCHANGE_REPLIES, order_id=order_id):
            response = f"Okay, I haven't arranged an exchange for order {order_id}. No operation was performed."
            return reply(response, {"outcome": "kept", "order_id": order_id, "executed": False}, **done)
    reference = f"DEMO-EXCH-{uuid4().hex[:10].upper()}"
    response = (f"Exchange request {reference} is prepared for order {order_id}.{detail} "
                "A pickup of the original item would be scheduled and the replacement shipped once it is checked. "
                "This is a demonstration; no real exchange was arranged.")
    output = reply(response, {"outcome": "exchange_prepared", "order_id": order_id, "reference": reference, "executed": True},
                   completed_actions={**completed, key: reference}, action_approved=False, **done)
    description = f"Exchange requested. Change: {request_text}" if request_text else "Exchange requested."
    return _with_ticket(output, _raise_ticket(
        state, "exchange_request", order_id, description, reference, {"exchange_details": request_text},
    ))


def respond_to_query(state: AgentState) -> dict[str, str]:
    """Retained Phase 1 deterministic node; not used by the support graph."""
    query = (state.get("query") or "").strip()
    if not query:
        return {"response": "Please provide a query."}
    return {"response": f"The test graph received: {query}"}

