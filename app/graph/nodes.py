"""Classifier and workflow nodes for the multilingual ecommerce support graph."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.classifier.classifier_service import classify_and_verify
from app.config import Config
from app.graph.state import AgentState
from app.graph.routing import INTENT_LABELS
from app.rag.service import (
    RAGConfigurationError,
    RAGProviderError,
    get_rag_service,
)
from app.rag.workflow import run_agentic_rag
from app.services.issue_resolver import resolve_issue_type
from app.services.product_service import search_products
from app.services.store_response import format_store_information
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


def _classify_pending_message(state: AgentState, pending: dict[str, Any]) -> dict[str, Any]:
    """Classify only when the current text does not validate against its pending slot."""
    query = (state.get("query") or "").strip()
    try:
        result = classify_and_verify(query)
    except Exception as exc:
        result = {"verifier_status": "CLASSIFICATION_ERROR", "final_intent": None, "error": str(exc)}
    result = {**result, "query": query}
    if _verified_new_request(result, pending):
        stack = list(state.get("suspended_workflows") or [])
        stack.append({**pending, "status": "suspended"})
        return {
            "continuity_action": "classify", "resuming_pending": False,
            "pending_workflow": None, "suspended_workflows": stack,
            "workflow_status": "suspended", "pending_clarification": pending.get("prompt"),
            "preclassified_result": result,
        }
    if state.get("confirmation_required") and state.get("confirmation_status") != "confirmed":
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
        match = re.fullmatch(r"(?:my order(?: id)? is\s+|order id[: ]+)?(ORD-\d{4})", query, re.IGNORECASE)
        if match:
            accepted = match.group(1).upper()
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
    response = "I could not safely resume that pending request. Please describe what you need again."
    return {"selected_route": "pending:unsupported", "response": response, "tool_result": {"status": "not_executed"}, "pending_workflow": None, "workflow_status": "idle", "resuming_pending": False, "messages": _append_message(state, "assistant", response)}


def finalize_turn_node(state: AgentState) -> dict[str, Any]:
    """Restore the most recently interrupted workflow after the current task finishes."""
    active = state.get("pending_workflow")
    if isinstance(active, dict) and active.get("status") in {"awaiting_input", "suspended", "awaiting_confirmation"}:
        return {}
    suspended = list(state.get("suspended_workflows") or [])
    if not suspended:
        return {}
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
        "pending_workflow": restored,
        "suspended_workflows": suspended,
        "workflow_status": "awaiting_input",
        "pending_clarification": prompt,
        "response": response,
        "messages": messages,
        "resuming_pending": False,
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
        result = state.get("preclassified_result") or classify_and_verify(query)
    except Exception as exc:
        result = {
            "predicted_intent": None, "confidence_score": None, "final_intent": None,
            "verifier_status": "CLASSIFICATION_ERROR", "routing": "ERROR",
            "latency_ms": None, "error": str(exc),
        }
    return {
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
            candidates=resolution.get("candidates", []),
        )
        return {
            "selected_route": "issue_support:clarification", "issue_type": None,
            "workflow_step": "awaiting_issue_clarification",
            "tool_result": {"outcome": "clarification_required", "candidates": resolution.get("candidates", []), "tools_executed": False},
            "response": response, "error": None,
            "messages": _append_message(state, "assistant", response),
            "pending_workflow": pending, "workflow_status": "awaiting_input",
            "pending_clarification": response, "resuming_pending": False,
        }

    issue_type = resolution["issue_type"]
    order_id = _order_id(state) or collected.get("order_id")
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
            return _issue_result(state, issue_type, result["outcome"], "return_request_prepared", response, result, issue_details=collected)

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
            return _issue_result(state, issue_type, result["outcome"], "damaged_case_recorded", response, result, issue_details=collected)

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
            return _issue_result(state, issue_type, result["outcome"], "missing_delivery_investigated", response, result, issue_details=collected)
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
    try:
        result = customer_handoff_tool(state.get("session_id"), reason, summary or reason)
        if result["outcome"] == "simulated_handoff_ready":
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
    return {
        "selected_route": "customer_care",
        "handoff_status": handoff_status,
        "support_case_reference": (result.get("handoff") or {}).get("reference"),
        "tool_result": result, "response": response,
        "error": None if handoff_status == "simulated_handoff_ready" else result.get("error"),
        "messages": _append_message(state, "assistant", response),
    }


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
        response = "Document knowledge lookup is unavailable because its provider configuration is incomplete."
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
                response += "\n\nRAG is not configured. Existing development sample data follows and is fictional:\n"
                response += format_store_information(demo)
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
        "rag_context": result.get("documents", []),
        "sources": result.get("sources", []),
        "messages": _append_message(state, "assistant", response),
    }


def payment_node(state: AgentState) -> dict[str, Any]:
    route = "payment_support" if state.get("final_intent") == "issue_support" else "payment"
    return _workflow_pending(
        state, route,
        f"The {route.replace('_', ' ')} workflow was selected, but its business integration is pending. No payment operation was performed.",
    )


def shipping_node(state: AgentState) -> dict[str, Any]:
    return _workflow_pending(
        state, "shipping",
        "Shipping information was selected, but its business integration is pending. No shipping lookup was performed.",
    )


def rag_node(state: AgentState) -> dict[str, Any]:
    return _workflow_pending(
        state, "rag",
        "The knowledge workflow was selected, but retrieval is pending. No documents or external models were queried.",
    )


SUPPORTED_ALTERNATIVES = (
    "I can check an order's status, help with a return, refund, damaged or missing item, "
    "look up product availability, or connect you with a support agent."
)
UNSUPPORTED_RESPONSES = {
    "account_management": "I can't make changes to your account here yet.",
    "add_to_cart": "I can't add items to your cart from this chat yet.",
    "order_management": "I can't cancel or change orders from this chat yet.",
    "place_order": "I can't place orders from this chat yet.",
    "remove_from_cart": "I can't change your cart from this chat yet.",
    "submit_feedback": "Thanks for sharing that. I can't record feedback from this chat yet.",
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
        "response": response, "error": None,
        "messages": _append_message(state, "assistant", response),
    }


def unknown_intent_node(state: AgentState) -> dict[str, Any]:
    response = "I could not verify this as one of the supported ecommerce intents. No business workflow was run."
    return {
        "selected_route": "unknown_intent",
        "tool_result": {"status": "not_executed", "reason": "unknown_intent"},
        "response": response, "error": state.get("error"),
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
        "response": response, "error": None,
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


def respond_to_query(state: AgentState) -> dict[str, str]:
    """Retained Phase 1 deterministic node; not used by the support graph."""
    query = (state.get("query") or "").strip()
    if not query:
        return {"response": "Please provide a query."}
    return {"response": f"The test graph received: {query}"}

