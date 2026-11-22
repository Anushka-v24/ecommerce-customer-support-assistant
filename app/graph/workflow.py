"""LangGraph construction and thread-aware invocation helpers."""

from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph import nodes
from app.graph.checkpointer import close_postgres_checkpointer, get_configured_checkpointer
from app.graph.risk import ABANDON, approval_answer
from app.graph.routing import ROUTE_TO_NODE, route_with_intent_confirmation
from app.graph.state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build the graph with an injected saver or the explicitly configured backend."""
    builder = StateGraph(AgentState)
    node_functions = {
        "continuity_node": nodes.continuity_node,
        "classify_intent": nodes.classify_intent_node,
        "confirm_intent_node": nodes.confirm_intent_node,
        "resume_pending_node": nodes.resume_pending_node,
        "pending_clarification_node": nodes.pending_clarification_node,
        "finalize_turn_node": nodes.finalize_turn_node,
        "localize_reply_node": nodes.localize_reply_node,
        "order_tracking_node": nodes.order_tracking_node,
        "order_cancellation_node": nodes.order_cancellation_node,
        "greeting_node": nodes.greeting_node,
        "goodbye_node": nodes.goodbye_node,
        "product_recommendation_node": nodes.product_recommendation_node,
        "warranty_node": nodes.warranty_node,
        "exchange_node": nodes.exchange_node,
        "feedback_node": nodes.feedback_node,
        "escalation_node": nodes.escalation_node,
        "product_inquiry_node": nodes.product_inquiry_node,
        "payment_node": nodes.payment_node,
        "issue_resolver_node": nodes.issue_resolver_node,
        "customer_care_node": nodes.customer_care_node,
        "store_information_node": nodes.store_information_node,
        "shipping_node": nodes.shipping_node,
        "rag_node": nodes.rag_node,
        "unsupported_node": nodes.unsupported_node,
        "unknown_intent_node": nodes.unknown_intent_node,
        "out_of_scope_node": nodes.out_of_scope_node,
        "verification_error_node": nodes.verification_error_node,
    }
    for name, function in node_functions.items():
        builder.add_node(name, function)
    builder.add_edge(START, "continuity_node")
    builder.add_conditional_edges(
        "continuity_node",
        nodes.route_continuity_action,
        {
            "classify": "classify_intent",
            "resume_pending": "resume_pending_node",
            "clarify_pending": "pending_clarification_node",
        },
    )
    builder.add_conditional_edges(
        "classify_intent", route_with_intent_confirmation,
        {**ROUTE_TO_NODE, "confirm_intent": "confirm_intent_node"},
    )
    for node_name in set(ROUTE_TO_NODE.values()) | {"confirm_intent_node", "resume_pending_node", "pending_clarification_node"}:
        builder.add_edge(node_name, "finalize_turn_node")
    builder.add_edge("finalize_turn_node", "localize_reply_node")
    builder.add_edge("localize_reply_node", END)
    selected_checkpointer = checkpointer if checkpointer is not None else get_configured_checkpointer()
    return builder.compile(checkpointer=selected_checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    """Return the process-wide compiled graph and its process-lifetime saver."""
    return build_graph()


def close_graph_resources() -> None:
    """Clear the compiled graph and close its shared database pools."""
    get_graph.cache_clear()
    close_postgres_checkpointer()
    from app.db.session import dispose_engines

    dispose_engines()


# Register only after this module (and the graph dependencies) has loaded. This
# avoids importing LangGraph during Python's interpreter-shutdown phase.
atexit.register(close_graph_resources)


def invoke_graph(
    query: str, session_id: str, graph: Any | None = None, customer_id: str | None = None,
) -> AgentState:
    """Invoke a conversation with its session ID as the checkpointer thread ID."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required to isolate conversation memory.")
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    runnable = graph or get_graph()
    config = {"configurable": {"thread_id": session_id}}
    payload = _turn_payload(runnable, config, query, session_id, customer_id)
    return _surface_approval(runnable.invoke(payload, config), query)


def stream_graph(
    query: str, session_id: str, graph: Any | None = None, customer_id: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Like invoke_graph, but yield ("node", name) as each node finishes, then ("result", state)."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required to isolate conversation memory.")
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    runnable = graph or get_graph()
    config = {"configurable": {"thread_id": session_id}}
    payload = _turn_payload(runnable, config, query, session_id, customer_id)
    interrupts = None
    for chunk in runnable.stream(payload, config, stream_mode="updates"):
        for name, update in chunk.items():
            if name == "__interrupt__":
                interrupts = update
            else:
                yield "node", name
    state = dict(runnable.get_state(config).values)
    if interrupts:
        state["__interrupt__"] = interrupts
    yield "result", _surface_approval(state, query)


def _turn_payload(runnable: Any, config: dict[str, Any], query: str, session_id: str, customer_id: str | None) -> Any:
    """Resume a paused approval with the answer, or start a fresh turn."""
    waiting = _pending_approval(runnable, config)
    if waiting is not None:
        if approval_answer(query) != "other":
            return Command(resume=query)
        # A new request instead of yes/no: close the approval without acting, then handle it.
        runnable.invoke(Command(resume=ABANDON), config)
    # Buttons, the raised ticket and approvals belong to one turn; reset them before each run.
    turn_input: dict[str, Any] = {
        "query": query, "session_id": session_id, "quick_replies": [], "ticket": None, "action_approved": False,
    }
    if customer_id:
        turn_input["customer_id"] = customer_id
    return turn_input


def _pending_approval(runnable: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """The approval payload the thread is paused on, if any."""
    try:
        snapshot = runnable.get_state(config)
    except ValueError:
        return None  # graphs compiled without a checkpointer cannot pause
    for task in snapshot.tasks or ():
        for pending in task.interrupts or ():
            if isinstance(pending.value, dict) and pending.value.get("type") == "approval":
                return pending.value
    return None


def _surface_approval(result: dict[str, Any], query: str) -> dict[str, Any]:
    """Present a paused approval as an ordinary assistant turn."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return result
    payload = interrupts[0].value
    response, replies, language = payload["prompt"], payload.get("quick_replies") or [], "en"
    translated = nodes.translate_reply(query, response, replies)
    if translated:
        response, replies, language = translated["response"], translated["quick_replies"], translated["lang"]
    return {
        **{k: v for k, v in result.items() if k != "__interrupt__"},
        "selected_route": payload["action"],
        "tool_result": {"outcome": "awaiting_confirmation", "order_id": payload.get("order_id"),
                        "risk": payload.get("risk"), "executed": False},
        "response": response, "quick_replies": replies, "reply_language": language,
        "workflow_status": "awaiting_approval",
    }


def build_test_graph():
    """Retain the original Phase 1 deterministic graph for its smoke tests."""
    builder = StateGraph(AgentState)
    builder.add_node("respond", nodes.respond_to_query)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()

