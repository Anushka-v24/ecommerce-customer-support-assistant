"""LangGraph construction and thread-aware invocation helpers."""

from __future__ import annotations

import atexit
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.checkpointer import close_postgres_checkpointer, get_configured_checkpointer
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
        "order_tracking_node": nodes.order_tracking_node,
        "order_cancellation_node": nodes.order_cancellation_node,
        "greeting_node": nodes.greeting_node,
        "goodbye_node": nodes.goodbye_node,
        "product_recommendation_node": nodes.product_recommendation_node,
        "warranty_node": nodes.warranty_node,
        "exchange_node": nodes.exchange_node,
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
    builder.add_edge("finalize_turn_node", END)
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
    # Buttons and the raised ticket belong to one turn; reset them before each run.
    turn_input: dict[str, Any] = {"query": query, "session_id": session_id, "quick_replies": [], "ticket": None}
    if customer_id:
        turn_input["customer_id"] = customer_id
    return runnable.invoke(turn_input, {"configurable": {"thread_id": session_id}})


def build_test_graph():
    """Retain the original Phase 1 deterministic graph for its smoke tests."""
    builder = StateGraph(AgentState)
    builder.add_node("respond", nodes.respond_to_query)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()

