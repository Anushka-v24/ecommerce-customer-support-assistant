"""Typed state passed between LangGraph nodes."""

from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    messages: list[dict[str, Any]]
    predicted_intent: str | None
    confidence_score: float | None
    final_intent: str | None
    verifier_status: str | None
    classifier_routing: str | None
    selected_route: str | None
    tool_result: dict[str, Any] | list[Any] | str | None
    response: str | None
    error: str | None
    pending_workflow: NotRequired[dict[str, Any] | None]
    suspended_workflows: NotRequired[list[dict[str, Any]]]
    workflow_status: NotRequired[str | None]
    pending_clarification: NotRequired[str | None]
    resuming_pending: NotRequired[bool]
    continuity_action: NotRequired[str | None]
    preclassified_result: NotRequired[dict[str, Any] | None]
    confirmation_required: NotRequired[bool]
    confirmation_status: NotRequired[str | None]
    order_id: NotRequired[str | None]
    product_id: NotRequired[str | None]
    product_name: NotRequired[str | None]
    issue_type: NotRequired[str | None]
    workflow_step: NotRequired[str | None]
    issue_details: NotRequired[dict[str, Any]]
    support_case_reference: NotRequired[str | None]
    handoff_status: NotRequired[str | None]
    entities: NotRequired[dict[str, Any]]
    rag_context: NotRequired[list[dict[str, Any]]]
    sources: NotRequired[list[str]]
    rag_original_query: NotRequired[str]
    rag_search_query: NotRequired[str]
    rag_retry_count: NotRequired[int]
    rag_documents: NotRequired[list[dict[str, Any]]]
    rag_evidence: NotRequired[dict[str, Any]]
    rag_answer: NotRequired[str]
    rag_status: NotRequired[str]
    rag_error: NotRequired[str | None]
    rag_error_kind: NotRequired[str | None]
