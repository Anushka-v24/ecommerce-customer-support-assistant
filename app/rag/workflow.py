"""Bounded LangGraph workflow for document-grounded store-information answers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.state import AgentState
from app.rag.service import (
    RAGConfigurationError,
    RAGProviderError,
    RAGService,
    UNAVAILABLE_ANSWER,
    get_rag_service,
)


_FOLLOW_UP = re.compile(r"\b(?:it|that|they|them|there|those|these|what about|and what about)\b", re.I)
_TRANSACTIONAL_CONTEXT = re.compile(
    r"\b(?:order|ord-\d+|refund|return my|return this|tracking|payment|damaged|missing parcel|"
    r"agent|courier|shipment status)\b",
    re.I,
)


def _contextual_query(state: AgentState) -> str:
    """Resolve a short knowledge follow-up from this thread's recent user turn only."""
    query = (state.get("query") or "").strip()
    if not query or not _FOLLOW_UP.search(query):
        return query
    for message in reversed(state.get("messages") or []):
        if message.get("role") != "user":
            continue
        previous = str(message.get("content") or "").strip()
        if not previous or previous == query or _TRANSACTIONAL_CONTEXT.search(previous):
            continue
        return f"{previous}\nFollow-up question: {query}"
    return query


def _service(service: RAGService | None) -> RAGService:
    return service if service is not None else get_rag_service()


def build_rag_graph(service: RAGService | None = None):
    """Build the small RAG subgraph; service injection keeps tests fully offline."""

    def analyze_query(state: AgentState) -> dict[str, Any]:
        original = (state.get("query") or "").strip()
        contextual = _contextual_query(state)
        return {
            "rag_original_query": original,
            "rag_search_query": contextual,
            "rag_retry_count": 0,
            "rag_documents": [],
            "rag_evidence": {},
            "rag_error": None,
            "rag_error_kind": None,
            "rag_status": "working",
        }

    def retrieve_documents(state: AgentState) -> dict[str, Any]:
        try:
            documents = _service(service).retrieve(state.get("rag_search_query") or state.get("query") or "")
            return {"rag_documents": documents, "rag_error": None, "rag_error_kind": None}
        except (RAGConfigurationError, RAGProviderError) as exc:
            kind = "configuration" if isinstance(exc, RAGConfigurationError) else "provider"
            return {"rag_documents": [], "rag_error": str(exc), "rag_error_kind": kind, "rag_status": "unavailable"}
        except Exception:
            return {"rag_documents": [], "rag_error": "Document retrieval failed.", "rag_error_kind": "provider", "rag_status": "unavailable"}

    def evaluate_evidence(state: AgentState) -> dict[str, Any]:
        documents = state.get("rag_documents") or []
        question = state.get("rag_search_query") or state.get("rag_original_query") or ""
        try:
            evidence = _service(service).evaluate_evidence(question, documents)
            if (
                not isinstance(evidence, dict)
                or set(evidence) != {"sufficient", "reason", "missing_information"}
                or not isinstance(evidence.get("sufficient"), bool)
                or not isinstance(evidence.get("reason"), str)
                or not evidence["reason"].strip()
                or (evidence.get("sufficient") is True and not documents)
                or (evidence.get("missing_information") is not None and not isinstance(evidence.get("missing_information"), str))
            ):
                raise RAGProviderError("Evidence evaluator returned an invalid structure")
            return {"rag_evidence": evidence, "rag_error": None, "rag_error_kind": None}
        except (RAGConfigurationError, RAGProviderError) as exc:
            kind = "configuration" if isinstance(exc, RAGConfigurationError) else "provider"
            return {"rag_evidence": {}, "rag_error": str(exc), "rag_error_kind": kind, "rag_status": "unavailable"}
        except Exception:
            return {"rag_evidence": {}, "rag_error": "Evidence evaluation failed.", "rag_error_kind": "provider", "rag_status": "unavailable"}

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        try:
            rewritten = _service(service).rewrite_query(
                state.get("rag_search_query") or state.get("rag_original_query") or "",
                state.get("rag_documents") or [],
                (state.get("rag_evidence") or {}).get("missing_information"),
            )
            return {
                "rag_search_query": rewritten,
                "rag_retry_count": (state.get("rag_retry_count") or 0) + 1,
                "rag_error": None,
                "rag_error_kind": None,
            }
        except (RAGConfigurationError, RAGProviderError) as exc:
            kind = "configuration" if isinstance(exc, RAGConfigurationError) else "provider"
            return {"rag_error": str(exc), "rag_error_kind": kind, "rag_status": "unavailable"}
        except Exception:
            return {"rag_error": "Query refinement failed.", "rag_error_kind": "provider", "rag_status": "unavailable"}

    def generate_answer(state: AgentState) -> dict[str, Any]:
        try:
            # Include the customer's current wording plus the same-thread expansion,
            # while the model still receives only retrieved excerpts as evidence.
            original = state.get("rag_original_query") or ""
            contextual = state.get("rag_search_query") or original
            question = original if contextual == original else f"{original}\nContext: {contextual}"
            answer = _service(service).generate_answer(question, state.get("rag_documents") or [])
            sources = list(dict.fromkeys(
                str(item.get("source", "unknown"))
                for item in (state.get("rag_documents") or [])
                if item.get("source")
            ))
            return {"rag_answer": answer, "sources": sources, "rag_status": "answered", "rag_error": None, "rag_error_kind": None}
        except (RAGConfigurationError, RAGProviderError) as exc:
            kind = "configuration" if isinstance(exc, RAGConfigurationError) else "provider"
            return {"rag_error": str(exc), "rag_error_kind": kind, "rag_status": "unavailable"}
        except Exception:
            return {"rag_error": "Answer generation failed.", "rag_error_kind": "provider", "rag_status": "unavailable"}

    def return_unavailable(state: AgentState) -> dict[str, Any]:
        return {
            "rag_answer": UNAVAILABLE_ANSWER,
            "sources": [],
            "rag_status": "unavailable" if state.get("rag_error") else "no_evidence",
        }

    def after_retrieval(state: AgentState) -> str:
        return "return_unavailable" if state.get("rag_error") else "evaluate_evidence"

    def after_evaluation(state: AgentState) -> str:
        if state.get("rag_error"):
            return "return_unavailable"
        evidence = state.get("rag_evidence") or {}
        if evidence.get("sufficient") is True:
            return "generate_answer"
        if (state.get("rag_retry_count") or 0) < 1:
            return "rewrite_query"
        return "return_unavailable"

    def after_rewrite(state: AgentState) -> str:
        return "return_unavailable" if state.get("rag_error") else "retrieve_documents"

    def after_generation(state: AgentState) -> str:
        return "return_unavailable" if state.get("rag_error") else END

    builder = StateGraph(AgentState)
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("evaluate_evidence", evaluate_evidence)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("return_unavailable", return_unavailable)
    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "retrieve_documents")
    builder.add_conditional_edges("retrieve_documents", after_retrieval, {
        "evaluate_evidence": "evaluate_evidence", "return_unavailable": "return_unavailable",
    })
    builder.add_conditional_edges("evaluate_evidence", after_evaluation, {
        "generate_answer": "generate_answer", "rewrite_query": "rewrite_query",
        "return_unavailable": "return_unavailable",
    })
    builder.add_conditional_edges("rewrite_query", after_rewrite, {
        "retrieve_documents": "retrieve_documents", "return_unavailable": "return_unavailable",
    })
    builder.add_conditional_edges("generate_answer", after_generation, {
        END: END, "return_unavailable": "return_unavailable",
    })
    builder.add_edge("return_unavailable", END)
    return builder.compile()


@lru_cache(maxsize=1)
def _compiled_graph_for(service: RAGService):
    """Reuse the compiled subgraph while the application service is unchanged."""
    return build_rag_graph(service)


def run_agentic_rag(
    query: str,
    messages: list[dict[str, Any]] | None = None,
    *,
    service: RAGService | None = None,
) -> dict[str, Any]:
    selected_service = service if service is not None else get_rag_service()
    try:
        graph = _compiled_graph_for(selected_service)
    except TypeError:
        # Unhashable test doubles can still inject a service without affecting
        # production graph caching or adding a dependency on mock behavior.
        graph = build_rag_graph(selected_service)
    return graph.invoke({"query": query, "messages": list(messages or [])})
