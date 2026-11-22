"""HTTP endpoints for health and LangGraph-backed conversations."""

import json
import logging
import time
from uuid import uuid4

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from app.api.blocks import build_blocks
from app.services.customer_service import get_customer, list_customers
from app.services.pii import mask_pii
from app.services.ticket_service import list_tickets, public_ticket
from app.services.turn_log import append_record

api_blueprint = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


def invoke_graph(query: str, session_id: str, customer_id: str | None = None):
    """Load and call the graph only for chat requests; health stays lightweight."""
    from app.graph.workflow import invoke_graph as graph_invoke

    return graph_invoke(query, session_id, customer_id=customer_id)


@api_blueprint.get("/health")
def health_check():
    return jsonify({"status": "healthy", "service": "ecom-langgraph"}), 200


def stream_graph(query: str, session_id: str, customer_id: str | None = None):
    from app.graph.workflow import stream_graph as graph_stream

    return graph_stream(query, session_id, customer_id=customer_id)


_UNAVAILABLE = {"error": "chat_unavailable", "message": "The chat service could not process this request. Please try again later."}
# Shown while a streamed turn runs; keyed by the node that just finished.
_NODE_STATUS = {
    "classify_intent": "Understanding what you need",
    "confirm_intent": "Checking I understood you",
    "order_tracking_node": "Checking your order",
    "order_cancellation_node": "Checking whether the order can be cancelled",
    "exchange_node": "Checking the exchange",
    "product_inquiry_node": "Looking up the product",
    "product_recommendation_node": "Finding products for you",
    "issue_resolver_node": "Working on your issue",
    "customer_care_node": "Preparing a hand-off to our team",
    "store_information_node": "Searching our help documents",
    "feedback_node": "Recording your feedback",
    "escalation_node": "Passing you to a person",
    "finalize_turn_node": "Writing the reply",
    "localize_reply_node": "Putting it in your language",
}


def _parse_chat_request():
    """Return (session_id, message, customer_id) or a ready (body, status) error."""
    if not request.is_json:
        return None, ({"error": "json_required", "message": "Send the request as application/json."}, 415)
    try:
        payload = request.get_json(silent=False)
    except BadRequest:
        return None, ({"error": "invalid_json", "message": "Request body must contain valid JSON."}, 400)

    if not isinstance(payload, dict):
        return None, ({"error": "invalid_request", "message": "JSON body must be an object."}, 400)
    unexpected = set(payload) - {"session_id", "message", "customer_id"}
    if unexpected:
        return None, ({"error": "invalid_request", "message": "Only session_id, message and customer_id fields are accepted."}, 400)

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None, ({"error": "invalid_session_id", "message": "session_id must be a non-empty string."}, 400)
    session_id = session_id.strip()
    if len(session_id) > current_app.config["MAX_SESSION_ID_LENGTH"]:
        return None, ({"error": "invalid_session_id", "message": "session_id exceeds the maximum allowed length."}, 400)

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return None, ({"error": "invalid_message", "message": "message must be a non-empty string."}, 400)
    if len(message) > current_app.config["MAX_CHAT_MESSAGE_LENGTH"]:
        return None, ({"error": "message_too_large", "message": "message exceeds the maximum allowed length."}, 413)

    # Demo sign-in only: the customer ID is chosen in the UI and is not authentication.
    customer_id = payload.get("customer_id")
    if customer_id is not None and get_customer(customer_id) is None:
        return None, ({"error": "invalid_customer_id", "message": "customer_id does not match a known customer."}, 400)
    return (session_id, message.strip(), customer_id), None


def _turn_body(result, session_id: str, message: str, started: float) -> dict:
    """Expose only safe fields from a finished graph state and log the turn."""
    answer = result.get("response") or "The assistant did not return a response."
    turn_id = uuid4().hex
    _log_turn(turn_id, session_id, message, answer, result, (time.perf_counter() - started) * 1000)
    body = {
        "session_id": session_id,
        "turn_id": turn_id,
        "answer": answer,
        "intent": result.get("final_intent"),
        "workflow_status": result.get("workflow_status"),
        "reply_language": result.get("reply_language") or "en",
    }
    quick_replies = _safe_quick_replies(result.get("quick_replies"))
    if quick_replies:
        body["quick_replies"] = quick_replies
    if isinstance(result.get("ticket"), dict):
        body["ticket"] = public_ticket(result["ticket"])
    blocks = build_blocks(result)
    if blocks:
        body["blocks"] = blocks
    return body


@api_blueprint.post("/chat")
def chat():
    """Validate a chat turn, pass it to the existing graph, and expose safe fields only."""
    parsed, error = _parse_chat_request()
    if error:
        return jsonify(error[0]), error[1]
    session_id, message, customer_id = parsed

    started = time.perf_counter()
    try:
        if customer_id:
            result = invoke_graph(message, session_id, customer_id=customer_id)
        else:
            result = invoke_graph(message, session_id)
    except Exception as exc:
        # Do not echo exception messages, which may contain provider/config details.
        logger.error("Chat graph invocation failed (%s)", type(exc).__name__)
        return jsonify(_UNAVAILABLE), 503

    if not isinstance(result, dict):
        logger.error("Chat graph returned a non-object state result")
        return jsonify(_UNAVAILABLE), 503
    return jsonify(_turn_body(result, session_id, message, started)), 200


@api_blueprint.post("/chat/stream")
def chat_stream():
    """Same turn as /chat, sent as Server-Sent Events: status updates, then the result."""
    parsed, error = _parse_chat_request()
    if error:
        return jsonify(error[0]), error[1]
    session_id, message, customer_id = parsed

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def events():
        started = time.perf_counter()
        yield sse("status", {"stage": "received", "label": "Reading your message"})
        try:
            for kind, value in stream_graph(message, session_id, customer_id=customer_id):
                if kind == "node" and value in _NODE_STATUS:
                    yield sse("status", {"stage": value, "label": _NODE_STATUS[value]})
                elif kind == "result":
                    yield sse("result", _turn_body(value, session_id, message, started))
                    return
        except Exception as exc:
            logger.error("Chat stream failed (%s)", type(exc).__name__)
        yield sse("error", _UNAVAILABLE)

    return Response(stream_with_context(events()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _safe_quick_replies(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {"label": item["label"], "value": item["value"]}
        for item in value
        if isinstance(item, dict) and isinstance(item.get("label"), str) and isinstance(item.get("value"), str)
    ]


@api_blueprint.get("/customers")
def customers():
    """List the fictional demo customers offered by the sign-in picker."""
    return jsonify({"customers": list_customers()}), 200


@api_blueprint.get("/tickets")
def tickets():
    """List one customer's (or one conversation's) tickets, newest first."""
    customer_id = (request.args.get("customer_id") or "").strip() or None
    session_id = (request.args.get("session_id") or "").strip() or None
    if customer_id is None and session_id is None:
        return jsonify({"error": "invalid_request", "message": "Provide customer_id or session_id."}), 400
    if customer_id is not None and get_customer(customer_id) is None:
        return jsonify({"error": "invalid_customer_id", "message": "customer_id does not match a known customer."}), 400
    if session_id is not None and len(session_id) > current_app.config["MAX_SESSION_ID_LENGTH"]:
        return jsonify({"error": "invalid_session_id", "message": "session_id exceeds the maximum allowed length."}), 400
    try:
        records = list_tickets(customer_id=customer_id, session_id=session_id)
    except Exception as exc:
        logger.error("Ticket listing failed (%s)", type(exc).__name__)
        return jsonify({"error": "tickets_unavailable", "message": "Tickets could not be loaded right now."}), 503
    return jsonify({"tickets": [public_ticket(record) for record in records]}), 200


def _log_turn(turn_id: str, session_id: str, message: str, answer: str, result: dict, latency_ms: float) -> None:
    """Record what the monitoring and retraining loop needs; never fail the chat turn."""
    record = {
        "turn_id": turn_id, "session_id": session_id, "message": mask_pii(message), "answer": mask_pii(answer),
        "predicted_intent": result.get("predicted_intent"), "confidence": result.get("confidence_score"),
        "final_intent": result.get("final_intent"), "verifier_status": result.get("verifier_status"),
        "classifier_routing": result.get("classifier_routing"), "route": result.get("selected_route"),
        "workflow_status": result.get("workflow_status"), "latency_ms": round(latency_ms, 1),
    }
    logger.info(
        "chat_turn intent=%s confidence=%s verifier=%s route=%s latency_ms=%.0f",
        record["final_intent"], record["confidence"], record["verifier_status"], record["route"], latency_ms,
    )
    if not current_app.config.get("TURN_LOG_ENABLED"):
        return
    try:
        append_record(current_app.config["TURN_LOG_DIR"], "turns.jsonl", record)
    except OSError as exc:
        logger.warning("Could not write turn log (%s)", type(exc).__name__)


@api_blueprint.post("/feedback")
def feedback():
    """Store a thumbs up/down on one answer; thumbs-down turns become new training data."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_request", "message": "Send a JSON object."}), 400
    allowed = {"session_id", "turn_id", "rating", "message", "answer", "comment"}
    if set(payload) - allowed:
        return jsonify({"error": "invalid_request", "message": "Unexpected fields in feedback."}), 400
    rating = payload.get("rating")
    if rating not in {"up", "down"}:
        return jsonify({"error": "invalid_rating", "message": "rating must be 'up' or 'down'."}), 400
    limits = {
        "session_id": current_app.config["MAX_SESSION_ID_LENGTH"],
        "turn_id": 64,
        "message": current_app.config["MAX_CHAT_MESSAGE_LENGTH"],
        "answer": 20000,
        "comment": current_app.config["MAX_FEEDBACK_COMMENT_LENGTH"],
    }
    record = {"rating": rating}
    for field, limit in limits.items():
        value = payload.get(field)
        if value is None and field in {"comment", "message", "answer"}:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            return jsonify({"error": f"invalid_{field}", "message": f"{field} must be a non-empty string up to {limit} characters."}), 400
        record[field] = mask_pii(value.strip()) if field in {"message", "answer", "comment"} else value.strip()
    if current_app.config.get("TURN_LOG_ENABLED"):
        try:
            append_record(current_app.config["TURN_LOG_DIR"], "feedback.jsonl", record)
        except OSError as exc:
            logger.error("Could not store feedback (%s)", type(exc).__name__)
            return jsonify({"error": "feedback_unavailable", "message": "Feedback could not be saved."}), 503
    return jsonify({"status": "recorded"}), 201


@api_blueprint.app_errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "request_too_large", "message": "Request body exceeds the allowed size."}), 413
