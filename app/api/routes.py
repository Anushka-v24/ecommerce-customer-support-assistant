"""HTTP endpoints for health and LangGraph-backed conversations."""

import logging
import time
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from app.services.customer_service import get_customer, list_customers
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


@api_blueprint.post("/chat")
def chat():
    """Validate a chat turn, pass it to the existing graph, and expose safe fields only."""
    if not request.is_json:
        return jsonify({"error": "json_required", "message": "Send the request as application/json."}), 415
    try:
        payload = request.get_json(silent=False)
    except BadRequest:
        return jsonify({"error": "invalid_json", "message": "Request body must contain valid JSON."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_request", "message": "JSON body must be an object."}), 400
    unexpected = set(payload) - {"session_id", "message", "customer_id"}
    if unexpected:
        return jsonify({"error": "invalid_request", "message": "Only session_id, message and customer_id fields are accepted."}), 400

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "invalid_session_id", "message": "session_id must be a non-empty string."}), 400
    session_id = session_id.strip()
    if len(session_id) > current_app.config["MAX_SESSION_ID_LENGTH"]:
        return jsonify({"error": "invalid_session_id", "message": "session_id exceeds the maximum allowed length."}), 400

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "invalid_message", "message": "message must be a non-empty string."}), 400
    if len(message) > current_app.config["MAX_CHAT_MESSAGE_LENGTH"]:
        return jsonify({"error": "message_too_large", "message": "message exceeds the maximum allowed length."}), 413

    # Demo sign-in only: the customer ID is chosen in the UI and is not authentication.
    customer_id = payload.get("customer_id")
    if customer_id is not None and get_customer(customer_id) is None:
        return jsonify({"error": "invalid_customer_id", "message": "customer_id does not match a known customer."}), 400

    started = time.perf_counter()
    try:
        if customer_id:
            result = invoke_graph(message.strip(), session_id, customer_id=customer_id)
        else:
            result = invoke_graph(message.strip(), session_id)
    except Exception as exc:
        # Do not echo exception messages, which may contain provider/config details.
        logger.error("Chat graph invocation failed (%s)", type(exc).__name__)
        return jsonify({
            "error": "chat_unavailable",
            "message": "The chat service could not process this request. Please try again later.",
        }), 503

    if not isinstance(result, dict):
        logger.error("Chat graph returned a non-object state result")
        return jsonify({
            "error": "chat_unavailable",
            "message": "The chat service could not process this request. Please try again later.",
        }), 503
    answer = result.get("response") or "The assistant did not return a response."
    turn_id = uuid4().hex
    _log_turn(turn_id, session_id, message.strip(), answer, result, (time.perf_counter() - started) * 1000)
    body = {
        "session_id": session_id,
        "turn_id": turn_id,
        "answer": answer,
        "intent": result.get("final_intent"),
        "workflow_status": result.get("workflow_status"),
    }
    quick_replies = _safe_quick_replies(result.get("quick_replies"))
    if quick_replies:
        body["quick_replies"] = quick_replies
    if isinstance(result.get("ticket"), dict):
        body["ticket"] = public_ticket(result["ticket"])
    return jsonify(body), 200


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
        "turn_id": turn_id, "session_id": session_id, "message": message, "answer": answer,
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
        record[field] = value.strip()
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
