"""HTTP endpoints for health and LangGraph-backed conversations."""

import logging

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

api_blueprint = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


def invoke_graph(query: str, session_id: str):
    """Load and call the graph only for chat requests; health stays lightweight."""
    from app.graph.workflow import invoke_graph as graph_invoke

    return graph_invoke(query, session_id)


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
    unexpected = set(payload) - {"session_id", "message"}
    if unexpected:
        return jsonify({"error": "invalid_request", "message": "Only session_id and message fields are accepted."}), 400

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

    try:
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
    return jsonify({
        "session_id": session_id,
        "answer": result.get("response") or "The assistant did not return a response.",
        "intent": result.get("final_intent"),
        "workflow_status": result.get("workflow_status"),
    }), 200


@api_blueprint.app_errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "request_too_large", "message": "Request body exceeds the allowed size."}), 413
