"""Flask chat endpoint tests; provider/model work is mocked or stubbed."""

from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services import case_repository
from app.services.case_repository import list_demo_records
from langgraph.checkpoint.memory import InMemorySaver
# Tests never share the developer's PostgreSQL checkpoints; .env may select them.
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
# These modules test the workflows reached after a confirmed request; test_intent_confirmation covers the confirmation step.
patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False).start()


class ApiTestConfig(Config):
    TESTING = True
    APP_ENV = "test"
    CORS_ALLOWED_ORIGINS = "https://shop.example, https://preview.example"
    MAX_CHAT_MESSAGE_LENGTH = 32
    MAX_SESSION_ID_LENGTH = 24
    MAX_CONTENT_LENGTH = 65536
    TURN_LOG_ENABLED = False


class WildcardCorsConfig(ApiTestConfig):
    CORS_ALLOWED_ORIGINS = "*"


class DamageApiTestConfig(ApiTestConfig):
    MAX_CHAT_MESSAGE_LENGTH = 256


class FlaskChatApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(ApiTestConfig)
        self.client = self.app.test_client()

    def post(self, body, **kwargs):
        return self.client.post("/api/chat", json=body, **kwargs)

    def test_success_returns_only_safe_graph_output_fields(self):
        graph_result = {
            "response": "Please provide your order ID.",
            "final_intent": "order_tracking",
            "workflow_status": "awaiting_input",
            "messages": [{"role": "assistant", "content": "private"}],
            "pending_workflow": {"secret_internal": "not returned"},
            "error": "internal diagnostic",
        }
        with patch("app.api.routes.invoke_graph", return_value=graph_result) as invoke:
            response = self.post({"session_id": "customer-a", "message": "Where is my order?"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertRegex(body.pop("turn_id"), r"^[0-9a-f]{32}$")
        self.assertEqual(body, {
            "session_id": "customer-a",
            "answer": "Please provide your order ID.",
            "intent": "order_tracking",
            "workflow_status": "awaiting_input",
        })
        invoke.assert_called_once_with("Where is my order?", "customer-a")

    def test_missing_session_id_is_rejected(self):
        response = self.post({"message": "Hello"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_session_id")

    def test_empty_or_oversized_session_id_is_rejected(self):
        self.assertEqual(self.post({"session_id": "  ", "message": "Hi"}).status_code, 400)
        self.assertEqual(self.post({"session_id": "x" * 25, "message": "Hi"}).status_code, 400)

    def test_missing_or_empty_message_is_rejected(self):
        self.assertEqual(self.post({"session_id": "s"}).status_code, 400)
        self.assertEqual(self.post({"session_id": "s", "message": "  "}).status_code, 400)

    def test_oversized_message_is_rejected_before_graph_invocation(self):
        with patch("app.api.routes.invoke_graph") as invoke:
            response = self.post({"session_id": "s", "message": "x" * 33})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "message_too_large")
        invoke.assert_not_called()

    def test_oversized_http_body_returns_json_error(self):
        response = self.client.post(
            "/api/chat",
            data='{"session_id":"s","message":"' + ("x" * 70000) + '"}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "request_too_large")

    def test_invalid_json_is_rejected(self):
        response = self.client.post("/api/chat", data="{invalid", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_json")

    def test_non_json_content_type_is_rejected(self):
        response = self.client.post("/api/chat", data="message=hello")
        self.assertEqual(response.status_code, 415)

    def test_internal_thread_data_is_rejected(self):
        with patch("app.api.routes.invoke_graph") as invoke:
            response = self.post({"session_id": "s", "message": "Hi", "thread_id": "other"})
        self.assertEqual(response.status_code, 400)
        invoke.assert_not_called()

    def test_graph_failure_is_safe_and_does_not_expose_exception(self):
        with patch("app.api.routes.invoke_graph", side_effect=RuntimeError("private stack and secret")):
            response = self.post({"session_id": "s", "message": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "chat_unavailable")
        self.assertNotIn("private", response.get_data(as_text=True))
        self.assertNotIn("secret", response.get_data(as_text=True))

    def test_allowed_origin_receives_cors_headers_and_disallowed_origin_does_not(self):
        allowed = self.client.get("/api/health", headers={"Origin": "https://shop.example"})
        self.assertEqual(allowed.headers.get("Access-Control-Allow-Origin"), "https://shop.example")
        self.assertIn("Origin", allowed.headers.get("Vary", ""))
        disallowed = self.client.get("/api/health", headers={"Origin": "https://untrusted.example"})
        self.assertIsNone(disallowed.headers.get("Access-Control-Allow-Origin"))

    def test_preflight_allows_json_from_allowlisted_origin(self):
        response = self.client.options("/api/chat", headers={
            "Origin": "https://shop.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "https://shop.example")
        self.assertIn("POST", response.headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("Content-Type", response.headers.get("Access-Control-Allow-Headers", ""))

    def test_wildcard_cors_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact origins"):
            create_app(WildcardCorsConfig)

    def test_health_endpoint_keeps_existing_response(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "healthy", "service": "ecom-langgraph"})

    def test_conversation_continues_and_remains_isolated_by_session(self):
        graph = build_graph(checkpointer=InMemorySaver())

        def classifier(query):
            intent = "order_tracking" if query == "Where is my order?" else "out_of_scope"
            return {
                "predicted_intent": intent,
                "confidence_score": 99.0,
                "final_intent": intent,
                "verifier_status": "ACCEPTED_BY_CLASSIFIER",
                "routing": intent,
                "latency_ms": 1.0,
                "error": None,
            }

        def use_test_graph(query, session_id):
            return invoke_graph(query, session_id, graph)

        with patch.object(nodes, "classify_and_verify", side_effect=classifier), \
             patch("app.api.routes.invoke_graph", side_effect=use_test_graph):
            first = self.post({"session_id": "thread-a", "message": "Where is my order?"})
            other = self.post({"session_id": "thread-b", "message": "ORD-1001"})
            resumed = self.post({"session_id": "thread-a", "message": "ORD-1001"})

        self.assertEqual(first.get_json()["workflow_status"], "awaiting_input")
        self.assertEqual(other.get_json()["intent"], "out_of_scope")
        self.assertEqual(resumed.get_json()["intent"], "order_tracking")
        self.assertIn("ORD-1001", resumed.get_json()["answer"])
        self.assertIn("demonstration", resumed.get_json()["answer"].lower())

    def test_damaged_order_api_conversation_resumes_without_reclassifying_slots(self):
        app = create_app(DamageApiTestConfig)
        client = app.test_client()
        graph = build_graph(checkpointer=InMemorySaver())
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        calls = []

        def classifier(query):
            calls.append(query)
            return {
                "predicted_intent": "issue_support", "confidence_score": 99.0,
                "final_intent": "issue_support", "verifier_status": "ACCEPTED_BY_CLASSIFIER",
                "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
            }

        def invoke(query, session_id):
            return invoke_graph(query, session_id, graph)

        with patch.object(Config, "BUSINESS_DATA_BACKEND", "json"), \
             patch.object(case_repository, "DEFAULT_DB_PATH", Path(temp_dir.name) / "cases.sqlite3"), \
             patch.object(nodes, "classify_and_verify", side_effect=classifier), \
             patch("app.api.routes.invoke_graph", side_effect=invoke):
            first = client.post("/api/chat", json={"session_id": "damage-api", "message": "My order arrived damaged"})
            second = client.post("/api/chat", json={"session_id": "damage-api", "message": "ORD-1001"})
            third = client.post("/api/chat", json={"session_id": "damage-api", "message": "The box was crushed and the screen is cracked."})
            records = list_demo_records("damaged_order", "damage-api")

        self.assertIn("provide your order ID", first.get_json()["answer"])
        self.assertIn("describe what is damaged", second.get_json()["answer"])
        self.assertIn("demonstration case", third.get_json()["answer"].lower())
        self.assertIn("no retailer or courier claim", third.get_json()["answer"].lower())
        self.assertEqual(calls, ["My order arrived damaged"])
        self.assertEqual(len(records), 1)
        self.assertIn("screen is cracked", records[0]["reason"])


class TurnLogAndFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.log_dir.cleanup)
        config = type("LoggingConfig", (ApiTestConfig,), {"TURN_LOG_ENABLED": True, "TURN_LOG_DIR": self.log_dir.name})
        self.client = create_app(config).test_client()

    def records(self, filename):
        path = Path(self.log_dir.name) / filename
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_chat_turn_is_logged_with_monitoring_fields(self):
        graph_result = {
            "response": "Order ORD-1001 has shipped.", "final_intent": "order_tracking",
            "predicted_intent": "order_tracking", "confidence_score": 99.1,
            "verifier_status": "ACCEPTED_BY_CLASSIFIER", "selected_route": "order_tracking",
            "workflow_status": "completed",
        }
        with patch("app.api.routes.invoke_graph", return_value=graph_result):
            body = self.client.post("/api/chat", json={"session_id": "s1", "message": "where is ORD-1001"}).get_json()
        [record] = self.records("turns.jsonl")
        self.assertEqual(record["turn_id"], body["turn_id"])
        self.assertEqual((record["final_intent"], record["confidence"], record["route"]), ("order_tracking", 99.1, "order_tracking"))
        self.assertIn("latency_ms", record)

    def test_thumbs_down_is_stored_with_message_and_answer(self):
        response = self.client.post("/api/feedback", json={
            "session_id": "s1", "turn_id": "abc123", "rating": "down",
            "message": "i do not like my order", "answer": "Not supported.", "comment": "wrong",
        })
        self.assertEqual(response.status_code, 201)
        [record] = self.records("feedback.jsonl")
        self.assertEqual((record["rating"], record["message"], record["comment"]), ("down", "i do not like my order", "wrong"))

    def test_invalid_feedback_is_rejected(self):
        for body in (
            {"session_id": "s1", "turn_id": "t", "rating": "meh"},
            {"session_id": "s1", "rating": "up"},
            {"session_id": "s1", "turn_id": "t", "rating": "up", "extra": 1},
            {"session_id": "s1", "turn_id": "t", "rating": "up", "message": "x" * 33},
        ):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/feedback", json=body).status_code, 400)
        self.assertFalse((Path(self.log_dir.name) / "feedback.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
