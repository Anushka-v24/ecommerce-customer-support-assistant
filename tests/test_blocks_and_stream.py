import json
import unittest
from unittest.mock import patch

from app import create_app
from app.api.blocks import build_blocks
from app.config import Config
from tests.test_chat_api import ApiTestConfig

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()


class BlockTests(unittest.TestCase):
    def test_order_card_has_timeline_and_product(self):
        blocks = build_blocks({"selected_route": "order_tracking", "tool_result": {
            "outcome": "found", "order_id": "ORD-1001", "order_status": "Shipped", "eta": "2025-11-08",
            "product_id": "PROD-001", "delivery_progress": {"explanation": "On its way."},
        }})
        card = blocks[0]
        self.assertEqual(card["type"], "order_card")
        self.assertEqual(card["product_name"], "Laptop Pro")
        states = {step["label"]: step["state"] for step in card["timeline"]}
        self.assertEqual(states["Processing"], "done")
        self.assertEqual(states["Shipped"], "current")
        self.assertEqual(states["Delivered"], "upcoming")

    def test_recommendations_become_product_cards(self):
        blocks = build_blocks({"selected_route": "product_recommendation", "tool_result": {"products": ["Laptop Pro"]}})
        self.assertEqual(blocks[0]["type"], "product_card")
        self.assertTrue(blocks[0]["in_stock"])

    def test_unknown_results_give_no_blocks(self):
        self.assertEqual(build_blocks({"selected_route": "order_tracking", "tool_result": {"outcome": "not_found"}}), [])
        self.assertEqual(build_blocks({}), [])


class StreamEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(ApiTestConfig).test_client()

    def events(self, response):
        parsed = []
        for chunk in response.get_data(as_text=True).strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in chunk.splitlines())
            parsed.append((lines["event"], json.loads(lines["data"])))
        return parsed

    def test_status_events_then_result(self):
        def fake_stream(query, session_id, customer_id=None):
            yield "node", "classify_intent"
            yield "node", "order_tracking_node"
            yield "result", {"response": "Shipped.", "final_intent": "order_tracking", "workflow_status": "completed"}
        with patch("app.api.routes.stream_graph", side_effect=fake_stream):
            response = self.client.post("/api/chat/stream", json={"session_id": "s1", "message": "where is ORD-1001"})
            # The body streams lazily, so read it while the patch is active.
            events = self.events(response)
        self.assertEqual(response.mimetype, "text/event-stream")
        self.assertEqual([e for e, _ in events], ["status", "status", "status", "result"])
        self.assertEqual(events[2][1]["label"], "Checking your order")
        self.assertEqual(events[-1][1]["answer"], "Shipped.")

    def test_failure_ends_with_error_event(self):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
            yield
        with patch("app.api.routes.stream_graph", side_effect=broken):
            response = self.client.post("/api/chat/stream", json={"session_id": "s1", "message": "hi"})
            body = response.get_data(as_text=True)
            events = self.events(response)
        self.assertEqual(events[-1][0], "error")
        self.assertNotIn("boom", body)

    def test_validation_errors_are_plain_json(self):
        response = self.client.post("/api/chat/stream", json={"message": "hi"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_session_id")


if __name__ == "__main__":
    unittest.main()
