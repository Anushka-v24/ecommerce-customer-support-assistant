"""A bare order number such as 1010 is read as ORD-1010."""

import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services.order_ids import with_order_prefix

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()


class OrderNumberReadingTests(unittest.TestCase):
    def test_bare_numbers_become_order_tracking_requests(self):
        for text in ("1010", "#1010", "1010?", " 1010 "):
            self.assertEqual(with_order_prefix(text), "track order ORD-1010", text)

    def test_numbers_inside_order_messages_get_the_prefix(self):
        self.assertEqual(with_order_prefix("order 1010"), "order ORD-1010")
        self.assertEqual(with_order_prefix("track 1010"), "track ORD-1010")
        self.assertEqual(with_order_prefix("where is #1010 now?"), "where is ORD-1010 now?")
        self.assertEqual(with_order_prefix("refund for 1010"), "refund for ORD-1010")
        self.assertEqual(with_order_prefix("1010 order status"), "ORD-1010 order status")

    def test_messages_that_are_not_about_an_order_number_are_left_alone(self):
        for text in (
            "ORD-1010", "ord 1010", "hello", "my pin is 4000", "return policy 2026", "what is the 2024 sale",
            "I paid Rs 1500 for a refund", "refund of ₹1500", "pay $2000 refund", "call me on 9876543210",
            "track 1010 and 1011", "price 1010.50 order", "",
        ):
            self.assertIsNone(with_order_prefix(text), text)

    def test_a_year_needs_a_clear_order_word(self):
        self.assertIsNone(with_order_prefix("refund 2026"))
        self.assertEqual(with_order_prefix("order 2026"), "order ORD-2026")


class OrderNumberThroughTheGraphTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Config, "TURN_LOG_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.seen = []

        def classify(query, history=None):
            self.seen.append(query)
            return {
                "predicted_intent": "order_tracking", "confidence_score": 98.0, "final_intent": "order_tracking",
                "verifier_status": "ACCEPTED_BY_CLASSIFIER", "routing": "DIRECT_CLASSIFIER_OUTPUT",
                "latency_ms": 1.0, "error": None,
            }

        classifier = patch.object(nodes, "classify_and_verify", side_effect=classify)
        classifier.start()
        self.addCleanup(classifier.stop)
        self.graph = build_graph(InMemorySaver())

    def test_every_way_of_writing_the_number_finds_the_order(self):
        for text in ("1010", "#1010", "order 1010", "track 1010", "where is 1010", "status of 1010 please", "1010 order status"):
            result = invoke_graph(text, f"num-{uuid4()}", graph=self.graph)
            self.assertIn("ORD-1010", result["response"], text)
            self.assertIn("Processing", result["response"], text)
            self.assertNotIn("Please provide your order ID", result["response"], text)
        self.assertTrue(all("ORD-1010" in query for query in self.seen))

    def test_an_unknown_number_gets_a_not_found_answer(self):
        result = invoke_graph("7777", f"num-{uuid4()}", graph=self.graph)
        self.assertIn("ORD-7777", result["response"])
        self.assertNotIn("Please provide your order ID", result["response"])


if __name__ == "__main__":
    unittest.main()
