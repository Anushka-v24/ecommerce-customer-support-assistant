"""End-to-end graph tests for Phase 4B business routes (classifier is mocked)."""

import unittest
from uuid import uuid4
from unittest.mock import patch

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()


def classifier_result(intent=None, status="ACCEPTED_BY_CLASSIFIER"):
    final_intent = intent if status == "ACCEPTED_BY_CLASSIFIER" else None
    return {
        "predicted_intent": intent,
        "confidence_score": 99.0,
        "final_intent": final_intent,
        "verifier_status": status,
        "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0,
        "error": None,
    }


class BusinessGraphTests(unittest.TestCase):
    def invoke_as(self, query, intent, status="ACCEPTED_BY_CLASSIFIER"):
        graph = build_graph()
        with patch.object(nodes, "classify_and_verify", return_value=classifier_result(intent, status)):
            return invoke_graph(query, str(uuid4()), graph)

    def test_order_route_looks_up_supplied_id(self):
        result = self.invoke_as("Where is order ORD-1001?", "order_tracking")
        self.assertEqual(result["selected_route"], "order_tracking")
        self.assertEqual(result["tool_result"]["outcome"], "found")
        self.assertEqual(result["tool_result"]["order_status"], "Shipped")
        self.assertIn("demonstration data", result["response"].lower())

    def test_missing_order_id_requests_id_without_lookup(self):
        with patch.object(nodes, "lookup_order_status", wraps=nodes.lookup_order_status) as tool:
            result = self.invoke_as("Where is my order?", "order_tracking")
        self.assertEqual(result["tool_result"]["outcome"], "missing_order_id")
        self.assertFalse(result["tool_result"]["lookup_performed"])
        self.assertIn("provide your order ID", result["response"])
        tool.assert_called_once_with(None)

    def test_unknown_order_id_does_not_claim_success(self):
        result = self.invoke_as("Track order ORD-9999", "order_tracking")
        self.assertEqual(result["tool_result"]["outcome"], "not_found")
        self.assertNotIn("has status", result["response"])

    def test_product_route_returns_sample_stock_for_identified_product(self):
        result = self.invoke_as("Stock for Laptop Pro", "product_inquiry")
        self.assertEqual(result["selected_route"], "product_inquiry")
        self.assertEqual(result["tool_result"]["outcome"], "found")
        self.assertEqual(result["tool_result"]["stock"], 8)
        self.assertIn("not live inventory", result["response"])

    def test_ambiguous_product_requests_clarification(self):
        result = self.invoke_as("Laptop Pro or Galaxy S21", "product_inquiry")
        self.assertEqual(result["tool_result"]["outcome"], "ambiguous")
        self.assertFalse(result["tool_result"]["lookup_performed"])
        self.assertIn("Which one did you mean", result["response"])

    def test_unidentified_product_requests_name_or_id(self):
        result = self.invoke_as("Do you sell a teleporting unicorn?", "product_inquiry")
        self.assertEqual(result["tool_result"]["outcome"], "not_found")
        self.assertIn("exact name or ID", result["response"])
        self.assertNotIn("in stock", result["response"])

    def test_unrelated_intents_do_not_execute_order_or_product_tools(self):
        cases = [
            ("billing_and_payment", "ACCEPTED_BY_CLASSIFIER", "payment"),
            ("add_to_cart", "ACCEPTED_BY_CLASSIFIER", "unsupported:add_to_cart"),
            (None, "OUT_OF_SCOPE", "out_of_scope"),
            (None, "UNKNOWN_ECOMMERCE_INTENT", "unknown_intent"),
        ]
        with patch.object(nodes, "lookup_order_status") as order_tool, patch.object(
            nodes, "check_product_stock"
        ) as product_tool:
            for intent, status, expected_route in cases:
                with self.subTest(intent=intent, status=status):
                    graph = build_graph()
                    classifier = classifier_result(intent, status)
                    with patch.object(nodes, "classify_and_verify", return_value=classifier):
                        result = invoke_graph("unrelated query", str(uuid4()), graph)
                    self.assertEqual(result["selected_route"], expected_route)
            order_tool.assert_not_called()
            product_tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()

