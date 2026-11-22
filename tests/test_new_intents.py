"""Greeting, goodbye, recommendation, warranty, and exchange routes added with the 20-label model."""

import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services import case_repository


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# These tests cover the exchange workflow itself; test_intent_confirmation covers its confirmation step.
patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False).start()
# Exchanges raise tickets; keep them out of the developer's data/runtime database.
_TICKET_DIR = tempfile.TemporaryDirectory()
patch.object(case_repository, "DEFAULT_DB_PATH", Path(_TICKET_DIR.name) / "support.sqlite3").start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()


def classified(intent):
    return {
        "predicted_intent": intent, "confidence_score": 98.0, "final_intent": intent,
        "verifier_status": "ACCEPTED_BY_CLASSIFIER", "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0, "error": None,
    }


class NewIntentRouteTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(InMemorySaver())

    def ask(self, intent, *messages):
        session = str(uuid4())
        with patch.object(nodes, "classify_and_verify", return_value=classified(intent)):
            results = [invoke_graph(message, session, self.graph) for message in messages]
        return results[-1]

    def test_greeting_and_goodbye(self):
        hello = self.ask("greeting", "namaste")
        self.assertEqual(hello["selected_route"], "greeting")
        self.assertIn("How can I help", hello["response"])
        bye = self.ask("goodbye_thanks", "thanks, that's all")
        self.assertEqual(bye["selected_route"], "goodbye")
        self.assertIn("You're welcome", bye["response"])

    def test_recommendation_focuses_on_mentioned_category_and_skips_out_of_stock(self):
        laptops = self.ask("product_recommendation", "which laptop should I buy for college?")
        self.assertEqual(laptops["selected_route"], "product_recommendation")
        self.assertEqual(laptops["tool_result"]["products"], ["Laptop Pro"])
        general = self.ask("product_recommendation", "what would you recommend?")
        self.assertNotIn("Galaxy S21", general["tool_result"]["products"])

    def test_warranty_for_named_product_and_general_terms(self):
        named = self.ask("warranty_inquiry", "Laptop Pro warranty kitni hai?")
        self.assertIn("Laptop Pro has a 24-month warranty", named["response"])
        general = self.ask("warranty_inquiry", "how do I claim warranty?")
        self.assertIn("12-month warranty", general["response"])

    def test_exchange_asks_for_order_then_prepares_for_delivered_order(self):
        result = self.ask("exchange_request", "I want a bigger size", "ORD-1003")
        self.assertEqual(result["selected_route"], "exchange")
        self.assertEqual(result["tool_result"]["outcome"], "exchange_prepared")
        self.assertIn("Requested change: I want a bigger size", result["response"])

    def test_exchange_before_delivery_offers_the_right_alternative(self):
        processing = self.ask("exchange_request", "exchange ORD-1002 for blue")
        self.assertEqual(processing["tool_result"]["outcome"], "not_delivered")
        self.assertIn("cancel it", processing["response"])
        shipped = self.ask("exchange_request", "exchange ORD-1001")
        self.assertIn("wait for it to arrive", shipped["response"])


if __name__ == "__main__":
    unittest.main()
