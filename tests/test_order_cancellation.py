"""Order cancellation: only unshipped orders, and only after an explicit yes."""

import unittest
from uuid import uuid4
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()


def order_management():
    return {
        "predicted_intent": "order_management", "confidence_score": 98.0,
        "final_intent": "order_management", "verifier_status": "ACCEPTED_BY_CLASSIFIER",
        "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
    }


class OrderCancellationTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(InMemorySaver())
        self.session = str(uuid4())
        classifier = patch.object(nodes, "classify_and_verify", return_value=order_management())
        self.classifier = classifier.start()
        self.addCleanup(classifier.stop)

    def say(self, message):
        return invoke_graph(message, self.session, self.graph)

    def test_processing_order_is_cancelled_only_after_yes(self):
        asked = self.say("please cancel my order ORD-1002")
        self.assertEqual(asked["selected_route"], "order_cancellation")
        self.assertEqual(asked["tool_result"]["outcome"], "awaiting_confirmation")
        self.assertFalse(asked["tool_result"]["executed"])
        self.assertIn("Reply 'yes'", asked["response"])

        done = self.say("yes")
        self.assertEqual(done["tool_result"]["outcome"], "cancelled")
        self.assertTrue(done["tool_result"]["executed"])
        self.assertIn("ORD-1002 has been cancelled", done["response"])
        self.assertFalse(done["confirmation_required"])
        self.assertIsNone(done["pending_workflow"])

    def test_no_keeps_the_order(self):
        self.say("cancel ORD-1004")
        kept = self.say("no")
        self.assertEqual(kept["tool_result"], {"status": "cancelled", "executed": False})
        self.assertIn("No operation was performed", kept["response"])

    def test_shipped_order_cannot_be_cancelled(self):
        result = self.say("I want to cancel ORD-1001")
        self.assertEqual(result["tool_result"]["outcome"], "not_cancellable")
        self.assertIn("already shipped", result["response"])
        self.assertFalse(result.get("confirmation_required"))

    def test_missing_order_id_is_asked_for_then_confirmed(self):
        asked = self.say("मेरा ऑर्डर कैंसिल कर दो")
        self.assertEqual(asked["tool_result"]["outcome"], "missing_order_id")
        confirm = self.say("ORD-1002")
        self.assertEqual(confirm["tool_result"]["outcome"], "awaiting_confirmation")
        done = self.say("हाँ")
        self.assertEqual(done["tool_result"]["outcome"], "cancelled")

    def test_fee_question_and_item_change_stay_unsupported(self):
        for message in ("is there a cancellation fee?", "change the size in my order"):
            with self.subTest(message=message):
                result = invoke_graph(message, str(uuid4()), self.graph)
                self.assertEqual(result["selected_route"], "unsupported:order_management")


if __name__ == "__main__":
    unittest.main()
