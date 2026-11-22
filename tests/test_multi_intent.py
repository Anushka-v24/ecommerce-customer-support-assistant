import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes, workflow
from app.graph.workflow import build_graph, invoke_graph
from app.services.multi_intent import plan_requests, split_request
from tests.test_graph import classifier_result

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()


def keyword_intent(text):
    """Stand-in for the classifier: intent by keyword, confident unless the text is vague."""
    lowered = text.lower()
    for word, intent in (("cancel", "order_management"), ("where", "order_tracking"), ("track", "order_tracking"),
                         ("stock", "product_inquiry"), ("hello", "greeting"), ("कहाँ", "order_tracking")):
        if word in lowered:
            return intent
    return None


def predict(text):
    intent = keyword_intent(text)
    return (intent, 97.0) if intent else ("issue_support", 40.0)


def classify(text, history=None):
    intent = keyword_intent(text) or "issue_support"
    return classifier_result(intent, intent)


class SplitTests(unittest.TestCase):
    def test_splits_on_english_hinglish_and_hindi_connectors(self):
        self.assertEqual(split_request("where is ORD-1001 and cancel ORD-1002"), ["where is ORD-1001", "cancel ORD-1002"])
        self.assertEqual(split_request("ORD-1001 kahan hai aur ORD-1002 cancel karo"), ["ORD-1001 kahan hai", "ORD-1002 cancel karo"])
        self.assertEqual(split_request("ORD-1001 कहाँ है और ORD-1002 कैंसिल करो"), ["ORD-1001 कहाँ है", "ORD-1002 कैंसिल करो"])

    def test_second_half_inherits_the_order_id(self):
        self.assertEqual(split_request("track ORD-1002 and then cancel it"), ["track ORD-1002", "cancel it ORD-1002"])

    def test_at_most_two_parts(self):
        self.assertEqual(len(split_request("where is ORD-1001 and cancel ORD-1002 and track ORD-1003")), 2)

    def test_gate_requires_confident_different_actionable_intents(self):
        self.assertIsNotNone(plan_requests("where is ORD-1001 and cancel ORD-1002", predict, 90))
        self.assertIsNone(plan_requests("where is ORD-1001 and where is ORD-1003", predict, 90))  # same intent
        self.assertIsNone(plan_requests("hello there and where is ORD-1001", predict, 90))       # small talk
        self.assertIsNone(plan_requests("black and white shirt please", predict, 90))           # not confident
        self.assertIsNone(plan_requests("where is my order", predict, 90))                      # nothing to split


class MultiRequestFlowTests(unittest.TestCase):
    def setUp(self):
        for patcher in (patch.object(Config, "MULTI_INTENT_ENABLED", True), patch.object(Config, "INTENT_CONFIRMATION_ENABLED", False),
                        patch.object(workflow, "predict_intent", side_effect=predict),
                        patch.object(nodes, "classify_and_verify", side_effect=classify)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = str(uuid4())

    def say(self, text):
        return invoke_graph(text, self.session, self.graph)

    def test_both_requests_answered_in_one_reply(self):
        result = self.say("where is ORD-1003 and do you have Laptop Pro in stock")
        self.assertIn("ORD-1003", result["response"])
        self.assertIn("Laptop Pro", result["response"])
        self.assertIsNone(result.get("queued_request"))

    def test_second_request_waits_for_approval_of_first(self):
        asked = self.say("cancel ORD-1002 and where is ORD-1001")
        self.assertEqual(asked["workflow_status"], "awaiting_approval")
        self.assertIn("other request", asked["response"])
        done = self.say("yes")
        self.assertIn("ORD-1002 has been cancelled", done["response"])
        self.assertIn("ORD-1001", done["response"].split("cancelled")[-1])
        self.assertIsNone(done.get("queued_request"))

    def test_single_request_is_unchanged(self):
        result = self.say("where is ORD-1001")
        self.assertEqual(result["selected_route"], "order_tracking")


if __name__ == "__main__":
    unittest.main()
