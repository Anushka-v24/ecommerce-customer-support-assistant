"""Questions about how cancelling works are answered; requests to cancel an order still start a cancellation."""

import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.api.blocks import build_blocks
from app.config import Config
from app.graph import nodes
from app.graph.nodes import CANCEL_FOLLOWUP_ANSWER
from app.graph.routing import _is_order_cancellation_request, route_classifier_result
from app.graph.workflow import build_graph, invoke_graph

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()

ACCEPTED = "ACCEPTED_BY_CLASSIFIER"


def state(query, intent="order_management"):
    return {"query": query, "final_intent": intent, "verifier_status": ACCEPTED}


class CancelRoutingTests(unittest.TestCase):
    def test_how_to_and_policy_questions_go_to_the_help_documents(self):
        for text in (
            "how to cancel order", "How do I cancel my order?", "how can I cancel an order",
            "can I cancel an order after it ships", "is it possible to cancel an order",
            "what are the steps to cancel", "order cancel kaise karte hain",
        ):
            self.assertFalse(_is_order_cancellation_request(text), text)
            self.assertEqual(route_classifier_result(state(text)), "store_information", text)

    def test_requests_to_cancel_still_start_a_cancellation(self):
        for text in ("cancel ORD-1002", "I want to cancel my order", "please cancel my order", "how can I cancel ORD-1002", "cancel it"):
            self.assertTrue(_is_order_cancellation_request(text), text)
            self.assertEqual(route_classifier_result(state(text)), "order_cancellation", text)

    def test_fee_questions_are_still_not_cancellations(self):
        self.assertFalse(_is_order_cancellation_request("is there a cancellation fee"))


class StaleSourcesTests(unittest.TestCase):
    def test_sources_from_an_earlier_answer_are_not_shown_on_a_later_reply(self):
        later = {"selected_route": "confirm:issue_support", "tool_result": {"outcome": "awaiting_intent_confirmation"},
                 "sources": ["help/order_changes_cancellation_en.md"]}
        self.assertFalse(any(block["type"] == "sources" for block in build_blocks(later)))

    def test_a_help_answer_still_cites_its_sources(self):
        answer = {"selected_route": "store_information", "tool_result": {"status": "answered"},
                  "sources": ["help/order_changes_cancellation_en.md"]}
        self.assertTrue(any(block["type"] == "sources" for block in build_blocks(answer)))


def classified(intent):
    return {
        "predicted_intent": intent, "confidence_score": 98.0, "final_intent": intent,
        "verifier_status": ACCEPTED, "routing": "DIRECT_CLASSIFIER_OUTPUT", "latency_ms": 1.0, "error": None,
    }


def fake_classifier(query, history=None):
    text = query.casefold()
    if "cancel" in text:
        return classified("order_management")
    if "speaker" in text:
        return classified("product_inquiry")
    return classified("issue_support")


class CancelFollowUpTests(unittest.TestCase):
    """'But I have not received the order', said while talking about cancelling, is about cancelling."""

    def setUp(self):
        for name, value in (("INTENT_CONFIRMATION_ENABLED", True), ("TURN_LOG_ENABLED", False)):
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, kwargs in (
            ("classify_and_verify", {"side_effect": fake_classifier}),
            ("get_rag_service", {}),
            ("run_agentic_rag", {"return_value": {"rag_status": "answered", "rag_answer": "Steps to cancel.", "sources": ["help/order_changes_cancellation_en.md"], "rag_documents": []}}),
        ):
            patcher = patch.object(nodes, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = f"cancel-ctx-{uuid4()}"

    def say(self, text):
        return invoke_graph(text, self.session, graph=self.graph)

    def test_after_a_how_to_cancel_question(self):
        self.say("how to cancel order")
        result = self.say("but i have not recieved the order")
        self.assertEqual(result["response"], CANCEL_FOLLOWUP_ANSWER)
        self.assertEqual(result["selected_route"], "cancel_followup")
        self.assertNotIn("missing delivery", result["response"])

    def test_after_a_request_to_cancel(self):
        self.say("I want to cancel my order")
        result = self.say("but i have not received the order")
        self.assertIn(CANCEL_FOLLOWUP_ANSWER, result["response"])

    def test_without_cancellation_talk_it_is_a_missing_delivery_report(self):
        result = self.say("but i have not recieved the order")
        self.assertNotEqual(result["selected_route"], "cancel_followup")
        self.assertNotIn(CANCEL_FOLLOWUP_ANSWER, result["response"])

    def test_the_context_ends_when_the_subject_changes(self):
        self.say("how to cancel order")
        self.say("do you have a speaker")
        result = self.say("but i have not recieved the order")
        self.assertNotEqual(result["selected_route"], "cancel_followup")
        self.assertNotIn(CANCEL_FOLLOWUP_ANSWER, result["response"])

    def test_an_order_id_in_the_message_goes_the_normal_way(self):
        self.say("how to cancel order")
        result = self.say("ORD-1010 has not arrived")
        self.assertNotEqual(result["selected_route"], "cancel_followup")


if __name__ == "__main__":
    unittest.main()
