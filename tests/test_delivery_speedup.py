"""Delivery speed-up requests, and order IDs typed loosely when the assistant asks for one."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.routing import is_delivery_speedup_request
from app.graph.workflow import build_graph, invoke_graph
from app.services import case_repository
from app.services.ticket_service import list_tickets

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "CHECKPOINT_BACKEND", "memory").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "ESCALATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()


def classified(intent):
    return {
        "predicted_intent": intent, "confidence_score": 98.0, "final_intent": intent,
        "verifier_status": "ACCEPTED_BY_CLASSIFIER", "routing": "DIRECT_CLASSIFIER_OUTPUT",
        "latency_ms": 1.0, "error": None,
    }


def fake_classifier(query, history=None):
    text = query.casefold()
    if "where is my order" in text or "status" in text:
        return classified("order_tracking")
    if "agent" in text or "talk" in text:
        return classified("customer_care_support")
    return classified("order_tracking")


class SpeedupDetectionTests(unittest.TestCase):
    def test_delivery_speedup_phrases(self):
        for text in (
            "please talk to delivery agent to send it sooner",
            "can you ask the courier to hurry",
            "I need my package faster",
            "please deliver my order urgently",
        ):
            self.assertTrue(is_delivery_speedup_request(text), text)

    def test_other_requests_are_not_speedups(self):
        for text in ("I want to talk to a person", "where is my refund", "return this urgently", "track ORD-1001", "hello"):
            self.assertFalse(is_delivery_speedup_request(text), text)


class SpeedupFlowTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("INTENT_CONFIRMATION_ENABLED", True), ("TURN_LOG_ENABLED", False)):
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_patch = patch.object(case_repository, "DEFAULT_DB_PATH", Path(temp_dir.name) / "support.sqlite3")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        classifier = patch.object(nodes, "classify_and_verify", side_effect=fake_classifier)
        classifier.start()
        self.addCleanup(classifier.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = f"speedup-{uuid4()}"

    def say(self, text, customer_id=None):
        return invoke_graph(text, self.session, graph=self.graph, customer_id=customer_id)

    def test_delivery_agent_request_is_honest_and_ticketed_as_speedup(self):
        asked = self.say("please talk to delivery agent to send it sooner", customer_id="CUST-002")
        self.assertEqual(asked["selected_route"], "confirm:customer_care")
        self.assertIn("can't reach the delivery agent live", asked["response"])
        self.assertNotIn("support team", asked["response"])
        done = self.say("yes")
        self.assertEqual(done["ticket"]["category"], "delivery_speedup")
        self.assertEqual(done["ticket"]["priority"], "high")
        self.assertIn("speed up your order", done["response"])
        self.assertEqual(len(list_tickets(customer_id="CUST-002")), 1)

    def test_speedup_names_the_order_when_given(self):
        asked = self.say("ask the delivery agent to bring ORD-1002 faster")
        self.assertIn("speed up your order for order ORD-1002", asked["response"])

    def test_plain_person_request_keeps_the_generic_prompt(self):
        asked = self.say("I want to talk to someone")
        self.assertIn("someone from our support team", asked["response"])


class LooseOrderIdTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Config, "TURN_LOG_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (("INTENT_CONFIRMATION_ENABLED", False),):
            p = patch.object(Config, name, value)
            p.start()
            self.addCleanup(p.stop)
        classifier = patch.object(nodes, "classify_and_verify", side_effect=lambda q, history=None: classified("order_tracking"))
        classifier.start()
        self.addCleanup(classifier.stop)
        self.graph = build_graph(InMemorySaver())

    def ask_then_reply(self, reply):
        session = f"loose-{uuid4()}"
        first = invoke_graph("where is my order", session, graph=self.graph)
        self.assertIn("order ID", first["response"])
        return invoke_graph(reply, session, graph=self.graph)

    def test_every_spelling_of_an_order_id_is_accepted(self):
        for reply in ("ORD-1002", "ORD 1002", "ord1002", "1002", "#1002", "order id is 1002"):
            result = self.ask_then_reply(reply)
            self.assertNotIn("could not identify that product", result["response"], reply)
            self.assertNotIn("Please provide your order ID", result["response"], reply)
            self.assertIn("ORD-1002", str(result.get("tool_result")) + result["response"], reply)


if __name__ == "__main__":
    unittest.main()
