import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.config import Config
from app.graph import nodes
from app.graph.workflow import build_graph, invoke_graph
from app.services import case_repository, frustration
from tests.test_graph import classifier_result

patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()
patch.object(Config, "REPLY_TRANSLATION_ENABLED", False).start()
patch.object(Config, "MULTI_INTENT_ENABLED", False).start()


class EscalationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for target, name, value in [
            (Config, "ESCALATION_ENABLED", True), (Config, "INTENT_CONFIRMATION_ENABLED", False),
            (Config, "TURN_LOG_ENABLED", False),
            (case_repository, "DEFAULT_DB_PATH", Path(tmp.name) / "support.sqlite3"),
            # Offline: the word list alone decides, no sentiment model.
            (frustration, "sentiment_probabilities", lambda text: None),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.graph = build_graph(InMemorySaver())
        self.session = str(uuid4())

    def say(self, text, intent="order_tracking"):
        with patch.object(nodes, "classify_and_verify", return_value=classifier_result(intent, intent)):
            return invoke_graph(text, self.session, self.graph)

    def test_calm_customer_is_not_escalated(self):
        result = self.say("where is my order ORD-1001?")
        self.assertEqual(result["selected_route"], "order_tracking")
        self.assertFalse(result["frustration"]["escalate"])

    def test_angry_customer_gets_a_person_with_summary_ticket(self):
        self.say("where is ORD-1001")
        result = self.say("WORST SERVICE EVER, this is bakwas, I'm fed up!!! where is ORD-1001, call me 9876543210")
        self.assertEqual(result["selected_route"], "escalation")
        self.assertTrue(result["escalated"])
        ticket = result["ticket"]
        self.assertEqual(ticket["category"], "escalation")
        self.assertEqual(ticket["priority"], "high")
        self.assertIn("ORD-1001", ticket["description"])
        self.assertNotIn("9876543210", ticket["description"])

    def test_escalation_happens_once_per_conversation(self):
        angry = "this is the worst service, fed up, I will go to consumer court!!!"
        self.assertEqual(self.say(angry)["selected_route"], "escalation")
        self.assertEqual(self.say(angry)["selected_route"], "order_tracking")


if __name__ == "__main__":
    unittest.main()
